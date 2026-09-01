"""
Subprocess lifecycle management for AWM sub-environments.
Each AWM scenario is a self-contained FastAPI application. This module handles:
- Patching generated code (DB path, FastApiMCP injection)
- Starting / stopping subprocess on a random port
- Persistent MCP connection for efficient tool calls

TRUST BOUNDARY:
    The ``full_code`` argument to ``ScenarioProcess.start`` originates from
    the curated HuggingFace dataset (Snowflake/AgentWorldModel-1K) and is
    treated as TRUSTED.

    This is *intentional*: scenario code is the data the env exists to serve,
    and the AWM design assumes it is benign. The container is the outer
    isolation boundary; per-subprocess sandboxing is not applied here.

    Verifier code (``_verifier_runner.py``) follows a different model with
    an explicit sandbox; that asymmetry is by design.

    If the dataset trust assumption changes, this module is the place that
    must be hardened (e.g. per-subprocess unshare/seccomp/non-root uid).
"""

import asyncio
import contextlib
import logging
import os
import re
import signal
import socket
import subprocess
import sys
import textwrap
import time
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from .config import (
    MAX_PORT_RETRIES,
    READY_POLL_INTERVAL,
    READY_TIMEOUT,
    RETRY_READY_TIMEOUT,
)


logger = logging.getLogger(__name__)


def _get_random_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        return s.getsockname()[1]


def _fix_datetime_response_fields(full_code: str) -> str:
    """Fix generated codegen bug: response models declare ORM ``DateTime``
    columns as ``str``, so ``model_validate(orm_row)`` raises ValidationError
    (datetime object is not a str) and every read of a populated table
    returns a bare 500.

    Rewrite those fields to a private datetime alias.  A private alias is
    important because generated scenarios inconsistently use both
    ``import datetime`` and ``from datetime import datetime``; annotating with
    the public name can therefore resolve to the datetime *module* instead of
    the datetime class. Only models with
    ``from_attributes=True`` (i.e. ORM-validated response models) are touched;
    request models keep ``str`` input fields as generated.
    """
    dt_cols: set[str] = set()
    nullable_dt_cols: set[str] = set()
    for cls in full_code.split("\nclass "):
        if "__tablename__" in cls:
            for match in re.finditer(
                r"(\w+)\s*=\s*Column\(([^)]*DateTime[^)]*)\)", cls
            ):
                column_name, arguments = match.groups()
                dt_cols.add(column_name)
                if "nullable=False" not in arguments:
                    nullable_dt_cols.add(column_name)
    if not dt_cols:
        return full_code

    # Split into top-level class blocks; patch only response models.
    parts = re.split(r"(?m)^class ", full_code)
    for i, part in enumerate(parts):
        if i == 0 or "from_attributes=True" not in part:
            continue
        for col in dt_cols:
            part = re.sub(
                rf"(?m)^(\s+{col}\s*:\s*)Optional\[str\]",
                r"\1Optional[_AWMDatetime]",
                part,
            )
            replacement = (
                r"\1Optional[_AWMDatetime]"
                if col in nullable_dt_cols
                else r"\1_AWMDatetime"
            )
            part = re.sub(rf"(?m)^(\s+{col}\s*:\s*)str\b", replacement, part)
        parts[i] = part

    patched = "class ".join(parts)
    if "from datetime import datetime as _AWMDatetime" not in patched:
        patched = "from datetime import datetime as _AWMDatetime\n" + patched
    return patched


def _fix_generated_code_errors(full_code: str) -> str:
    """Apply narrow compatibility fixes for recurring AWM codegen defects.

    These transformations repair generated Python at launch time; the source
    dataset and trajectory files remain untouched.  Keep replacements generic
    only when the intended correction is unambiguous.
    """

    # Generated handlers frequently close a session and only then serialize an
    # ORM object.  SQLAlchemy expires attributes on commit by default, causing
    # DetachedInstanceError after the close.  Retaining loaded values is safe
    # for these short-lived, per-request sessions.
    def _sessionmaker_repl(match: re.Match[str]) -> str:
        arguments = match.group(1)
        if "expire_on_commit" in arguments:
            return match.group(0)
        return f"sessionmaker(expire_on_commit=False, {arguments})"

    full_code = re.sub(r"sessionmaker\(([^()\n]*)\)", _sessionmaker_repl, full_code)

    # ORM response models should accept both their Python field names and
    # generated aliases.  The generator commonly configured only the output
    # alias, which makes explicit model construction and ORM validation fail.
    full_code = full_code.replace(
        "ConfigDict(from_attributes=True)",
        "ConfigDict(from_attributes=True, populate_by_name=True)",
    )
    full_code = re.sub(
        r'(?<!validation_alias=")serialization_alias="([^"]+)"',
        r'validation_alias="\1", serialization_alias="\1"',
        full_code,
    )
    class_parts = re.split(r"(?m)^class ", full_code)
    for index, class_part in enumerate(class_parts):
        if index == 0:
            continue
        declaration = class_part.split("\n", 1)[0]
        class_name = declaration.split("(", 1)[0].strip()
        if class_name.endswith(("Request", "Payload", "Body", "In")):
            class_part = re.sub(r'validation_alias="[^"]+",\s*', "", class_part)
        if "validation_alias=" in class_part and "model_config =" not in class_part:
            declaration, separator, body = class_part.partition("\n")
            class_part = (
                declaration
                + separator
                + "    model_config = ConfigDict(populate_by_name=True)\n"
                + body
            )
        if "from_attributes=True" in class_part:
            class_part = re.sub(
                r"(?m)^(\s+metadata\s*:\s*Optional\[str\]\s*=\s*Field\(None,\s*)",
                r'\1validation_alias="metadata_", ',
                class_part,
            )
            if re.search(r"(?m)^\s+metadata_\s*:", class_part):
                class_part = class_part.replace(
                    'validation_alias="metadata", ', ""
                )
        class_parts[index] = class_part
    full_code = "class ".join(class_parts)

    # SQLAlchemy API spelling errors emitted by the generator.
    full_code = full_code.replace(".nullsLast()", ".nullslast()")
    full_code = full_code.replace(".in__(", ".in_(")
    if "Integer.__clause_element__().count()" in full_code:
        full_code = "from sqlalchemy import func as _AWMFunc\n" + full_code
        full_code = full_code.replace(
            "Integer.__clause_element__().count()", "_AWMFunc.count()"
        )
    if re.search(r"\bdatetime\.utc\b", full_code):
        full_code = "from datetime import timezone as _AWMTimezone\n" + full_code
        full_code = re.sub(
            r"\bdatetime\.utc\b", "_AWMTimezone.utc", full_code
        )

    # practice_management_1 declares Appointment.invoices against
    # Invoice.encounter even though invoices only carry encounter_id.  The
    # invalid relationship prevents SQLAlchemy from configuring *any* mapper
    # in the scenario.  Encounter.invoices, which is the valid relationship,
    # is intentionally preserved.
    full_code = full_code.replace(
        '    encounters = relationship("Encounter", back_populates="appointment")\n'
        '    invoices = relationship("Invoice", back_populates="encounter")\n',
        '    encounters = relationship("Encounter", back_populates="appointment")\n',
    )

    # Response schema corrections where generated JSON types are narrower
    # than the actual stored documents.
    full_code = full_code.replace(
        "metadata_: Optional[Dict[str, str]] = Field(None, description=\"JSON metadata\"",
        "metadata_: Optional[Dict[str, object]] = Field(None, description=\"JSON metadata\"",
    )
    full_code = full_code.replace(
        "specifications: Optional[Dict[str, float]] = Field(",
        "specifications: Optional[Dict[str, object]] = Field(",
    )
    full_code = full_code.replace(
        "latitude: Optional[int] = Field(", "latitude: Optional[float] = Field("
    )
    full_code = full_code.replace(
        "longitude: Optional[int] = Field(", "longitude: Optional[float] = Field("
    )
    full_code = full_code.replace(
        "skills: Optional[Dict[str, object]] = Field(None, description=\"Skills metadata JSON\"",
        "skills: Optional[object] = Field(None, description=\"Skills metadata JSON\"",
    )
    full_code = full_code.replace(
        "assigned_projects: Optional[Dict[str, object]] = Field(None, description=\"Assigned projects metadata JSON\"",
        "assigned_projects: Optional[object] = Field(None, description=\"Assigned projects metadata JSON\"",
    )
    full_code = full_code.replace(
        "current_plan: Optional[Dict[str, Optional[str]]] = Field(",
        "current_plan: Optional[Dict[str, object]] = Field(",
    )
    full_code = full_code.replace(
        "target_plan: Optional[Dict[str, Optional[str]]] = Field(",
        "target_plan: Optional[Dict[str, object]] = Field(",
    )
    full_code = full_code.replace(
        "topics_json: Optional[Dict[str, object]] = Field(",
        "topics_json: Optional[object] = Field(",
    )
    full_code = full_code.replace(
        "equipment_flags_json: Optional[Dict[str, object]] = Field(",
        "equipment_flags_json: Optional[object] = Field(",
    )

    # Straightforward generated symbol/field-name typos.
    full_code = full_code.replace(
        "UpdateListingStatusOrderModel", "UpdateListingStatusListingModel"
    )
    full_code = full_code.replace("article_id=existing.tag_id", "tag_id=existing.tag_id")
    full_code = full_code.replace("article_id=link.tag_id", "tag_id=link.tag_id")

    # Repeated joins against the same association table create ambiguous SQL.
    full_code = full_code.replace(
        "q = q.join(TicketTag, TicketTag.ticket_id == Ticket.id).filter(TicketTag.tag == tag_val)",
        "q = q.filter(Ticket.id.in_(session.query(TicketTag.ticket_id).filter(TicketTag.tag == tag_val)))",
    )
    full_code = full_code.replace(
        "query = query.join(UserSkill, (UserSkill.user_id == User.id) & (UserSkill.skill_id == skill_id))",
        "query = query.filter(User.id.in_(session.query(UserSkill.user_id).filter(UserSkill.skill_id == skill_id)))",
    )
    full_code = full_code.replace(
        "cp_query = cp_query.order_by(desc(CareerProfile.average_rating_per_user.nullslast()))",
        "cp_query = cp_query.order_by(CareerProfile.average_rating_per_user.desc().nullslast())",
    )

    # A generated list comprehension already extracts integer IDs and then
    # incorrectly tries to unpack each integer a second time.
    full_code = full_code.replace(
        "ids_list = [i for (i,) in ts_ids]", "ids_list = list(ts_ids)"
    )

    # The payments schema has no company_id column.  Remove the stray ORM
    # mapping and constructor argument so reads and writes match the DB.
    full_code = full_code.replace(
        '    updated_at = Column(DateTime, default=datetime.utcnow)\n'
        '    company_id = Column(Integer, ForeignKey("companies.id"), nullable=True)\n',
        '    updated_at = Column(DateTime, default=datetime.utcnow)\n',
    )
    full_code = full_code.replace("        company_id=company_id,\n", "")

    # A bulk replace marks old rows deleted and then re-queries before the
    # delete is flushed, leaving deleted objects in the response list.
    full_code = full_code.replace(
        "        for r in existing:\n"
        "            session.delete(r)\n"
        "    roster_rows: List[TournamentTeamRoster] = []",
        "        for r in existing:\n"
        "            session.delete(r)\n"
        "        session.flush()\n"
        "    roster_rows: List[TournamentTeamRoster] = []",
    )

    # Empty/whitespace and malformed optional JSON values should behave as
    # absent metadata, not crash otherwise valid list/get tools.
    json_list_markers = (
        "json.loads(mi.allergen_flags) if mi.allergen_flags else []",
        "json.loads(b.utility_types_supported) if b.utility_types_supported else []",
        "skills_list = json.loads(profile.skills)",
        "preferences_list = json.loads(profile.preferences)",
    )
    needs_json_list_helper = any(marker in full_code for marker in json_list_markers)
    full_code = full_code.replace(
        json_list_markers[0], "_awm_json_list(mi.allergen_flags)"
    )
    full_code = full_code.replace(
        json_list_markers[1], "_awm_json_list(b.utility_types_supported)"
    )
    full_code = full_code.replace(
        json_list_markers[2], "skills_list = _awm_json_list(profile.skills)"
    )
    full_code = full_code.replace(
        json_list_markers[3],
        "preferences_list = _awm_json_list(profile.preferences)",
    )
    if needs_json_list_helper:
        json_list_helper = '''import json as _awm_json

def _awm_json_list(value):
    if value is None or (isinstance(value, str) and not value.strip()):
        return []
    if isinstance(value, list):
        return value
    try:
        parsed = _awm_json.loads(value)
    except (ValueError, TypeError):
        return [item.strip() for item in str(value).split(",") if item.strip()]
    return parsed if isinstance(parsed, list) else [parsed]

'''
        full_code = json_list_helper + full_code
    full_code = full_code.replace(
        '    if text_value is None or text_value == "":\n'
        "        return None\n"
        "    import json\n\n"
        "    return json.loads(text_value)",
        '    if text_value is None or text_value.strip() == "":\n'
        "        return None\n"
        "    import json\n\n"
        "    try:\n"
        "        return json.loads(text_value)\n"
        "    except (json.JSONDecodeError, TypeError):\n"
        "        return None",
    )

    # Generated time-only values are valid for these APIs but were passed to
    # datetime.fromisoformat, which requires a date component.
    full_code = full_code.replace(
        "start_time_part = datetime.fromisoformat(start_dt_str)",
        'start_time_part = datetime.strptime(start_dt_str, "%H:%M")',
    )
    full_code = full_code.replace(
        "end_time_part = datetime.fromisoformat(end_dt_str)",
        'end_time_part = datetime.strptime(end_dt_str, "%H:%M")',
    )
    full_code = full_code.replace(
        "started_at_value = datetime.fromisoformat(payload.started_at)",
        "started_at_value = (\n"
        "                datetime.combine(datetime.utcnow().date(), datetime.strptime(payload.started_at, \"%H:%M\").time())\n"
        "                if len(payload.started_at) == 5 else datetime.fromisoformat(payload.started_at)\n"
        "            )",
    )
    full_code = full_code.replace(
        "    if value is None:\n        return None\n    return datetime.strptime(value, \"%H:%M\").time()",
        "    if value is None or value.strip() == \"\":\n"
        "        return None\n"
        "    return datetime.strptime(value, \"%H:%M\").time()",
    )

    # Normalize aware query parameters before comparing them with SQLite's
    # timezone-naive DateTime values.
    full_code = full_code.replace(
        "    session = SessionLocal()\n    facilities = session.query(Facility).all()",
        "    session = SessionLocal()\n"
        "    start_datetime = start_datetime.replace(tzinfo=None)\n"
        "    end_datetime = end_datetime.replace(tzinfo=None)\n"
        "    facilities = session.query(Facility).all()",
    )
    full_code = full_code.replace(
        'window_start = datetime.fromisoformat(start_time.replace("Z", "+00:00"))',
        'window_start = datetime.fromisoformat(start_time.replace("Z", "+00:00")).replace(tzinfo=None)',
    )
    full_code = full_code.replace(
        'window_end = datetime.fromisoformat(end_time.replace("Z", "+00:00"))',
        'window_end = datetime.fromisoformat(end_time.replace("Z", "+00:00")).replace(tzinfo=None)',
    )
    full_code = full_code.replace(
        'shift_start = datetime.fromisoformat(shift_start_datetime.replace("Z", "+00:00"))',
        'shift_start = datetime.fromisoformat(shift_start_datetime.replace("Z", "+00:00")).replace(tzinfo=None)',
    )
    full_code = full_code.replace(
        'shift_end = datetime.fromisoformat(shift_end_datetime.replace("Z", "+00:00"))',
        'shift_end = datetime.fromisoformat(shift_end_datetime.replace("Z", "+00:00")).replace(tzinfo=None)',
    )
    full_code = full_code.replace(
        "appt.appointment_datetime < appointment_start",
        "appt.appointment_datetime < appointment_start.replace(tzinfo=None)",
    )
    full_code = full_code.replace(
        "appt.appointment_datetime > appointment_end",
        "appt.appointment_datetime > appointment_end.replace(tzinfo=None)",
    )

    # Ticket SLA helpers return aware UTC while SQLite returns naive values.
    if "def utcnow() -> datetime:" in full_code:
        full_code = full_code.replace("now = utcnow()", "now = utcnow().replace(tzinfo=None)")

    # Missing employees are a normal lookup miss, not a response-model crash.
    full_code = full_code.replace(
        "return EmployeeResponse(personal_details=None)  # type: ignore",
        'return JSONResponse(status_code=404, content={"detail": "Employee not found"})',
    )

    # The search-profile UNION query loses its generated ``uid`` label on
    # some SQLAlchemy versions.  Use a correlated count for sorting and a
    # direct row pass for the response counts.
    old_sort = '''        sub_req = (
            session.query(Connection.requester_id.label("uid"))
            .filter(Connection.status == "accepted")
            .subquery()
        )
        sub_add = (
            session.query(Connection.addressee_id.label("uid"))
            .filter(Connection.status == "accepted")
            .subquery()
        )
        conn_counts = (
            session.query(sub_req.c.uid.label("uid"))
            .union_all(session.query(sub_add.c.uid))
            .subquery()
        )
        count_table = (
            session.query(conn_counts.c.uid, func.count().label("cnt"))
            .group_by(conn_counts.c.uid)
            .subquery()
        )
        q = q.outerjoin(count_table, User.id == count_table.c.uid).order_by(func.coalesce(count_table.c.cnt, 0).desc(), User.created_at.desc())'''
    new_sort = '''        connection_count = (
            session.query(func.count(Connection.id))
            .filter(
                Connection.status == "accepted",
                (Connection.requester_id == User.id) | (Connection.addressee_id == User.id),
            )
            .correlate(User)
            .scalar_subquery()
        )
        q = q.order_by(connection_count.desc(), User.created_at.desc())'''
    full_code = full_code.replace(old_sort, new_sort)
    old_counts = '''        sub_req2 = (
            session.query(Connection.requester_id.label("uid"))
            .filter(Connection.status == "accepted")
            .filter(Connection.requester_id.in_(user_ids))
            .subquery()
        )
        sub_add2 = (
            session.query(Connection.addressee_id.label("uid"))
            .filter(Connection.status == "accepted")
            .filter(Connection.addressee_id.in_(user_ids))
            .subquery()
        )
        conn_union = (
            session.query(sub_req2.c.uid)
            .union_all(session.query(sub_add2.c.uid))
            .subquery()
        )
        conn_counts_rows = (
            session.query(conn_union.c.uid, func.count().label("cnt"))
            .group_by(conn_union.c.uid)
            .all()
        )
        for uid, cnt in conn_counts_rows:
            conn_counts_map[uid] = cnt'''
    new_counts = '''        connection_rows = session.query(Connection).filter(
            Connection.status == "accepted",
            (Connection.requester_id.in_(user_ids)) | (Connection.addressee_id.in_(user_ids)),
        ).all()
        user_id_set = set(user_ids)
        for connection in connection_rows:
            if connection.requester_id in user_id_set:
                conn_counts_map[connection.requester_id] = conn_counts_map.get(connection.requester_id, 0) + 1
            if connection.addressee_id in user_id_set:
                conn_counts_map[connection.addressee_id] = conn_counts_map.get(connection.addressee_id, 0) + 1'''
    full_code = full_code.replace(old_counts, new_counts)

    # A handful of sample rows contain syntactically malformed timestamp
    # strings (for example ``2025-08:20`` or a quoted SQLite datetime()
    # expression).  Keep the source data unchanged and make only the affected
    # ORM columns tolerant when decoding those existing values.
    tolerant_columns = {
        "MenuCategory": {"created_at"},
        "Campaign": {"updated_at"},
        "Unit": {"updated_at"},
        "Booking": {"updated_at"},
        "AvailabilitySlot": {"start_time", "end_time"},
        "Appointment": {"start_time", "end_time"},
        "VideoSession": {"start_time", "end_time"},
        "ProviderQualityReport": {"period_start", "period_end"},
    }
    tolerant_datetime_used = False
    for model_name, column_names in tolerant_columns.items():
        model_pattern = re.compile(
            rf"(?ms)(^class {model_name}\(Base\):\n.*?)(?=^class |^Base\.metadata|^app\s*=)"
        )

        def _patch_model(match: re.Match[str]) -> str:
            nonlocal tolerant_datetime_used
            model_block = match.group(1)
            for column_name in column_names:
                model_block, count = re.subn(
                    rf"(?m)^(\s+{column_name}\s*=\s*Column\()DateTime\b",
                    r"\1_AWMTolerantDateTime",
                    model_block,
                )
                tolerant_datetime_used = tolerant_datetime_used or count > 0
            return model_block

        full_code = model_pattern.sub(_patch_model, full_code)

    if tolerant_datetime_used:
        tolerant_type = '''from datetime import datetime as _AWMCompatDatetime, timedelta as _AWMCompatTimedelta
import re as _awm_re
from sqlalchemy.types import String as _AWMString, TypeDecorator as _AWMTypeDecorator

class _AWMTolerantDateTime(_AWMTypeDecorator):
    impl = _AWMString
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if isinstance(value, _AWMCompatDatetime):
            return value.isoformat(sep=" ")
        return value

    def process_result_value(self, value, dialect):
        if value is None or isinstance(value, _AWMCompatDatetime):
            return value
        text = str(value).strip()
        if not text:
            return None
        text = _awm_re.sub(r"^(\\d{4}-\\d{2}):(\\d{2})", r"\\1-\\2", text)
        sqlite_expr = _awm_re.fullmatch(r"datetime\\('now'(.*)\\)", text)
        if sqlite_expr:
            result = _AWMCompatDatetime.utcnow()
            modifiers = _awm_re.findall(r"'([^']+)'", sqlite_expr.group(1))
            for modifier in modifiers:
                delta = _awm_re.fullmatch(r"([+-]?\\d+)\\s+(days?|hours?|minutes?)", modifier)
                if delta:
                    amount = int(delta.group(1))
                    unit = delta.group(2).rstrip("s") + "s"
                    result += _AWMCompatTimedelta(**{unit: amount})
                    continue
                clock = _awm_re.fullmatch(r"(\\d{2}):(\\d{2})(?::(\\d{2}))?", modifier)
                if clock:
                    result = result.replace(
                        hour=int(clock.group(1)),
                        minute=int(clock.group(2)),
                        second=int(clock.group(3) or 0),
                        microsecond=0,
                    )
            return result
        return _AWMCompatDatetime.fromisoformat(text.replace("Z", "+00:00"))

'''
        full_code = tolerant_type + full_code

    return full_code


def _patch_env_code(full_code: str, db_path: str, host: str, port: int) -> str:
    """
    Patch the generated FastAPI code:
    1. Replace create_engine() call to point to the session-specific DB.
    2. Fix DateTime/str response-model mismatch (codegen bug -> bare 500s).
    3. Register exception handlers so DB errors surface as informative
       JSON payloads instead of a content-free "Internal Server Error".
    """
    full_code = _fix_datetime_response_fields(full_code)
    full_code = _fix_generated_code_errors(full_code)

    new_lines = [
        "import warnings",
        'warnings.filterwarnings("ignore", category=DeprecationWarning)',
    ]

    sql_path = f"sqlite:///{db_path}"

    for line in full_code.split("\n"):
        if "create_engine(" in line:
            left = line.split("create_engine(")[0]
            line = f"{left}create_engine('{sql_path}', connect_args={{'check_same_thread': False}})"

        if "uvicorn.run(app" in line:
            mcp_inject = textwrap.dedent("""\
                # Route-shadowing fix: static paths before dynamic {param} paths.
                app.router.routes.sort(key=lambda _r: tuple(1 if _s.startswith('{') else 0 for _s in (getattr(_r, 'path', '') or '').strip('/').split('/')))
                from fastapi_mcp import FastApiMCP
                mcp = FastApiMCP(app)
                mcp.mount_http()

                # Surface DB errors to the agent instead of bare 500s:
                # UNIQUE violations (task wants to create an entity that the
                # seed already contains) become 409 + constraint detail, so
                # the model can react (look up the existing row) rather than
                # blind-retry. Any other unhandled error returns its type and
                # message instead of "Internal Server Error".
                from fastapi.responses import JSONResponse
                from fastapi.encoders import jsonable_encoder
                from fastapi.exceptions import ResponseValidationError
                from sqlalchemy.exc import IntegrityError

                @app.exception_handler(IntegrityError)
                async def _awm_integrity_error_handler(request, exc):
                    return JSONResponse(
                        status_code=409,
                        content={"detail": f"IntegrityError: {exc.orig}"},
                    )

                @app.exception_handler(ResponseValidationError)
                async def _awm_response_validation_error_handler(request, exc):
                    # The generated response schema is occasionally wrong even
                    # though the endpoint produced useful data.  Preserve that
                    # raw body instead of turning a read into a content-free 500.
                    return JSONResponse(
                        status_code=200,
                        content=jsonable_encoder(exc.body),
                    )

                @app.exception_handler(Exception)
                async def _awm_generic_error_handler(request, exc):
                    status_code = 500
                    if type(exc) is ValueError:
                        # Invalid caller values (for example 2026-04-31) are
                        # request errors, not broken tools.
                        status_code = 422
                    elif (
                        isinstance(exc, AttributeError)
                        and "'NoneType' object has no attribute" in str(exc)
                    ):
                        # Generated lookup handlers often dereference a miss
                        # instead of returning an explicit not-found result.
                        status_code = 404
                    return JSONResponse(
                        status_code=status_code,
                        content={"detail": f"{type(exc).__name__}: {exc}"},
                    )
            """)
            for inject_line in mcp_inject.strip().split("\n"):
                new_lines.append(f"    {inject_line}")

            line = f"    uvicorn.run(app, host='{host}', port={port})"

        new_lines.append(line)

    return "\n".join(new_lines)


# ---------------------------------------------------------------------------
# Persistent MCP connection
# ---------------------------------------------------------------------------
class _MCPConnection:
    """Persistent MCP client session backed by AsyncExitStack.

    Keeps the streamable HTTP transport and ClientSession alive across
    multiple ``call_tool`` / ``list_tools`` invocations, avoiding the
    overhead of creating a new connection per call.
    """

    def __init__(self) -> None:
        self._stack: contextlib.AsyncExitStack | None = None
        self._session: ClientSession | None = None

    @property
    def connected(self) -> bool:
        return self._session is not None

    async def connect(self, mcp_url: str) -> None:
        self._stack = contextlib.AsyncExitStack()
        try:
            read_stream, write_stream, _ = await self._stack.enter_async_context(
                streamable_http_client(mcp_url)
            )
            self._session = await self._stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await self._session.initialize()
        except Exception:
            await self.close()
            raise

    async def list_tools(self) -> list[dict]:
        assert self._session is not None, "Not connected"
        result = await self._session.list_tools()
        return [
            {
                "name": t.name,
                "description": t.description or "",
                "inputSchema": t.inputSchema or {},
            }
            for t in result.tools
        ]

    async def call_tool(self, tool_name: str, arguments: dict) -> dict[str, Any]:
        assert self._session is not None, "Not connected"
        result = await self._session.call_tool(tool_name, arguments)

        parts = []
        for c in result.content:
            if hasattr(c, "text"):
                parts.append(c.text)
            else:
                parts.append(str(c))

        text = "\n".join(parts)
        if result.isError:
            return {"success": False, "result": None, "error": text}
        return {"success": True, "result": text, "error": None}

    async def close(self) -> None:
        if self._stack is not None:
            try:
                await self._stack.aclose()
            except Exception:
                pass
            self._stack = None
            self._session = None


# ---------------------------------------------------------------------------
# ScenarioProcess
# ---------------------------------------------------------------------------
class ScenarioProcess:
    """Manages a single sub-environment subprocess with persistent MCP connection."""

    def __init__(self):
        self._process: subprocess.Popen | None = None
        self._port: int | None = None
        self._temp_dir: str | None = None
        self._owns_temp_dir: bool = False
        self._server_py: str | None = None
        self._log_file = None
        self._log_path: str | None = None
        # Persistent MCP connection + dedicated event loop
        self._mcp: _MCPConnection | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    @property
    def port(self) -> int | None:
        return self._port

    @property
    def mcp_url(self) -> str | None:
        if self._port is None:
            return None
        return f"http://127.0.0.1:{self._port}/mcp"

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self, full_code: str, db_path: str, session_dir: str) -> str:
        """
        Start the sub-environment subprocess and establish persistent MCP
        connection.

        Returns:
            The MCP URL of the started server.

        Raises:
            RuntimeError: If the server fails to start within the timeout.
        """
        self.stop()

        self._temp_dir = session_dir
        self._owns_temp_dir = False  # caller owns the directory
        host = "127.0.0.1"

        last_error = ""
        for attempt in range(1 + MAX_PORT_RETRIES):
            self._port = _get_random_port()
            timeout = READY_TIMEOUT if attempt == 0 else RETRY_READY_TIMEOUT

            patched_code = _patch_env_code(full_code, db_path, host, self._port)

            self._server_py = f"{self._temp_dir}/server.py"
            with open(self._server_py, "w", encoding="utf-8") as f:
                f.write(patched_code)

            if attempt > 0:
                logger.info(
                    f"Retry {attempt}/{MAX_PORT_RETRIES} on port {self._port} "
                    f"(timeout={timeout}s) ..."
                )
            else:
                logger.info(f"Starting sub-env on port {self._port} ...")

            self._log_path = f"{self._temp_dir}/server.log"
            self._log_file = open(self._log_path, "w", encoding="utf-8")

            self._process = subprocess.Popen(
                [sys.executable, self._server_py],
                stdout=self._log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                text=True,
            )

            if self._wait_for_ready(timeout):
                # Establish persistent MCP connection
                try:
                    self._connect_mcp()
                    logger.info(
                        f"Sub-env ready on port {self._port}, "
                        f"mcp=persistent, log={self._log_path}"
                    )
                    return self.mcp_url
                except Exception as e:
                    last_error = (
                        f"Sub-env started on port {self._port} but "
                        f"MCP connection failed: {e}"
                    )
                    logger.warning(last_error)
                    self._disconnect_mcp()
                    self.stop()
                    continue

            # Failed — collect error output for diagnostics
            failed_port = self._port
            self._log_file.flush()
            try:
                with open(self._log_path, "r", encoding="utf-8") as lf:
                    logged_output = lf.read()
            except OSError:
                logged_output = ""
            last_error = (
                f"Sub-env failed to start on port {failed_port} "
                f"(timeout {timeout}s).\nOutput: {logged_output}"
            )
            logger.warning(last_error)
            self.stop()

        raise RuntimeError(
            f"Sub-env failed after {1 + MAX_PORT_RETRIES} attempts. "
            f"Last error: {last_error}"
        )

    def _wait_for_ready(self, timeout: float = READY_TIMEOUT) -> bool:
        """Poll the MCP endpoint until the server is ready."""
        start = time.time()
        while time.time() - start < timeout:
            if self._process.poll() is not None:
                return False

            try:
                with socket.create_connection(("127.0.0.1", self._port), timeout=1):
                    time.sleep(0.5)
                    return True
            except (ConnectionRefusedError, OSError, socket.timeout):
                pass

            time.sleep(READY_POLL_INTERVAL)

        return False

    # -- Persistent MCP connection management ---------------------------------

    def _connect_mcp(self) -> None:
        """Create a dedicated event loop and establish persistent MCP session."""
        self._loop = asyncio.new_event_loop()
        self._mcp = _MCPConnection()
        self._loop.run_until_complete(self._mcp.connect(self.mcp_url))

    def _disconnect_mcp(self) -> None:
        """Tear down the persistent MCP session and event loop."""
        if self._mcp is not None:
            if self._loop is not None and not self._loop.is_closed():
                try:
                    self._loop.run_until_complete(self._mcp.close())
                except Exception:
                    pass
            self._mcp = None
        if self._loop is not None:
            try:
                self._loop.close()
            except Exception:
                pass
            self._loop = None

    def list_tools(self, timeout: float = 15.0) -> list[dict]:
        """List MCP tools via persistent connection (sync)."""
        if self._mcp is None or self._loop is None:
            raise RuntimeError("MCP connection not established")
        try:
            return self._loop.run_until_complete(
                asyncio.wait_for(self._mcp.list_tools(), timeout=timeout)
            )
        except Exception as e:
            logger.error(f"Failed to list MCP tools: {e}")
            return []

    def call_tool(
        self, tool_name: str, arguments: dict, timeout: float = 30.0
    ) -> dict[str, Any]:
        """Call an MCP tool via persistent connection (sync).

        Returns:
            dict with keys: "success" (bool), "result" (Any), "error" (str | None)
        """
        if self._mcp is None or self._loop is None:
            return {
                "success": False,
                "result": None,
                "error": "MCP connection not established",
            }
        try:
            return self._loop.run_until_complete(
                asyncio.wait_for(
                    self._mcp.call_tool(tool_name, arguments),
                    timeout=timeout,
                )
            )
        except asyncio.TimeoutError:
            return {
                "success": False,
                "result": None,
                "error": f"Tool call timed out after {timeout}s",
            }
        except Exception as e:
            return {"success": False, "result": None, "error": str(e)}

    # -- Subprocess lifecycle --------------------------------------------------

    def stop(self) -> None:
        """Stop the subprocess and clean up resources.

        Thread-safe: captures ``self._process`` in a local variable and
        sets the attribute to ``None`` immediately so concurrent callers
        (e.g. cleanup thread + ``_handle_done``) don't double-kill.
        """
        # Close MCP connection first (before killing subprocess)
        self._disconnect_mcp()

        proc = self._process
        if proc is not None:
            self._process = None  # claim ownership immediately

            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass

            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    pass

        if self._log_file is not None:
            try:
                self._log_file.close()
            except OSError:
                pass
            self._log_file = None

        if self._owns_temp_dir and self._temp_dir and os.path.isdir(self._temp_dir):
            import shutil

            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None

        self._port = None
        self._server_py = None
