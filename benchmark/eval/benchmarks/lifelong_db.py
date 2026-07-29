"""LifelongAgentBench / db 适配层。

复用官方 reference 仓库的 DB 建表与打分纯逻辑（DBBench / DBBenchContainer /
DirectTypeAnswerValidator / AnswerType），环境交互逻辑移植并瘦身自
opd_evolver/benchmark/bench_lifelong_agent.py 的 db 分支。不 import opd_evolver。
"""
from __future__ import annotations

import ast
import asyncio
import json
import re
import sys
import threading
from pathlib import Path
from typing import Any, List, Optional

from ..core import registry
from ..core.scorer import ProgrammaticScorer, Scorer
from ..core.types import Action, BasicInfo, Observation, StepReturn
from .base import Task, TaskSuite

# reference 官方仓库路径（相对项目根 self_evolver）。
_REFERENCE_ROOT = (
    Path(__file__).resolve().parents[2]  # -> benchmark/
    / "opd-evolver" / "reference" / "LifelongAgentBench"
)

DEFAULT_MAX_STEPS = 6


def _ensure_reference_on_path() -> None:
    if not _REFERENCE_ROOT.exists():
        raise FileNotFoundError(
            f"LifelongAgentBench reference repo not found at {_REFERENCE_ROOT}."
        )
    ref = str(_REFERENCE_ROOT)
    if ref not in sys.path:
        sys.path.insert(0, ref)


def _maybe_parse_obj(value: Any) -> Any:
    """jsonl 字段可能是字符串化的对象，尽力还原为 dict/list。"""
    if value is None or not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        return ast.literal_eval(text)
    except Exception:
        return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"split not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _compact_action_space() -> str:
    return (
        "Execute SQL against the initialized MySQL database.\n"
        'Actions:\n  - {"action":"execute","params":{"command":"SQL query"}}\n'
        '  - {"action":"submit","params":{}}\n'
        "For SELECT tasks, submit evaluates the latest query output. For INSERT/UPDATE/DELETE "
        "tasks, submit evaluates the final table state."
    )


def _run_with_timeout(fn: Any, timeout_s: float, label: str) -> None:
    err: list[BaseException] = []

    def _target() -> None:
        try:
            fn()
        except BaseException as exc:  # noqa
            err.append(exc)

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout=timeout_s)
    if t.is_alive():
        print(f"[lifelong_db] {label} timed out after {timeout_s:.1f}s; skipping cleanup.")
        return
    if err:
        raise err[0]


class _DBRuntime:
    """封装 reference 的 DBBench + Docker MySQL 容器。"""

    def __init__(self, entry: dict[str, Any], mysql_image: str = "mysql:8.0"):
        _ensure_reference_on_path()
        from src.tasks.instance.db_bench.task import (
            AnswerType,
            DBBench,
            DirectTypeAnswerValidator,
        )
        from src.tasks.instance.db_bench.container import DBBenchContainer

        self.AnswerType = AnswerType
        self.DirectTypeAnswerValidator = DirectTypeAnswerValidator

        cleaned: dict[str, Any] = dict(entry)
        for key in ("answer_info", "table_info", "skill_list"):
            if key in cleaned:
                cleaned[key] = _maybe_parse_obj(cleaned[key])
        answer_info = cleaned.get("answer_info")
        if isinstance(answer_info, dict):
            for sub in ("direct", "sql", "md5"):
                if sub in answer_info:
                    answer_info[sub] = _maybe_parse_obj(answer_info[sub])
            cleaned["answer_info"] = answer_info
        table_info = cleaned.get("table_info")
        if isinstance(table_info, dict):
            for sub in ("row_list", "column_info_list", "name"):
                if sub in table_info:
                    table_info[sub] = _maybe_parse_obj(table_info[sub])
            cleaned["table_info"] = table_info
        skill_list = cleaned.get("skill_list")
        if isinstance(skill_list, str):
            skill_list = _maybe_parse_obj(skill_list)
        if skill_list is None:
            skill_list = []
        if not isinstance(skill_list, list):
            skill_list = [skill_list]
        cleaned["skill_list"] = skill_list

        self.dataset_item = DBBench._construct_dataset_item(cleaned)
        self.container = DBBenchContainer(image=mysql_image)
        self.last_output = ""
        self._DBBench = DBBench

    def reset(self) -> str:
        self.container.execute(self._DBBench._build_init_sql(self.dataset_item))
        return "Database initialized."

    def execute(self, sql: str) -> str:
        self.last_output = self.container.execute(sql, self.dataset_item.database_name)
        return self.last_output

    def submit(self, answer: Optional[str] = None) -> tuple[bool, str]:
        answer_info = self.dataset_item.answer_info
        candidate = self.last_output if (answer is None or not str(answer).strip()) else str(answer)
        if answer_info.answer_type == self.AnswerType.MD5:
            column_name_str = ",".join(
                f"`{col.name}`" for col in self.dataset_item.table_info.column_info_list
            )
            table_name = self.dataset_item.table_info.name
            md5_query = (
                "select md5(group_concat(rowhash order by rowhash)) as hash "
                f"from( SELECT substring(MD5(CONCAT_WS(',', {column_name_str})), 1, 5) AS rowhash "
                f"FROM `{table_name}`) as sub;"
            )
            candidate = self.container.execute(md5_query, self.dataset_item.database_name)
            match = re.search(r"\('?(.*?)'?,\)", candidate)
            candidate = match.group(1) if match else candidate
            ok = candidate == answer_info.answer_md5
        else:
            ok = self.DirectTypeAnswerValidator.validate(candidate, answer_info.answer_direct)
        return bool(ok), str(candidate)

    def close(self) -> None:
        try:
            self.container.execute(f"drop database `{self.dataset_item.database_name}`")
        except Exception:  # noqa
            pass
        try:
            _run_with_timeout(self.container.delete, timeout_s=10.0, label="DB container delete")
        except Exception as exc:  # noqa
            print(f"[lifelong_db] runtime cleanup failed: {exc}")


class LifelongDBTask(Task):
    def __init__(self, task_id: str, entry: dict[str, Any], max_steps: int,
                 mysql_image: str = "mysql:8.0"):
        self.task_id = task_id
        self.entry = entry
        self.max_steps = max_steps
        self.mysql_image = mysql_image
        self.runtime: Optional[_DBRuntime] = None
        self.done = False
        self.steps = 0

    def _instruction(self) -> str:
        table = _maybe_parse_obj(self.entry.get("table_info", {}))
        if not isinstance(table, dict):
            table = {}
        col_list = _maybe_parse_obj(table.get("column_info_list", []))
        if not isinstance(col_list, list):
            col_list = []
        cols = [
            c.get("name")
            for c in col_list
            if isinstance(c, dict) and isinstance(c.get("name"), (str, int, float))
        ]
        suffix = f"Table: {table.get('name')}; columns: {', '.join(str(c) for c in cols if c is not None)}."
        return f"{self.entry.get('instruction', '')}\n{suffix}"

    def get_basic_info(self) -> BasicInfo:
        return BasicInfo(
            env_id=self.task_id,
            instruction=self._instruction(),
            action_space=_compact_action_space(),
            max_steps=self.max_steps,
            prompt_type="db",
            meta_data={
                "task_type": "db",
                "skill_tags": self.entry.get("skill_list") or [],
            },
        )

    async def reset(self) -> Observation:
        self.done = False
        self.steps = 0
        # DBBenchContainer 构造是阻塞的（起 docker + 连 mysql），放线程池。
        def _spawn() -> tuple[_DBRuntime, str]:
            rt = _DBRuntime(self.entry, mysql_image=self.mysql_image)
            return rt, rt.reset()

        self.runtime, message = await asyncio.to_thread(_spawn)
        return {
            "message": message,
            "instruction": self._instruction(),
            "current_step": 0,
            "max_steps": self.max_steps,
        }

    async def step(self, action: Action) -> StepReturn:
        if self.done:
            return {"error": "Environment already finished"}, 0.0, True, {"error": "already_done"}
        self.steps += 1
        action_type = str(action.get("action", ""))
        params = action.get("params", {}) if isinstance(action.get("params"), dict) else {}

        if action_type == "execute":
            command = str(params.get("command", "")).strip()
            if not command:
                return {"error": "No command provided"}, 0.0, False, {"error": "no_command"}
            output = await asyncio.to_thread(self.runtime.execute, command)
            done = self.steps >= self.max_steps
            self.done = done
            return (
                {"command": command, "output": output, "current_step": self.steps,
                 "max_steps": self.max_steps},
                0.0,
                done,
                {"max_steps_reached": done} if done else {},
            )

        if action_type == "submit":
            answer = params.get("answer")
            success, observed = await asyncio.to_thread(
                self.runtime.submit, str(answer) if answer is not None else None
            )
            self.done = True
            return (
                {"message": "Solution submitted", "success": success, "output": observed,
                 "current_step": self.steps},
                1.0 if success else 0.0,
                True,
                {"submitted": True, "observed": observed, "success": success},
            )

        return {"error": f"Unknown action type: {action_type}"}, 0.0, False, {"error": action_type}

    async def close(self) -> None:
        if self.runtime is not None:
            await asyncio.to_thread(self.runtime.close)
            self.runtime = None


class LifelongDBSuite(TaskSuite):
    name = "lifelong_db"

    def __init__(self, data_dir: str, split: str = "test", max_steps: Optional[int] = None,
                 mysql_image: str = "mysql:8.0"):
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_steps = max_steps or DEFAULT_MAX_STEPS
        self.mysql_image = mysql_image
        # 数据既可能在 <data_dir>/db/<split>.jsonl，也可能 data_dir 已指向 db 目录。
        candidate = self.data_dir / "db" / f"{split}.jsonl"
        if not candidate.is_file():
            candidate = self.data_dir / f"{split}.jsonl"
        self.rows = _load_jsonl(candidate)
        self._ids = [str(r.get("task_id") or f"db_{i}") for i, r in enumerate(self.rows)]

    def list_task_ids(self) -> List[str]:
        return self._ids

    def make_task(self, index: int) -> Task:
        return LifelongDBTask(self._ids[index], self.rows[index], self.max_steps,
                              mysql_image=self.mysql_image)

    def scorer(self) -> Scorer:
        return ProgrammaticScorer(reward_threshold=1.0)

    def default_max_steps(self) -> int:
        return self.max_steps


def _factory(args: Any) -> TaskSuite:
    return LifelongDBSuite(
        data_dir=args.data_dir,
        split=args.split,
        max_steps=args.max_steps,
        mysql_image=getattr(args, "mysql_image", None) or "mysql:8.0",
    )


registry.register("lifelong_db", _factory)
