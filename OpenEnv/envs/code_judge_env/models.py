"""Wire models shared by CodeJudge clients and the OpenEnv server."""

from typing import Annotated, Any

from openenv.core.env_server.mcp_types import (
    CallToolAction,
    ListToolsAction,
    ListToolsObservation,
)
from openenv.core.env_server.types import Action, Observation
from pydantic import ConfigDict, Field, TypeAdapter


_ACTION_UNION = Annotated[
    ListToolsAction | CallToolAction,
    Field(discriminator="type"),
]
_ACTION_ADAPTER = TypeAdapter(_ACTION_UNION)


class CodeJudgeAction(Action):
    """Discriminated union of list-tools and call-tool actions."""

    @classmethod
    def model_validate(cls, obj: Any, **kwargs: Any) -> Action:  # type: ignore[override]
        return _ACTION_ADAPTER.validate_python(obj)

    @classmethod
    def model_json_schema(cls, **kwargs: Any) -> dict[str, Any]:  # type: ignore[override]
        return _ACTION_ADAPTER.json_schema(**kwargs)


class CodeJudgeObservation(Observation):
    """Observation fields consumed by GenericEnvClient and AWMAgentLoop."""

    model_config = ConfigDict(extra="forbid")

    reward_type: str | None = None
    scenario: str | None = None
    task: str | None = None
    task_idx: int | None = None
    task_id: str | None = None
    has_verifier: dict[str, bool] | None = None
    num_tools: int | None = None
    error: str | None = None
    verify_result: dict[str, Any] | None = None
    steps_taken: int | None = None

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        kwargs.setdefault("exclude_none", True)
        return super().model_dump(**kwargs)


class CodeJudgeListToolsObservation(ListToolsObservation):
    """Empty tool listing plus an optional application error."""

    model_config = ConfigDict(extra="forbid")

    error: str | None = None
