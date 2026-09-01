"""Typed client for the CodeJudge OpenEnv service."""

from typing import Any

from openenv.core.client_types import StepResult
from openenv.core.mcp_client import MCPToolClient

from .models import CodeJudgeListToolsObservation, CodeJudgeObservation


class CodeJudgeEnv(MCPToolClient):
    """MCP client with CodeJudge-specific observation parsing."""

    def _parse_result(self, payload: dict[str, Any]) -> StepResult:
        observation_data = payload.get("observation", {})
        if "tools" in observation_data:
            observation = CodeJudgeListToolsObservation(**observation_data)
        else:
            observation = CodeJudgeObservation(**observation_data)
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )
