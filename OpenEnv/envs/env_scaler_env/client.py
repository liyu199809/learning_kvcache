"""Client for the EnvScaler OpenEnv service."""

from typing import Any

from openenv.core.client_types import StepResult
from openenv.core.mcp_client import MCPToolClient

from .models import EnvScalerListToolsObservation, EnvScalerObservation


class EnvScalerEnv(MCPToolClient):
    """MCP client with EnvScaler-specific observation parsing."""

    def _parse_result(self, payload: dict[str, Any]) -> StepResult:
        observation_data = payload.get("observation", {})
        if "tools" in observation_data:
            observation = EnvScalerListToolsObservation(**observation_data)
        else:
            observation = EnvScalerObservation(**observation_data)
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )
