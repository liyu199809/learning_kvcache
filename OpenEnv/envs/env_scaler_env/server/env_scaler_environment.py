"""OpenEnv adapter for the official EnvScaler executable environment data."""

from __future__ import annotations

import logging
import shutil
import tempfile
from typing import Any
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction, Tool
from openenv.core.env_server.types import Action, State

from ..models import EnvScalerListToolsObservation, EnvScalerObservation
from .config import TOOL_CALL_TIMEOUT
from .data_loader import EnvScalerDataLoader
from .episode_worker import EpisodeWorker


logger = logging.getLogger(__name__)


class EnvScalerEnvironment(Environment):
    """Serve one isolated EnvScaler task per OpenEnv WebSocket session."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, data_loader: EnvScalerDataLoader | None = None) -> None:
        super().__init__()
        self._data_loader = data_loader or EnvScalerDataLoader()
        self._worker = EpisodeWorker()
        self._state = State(episode_id=None, step_count=0)
        self._scenario: str | None = None
        self._task_idx: int | None = None
        self._task: dict[str, Any] | None = None
        self._tools: list[dict[str, Any]] = []
        self._work_dir: str | None = None
        self._reset_ok = False
        self._episode_done = False
        self._trajectory: list[dict[str, Any]] = []

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        scenario: str | None = None,
        task_idx: int | None = None,
        **kwargs: Any,
    ) -> EnvScalerObservation:
        del seed, kwargs
        self._cleanup()
        if not scenario:
            return EnvScalerObservation(
                reward_type="reset_error",
                error="Parameter 'scenario' is required",
            )
        if task_idx is None:
            return EnvScalerObservation(
                reward_type="reset_error",
                scenario=scenario,
                error="Parameter 'task_idx' is required",
            )
        try:
            environment_item = self._data_loader.get_environment(scenario)
            task = self._data_loader.get_task(scenario, int(task_idx))
            work_dir = tempfile.mkdtemp(prefix=f"openenv_envscaler_{scenario}_")
            payload = {
                "env_class_code": environment_item["env_class_code"],
                "env_class_name": task["env_class_name"],
                "init_config": task["init_config"],
                "tools": environment_item["tools"],
                "checklist_with_func": task["checklist_with_func"],
            }
            self._worker.start(payload, work_dir)
        except Exception as exc:
            if "work_dir" in locals():
                shutil.rmtree(work_dir, ignore_errors=True)
            logger.exception("Failed to reset EnvScaler task %s:%s", scenario, task_idx)
            return EnvScalerObservation(
                reward_type="reset_error",
                scenario=scenario,
                task_idx=int(task_idx),
                error=f"{type(exc).__name__}: {exc}",
            )

        self._scenario = scenario
        self._task_idx = int(task_idx)
        self._task = task
        self._tools = environment_item["tools"]
        self._work_dir = work_dir
        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        self._reset_ok = True
        self._episode_done = False
        self._trajectory = []
        return EnvScalerObservation(
            reward_type="reset_ok",
            scenario=scenario,
            task=str(task["task"]),
            task_idx=self._task_idx,
            task_id=str(task["task_id"]),
            has_verifier={"sql": False, "code": True},
            num_tools=len(self._tools),
        )

    def step(
        self,
        action: Action,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> EnvScalerObservation | EnvScalerListToolsObservation:
        del kwargs
        if self._episode_done:
            return EnvScalerObservation(
                done=True,
                reward_type="episode_already_done",
                error="Episode has ended. Call reset() to start a new episode.",
            )
        self._state.step_count += 1
        if isinstance(action, ListToolsAction):
            return self._list_tools()
        if not isinstance(action, CallToolAction):
            return EnvScalerObservation(
                reward=0.0,
                reward_type="invalid_action",
                error=f"Unknown action type: {type(action).__name__}",
            )
        if action.tool_name == "verify":
            return self._verify(action)
        if action.tool_name == "done":
            return self._done()
        if action.tool_name == "__list_scenarios__":
            scenarios = self._data_loader.list_scenarios()
            return EnvScalerObservation(
                reward_type="tool_call_ok",
                scenarios=scenarios,
                total=len(scenarios),
            )
        return self._call_tool(action, timeout_s)

    def _list_tools(self) -> EnvScalerListToolsObservation:
        if not self._reset_ok:
            return EnvScalerListToolsObservation(tools=[], error="Call reset() before list_tools")
        tools = [
            Tool(
                name=tool["function"]["name"],
                description=tool["function"].get("description", ""),
                input_schema=tool["function"].get("parameters", {"type": "object"}),
            )
            for tool in self._tools
        ]
        return EnvScalerListToolsObservation(tools=tools)

    def _call_tool(self, action: CallToolAction, timeout_s: float | None) -> EnvScalerObservation:
        if not self._reset_ok or not self._worker.is_running:
            return EnvScalerObservation(
                reward=0.0,
                reward_type="server_error",
                tool_name=action.tool_name,
                error="Environment is not initialized. Call reset() first.",
            )
        try:
            response = self._worker.call_tool(
                action.tool_name,
                action.arguments,
                timeout_s or TOOL_CALL_TIMEOUT,
            )
        except Exception as exc:
            response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        self._trajectory.append(
            {
                "tool_name": action.tool_name,
                "arguments": action.arguments,
                "success": bool(response.get("ok")),
                "error": response.get("error"),
            }
        )
        if response.get("ok"):
            return EnvScalerObservation(
                reward=0.0,
                reward_type="tool_call_ok",
                tool_name=action.tool_name,
                tool_result=response.get("result"),
            )
        error = str(response.get("error", "Unknown tool error"))
        if error.startswith("ValueError: Unknown tool"):
            reward_type = "tool_not_found"
        elif error.startswith("ValidationError:"):
            reward_type = "invalid_args"
        else:
            reward_type = "tool_error"
        return EnvScalerObservation(
            reward=0.0,
            reward_type=reward_type,
            tool_name=action.tool_name,
            error=error,
        )

    def _verify(self, action: CallToolAction) -> EnvScalerObservation:
        if not self._reset_ok or self._task is None:
            return EnvScalerObservation(
                reward=0.0,
                reward_type="server_error",
                error="Cannot verify before a successful reset",
            )
        verifier_mode = (action.arguments or {}).get("verifier_mode", "code")
        if verifier_mode != "code":
            return EnvScalerObservation(
                reward=0.0,
                reward_type="invalid_args",
                error="EnvScaler supports verifier_mode='code' only",
            )
        try:
            response = self._worker.verify()
        except Exception as exc:
            response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        if not response.get("ok"):
            return EnvScalerObservation(
                reward=0.0,
                reward_type="judge_error",
                scenario=self._scenario,
                task=str(self._task["task"]),
                task_idx=self._task_idx,
                task_id=str(self._task["task_id"]),
                error=str(response.get("error", "Verifier failed")),
            )
        result = response["result"]
        score = float(result["score"])
        reward_type = "complete" if score == 1.0 and result["checker_errors"] == 0 else "incomplete"
        self._trajectory.append(
            {
                "action": "verify",
                "final_answer": (action.arguments or {}).get("final_answer"),
                "reward": score,
                "reward_type": reward_type,
                "verify_result": result,
            }
        )
        return EnvScalerObservation(
            reward=score,
            reward_type=reward_type,
            verify_result=result,
            scenario=self._scenario,
            task=str(self._task["task"]),
            task_idx=self._task_idx,
            task_id=str(self._task["task_id"]),
            steps_taken=self._state.step_count,
        )

    def _done(self) -> EnvScalerObservation:
        self._episode_done = True
        self._worker.stop()
        return EnvScalerObservation(
            done=True,
            reward=0.0,
            reward_type="episode_done",
            scenario=self._scenario,
            task_idx=self._task_idx,
            task_id=str(self._task["task_id"]) if self._task else None,
            steps_taken=self._state.step_count,
        )

    @property
    def state(self) -> State:
        return self._state

    def close(self) -> None:
        self._cleanup()

    def _cleanup(self) -> None:
        self._worker.stop()
        if self._work_dir:
            shutil.rmtree(self._work_dir, ignore_errors=True)
        self._work_dir = None
        self._scenario = None
        self._task_idx = None
        self._task = None
        self._tools = []
        self._reset_ok = False
        self._episode_done = False
        self._trajectory = []
