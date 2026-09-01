"""OpenEnv environment that verifies one DeepCoder TACO response."""

from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction
from openenv.core.env_server.types import Action, State

from ..models import CodeJudgeListToolsObservation, CodeJudgeObservation
from .data_loader import TacoDataLoader
from .judge import judge_answer


logger = logging.getLogger(__name__)


class CodeJudgeEnvironment(Environment):
    """Bind one hidden-test task to each persistent OpenEnv session."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, data_loader: TacoDataLoader) -> None:
        super().__init__()
        self._data_loader = data_loader
        self._state = State(episode_id=None, step_count=0)
        self._task: dict[str, Any] | None = None
        self._task_idx: int | None = None
        self._reset_ok = False
        self._episode_done = False

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        scenario: str | None = None,
        task_idx: int | None = None,
        **kwargs: Any,
    ) -> CodeJudgeObservation:
        del seed, kwargs
        self._clear()
        if scenario != "taco":
            return CodeJudgeObservation(
                reward_type="reset_error",
                error="CodeJudge requires scenario='taco'",
            )
        if task_idx is None:
            return CodeJudgeObservation(
                reward_type="reset_error",
                scenario=scenario,
                error="Parameter 'task_idx' is required",
            )
        try:
            task = self._data_loader.get_task(int(task_idx))
        except Exception as exc:
            return CodeJudgeObservation(
                reward_type="reset_error",
                scenario=scenario,
                task_idx=int(task_idx),
                error=f"{type(exc).__name__}: {exc}",
            )

        self._task = task
        self._task_idx = int(task_idx)
        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        self._reset_ok = True
        return CodeJudgeObservation(
            reward_type="reset_ok",
            scenario="taco",
            task=task["problem"],
            task_idx=self._task_idx,
            task_id=task["task_id"],
            has_verifier={"sql": False, "code": True},
            num_tools=0,
        )

    def step(
        self,
        action: Action,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> CodeJudgeObservation | CodeJudgeListToolsObservation:
        del timeout_s, kwargs
        if self._episode_done:
            return CodeJudgeObservation(
                done=True,
                reward=0.0,
                reward_type="episode_already_done",
                error="Episode has ended. Call reset() to start a new episode.",
            )
        self._state.step_count += 1
        if isinstance(action, ListToolsAction):
            return CodeJudgeListToolsObservation(tools=[])
        if not isinstance(action, CallToolAction):
            return CodeJudgeObservation(
                reward=0.0,
                reward_type="invalid_action",
                error=f"Unknown action type: {type(action).__name__}",
            )
        if action.tool_name == "verify":
            return self._verify(action)
        if action.tool_name == "done":
            return self._done()
        return CodeJudgeObservation(
            reward=0.0,
            reward_type="tool_not_found",
            error=f"CodeJudge exposes no agent tools; unknown tool {action.tool_name!r}",
        )

    def _verify(self, action: CallToolAction) -> CodeJudgeObservation:
        if not self._reset_ok or self._task is None or self._task_idx is None:
            return CodeJudgeObservation(
                reward=0.0,
                reward_type="server_error",
                error="Cannot verify before a successful reset",
            )
        arguments = action.arguments or {}
        if arguments.get("verifier_mode", "code") != "code":
            return CodeJudgeObservation(
                reward=0.0,
                reward_type="invalid_args",
                error="CodeJudge supports verifier_mode='code' only",
            )
        final_answer = arguments.get("final_answer")
        if not isinstance(final_answer, str):
            return CodeJudgeObservation(
                reward=0.0,
                reward_type="invalid_args",
                error="verify requires string final_answer",
            )
        try:
            response = judge_answer(final_answer, self._task["tests"])
        except Exception as exc:
            logger.exception("CodeJudge failed for task_idx=%s", self._task_idx)
            response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        self._episode_done = True
        if not response.get("ok"):
            return CodeJudgeObservation(
                done=True,
                reward=0.0,
                reward_type="judge_error",
                scenario="taco",
                task_idx=self._task_idx,
                task_id=self._task["task_id"],
                steps_taken=self._state.step_count,
                error=str(response.get("error", "Judge failed")),
            )
        return CodeJudgeObservation(
            done=True,
            reward=float(response["reward"]),
            reward_type=str(response["reward_type"]),
            verify_result=response["verify_result"],
            scenario="taco",
            task_idx=self._task_idx,
            task_id=self._task["task_id"],
            steps_taken=self._state.step_count,
        )

    def _done(self) -> CodeJudgeObservation:
        self._episode_done = True
        return CodeJudgeObservation(
            done=True,
            reward=0.0,
            reward_type="episode_done",
            scenario="taco" if self._reset_ok else None,
            task_idx=self._task_idx,
            task_id=self._task["task_id"] if self._task else None,
            steps_taken=self._state.step_count,
        )

    @property
    def state(self) -> State:
        return self._state

    def close(self) -> None:
        self._clear()

    def _clear(self) -> None:
        self._task = None
        self._task_idx = None
        self._reset_ok = False
        self._episode_done = False
        self._state = State(episode_id=None, step_count=0)
