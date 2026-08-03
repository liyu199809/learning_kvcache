"""LifelongAgentBench / os_interaction 适配层（native function-calling）。

复用官方 reference 仓库的 OSInteraction / OSInteractionContainer / CommandItem
纯逻辑；环境交互移植并瘦身自 opd_evolver bench_lifelong_agent.py 的 os 分支。
不 import opd_evolver。

与 DB 的差异：
  - 工具是 execute(bash) + finish（finish 无参数，触发隐藏评测）。
  - 判分：运行 evaluation_command_item，exit_code == 0 即成功（不看模型输出）。
  - 容器镜像为 local-os/default（官方 OS 镜像）。
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, List, Optional

from ..core import registry
from ..core.scorer import ProgrammaticScorer, Scorer
from ..core.tool_schemas import OS_TOOLS, OS_NATIVE_SYSTEM_PROMPT
from ..core.types import Action, BasicInfo, Observation, StepReturn
from .base import Task, TaskSuite
from .lifelong_db import (
    _REFERENCE_ROOT,
    _ensure_reference_on_path,
    _load_jsonl,
    _maybe_parse_obj,
    _run_with_timeout,
)

DEFAULT_MAX_STEPS = 8
DEFAULT_OS_TIMEOUT = 20


class _OSRuntime:
    """封装 reference 的 OSInteraction + Docker 容器（local-os/default）。"""

    def __init__(self, entry: dict[str, Any], timeout: int = DEFAULT_OS_TIMEOUT):
        _ensure_reference_on_path()
        from src.tasks.instance.os_interaction.task import OSInteraction
        from src.tasks.instance.os_interaction.container import OSInteractionContainer
        from src.tasks.instance.os_interaction.utility import CommandItem, CommandName

        self.CommandItem = CommandItem
        self.CommandName = CommandName

        cleaned = dict(entry)
        for key in ("initialization_command_item", "evaluation_info", "skill_list"):
            if key in cleaned:
                cleaned[key] = _maybe_parse_obj(cleaned[key])
        eval_info = cleaned.get("evaluation_info")
        if isinstance(eval_info, dict):
            for sub in ("evaluation_command_item", "ground_truth_command_item",
                        "extra_evaluation_command_item"):
                if sub in eval_info:
                    eval_info[sub] = _maybe_parse_obj(eval_info[sub])
            cleaned["evaluation_info"] = eval_info
        skills = cleaned.get("skill_list")
        if isinstance(skills, str):
            skills = _maybe_parse_obj(skills)
        if skills is None:
            skills = []
        if not isinstance(skills, list):
            skills = [skills]
        cleaned["skill_list"] = skills
        cleaned.setdefault("raw_entry_hash", "")

        self.dataset_item = OSInteraction._construct_dataset_item(cleaned)
        self.container = OSInteractionContainer(timeout)
        self.timeout = timeout

    def reset(self) -> str:
        result = self.container.execute_independent(
            self.dataset_item.initialization_command_item
        )
        if result.timeout_flag or result.exit_code != 0:
            raise RuntimeError(
                f"OS initialization failed: exit={result.exit_code} output={result.output}"
            )
        return "Container initialized."

    def execute(self, command: str) -> str:
        result = self.container.execute_independent(
            self.CommandItem(command_name=self.CommandName.BASH, script=command)
        )
        if result.timeout_flag:
            return (
                f"The command is marked as timeout since it did not finish "
                f"within {self.timeout} seconds."
            )
        return result.output or ""

    def submit(self) -> tuple[bool, str]:
        """运行官方隐藏评测命令，exit_code == 0 即通过。"""
        result = self.container.execute_independent(
            self.dataset_item.evaluation_info.evaluation_command_item
        )
        output = "timeout" if result.timeout_flag else (result.output or "")
        ok = (not result.timeout_flag) and result.exit_code == 0
        return bool(ok), str(output)

    def close(self) -> None:
        try:
            _run_with_timeout(self.container.terminate, timeout_s=10.0, label="OS container terminate")
        except Exception as exc:  # noqa
            print(f"[lifelong_os] runtime cleanup failed: {exc}")


class LifelongOSTask(Task):
    def __init__(self, task_id: str, entry: dict[str, Any], max_steps: int,
                 os_timeout: int = DEFAULT_OS_TIMEOUT):
        self.task_id = task_id
        self.entry = entry
        self.max_steps = max_steps
        self.os_timeout = os_timeout
        self.runtime: Optional[_OSRuntime] = None
        self.done = False
        self.steps = 0

    def _instruction(self) -> str:
        return str(self.entry.get("instruction", ""))

    def get_basic_info(self) -> BasicInfo:
        return BasicInfo(
            env_id=self.task_id,
            instruction=self._instruction(),
            action_space="execute(bash) / finish",
            max_steps=self.max_steps,
            prompt_type="os",
            meta_data={
                "task_type": "os",
                "skill_tags": self.entry.get("skill_list") or [],
                # native agent 配置：OS 用 execute+finish，finish 无参数、无守卫。
                "native_tools": OS_TOOLS,
                "native_system_prompt": OS_NATIVE_SYSTEM_PROMPT,
                "finish_action": "finish",
                "guardrail_command": None,
            },
        )

    async def reset(self) -> Observation:
        self.done = False
        self.steps = 0

        def _spawn() -> tuple[_OSRuntime, str]:
            rt = _OSRuntime(self.entry, timeout=self.os_timeout)
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
        action_type = str(action.get("action", "")).lower()
        params = action.get("params", {}) if isinstance(action.get("params"), dict) else {}

        if action_type == "execute":
            command = str(params.get("command", "")).strip()
            if not command:
                return {"error": "No command provided"}, 0.0, False, {"error": "no_command"}
            output = await asyncio.to_thread(self.runtime.execute, command)
            # execute 从不结束 episode：到达 max_steps 由 runner 循环边界收束。
            return (
                {"command": command, "output": output, "current_step": self.steps,
                 "max_steps": self.max_steps},
                0.0,
                False,
                {},
            )

        if action_type == "finish":
            success, observed = await asyncio.to_thread(self.runtime.submit)
            self.done = True
            return (
                {"message": "Task finished", "success": success, "output": observed,
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


class LifelongOSSuite(TaskSuite):
    name = "lifelong_os"

    def __init__(self, data_dir: str, split: str = "test", max_steps: Optional[int] = None,
                 os_timeout: int = DEFAULT_OS_TIMEOUT):
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_steps = max_steps or DEFAULT_MAX_STEPS
        self.os_timeout = os_timeout
        candidate = self.data_dir / "os" / f"{split}.jsonl"
        if not candidate.is_file():
            candidate = self.data_dir / f"{split}.jsonl"
        self.rows = _load_jsonl(candidate)
        self._ids = [str(r.get("task_id") or f"os_{i}") for i, r in enumerate(self.rows)]

    def list_task_ids(self) -> List[str]:
        return self._ids

    def make_task(self, index: int) -> Task:
        return LifelongOSTask(self._ids[index], self.rows[index], self.max_steps,
                              os_timeout=self.os_timeout)

    def scorer(self) -> Scorer:
        return ProgrammaticScorer(reward_threshold=1.0)

    def default_max_steps(self) -> int:
        return self.max_steps


def _factory(args: Any) -> TaskSuite:
    return LifelongOSSuite(
        data_dir=args.data_dir,
        split=args.split,
        max_steps=args.max_steps,
        os_timeout=getattr(args, "os_timeout", None) or DEFAULT_OS_TIMEOUT,
    )


registry.register("lifelong_os", _factory)
