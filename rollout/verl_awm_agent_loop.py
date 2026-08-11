"""verl agent loop for Agent World Model environments served by OpenEnv.

One :class:`AWMAgentLoop` instance is created for each rollout trajectory by
verl.  The instance therefore owns exactly one OpenEnv WebSocket session.  The
session is reset once, reused by every tool call and by the final verifier, and
closed when the trajectory finishes.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from openenv.core.generic_client import GenericEnvClient

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, ToolAgentLoop
from verl.tools.function_tool import FunctionTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)


class AWMAgentLoop(ToolAgentLoop):
    """ToolAgentLoop backed by one persistent AWM OpenEnv session."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._env_client = None
        self._env_lock = asyncio.Lock()
        self._reset_succeeded = False
        self._transport_failed = False
        self._transport_error = ""
        self._last_assistant_text = ""
        self._verify_reward_type = "not_verified"
        self._close_error = ""

    @staticmethod
    def _as_python(value: Any) -> Any:
        """Unwrap scalar numpy/Arrow containers passed through non-tensor data."""
        if hasattr(value, "item"):
            try:
                return value.item()
            except (TypeError, ValueError):
                pass
        return value

    @classmethod
    def parse_tool_schemas(cls, raw_tools: Any) -> list[OpenAIFunctionToolSchema]:
        """Parse the per-sample OpenAI tool schemas from JSON text or a list."""
        raw_tools = cls._as_python(raw_tools)
        if isinstance(raw_tools, str):
            try:
                raw_tools = json.loads(raw_tools)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in sample tools: {exc}") from exc
        elif hasattr(raw_tools, "tolist"):
            raw_tools = raw_tools.tolist()
        if isinstance(raw_tools, tuple):
            raw_tools = list(raw_tools)
        if not isinstance(raw_tools, list):
            raise TypeError(f"Sample tools must be a JSON list or list, got {type(raw_tools).__name__}")

        schemas = [OpenAIFunctionToolSchema.model_validate(schema) for schema in raw_tools]
        names = [schema.function.name for schema in schemas]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"Duplicate tool names in sample: {duplicates}")
        return schemas

    def _install_sample_tools(self, raw_tools: Any) -> None:
        schemas = self.parse_tool_schemas(raw_tools)
        tools: dict[str, FunctionTool] = {}
        for schema in schemas:
            name = schema.function.name

            def make_proxy(tool_name: str):
                async def proxy(**parameters):
                    return await self._call_openenv_tool(tool_name, parameters)

                return proxy

            tools[name] = FunctionTool(
                name=name,
                fn=make_proxy(name),
                tool_schema=schema,
                is_async=True,
            )

        # These tools belong only to this trajectory.  Do not touch verl's
        # process-global FunctionTool registry.
        self.tools = tools
        self.tool_schemas = [schema.model_dump(exclude_unset=True, exclude_none=True) for schema in schemas]

    @staticmethod
    def _normalize_env_config(raw_config: Any) -> dict[str, Any]:
        raw_config = AWMAgentLoop._as_python(raw_config)
        if not isinstance(raw_config, dict):
            raise TypeError(f"env_config must be a dict, got {type(raw_config).__name__}")
        config = dict(raw_config)
        scenario = AWMAgentLoop._as_python(config.get("scenario"))
        task_idx = AWMAgentLoop._as_python(config.get("task_idx"))
        if scenario is None or task_idx is None:
            raise ValueError(f"env_config missing scenario/task_idx: {config}")
        return {
            "scenario": str(scenario),
            "task_idx": int(task_idx),
            "awm_base_url": str(config.get("awm_base_url") or "http://localhost:8899").rstrip("/"),
        }

    def _create_sync_client(self, base_url: str):
        """Factory seam used by CPU tests; the production path uses OpenEnv."""
        return GenericEnvClient(base_url=base_url).sync()

    @staticmethod
    def _observation(result: Any) -> dict[str, Any]:
        observation = getattr(result, "observation", result)
        return observation if isinstance(observation, dict) else {"tool_result": observation}

    @staticmethod
    def _observation_text(observation: dict[str, Any]) -> str:
        tool_result = observation.get("tool_result")
        if tool_result is not None:
            if isinstance(tool_result, str):
                return tool_result
            return json.dumps(tool_result, ensure_ascii=False)
        if observation.get("error"):
            return f"Error: {observation['error']}"
        return json.dumps(observation, ensure_ascii=False)

    def _mark_transport_failed(self, stage: str, exc: BaseException | str) -> None:
        if not self._transport_failed:
            self._transport_error = f"{stage}: {exc}"
        self._transport_failed = True
        logger.warning("AWM trajectory transport failure: %s", self._transport_error)

    async def _reset_openenv(self, env_config: dict[str, Any]) -> None:
        try:
            self._env_client = self._create_sync_client(env_config["awm_base_url"])
            result = await asyncio.to_thread(
                self._env_client.reset,
                scenario=env_config["scenario"],
                task_idx=env_config["task_idx"],
            )
            observation = self._observation(result)
            if observation.get("error"):
                raise RuntimeError(observation["error"])
            reward_type = observation.get("reward_type")
            if reward_type not in (None, "reset_ok"):
                raise RuntimeError(f"unexpected reset reward_type={reward_type!r}: {observation}")
            self._reset_succeeded = True
        except Exception as exc:  # OpenEnv exposes several transport exception types.
            self._mark_transport_failed("reset", exc)

    async def _call_openenv_tool(self, tool_name: str, parameters: dict[str, Any]) -> ToolResponse:
        """Forward one tool call through the trajectory's persistent session."""
        async with self._env_lock:
            if self._transport_failed or self._env_client is None:
                return ToolResponse(text=f"OpenEnv trajectory unavailable: {self._transport_error}")
            action = {
                "type": "call_tool",
                "tool_name": tool_name,
                "arguments": parameters,
            }
            try:
                result = await asyncio.to_thread(self._env_client.step, action)
            except Exception as exc:
                self._mark_transport_failed(f"step({tool_name})", exc)
                return ToolResponse(text=f"OpenEnv transport error: {self._transport_error}")
            # An observation-level error is a normal tool/application error.  It
            # is returned to the model and does not discard the environment.
            return ToolResponse(text=self._observation_text(self._observation(result)))

    async def _verify(self) -> float:
        if self._transport_failed or not self._reset_succeeded or self._env_client is None:
            self._verify_reward_type = "transport_error"
            return 0.0
        action = {
            "type": "call_tool",
            "tool_name": "verify",
            "arguments": {"verifier_mode": "code", "final_answer": self._last_assistant_text},
        }
        async with self._env_lock:
            try:
                result = await asyncio.to_thread(self._env_client.step, action)
            except Exception as exc:
                self._mark_transport_failed("verify", exc)
                self._verify_reward_type = "transport_error"
                return 0.0
        observation = self._observation(result)
        self._verify_reward_type = str(observation.get("reward_type") or "unknown")
        if observation.get("error") and not self._transport_error:
            self._transport_error = f"verify: {observation['error']}"
        return 1.0 if self._verify_reward_type == "complete" else 0.0

    async def _close_openenv(self) -> None:
        client, self._env_client = self._env_client, None
        if client is None:
            return
        try:
            await asyncio.to_thread(client.close)
        except Exception as exc:
            self._close_error = str(exc)
            logger.warning("Failed to close AWM OpenEnv session: %s", exc)

    def _clean_assistant_text(self, text: str) -> str:
        for token in {self.tokenizer.eos_token, self.tokenizer.pad_token}:
            if token:
                text = text.replace(token, "")
        return text.strip()

    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:
        state = await super()._handle_generating_state(agent_data, sampling_params, ignore_termination)
        try:
            active_tools = getattr(agent_data, "_active_tools", self.tools)
            schemas = [tool.tool_schema for tool in active_tools.values()]
            content, _ = await self.tool_parser.extract_tool_calls(agent_data.response_ids, schemas)
        except Exception:
            content = self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
        self._last_assistant_text = self._clean_assistant_text(content or "")
        return state

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        state = await super()._handle_processing_tools_state(agent_data)
        # The current batch of tool messages is retained for debugging, but no
        # further model turn is generated after a transport failure.
        return AgentState.TERMINATED if self._transport_failed else state

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """Run one model trajectory against one persistent OpenEnv session."""
        if "tools" not in kwargs:
            raise ValueError("AWMAgentLoop requires a per-sample 'tools' field")
        breakpoint()
        self._install_sample_tools(kwargs["tools"])
        env_config = self._normalize_env_config(kwargs.get("env_config"))

        output: AgentLoopOutput | None = None
        try:
            await self._reset_openenv(env_config)
            # Even if reset failed, produce at most one valid model turn so the
            # failed sample can carry reward=0 instead of crashing its batch.
            output = await super().run(sampling_params, **kwargs)
            output.reward_score = await self._verify()
        finally:
            await self._close_openenv()

        assert output is not None
        error = self._transport_error
        if self._close_error:
            error = f"{error}; close: {self._close_error}" if error else f"close: {self._close_error}"
        diagnostics = {
            "awm_scenario": env_config["scenario"],
            "awm_task_idx": env_config["task_idx"],
            "awm_reward_type": self._verify_reward_type,
            "awm_error": error,
            "awm_final_answer": self._last_assistant_text,
        }
        output.extra_fields.update(diagnostics)
        output.extra_fields.setdefault("reward_extra_info", {}).update(diagnostics)
        return output


__all__ = ["AWMAgentLoop"]
