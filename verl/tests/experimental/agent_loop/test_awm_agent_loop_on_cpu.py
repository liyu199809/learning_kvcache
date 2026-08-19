"""CPU tests for the project-local OpenEnv AWM agent loop."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import dataclass
from functools import wraps
from types import SimpleNamespace

import pytest

from rollout.verl_awm_agent_loop import AWMAgentLoop
from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput
from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop
from verl.tools.schemas import ToolResponse


TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "set_value",
        "description": "Set a value in the environment.",
        "parameters": {
            "type": "object",
            "properties": {"value": {"type": "integer", "description": "Value to store."}},
            "required": ["value"],
        },
    },
}


def run_async_test(function):
    """Run async CPU tests without requiring the optional pytest-asyncio plugin."""

    @wraps(function)
    def wrapper(*args, **kwargs):
        return asyncio.run(function(*args, **kwargs))

    return wrapper


@dataclass
class FakeResult:
    observation: dict
    reward: float | None = None
    done: bool = False
    metadata: dict | None = None


class FakeClient:
    def __init__(self, *, reset_error=None, step_error=None, verify_reward_type="complete", delay=0.0):
        self.reset_error = reset_error
        self.step_error = step_error
        self.verify_reward_type = verify_reward_type
        self.delay = delay
        self.reset_calls = []
        self.step_calls = []
        self.close_calls = 0
        self.active = 0
        self.max_active = 0
        self.guard = threading.Lock()

    def reset(self, **kwargs):
        self.reset_calls.append(kwargs)
        if self.reset_error:
            raise self.reset_error
        return FakeResult({"reward_type": "reset_ok"})

    def step(self, action):
        with self.guard:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if self.delay:
                time.sleep(self.delay)
            self.step_calls.append(action)
            if self.step_error and action["tool_name"] != "verify":
                raise self.step_error
            if action["tool_name"] == "verify":
                return FakeResult({"reward_type": self.verify_reward_type})
            return FakeResult({"tool_result": {"stored": action["arguments"]["value"]}})
        finally:
            with self.guard:
                self.active -= 1

    def close(self):
        self.close_calls += 1


def make_loop(client: FakeClient) -> AWMAgentLoop:
    loop = object.__new__(AWMAgentLoop)
    loop._env_client = None
    loop._env_lock = asyncio.Lock()
    loop._reset_succeeded = False
    loop._transport_failed = False
    loop._transport_error = ""
    loop._last_assistant_text = "final"
    loop._verify_reward_type = "not_verified"
    loop._close_error = ""
    loop.tokenizer = SimpleNamespace(eos_token="<eos>", pad_token="<pad>")
    loop.tools = {}
    loop.tool_schemas = []
    loop._create_sync_client = lambda _base_url: client
    return loop


def make_output() -> AgentLoopOutput:
    return AgentLoopOutput(
        prompt_ids=[1],
        response_ids=[2],
        response_mask=[1],
        num_turns=2,
        metrics=AgentLoopMetrics(),
        extra_fields={},
    )


@pytest.mark.parametrize("raw_tools", [[TOOL_SCHEMA], json.dumps([TOOL_SCHEMA])])
def test_tool_schemas_accept_list_and_json(raw_tools):
    schemas = AWMAgentLoop.parse_tool_schemas(raw_tools)
    assert [schema.function.name for schema in schemas] == ["set_value"]


def test_duplicate_tool_names_are_rejected():
    with pytest.raises(ValueError, match="Duplicate tool names"):
        AWMAgentLoop.parse_tool_schemas([TOOL_SCHEMA, TOOL_SCHEMA])


@run_async_test
async def test_calls_share_one_client_and_are_serialized():
    client = FakeClient(delay=0.02)
    loop = make_loop(client)
    config = loop._normalize_env_config({"scenario": "marketplace_1", "task_idx": 3})
    await loop._reset_openenv(config)

    first, second = await asyncio.gather(
        loop._call_openenv_tool("set_value", {"value": 1}),
        loop._call_openenv_tool("set_value", {"value": 2}),
    )
    loop._last_assistant_text = "done"
    reward = await loop._verify()
    await loop._close_openenv()

    assert json.loads(first.text) == {"stored": 1}
    assert json.loads(second.text) == {"stored": 2}
    assert client.max_active == 1
    assert len(client.reset_calls) == 1
    assert [call["tool_name"] for call in client.step_calls] == ["set_value", "set_value", "verify"]
    assert client.step_calls[-1]["arguments"] == {"verifier_mode": "code", "final_answer": "done"}
    assert reward == 1.0
    assert client.close_calls == 1


@run_async_test
async def test_observation_error_is_returned_without_losing_session():
    client = FakeClient()

    def application_error(_action):
        client.step_calls.append(_action)
        return FakeResult({"error": "invalid business input"})

    client.step = application_error
    loop = make_loop(client)
    await loop._reset_openenv(loop._normalize_env_config({"scenario": "s", "task_idx": 1}))
    response = await loop._call_openenv_tool("set_value", {"value": 1})

    assert response.text == "Error: invalid business input"
    assert loop._transport_failed is False


@run_async_test
async def test_tool_result_gets_budget_reminder_after_native_truncation(monkeypatch):
    loop = make_loop(FakeClient())
    loop.max_user_turns = 5
    loop.max_assistant_turns = 6

    async def fake_native_call(_self, _tool_call, _tools_kwargs, _agent_data):
        # ToolAgentLoop has already applied max_tool_response_length at this point.
        return ToolResponse(text="x" * 800 + "...(truncated)"), 0.0, {}

    monkeypatch.setattr(ToolAgentLoop, "_call_tool", fake_native_call)

    first = SimpleNamespace(user_turns=0, assistant_turns=1)
    first_response, _, _ = await AWMAgentLoop._call_tool(loop, None, {}, first)
    assert first_response.text.startswith("x" * 800 + "...(truncated)")
    assert "You have 4 remaining opportunities" in first_response.text

    last = SimpleNamespace(user_turns=4, assistant_turns=5)
    last_response, _, _ = await AWMAgentLoop._call_tool(loop, None, {}, last)
    assert "You have 0 remaining opportunities" in last_response.text
    assert "must respond to the user directly" in last_response.text


@run_async_test
async def test_disconnect_fails_closed_without_retry_or_verify():
    client = FakeClient(step_error=ConnectionError("socket closed"))
    loop = make_loop(client)
    await loop._reset_openenv(loop._normalize_env_config({"scenario": "s", "task_idx": 1}))

    first = await loop._call_openenv_tool("set_value", {"value": 1})
    second = await loop._call_openenv_tool("set_value", {"value": 2})
    reward = await loop._verify()
    await loop._close_openenv()

    assert "transport error" in first.text.lower()
    assert "unavailable" in second.text.lower()
    assert len(client.reset_calls) == 1
    assert len(client.step_calls) == 1
    assert reward == 0.0
    assert loop._verify_reward_type == "transport_error"
    assert client.close_calls == 1


@run_async_test
@pytest.mark.parametrize("reward_type, expected", [("complete", 1.0), ("incomplete", 0.0), ("format_error", 0.0)])
async def test_verify_uses_binary_reward(reward_type, expected):
    client = FakeClient(verify_reward_type=reward_type)
    loop = make_loop(client)
    await loop._reset_openenv(loop._normalize_env_config({"scenario": "s", "task_idx": 1}))

    assert await loop._verify() == expected
    assert loop._verify_reward_type == reward_type


@run_async_test
async def test_run_reset_failure_returns_zero_and_closes(monkeypatch):
    client = FakeClient(reset_error=ConnectionError("server unavailable"))
    loop = make_loop(client)

    async def fake_tool_agent_run(_self, _sampling_params, **_kwargs):
        return make_output()

    monkeypatch.setattr(ToolAgentLoop, "run", fake_tool_agent_run)
    output = await AWMAgentLoop.run(
        loop,
        {},
        tools=json.dumps([TOOL_SCHEMA]),
        env_config={"scenario": "marketplace_1", "task_idx": 3},
    )

    assert output.reward_score == 0.0
    assert output.extra_fields["awm_reward_type"] == "transport_error"
    assert "reset" in output.extra_fields["awm_error"]
    assert len(client.reset_calls) == 1
    assert client.step_calls == []
    assert client.close_calls == 1


@run_async_test
async def test_run_records_reward_and_diagnostics(monkeypatch):
    client = FakeClient()
    loop = make_loop(client)

    async def fake_tool_agent_run(self, _sampling_params, **_kwargs):
        response = await self.tools["set_value"].call({"value": 7})
        assert json.loads(response.text) == {"stored": 7}
        self._last_assistant_text = "task finished"
        return make_output()

    monkeypatch.setattr(ToolAgentLoop, "run", fake_tool_agent_run)
    output = await AWMAgentLoop.run(
        loop,
        {},
        tools=[TOOL_SCHEMA],
        env_config={
            "scenario": "marketplace_1",
            "task_idx": 3,
            "awm_base_url": "http://localhost:8899/",
        },
    )

    assert output.reward_score == 1.0
    assert output.extra_fields["awm_reward_type"] == "complete"
    assert output.extra_fields["awm_final_answer"] == "task finished"
    assert output.extra_fields["reward_extra_info"]["awm_task_idx"] == 3
    assert [call["tool_name"] for call in client.step_calls] == ["set_value", "verify"]
    assert client.close_calls == 1
