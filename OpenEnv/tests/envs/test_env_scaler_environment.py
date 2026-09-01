"""CPU-only tests for the EnvScaler OpenEnv adapter."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

from envs.env_scaler_env.server.data_loader import EnvScalerDataLoader
from envs.env_scaler_env.server.env_scaler_environment import EnvScalerEnvironment


ENV_CODE = '''
class CounterEnvironment:
    def __init__(self, config=None):
        self.value = 0

    def get_value(self):
        return {"value": self.value}

    def set_value(self, value):
        self.value = value
        return {"success": True, "value": self.value}
'''


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_value",
            "description": "Read the current value",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_value",
            "description": "Set the current value",
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
                "additionalProperties": False,
            },
        },
    },
]


def _write_fixture_data(path: Path) -> None:
    environments = {
        "env_1_rl": {
            "env_id": "env_1_rl",
            "environment_summary": "Counter",
            "env_class_name": "CounterEnvironment",
            "env_class_code": ENV_CODE,
            "tools": TOOLS,
        }
    }
    tasks = []
    for suffix in (10, 2, 1):
        tasks.append(
            {
                "env_id": "env_1_rl",
                "env_class_name": "CounterEnvironment",
                "task_id": f"env_1_rl-task_{suffix}",
                "init_config": {"value": 1},
                "task": f"Set the value for task {suffix}",
                "checklist_with_func": [
                    {
                        "check_item": "Final value is three",
                        "check_func": "def check_func(final_state):\n    return final_state['value'] == 3",
                    },
                    {
                        "check_item": "Initial value was one",
                        "check_func": "def check_func(final_state):\n    return initial_state['value'] == 1",
                    },
                ],
            }
        )
    (path / "191_env_metadata.json").write_text(json.dumps(environments), encoding="utf-8")
    (path / "envscaler_rl_scenario_metadata.json").write_text(json.dumps(tasks), encoding="utf-8")


def _loader(tmp_path: Path) -> EnvScalerDataLoader:
    _write_fixture_data(tmp_path)
    return EnvScalerDataLoader(cache_dir=tmp_path, download_if_missing=False)


def test_data_loader_builds_stable_per_environment_task_index(tmp_path: Path) -> None:
    loader = _loader(tmp_path)

    assert loader.get_task("env_1_rl", 0)["task_id"] == "env_1_rl-task_1"
    assert loader.get_task("env_1_rl", 1)["task_id"] == "env_1_rl-task_2"
    assert loader.get_task("env_1_rl", 2)["task_id"] == "env_1_rl-task_10"
    assert loader.stats()["task_count"] == 3


def test_reset_tools_calls_and_dense_verification(tmp_path: Path) -> None:
    environment = EnvScalerEnvironment(data_loader=_loader(tmp_path))
    try:
        reset = environment.reset(scenario="env_1_rl", task_idx=0)
        assert reset.reward_type == "reset_ok"
        assert reset.task_id == "env_1_rl-task_1"
        assert reset.num_tools == 2

        listed = environment.step(ListToolsAction())
        assert [tool.name for tool in listed.tools] == ["get_value", "set_value"]

        before = environment.step(
            CallToolAction(tool_name="get_value", arguments={})
        )
        assert before.tool_result == {"value": 1}

        partial = environment.step(
            CallToolAction(
                tool_name="verify",
                arguments={"verifier_mode": "code", "final_answer": "not done"},
            )
        )
        assert partial.reward_type == "incomplete"
        assert partial.reward == 0.5

        updated = environment.step(
            CallToolAction(tool_name="set_value", arguments={"value": 3})
        )
        assert updated.tool_result["value"] == 3

        complete = environment.step(
            CallToolAction(
                tool_name="verify",
                arguments={"verifier_mode": "code", "final_answer": "done"},
            )
        )
        assert complete.reward_type == "complete"
        assert complete.reward == 1.0
        assert complete.verify_result["passed_checks"] == 2
    finally:
        environment.close()


def test_invalid_arguments_do_not_kill_episode(tmp_path: Path) -> None:
    environment = EnvScalerEnvironment(data_loader=_loader(tmp_path))
    try:
        assert environment.reset(scenario="env_1_rl", task_idx=0).reward_type == "reset_ok"
        invalid = environment.step(
            CallToolAction(tool_name="set_value", arguments={"value": "three"})
        )
        assert invalid.reward_type == "invalid_args"
        assert environment.step(
            CallToolAction(tool_name="get_value", arguments={})
        ).tool_result == {"value": 1}
    finally:
        environment.close()


def test_sessions_have_isolated_state(tmp_path: Path) -> None:
    loader = _loader(tmp_path)
    first = EnvScalerEnvironment(data_loader=loader)
    second = EnvScalerEnvironment(data_loader=loader)
    try:
        assert first.reset(scenario="env_1_rl", task_idx=0).reward_type == "reset_ok"
        assert second.reset(scenario="env_1_rl", task_idx=0).reward_type == "reset_ok"
        first.step(CallToolAction(tool_name="set_value", arguments={"value": 3}))
        second_value = second.step(CallToolAction(tool_name="get_value", arguments={}))
        assert second_value.tool_result == {"value": 1}
    finally:
        first.close()
        second.close()
