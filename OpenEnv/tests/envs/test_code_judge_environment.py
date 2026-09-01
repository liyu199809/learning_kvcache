"""CPU-only tests for the DeepCoder/TACO CodeJudge OpenEnv adapter."""

from __future__ import annotations

import os
import sys
from typing import Any

import pytest
from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

from envs.code_judge_env.server.code_judge_environment import CodeJudgeEnvironment
from envs.code_judge_env.server.data_loader import parse_tests
from envs.code_judge_env.server.judge import extract_python_code, judge_answer


class FakeLoader:
    def get_task(self, task_idx: int) -> dict[str, Any]:
        if task_idx != 0:
            raise ValueError("unknown task")
        return {
            "task_id": "taco_0",
            "task_idx": 0,
            "problem": "Read one integer and print it.",
            "tests": {"inputs": ["1\n"], "outputs": ["1\n"]},
        }


def fenced(code: str) -> str:
    return f"Reasoning before code.\n```python\n{code}\n```"


def test_parse_tests_accepts_stdin_and_functional_formats() -> None:
    stdin = parse_tests('{"inputs":["1\\n"],"outputs":["1\\n"]}')
    functional = parse_tests(
        '{"inputs":[[1,2]],"outputs":[3],"fn_name":"add"}'
    )

    assert stdin == {"inputs": ["1\n"], "outputs": ["1\n"]}
    assert functional["fn_name"] == "add"


@pytest.mark.parametrize(
    "payload",
    [
        '{"inputs":[],"outputs":[]}',
        '{"inputs":[1],"outputs":[]}',
        '{"inputs":[1],"outputs":[1],"fn_name":""}',
    ],
)
def test_parse_tests_rejects_invalid_payloads(payload: str) -> None:
    with pytest.raises((TypeError, ValueError)):
        parse_tests(payload)


def test_extract_python_code_uses_last_fenced_block() -> None:
    response = "```python\nprint('old')\n```\n```py\nprint('new')\n```"
    assert extract_python_code(response) == "print('new')"
    assert extract_python_code("print('no fence')") is None


def test_stdin_judge_returns_fractional_reward() -> None:
    response = judge_answer(
        fenced("value = int(input())\nprint(value if value == 1 else 0)"),
        {"inputs": ["1\n", "2\n"], "outputs": ["1\n", "2\n"]},
    )

    assert response["ok"] is True
    assert response["reward"] == 0.5
    assert response["reward_type"] == "incomplete"
    assert response["verify_result"]["passed_tests"] == 1


def test_functional_judge_supports_solution_class() -> None:
    response = judge_answer(
        fenced("class Solution:\n    def add(self, left, right):\n        return left + right"),
        {
            "fn_name": "add",
            "inputs": [[1, 2], [-4, 7]],
            "outputs": [3, 3],
        },
    )

    assert response["ok"] is True
    assert response["reward"] == 1.0
    assert response["reward_type"] == "complete"
    assert response["verify_result"]["test_type"] == "functional"


def test_format_and_compile_errors_are_distinct() -> None:
    tests = {"inputs": ["1\n"], "outputs": ["1\n"]}
    format_error = judge_answer("print(1)", tests)
    compile_error = judge_answer(fenced("if:"), tests)

    assert format_error["reward_type"] == "format_error"
    assert format_error["reward"] == 0.0
    assert compile_error["reward_type"] == "compile_error"
    assert compile_error["reward"] == 0.0


def test_environment_has_no_agent_tools_and_forwards_dense_reward(monkeypatch) -> None:
    def fake_judge(_answer: str, _tests: dict[str, Any]) -> dict[str, Any]:
        return {
            "ok": True,
            "reward": 0.5,
            "reward_type": "incomplete",
            "verify_result": {"passed_tests": 1, "total_tests": 2, "all_passed": False},
        }

    monkeypatch.setattr(
        "envs.code_judge_env.server.code_judge_environment.judge_answer",
        fake_judge,
    )
    environment = CodeJudgeEnvironment(FakeLoader())
    try:
        reset = environment.reset(scenario="taco", task_idx=0)
        assert reset.reward_type == "reset_ok"
        assert reset.num_tools == 0
        assert environment.step(ListToolsAction()).tools == []

        verified = environment.step(
            CallToolAction(
                tool_name="verify",
                arguments={"verifier_mode": "code", "final_answer": fenced("print(1)")},
            )
        )
        assert verified.reward == 0.5
        assert verified.reward_type == "incomplete"
        assert verified.done is True
    finally:
        environment.close()
