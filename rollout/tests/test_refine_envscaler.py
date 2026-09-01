import asyncio
import json
from types import SimpleNamespace

import pytest

from agent_world_model_env.models import AWMObservation
from rollout import refine
from rollout.common import tools_to_openai_schema
from rollout.refine2swift import convert


def _detail(round_no, score, reward_type="incomplete", advice=None,
            message_range=None, checker_errors=0, final_answer=""):
    return {
        "round": round_no,
        "message_range": message_range or [0, 0],
        "verify_reward": score,
        "verify_reward_type": reward_type,
        "final_reward_type": reward_type,
        "verify_result": {"checker_errors": checker_errors},
        "verify_error": None,
        "teacher_advice": advice,
        "student_final_answer": final_answer,
    }


def _conversation(tag):
    return [
        {"role": "system", "content": "system"},
        {"role": "user", "content": f"task-{tag}"},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "r1", "type": "function",
            "function": {"name": "round1_tool", "arguments": "{}"},
        }]},
        {"role": "tool", "tool_call_id": "r1", "content": "ok"},
        {"role": "assistant", "content": "round1 final"},
        {"role": "user", "content": "[Expert advice] advice one"},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "r2", "type": "function",
            "function": {"name": "round2_tool", "arguments": "{}"},
        }]},
        {"role": "tool", "tool_call_id": "r2", "content": "ok"},
        {"role": "assistant", "content": "round2 final"},
        {"role": "user", "content": "[Expert advice] advice two"},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "r3", "type": "function",
            "function": {"name": "round3_tool", "arguments": "{}"},
        }]},
        {"role": "tool", "tool_call_id": "r3", "content": "ok"},
        {"role": "assistant", "content": "round3 final"},
    ]


def _record(task_idx, details):
    return {
        "data_source": "envscaler_rl",
        "scenario": "env_test_rl",
        "task_idx": task_idx,
        "task_id": f"task-{task_idx}",
        "student_conversation": _conversation(task_idx),
        "rounds_detail": details,
    }


def test_envscaler_backend_defaults_and_all_task_enumeration(monkeypatch):
    monkeypatch.delenv("ENV_BASE_URL", raising=False)
    monkeypatch.delenv("ENVSCALER_BASE_URL", raising=False)
    args = refine._build_parser().parse_args(["--env-backend", "envscaler"])
    profile = refine._resolve_backend_args(args)
    assert args.awm_base_url == "http://127.0.0.1:8900"
    assert args.tasks_per_scenario is None
    assert args.student_max_iterations == 16
    assert args.llm_judge is False
    assert profile.data_source == "envscaler_rl"
    assert profile.judge_authoritative is False

    async def fake_list(_):
        return [
            {"name": "env_a_rl", "num_tasks": 2},
            {"name": "env_b_rl", "num_tasks": 3},
        ]

    monkeypatch.setattr(refine, "_list_all_scenarios", fake_list)
    args.num_scenarios = 100
    assert asyncio.run(refine._select_work(args)) == [
        ("env_a_rl", 0), ("env_a_rl", 1),
        ("env_b_rl", 0), ("env_b_rl", 1), ("env_b_rl", 2),
    ]


def test_deepcoder_profile_discovers_all_tasks(monkeypatch):
    monkeypatch.delenv("ENV_BASE_URL", raising=False)
    monkeypatch.delenv("CODE_JUDGE_BASE_URL", raising=False)
    args = refine._build_parser().parse_args(
        ["--dataset", "deepcoder-taco"])
    profile = refine._resolve_backend_args(args)
    assert args.awm_base_url == "http://127.0.0.1:8901"
    assert args.tasks_per_scenario is None
    assert args.student_max_iterations == 1
    assert args.student_max_tokens == 8192
    assert args.teacher_max_tool_calls == 0
    assert args.llm_judge is False
    assert profile.data_source == "deepcoder_taco"
    assert profile.include_verify_summary is True

    async def fake_discover(_profile, _url):
        return [{"name": "taco", "num_tasks": 7436}]

    monkeypatch.setattr(refine, "_discover_scenarios", fake_discover)
    work = asyncio.run(refine._select_work(args, profile))
    assert len(work) == 7436
    assert work[0] == ("taco", 0)
    assert work[-1] == ("taco", 7435)


def test_awm_judge_disables_thinking_without_changing_teacher(monkeypatch):
    captured = {}

    async def fake_judge(*args, **kwargs):
        captured.update(kwargs)
        return {
            "classification": "complete", "reasoning": "", "error": None,
            "confidence_score": 1.0,
        }

    monkeypatch.setattr(refine, "_llm_judge_round", fake_judge)
    profile = refine._backend_profile("awm")
    job = refine.RefineJob(
        scenario="env", task_idx=0, student_llm=None, teacher_llm=None,
        awm_base_url="http://unused", judge_llm=object(), use_llm_judge=True,
        judge_authoritative=True, judge_extra_body=profile.judge_extra_body,
    )
    _, final_reward, _ = asyncio.run(job._judge(
        "task", [], {"final_answer": ""},
        {"reward_type": "incomplete", "verify_result": {}},
    ))
    assert final_reward == "complete"
    assert captured["extra_body"] == {"thinking": {"type": "disabled"}}
    assert profile.teacher_system_prompt == refine.TEACHER_ADVICE_SYSTEM_PROMPT


def test_code_judge_verify_summary_contains_only_aggregate_feedback():
    summary = refine._verify_summary({
        "reward": 0.5,
        "verify_result": {
            "passed_tests": 2, "total_tests": 4, "all_passed": False,
            "runtime_errors": 1, "hidden_output": "must-not-leak",
        },
    })
    assert "score=0.5" in summary
    assert "passed_tests=2" in summary
    assert "runtime_errors=1" in summary
    assert "hidden_output" not in summary


def test_envscaler_verifier_remains_authoritative(monkeypatch):
    async def fake_judge(*args, **kwargs):
        return {
            "classification": "complete", "reasoning": "", "error": None,
            "confidence_score": 1.0,
        }

    monkeypatch.setattr(refine, "_llm_judge_round", fake_judge)
    job = refine.RefineJob(
        scenario="env", task_idx=0, student_llm=None, teacher_llm=None,
        awm_base_url="http://unused", judge_llm=object(), use_llm_judge=True,
        judge_authoritative=False,
    )
    _, final_reward, fallback = asyncio.run(job._judge(
        "task", [], {"final_answer": ""},
        {"reward_type": "incomplete", "verify_result": {}},
    ))
    assert final_reward == "incomplete"
    assert fallback is False


def test_score_outcome_classifies_partial_improvement():
    details = [
        _detail(1, 0.2),
        _detail(2, 0.6),
        _detail(3, 0.6),
    ]
    summary = refine._score_outcome(details, None)
    assert summary == {
        "baseline_score": 0.2,
        "best_score": 0.6,
        "best_round": 2,
        "score_improved": True,
        "sample_candidate_type": "improved_partial",
    }


def test_envscaler_task_id_and_unconstrained_schema_are_compatible():
    observation = AWMObservation(task_id="abc", task="do it")
    assert observation.task_id == "abc"

    original = {"type": "object", "properties": {"payload": {}}}
    tool = SimpleNamespace(
        name="update", description="", input_schema=original)
    schema = tools_to_openai_schema([tool])[0]["function"]["parameters"]
    assert set(schema["properties"]["payload"]["type"]) == {
        "string", "number", "boolean", "object", "array", "null",
    }
    assert original["properties"]["payload"] == {}


def test_converter_keeps_three_categories_and_matches_immediate_advice(tmp_path):
    records = [
        _record(0, [_detail(1, 1.0, "complete", message_range=[2, 5])]),
        _record(1, [
            _detail(1, 0.2, advice="advice one", message_range=[2, 5]),
            _detail(2, 1.0, "complete", message_range=[6, 9]),
        ]),
        _record(2, [
            _detail(1, 0.2, advice="advice one", message_range=[2, 5]),
            _detail(2, 0.4, advice="advice two", message_range=[6, 9]),
            _detail(3, 0.7, message_range=[10, 13]),
        ]),
        _record(3, [
            _detail(1, 0.5, advice="unused", message_range=[2, 5]),
            _detail(2, 0.5, message_range=[6, 9]),
        ]),
        _record(4, [
            _detail(1, 0.1, advice="bad", message_range=[2, 5]),
            _detail(2, 0.8, message_range=[6, 9], checker_errors=1),
        ]),
    ]
    source = tmp_path / "refine.jsonl"
    output = tmp_path / "swift.jsonl"
    source.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    convert(str(source), str(output), with_tools=False)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert {row["sample_type"] for row in rows} == {
        "complete_first_try", "complete_after_refine", "improved_partial",
    }

    partial = next(row for row in rows
                   if row["sample_type"] == "improved_partial")
    assert partial["source_round"] == 2
    assert partial["target_round"] == 3
    assert partial["verify_reward_type"] == "incomplete"
    assert partial["score_delta"] == pytest.approx(0.5)
    assert "advice two" in partial["teacher_prompt"]
    assert "advice one" not in partial["teacher_prompt"]
    rendered = "\n".join(message["content"] for message in partial["messages"])
    assert "round2_tool" in rendered
    assert "round1_tool" not in rendered


def test_converter_uses_verified_code_as_first_try_reference(tmp_path):
    solution = "```python\nprint(input())\n```"
    record = _record(9, [
        _detail(1, 1.0, "complete", message_range=[2, 5],
                final_answer=solution),
    ])
    record["data_source"] = "deepcoder_taco"
    record["scenario"] = "taco"
    record["task_id"] = "taco_9"
    source = tmp_path / "code-refine.jsonl"
    output = tmp_path / "code-swift.jsonl"
    source.write_text(json.dumps(record) + "\n", encoding="utf-8")

    convert(str(source), str(output), with_tools=False)
    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["sample_type"] == "complete_first_try"
    assert row["verify_reward_type"] == "complete"
    assert "[Reference solution]" in row["teacher_prompt"]
    assert solution in row["teacher_prompt"]
