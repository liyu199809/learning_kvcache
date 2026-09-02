"""Dataset filtering and iterative experience extraction.

Supported profiles are AWM tool tasks, EnvScaler checklist tasks, and
DeepCoder/TACO programming tasks. Every round starts from a fresh environment,
runs the student, verifies the result, and optionally asks a teacher for one
actionable ``# Advice:`` experience before retrying.

AWM keeps its LLM judge as the authoritative verdict, with judge thinking
disabled. EnvScaler and CodeJudge use their deterministic dense verifier scores.
Records retain the complete conversation, per-round score/evidence/advice, and a
single candidate classification: first-try complete, complete after refinement,
improved partial, or no improvement.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import time
import urllib.request
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction

from agent_world_model_env import AWMEnv

from rollout.common import (
    JsonlCheckpoint,
    RateLimiter,
    RetryLLM,
    execute_native_tool_call,
    llm_turn_native,
    loads_lenient,
    run_jobs,
    tools_to_openai_schema,
)
from rollout.prompt import (
    CODE_JUDGE_STUDENT_FINAL_SYSTEM_PROMPT,
    CODE_JUDGE_STUDENT_SYSTEM_PROMPT,
    CODE_JUDGE_TEACHER_ADVICE_SYSTEM_PROMPT,
    CODE_JUDGE_TEACHER_FINALIZE_SYSTEM_PROMPT,
    CODE_JUDGE_TEACHER_SOLUTION_FINALIZE_SYSTEM_PROMPT,
    CODE_JUDGE_TEACHER_SOLUTION_SYSTEM_PROMPT,
    ENVSCALER_STUDENT_NATIVE_FINAL_SYSTEM_PROMPT,
    ENVSCALER_STUDENT_NATIVE_SYSTEM_PROMPT,
    JUDGE_SYSTEM_PROMPT,
    STUDENT_NATIVE_SYSTEM_PROMPT,
    STUDENT_NATIVE_FINAL_SYSTEM_PROMPT,
    TEACHER_ADVICE_SYSTEM_PROMPT,
    TEACHER_FINALIZE_SYSTEM_PROMPT,
    build_judge_input,
    build_teacher_advice_input,
)

# 统一加载仓库根目录的 .env（student/teacher/judge 的 LLM 端点与 API key）。
# 与 benchmark/eval/run_eval.py 相同的约定：不覆盖 shell 里已有的变量，
# 因此临时 export 依然优先生效。文件已 gitignore，不装载入仓库。
# 可配置键：AWM_BASE_URL / ENDPOINT_URL / AWM_EXAMPLE_AGENT_MODEL /
# TEACHER_BASE_URL / TEACHER_MODEL / ARK_API_KEY /
# JUDGE_BASE_URL / JUDGE_API_KEY / JUDGE_MODEL。
load_dotenv(Path(__file__).resolve().parents[1] / ".env")


# Relaxed: matches "# Advice:" / "Advice:" / "advice:" with optional leading
# whitespace, and captures everything until the next blank line or end of text.
# Format is not important — we just want the actionable content.
ADVICE_LINE_RE = re.compile(
    r"^\s*#?\s*[Aa]dvice\s*:\s*(.+?)(?:\n\s*\n|\Z)",
    re.MULTILINE | re.DOTALL,
)

_HARNESS_TOOLS = frozenset({"verify", "done"})
_STUDENT_FORBIDDEN_TOOL_ERROR = (
    "Error: 'verify' and 'done' are managed by the harness; "
    "To finish, STOP calling tools and reply with your final answer as plain text."
)
_TEACHER_FORBIDDEN_TOOL_ERROR = (
    "Error: teacher may not call verify/done. To finish, STOP calling tools "
    "and reply with your final answer as plain text."
)
_TEACHER_FINALIZE_REQUEST = (
    "Probing phase is over. Do NOT call any tools. Emit your final diagnosis "
    "now, starting with `# Advice:` on its own line."
)
_EXPERT_ADVICE_TEMPLATE = (
    "[Expert advice]\n# Advice: {advice}\n\n"
    "Notice: Your first tried failed and the environment has been reset to its "
    "initial state. please try again."
)

# Ark request body used only by the authoritative AWM judge. Teacher and
# student calls keep their existing thinking behavior.
_JUDGE_THINKING_OFF_BODY = {"thinking": {"type": "disabled"}}

# vLLM-specific sampling option used only for DeepCoder/TACO student calls.
_CODE_STUDENT_EXTRA_BODY = {"repetition_penalty": 1.0}
_DEFAULT_TEACHER_MODEL = "ep-20260707130305-26bjx"
_CODE_TEACHER_MODEL = "ep-20260716095030-rdv28"


@dataclass(frozen=True)
class BackendProfile:
    name: str
    data_source: str
    default_base_url: str
    default_tasks_per_scenario: int | None
    default_student_iterations: int
    default_student_max_tokens: int
    default_teacher_model: str
    default_teacher_max_tokens: int
    default_teacher_tool_calls: int
    default_llm_judge: bool
    judge_authoritative: bool
    judge_extra_body: dict | None
    student_extra_body: dict | None
    reconnect_before_verify: bool
    include_verify_summary: bool
    student_system_prompt: str
    student_final_system_prompt: str
    teacher_system_prompt: str
    teacher_finalize_system_prompt: str
    output_stem: str


def _backend_profile(name: str) -> BackendProfile:
    name = {
        "codejudge": "deepcoder-taco",
        "deepcoder_taco": "deepcoder-taco",
    }.get(name, name)
    if name == "envscaler":
        return BackendProfile(
            name=name,
            data_source="envscaler_rl",
            default_base_url="http://127.0.0.1:8900",
            default_tasks_per_scenario=None,
            default_student_iterations=16,
            default_student_max_tokens=2048,
            default_teacher_model=_DEFAULT_TEACHER_MODEL,
            default_teacher_max_tokens=8192,
            default_teacher_tool_calls=3,
            default_llm_judge=False,
            judge_authoritative=False,
            judge_extra_body=None,
            student_extra_body=None,
            reconnect_before_verify=False,
            include_verify_summary=True,
            student_system_prompt=ENVSCALER_STUDENT_NATIVE_SYSTEM_PROMPT.format(
                current_date=date.today().isoformat()),
            student_final_system_prompt=ENVSCALER_STUDENT_NATIVE_FINAL_SYSTEM_PROMPT,
            teacher_system_prompt=TEACHER_ADVICE_SYSTEM_PROMPT,
            teacher_finalize_system_prompt=TEACHER_FINALIZE_SYSTEM_PROMPT,
            output_stem="envscaler_epoch1",
        )
    if name == "deepcoder-taco":
        return BackendProfile(
            name=name,
            data_source="deepcoder_taco",
            default_base_url="http://127.0.0.1:8901",
            default_tasks_per_scenario=None,
            default_student_iterations=1,
            default_student_max_tokens=16384,
            default_teacher_model=_CODE_TEACHER_MODEL,
            default_teacher_max_tokens=16384,
            default_teacher_tool_calls=0,
            default_llm_judge=False,
            judge_authoritative=False,
            judge_extra_body=None,
            student_extra_body=_CODE_STUDENT_EXTRA_BODY,
            reconnect_before_verify=True,
            include_verify_summary=True,
            student_system_prompt=CODE_JUDGE_STUDENT_SYSTEM_PROMPT,
            student_final_system_prompt=CODE_JUDGE_STUDENT_FINAL_SYSTEM_PROMPT,
            teacher_system_prompt=CODE_JUDGE_TEACHER_ADVICE_SYSTEM_PROMPT,
            teacher_finalize_system_prompt=CODE_JUDGE_TEACHER_FINALIZE_SYSTEM_PROMPT,
            output_stem="deepcoder_taco_epoch1",
        )
    if name == "awm":
        return BackendProfile(
            name=name,
            data_source="awm",
            default_base_url="http://localhost:8899",
            default_tasks_per_scenario=10,
            default_student_iterations=4,
            default_student_max_tokens=2048,
            default_teacher_model=_DEFAULT_TEACHER_MODEL,
            default_teacher_max_tokens=8192,
            default_teacher_tool_calls=3,
            default_llm_judge=True,
            judge_authoritative=True,
            judge_extra_body=_JUDGE_THINKING_OFF_BODY,
            student_extra_body=None,
            reconnect_before_verify=False,
            include_verify_summary=False,
            student_system_prompt=STUDENT_NATIVE_SYSTEM_PROMPT,
            student_final_system_prompt=STUDENT_NATIVE_FINAL_SYSTEM_PROMPT,
            teacher_system_prompt=TEACHER_ADVICE_SYSTEM_PROMPT,
            teacher_finalize_system_prompt=TEACHER_FINALIZE_SYSTEM_PROMPT,
            output_stem="refine_epoch1",
        )
    raise ValueError(f"unsupported dataset backend: {name!r}")


def _parse_tool_call(tool_call: dict) -> tuple[str, str, dict]:
    """Normalize one OpenAI-style native tool call."""
    function = tool_call.get("function") or {}
    raw_args = function.get("arguments", "{}")
    args = loads_lenient(raw_args) if isinstance(raw_args, str) else raw_args
    return (
        tool_call.get("id", ""),
        function.get("name", ""),
        args if isinstance(args, dict) else {},
    )


async def _execute_tool(env, name: str, args: dict, *,
                        forbidden_error: str) -> str:
    if name in _HARNESS_TOOLS:
        response = forbidden_error
    else:
        try:
            response = (await execute_native_tool_call(env, name, args)).text
        except Exception as exc:
            response = f"Error executing tool: {exc!r}"
    return response or ""


def _model_text(turn) -> str:
    """Prefer content, falling back to reasoning when content is blank."""
    content = turn.content or ""
    if not content.strip() and turn.reasoning:
        return turn.reasoning
    return content


def _json_object_from_text(text: str) -> dict | None:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return None
        try:
            parsed = json.loads(match.group())
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


async def _close_quietly(env) -> None:
    try:
        await env.close()
    except Exception:
        pass


async def _student_step(env, llm: RetryLLM, messages: list[dict],
                         tools_schema: list[dict],
                         max_iterations: int, max_tokens: int,
                         temperature: float,
                         extra_body: dict | None = None,
                         tool_response_cap: int = 4000,
                         final_system_prompt: str =
                         STUDENT_NATIVE_FINAL_SYSTEM_PROMPT) -> dict:
    trace: list[dict] = []
    executed: list[dict] = []
    content = ""
    made_tool_call = False
    failure_type: str | None = None
    error: str | None = None
    step_used = 0

    for step in range(1, max_iterations + 1):
        step_used = step
        is_last = step == max_iterations
        try:
            turn = await llm_turn_native(
                llm,
                messages,
                tools=None if is_last else tools_schema,
                temperature=temperature,
                max_tokens=max_tokens,
                tool_choice="none" if is_last else "auto",
                system_override=(
                    final_system_prompt if is_last else None
                ),
                extra_body=extra_body,
            )
        except Exception as exc:
            error = f"student LLM error at step {step}: {exc!r}"[:500]
            break

        content = turn.content
        reasoning = getattr(turn, "reasoning", "") or ""
        trace.append({
            "step": step,
            "assistant": content or "",
            "reasoning": reasoning,
        })

        if (not content.strip() and not turn.tool_calls
                and bool(reasoning.strip())):
            failure_type = "reasoning_without_answer"
            error = (
                "student generated substantive reasoning but no final answer"
            )
            break

        if not turn.tool_calls:
            break

        made_tool_call = True
        for tool_call in turn.tool_calls[:3]:
            call_id, name, args = _parse_tool_call(tool_call)
            tool_response = await _execute_tool(
                env,
                name,
                args,
                forbidden_error=_STUDENT_FORBIDDEN_TOOL_ERROR,
            )
            trace.append({
                "step": step,
                "tool": name,
                "arguments": args,
                "response": tool_response,
            })
            if name not in _HARNESS_TOOLS and not tool_response.startswith("Error"):
                executed.append({"tool_name": name, "arguments": args})
            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": f"{tool_response[:tool_response_cap]}\n\nYou have {max_iterations-step-1} remaining opportunities for parallel tool calls. Once the count hits 0, you must respond to the user directly whether the task is completed or not.",
            })

    return {
        "messages": messages,
        "trace": trace,
        "final_answer": content,
        "made_tool_call": made_tool_call,
        "executed_tool_calls": executed,
        "steps": step_used,
        "error": error,
        "failure_type": failure_type,
    }


async def _verify_only(env, final_answer: str) -> dict:
    """Call verify (code mode) on the current student env. Read-only wrt
    the DB state; safe to call every round."""
    verify = await env.step(CallToolAction(
        tool_name="verify",
        arguments={"verifier_mode": "code", "final_answer": final_answer},
    ))
    return {
        "reward": verify.reward,
        "reward_type": verify.observation.reward_type,
        "verify_result": verify.observation.verify_result,
        "error": getattr(verify.observation, "error", None),
    }


_JUDGE_CLASSIFICATIONS = ("complete", "incomplete", "server_error", "agent_error")

def _judge_result(classification="judge_error") -> dict:
    return {
        "classification": classification,
        "reasoning": None,
        "confidence_score": None,
        "error": None,
    }


async def _llm_judge_round(judge_llm: RetryLLM, task: str,
                           round_messages: list[dict],
                           final_answer: str,
                           code_verify: dict,
                           temperature: float, max_tokens: int,
                           timeout: float,
                           extra_body: dict | None = None) -> dict:
    """LLM-as-judge verdict for ONE round. Never raises: failures come back
    as {'classification': 'judge_error', 'error': ...} and the caller falls
    back to the code verifier's verdict.

    `round_messages` is passed BY VALUE into llm_turn_native - that helper
    appends the assistant reply to the list it is given, so a throwaway list
    keeps the judge's reply out of the student conversation.

    `extra_body` carries endpoint-specific request fields. The AWM profile uses
    it to disable judge thinking (see _JUDGE_THINKING_OFF_BODY)."""
    result = _judge_result()
    try:
        user_text = build_judge_input(
            task=task,
            round_messages=round_messages,
            final_answer=final_answer,
            code_verify=code_verify,
        )
        # Fresh list on purpose (see docstring); tools stay disabled.
        turn = await asyncio.wait_for(
            llm_turn_native(
                judge_llm,
                [{"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                 {"role": "user", "content": user_text}],
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=extra_body,
            ),
            timeout=timeout,
        )
        raw = _model_text(turn).strip()
        parsed = _json_object_from_text(raw)
        if parsed is None:
            result["error"] = f"failed to parse judge output (len={len(raw)})"
            return result

        classification = str(parsed.get("classification", "")).lower().strip()
        if classification not in _JUDGE_CLASSIFICATIONS:
            result["error"] = f"invalid classification {classification!r}"
            return result
        result["classification"] = classification
        result["reasoning"] = str(parsed.get("reasoning") or "")[:1000] or None
        result["confidence_score"] = parsed.get("confidence_score")
    except asyncio.TimeoutError:
        result["error"] = f"judge timed out after {timeout}s"
    except Exception as exc:
        result["error"] = f"judge failed: {exc!r}"[:500]
    return result


def _extract_advice(final_text: str, error: str | None,
                    full_block: bool = False) -> tuple[str | None, str | None]:
    raw = final_text.strip()
    clean = re.sub(r"<tool_call>.*?</tool_call>", "", raw,
                   flags=re.DOTALL).strip()
    pattern = (
        re.compile(r"^\s*#?\s*[Aa]dvice\s*:\s*(.+)\Z", re.MULTILINE | re.DOTALL)
        if full_block else ADVICE_LINE_RE
    )
    match = pattern.search(clean)
    advice = match.group(1).strip() if match else (clean if len(clean) > 20 else None)
    if advice is not None or error is not None:
        return advice, error
    if not raw:
        return None, "finalize returned empty text"
    if not clean:
        return None, "finalize returned only <tool_call> blocks"
    return None, f"finalize text unparseable (len={len(clean)})"


async def _teacher_advise(awm_base_url: str, scenario: str, task_idx: int,
                          task: str, student_messages: list[dict],
                          past_tool_calls: list[dict],
                          verify_reward_type: str,
                          verify_error: str | None,
                          teacher_llm: RetryLLM,
                          student_error: str | None = None,
                          max_iterations: int = 4,
                          max_tool_calls: int = 3,
                          max_tokens: int = 8192,
                          temperature: float = 0.4,
                          tool_response_cap: int = 4000,
                          verify_summary: str | None = None,
                          teacher_system_prompt: str = TEACHER_ADVICE_SYSTEM_PROMPT,
                          teacher_finalize_system_prompt: str =
                          TEACHER_FINALIZE_SYSTEM_PROMPT,
                          full_advice: bool = False) -> dict:
    """Return {'advice': str | None, 'rounds_used': int, 'tool_calls_made': int,
    'replay_ok': bool, 'error': str | None}. The teacher's private conversation
    is intentionally discarded. Normal paths retain one advice block; recovery
    paths may retain a complete multiline solution."""
    env = None
    advice: str | None = None
    rounds_used = 0
    tool_calls_made = 0
    replay_ok = True
    error: str | None = None
    final_text: str = ""
    try:
        tools_schema: list[dict] = []
        if max_tool_calls > 0:
            env = AWMEnv(base_url=awm_base_url,
                         message_timeout_s=180.0, connect_timeout_s=60.0)
            await env.connect()
            reset = await env.reset(scenario=scenario, task_idx=task_idx)
            reset_observation = getattr(reset, "observation", None)
            reset_error = getattr(reset_observation, "error", None)
            if reset_error:
                raise RuntimeError(
                    f"teacher environment reset failed: {reset_error}")
            list_res = await env.step(ListToolsAction())
            tools_schema = tools_to_openai_schema(list_res.observation.tools)

            for tool_call in past_tool_calls:
                try:
                    await env.step(CallToolAction(
                        tool_name=tool_call["tool_name"],
                        arguments=tool_call["arguments"],
                    ))
                except Exception:
                    replay_ok = False

        messages: list[dict] = [
            {"role": "system", "content": teacher_system_prompt},
            {"role": "user", "content": build_teacher_advice_input(
                task=task,
                student_conversation=student_messages,
                verify_reward_type=verify_reward_type,
                verify_error=verify_error,
                verify_summary=verify_summary,
                student_error=student_error,
                request_complete_solution=full_advice,
                allow_tool_probes=max_tool_calls > 0,
            )},
        ]

        for step in range(1, max_tool_calls + 1):
            rounds_used = step
            try:
                turn = await llm_turn_native(
                    teacher_llm,
                    messages,
                    tools=tools_schema,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tool_choice="auto",
                )
            except Exception as exc:
                error = f"teacher LLM error at probe step {step}: {exc!r}"[:500]
                break
            if not turn.tool_calls:
                break

            for tool_call in turn.tool_calls[:3]:
                call_id, name, args = _parse_tool_call(tool_call)
                tool_calls_made += 1
                tool_response = await _execute_tool(
                    env,
                    name,
                    args,
                    forbidden_error=_TEACHER_FORBIDDEN_TOOL_ERROR,
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": tool_response[:tool_response_cap],
                })

        finalize_messages = (
            [{"role": "system", "content": teacher_finalize_system_prompt}]
            + messages[1:]
            + [{"role": "user", "content": _TEACHER_FINALIZE_REQUEST}]
        )
        try:
            final_turn = await llm_turn_native(
                teacher_llm, finalize_messages,
                temperature=temperature, max_tokens=max_tokens,
                stream=True)
            final_text = _model_text(final_turn)
        except Exception as exc:
            error = f"teacher finalize failed: {exc!r}"[:500]

        advice, error = _extract_advice(
            final_text, error, full_block=full_advice)
    except Exception as exc:
        error = f"teacher run failed: {exc!r}"[:500]
    finally:
        if env is not None:
            await _close_quietly(env)
    return {
        "advice": advice,
        "rounds_used": rounds_used,
        "tool_calls_made": tool_calls_made,
        "replay_ok": replay_ok,
        "error": error,
        "final_raw": final_text or "",
    }


def _verify_error(verify: dict) -> str | None:
    observation_error = verify.get("error")
    if observation_error:
        return str(observation_error)[:400]
    result = verify.get("verify_result") or {}
    if not isinstance(result, dict):
        return None
    error = result.get("error") or result.get("message")
    return error[:400] if isinstance(error, str) else error


def _safe_score(value) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if math.isfinite(score) else None


def _verify_summary(verify: dict) -> str | None:
    """Expose aggregate checker evidence without leaking verifier internals."""
    result = verify.get("verify_result")
    if not isinstance(result, dict):
        return None
    fields = []
    score = _safe_score(verify.get("reward"))
    if score is not None:
        fields.append(f"score={score:.6g}")
    for key in (
        "passed_checks", "total_checks", "checker_errors",
        "passed_tests", "total_tests", "all_passed", "format_error",
        "compile_error", "timeouts", "runtime_errors",
    ):
        value = result.get(key)
        if value is not None:
            fields.append(f"{key}={value}")
    return ", ".join(fields) or None


def _score_outcome(rounds_detail: list[dict],
                   success_at_round: int | None) -> dict:
    """Summarize the score path and classify at most one training candidate."""
    scored = [
        (idx + 1, score)
        for idx, detail in enumerate(rounds_detail)
        if (score := _safe_score(detail.get("verify_reward"))) is not None
    ]
    baseline = scored[0][1] if scored else None
    best_round, best = max(scored, key=lambda item: item[1]) if scored else (None, None)

    sample_type = "no_improvement"
    if success_at_round == 1:
        sample_type = "complete_first_try"
    elif success_at_round is not None:
        sample_type = "complete_after_refine"
    elif baseline is not None:
        improved = [(round_no, score) for round_no, score in scored[1:]
                    if score > baseline + 1e-9]
        if improved:
            best_value = max(score for _, score in improved)
            best_round = next(round_no for round_no, score in improved
                              if abs(score - best_value) <= 1e-12)
            best = best_value
            sample_type = "improved_partial"

    return {
        "baseline_score": baseline,
        "best_score": best,
        "best_round": best_round,
        "score_improved": (
            baseline is not None and best is not None and best > baseline + 1e-9
        ),
        "sample_candidate_type": sample_type,
    }


def _round_record(round_number: int, message_range: list[int],
                  step: dict, verify: dict, judge: dict,
                  final_reward_type: str, judge_fallback: bool) -> dict:
    return {
        "round": round_number,
        "message_range": message_range,
        "student_final_answer": step["final_answer"],
        "student_trace": step["trace"],
        "student_steps": step["steps"],
        "student_made_tool_call": step["made_tool_call"],
        "student_error": step["error"],
        "student_failure_type": step.get("failure_type"),
        "student_new_tool_calls": step["executed_tool_calls"],
        "verify_reward": verify["reward"],
        "verify_reward_type": verify["reward_type"],
        "verify_result": verify.get("verify_result"),
        "verify_error": _verify_error(verify),
        "judge_classification": judge["classification"],
        "judge_reasoning": judge["reasoning"],
        "judge_confidence_score": judge["confidence_score"],
        "judge_error": judge["error"],
        "judge_fallback": judge_fallback,
        "final_reward_type": final_reward_type,
        "teacher_advice": None,
        "teacher_rounds_used": 0,
        "teacher_tool_calls_made": 0,
        "teacher_replay_ok": None,
        "teacher_error": None,
        "teacher_final_raw": "",
    }


def _add_teacher_result(round_record: dict, teacher: dict) -> None:
    for record_key, teacher_key in (
        ("teacher_advice", "advice"),
        ("teacher_rounds_used", "rounds_used"),
        ("teacher_tool_calls_made", "tool_calls_made"),
        ("teacher_replay_ok", "replay_ok"),
        ("teacher_error", "error"),
    ):
        round_record[record_key] = teacher[teacher_key]
    round_record["teacher_final_raw"] = teacher.get("final_raw", "")


@dataclass
class RefineJob:
    scenario: str
    task_idx: int
    student_llm: RetryLLM
    teacher_llm: RetryLLM | None
    awm_base_url: str
    data_source: str = "awm"
    student_system_prompt: str = STUDENT_NATIVE_SYSTEM_PROMPT
    student_final_system_prompt: str = STUDENT_NATIVE_FINAL_SYSTEM_PROMPT
    teacher_system_prompt: str = TEACHER_ADVICE_SYSTEM_PROMPT
    teacher_finalize_system_prompt: str = TEACHER_FINALIZE_SYSTEM_PROMPT
    include_verify_summary: bool = False
    only_infer: bool = True
    max_rounds: int = 3
    student_max_iterations: int = 10
    teacher_max_iterations: int = 4
    teacher_max_tool_calls: int = 3
    student_max_tokens: int = 2048
    teacher_max_tokens: int = 1024
    student_temperature: float = 1.0
    student_extra_body: dict | None = None
    reconnect_before_verify: bool = False
    teacher_temperature: float = 0.4
    episode_timeout: float = 900.0
    judge_llm: RetryLLM | None = None
    use_llm_judge: bool = True
    judge_authoritative: bool = True
    judge_extra_body: dict | None = None
    judge_temperature: float = 1.0
    judge_max_tokens: int = 8192
    judge_timeout: float = 180.0

    def key(self) -> tuple:
        return (self.scenario, self.task_idx)

    async def _judge(self, task: str, messages: list[dict],
                     step: dict, verify: dict) -> tuple[dict, str, bool]:
        enabled = self.use_llm_judge and self.judge_llm is not None
        judge = await _llm_judge_round(
            self.judge_llm,
            task=task,
            round_messages=messages,
            final_answer=step["final_answer"],
            code_verify={
                "reward_type": verify["reward_type"],
                "verify_result": verify.get("verify_result"),
            },
            temperature=self.judge_temperature,
            max_tokens=self.judge_max_tokens,
            timeout=self.judge_timeout,
            extra_body=self.judge_extra_body,
        ) if enabled else _judge_result(None)
        fallback = (self.judge_authoritative and enabled
                    and judge["classification"] == "judge_error")
        final_reward = verify["reward_type"]
        if (self.judge_authoritative
                and judge["classification"] in _JUDGE_CLASSIFICATIONS):
            final_reward = judge["classification"]
        return judge, final_reward, fallback

    async def _advise(self, task: str, messages: list[dict],
                      tool_calls: list[dict], reward_type: str,
                      verify: dict, student_error: str | None = None,
                      reasoning_without_answer: bool = False) -> dict:
        recover_with_solution = (
            reasoning_without_answer and self.data_source == "deepcoder_taco"
        )
        return await _teacher_advise(
            awm_base_url=self.awm_base_url,
            scenario=self.scenario,
            task_idx=self.task_idx,
            task=task,
            student_messages=messages,
            past_tool_calls=tool_calls,
            verify_reward_type=reward_type,
            verify_error=_verify_error(verify),
            student_error=student_error,
            teacher_llm=self.teacher_llm,
            max_iterations=self.teacher_max_iterations,
            max_tool_calls=self.teacher_max_tool_calls,
            max_tokens=self.teacher_max_tokens,
            temperature=self.teacher_temperature,
            verify_summary=(
                _verify_summary(verify) if self.include_verify_summary else None
            ),
            teacher_system_prompt=(
                CODE_JUDGE_TEACHER_SOLUTION_SYSTEM_PROMPT
                if recover_with_solution else self.teacher_system_prompt
            ),
            teacher_finalize_system_prompt=(
                CODE_JUDGE_TEACHER_SOLUTION_FINALIZE_SYSTEM_PROMPT
                if recover_with_solution else self.teacher_finalize_system_prompt
            ),
            full_advice=recover_with_solution,
        )

    async def run(self) -> dict:
        t0 = time.monotonic()

        def new_env():
            return AWMEnv(
                base_url=self.awm_base_url,
                message_timeout_s=self.episode_timeout,
                connect_timeout_s=60.0,
            )

        env = new_env()
        rounds_detail: list[dict] = []
        student_messages: list[dict] = []
        task_description: str = ""
        task_id: str | None = None
        success = False
        success_at_round: int | None = None
        error: str | None = None

        try:
            await env.connect()
            reset_res = await env.reset(
                scenario=self.scenario, task_idx=self.task_idx)
            reset_error = getattr(reset_res.observation, "error", None)
            if reset_error:
                raise RuntimeError(f"environment reset failed: {reset_error}")
            task_description = reset_res.observation.task
            task_id = getattr(reset_res.observation, "task_id", None)
            list_res = await env.step(ListToolsAction())
            tools_schema = tools_to_openai_schema(list_res.observation.tools)
            student_messages = [
                {"role": "system", "content": self.student_system_prompt},
                {"role": "user", "content": task_description},
            ]
            if self.reconnect_before_verify:
                await _close_quietly(env)

            for k in range(1, self.max_rounds + 1):
                if k > 1 and not self.reconnect_before_verify:
                    round_reset = await env.reset(
                        scenario=self.scenario, task_idx=self.task_idx)
                    reset_error = getattr(round_reset.observation, "error", None)
                    if reset_error:
                        raise RuntimeError(
                            f"environment reset failed before round {k}: "
                            f"{reset_error}")

                start = len(student_messages)
                step = await asyncio.wait_for(
                    _student_step(
                        env, self.student_llm, student_messages,
                        tools_schema,
                        max_iterations=self.student_max_iterations,
                        max_tokens=self.student_max_tokens,
                        temperature=self.student_temperature,
                        extra_body=self.student_extra_body,
                        final_system_prompt=self.student_final_system_prompt,
                    ),
                    timeout=self.episode_timeout,
                )
                student_messages = step["messages"]
                if self.reconnect_before_verify:
                    env = new_env()
                    await env.connect()
                    verify_reset = await env.reset(
                        scenario=self.scenario, task_idx=self.task_idx)
                    reset_error = getattr(
                        verify_reset.observation, "error", None)
                    if reset_error:
                        raise RuntimeError(
                            f"environment reset failed before verify round "
                            f"{k}: {reset_error}"
                        )
                    rebound_task_id = getattr(
                        verify_reset.observation, "task_id", None)
                    if (task_id is not None and rebound_task_id is not None
                            and rebound_task_id != task_id):
                        raise RuntimeError(
                            "environment rebound to a different task before "
                            f"verify: expected {task_id!r}, got "
                            f"{rebound_task_id!r}"
                        )
                verify = await _verify_only(env, step["final_answer"])
                judge, final_reward_type, judge_fallback = await self._judge(
                    task_description,
                    student_messages[start:],
                    step,
                    verify,
                )
                round_record = _round_record(
                    k,
                    [start, len(student_messages)],
                    step,
                    verify,
                    judge,
                    final_reward_type,
                    judge_fallback,
                )
                rounds_detail.append(round_record)

                if final_reward_type == "complete":
                    success = True
                    success_at_round = k
                    break
                if k == self.max_rounds or self.only_infer:
                    break

                if self.reconnect_before_verify:
                    await _close_quietly(env)

                teacher = await self._advise(
                    task_description,
                    student_messages,
                    step["executed_tool_calls"],
                    final_reward_type,
                    verify,
                    student_error=step["error"],
                    reasoning_without_answer=(
                        step["failure_type"] == "reasoning_without_answer"
                    ),
                )
                _add_teacher_result(round_record, teacher)

                if not teacher["advice"]:
                    reason = teacher.get("error") or "no advice; unknown reason"
                    error = (error or "") + f"[round {k}] teacher no-advice: {reason}; "
                    break

                student_messages.append({
                    "role": "user",
                    "content": _EXPERT_ADVICE_TEMPLATE.format(
                        advice=teacher["advice"]),
                })

        except asyncio.TimeoutError:
            error = f"episode timeout > {self.episode_timeout}s"
        except Exception as exc:
            import traceback
            error = (
                f"refine run failed: {exc!r} | tb: {traceback.format_exc()}"
            )[:2000]
        finally:
            await _close_quietly(env)

        elapsed = round(time.monotonic() - t0, 3)
        record = {
            "data_source": self.data_source,
            "scenario": self.scenario,
            "task_idx": self.task_idx,
            "task_id": task_id,
            "task": task_description,
            "max_rounds": self.max_rounds,
            "rounds": len(rounds_detail),
            "success": success,
            "success_at_round": success_at_round,
            "student_conversation": student_messages,
            "rounds_detail": rounds_detail,
            "elapsed_sec": elapsed,
            "error": error,
        }
        record.update(_score_outcome(rounds_detail, success_at_round))
        return record


async def _list_all_scenarios(awm_base_url: str) -> list[dict]:
    async with AWMEnv(base_url=awm_base_url) as env:
        list_res = await env.step(
            CallToolAction(tool_name="__list_scenarios__", arguments={}))
        return list_res.observation.scenarios


def _code_judge_scenarios(base_url: str) -> list[dict]:
    """Discover CodeJudge task count from its lightweight monitoring API."""
    url = f"{base_url.rstrip('/')}/dataset"
    with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310
        payload = json.loads(response.read().decode("utf-8"))
    if payload.get("status") != "ok":
        raise RuntimeError(f"CodeJudge dataset endpoint is unhealthy: {payload}")
    task_count = int(payload.get("task_count") or 0)
    if task_count < 1:
        raise RuntimeError(f"CodeJudge reported invalid task_count={task_count}")
    return [{"name": "taco", "num_tasks": task_count}]


async def _discover_scenarios(profile: BackendProfile,
                              base_url: str) -> list[dict]:
    if profile.name == "deepcoder-taco":
        return await asyncio.to_thread(_code_judge_scenarios, base_url)
    return await _list_all_scenarios(base_url)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add = parser.add_argument
    add("--dataset", "--env-backend", dest="env_backend",
        choices=("awm", "envscaler", "deepcoder-taco", "deepcoder_taco",
                 "codejudge"), default="awm",
        help="Dataset profile. EnvScaler and DeepCoder/TACO use their code "
             "verifiers as ground truth and include every advertised task.")
    add("--num-scenarios", type=int, default=1000)
    add("--tasks-per-scenario", type=int, default=None,
        help="Optional task cap per scenario. Defaults to 10 for AWM and all "
             "advertised tasks for EnvScaler and DeepCoder/TACO.")
    parser.add_argument("--scenarios", default=None,
                        help="Comma-separated scenario names; overrides --num-scenarios.")
    for option, default in (
        ("--start-scenario-idx", 0),
        ("--end-scenario-idx", None),
    ):
        add(option, type=int, default=default)

    add("--only-infer", action=argparse.BooleanOptionalAction, default=False,
        help="Inference-only: run the student once per task and skip the teacher "
             "advice module. Pass --no-only-infer to enable the full refine "
             "(teacher-advice) loop.")
    add("--max-rounds", type=int, default=2,
        help="Max number of student→verify→teacher_advice iterations.")
    add("--student-max-iterations", type=int, default=None,
        help="Max LLM turns within one round (AWM: 4; EnvScaler: 16; "
             "DeepCoder/TACO: 1).")
    for option, value_type, default in (
        ("--student-temperature", float, 1.0),
        ("--teacher-max-iterations", int, 4),
        ("--teacher-temperature", float, 0.4),
        ("--concurrency", int, 32),
        ("--episode-timeout", float, 1200.0),
        ("--llm-timeout", float, 600.0),
        ("--progress-interval", float, 10.0),
    ):
        add(option, type=value_type, default=default)
    add("--teacher-timeout", type=float, default=600.0,
        help="Per-call teacher timeout in seconds. Kept separate from "
             "--llm-timeout because complex advice turns can be much slower "
             "than student inference.")
    add("--teacher-max-tokens", type=int, default=None,
        help="Defaults to 16384 for code tasks and 8192 for tool tasks.")
    add("--student-max-tokens", type=int, default=None,
        help="Defaults to 2048 for tool datasets and 16384 for code tasks.")
    add("--teacher-max-tool-calls", type=int, default=None,
        help="Defaults to 3 for tool datasets and 0 for CodeJudge.")

    add("--env-base-url", "--awm-base-url", dest="awm_base_url", default=None,
        help="Environment server URL (AWM_BASE_URL also supported).")
    add("--student-base-url",
        default=os.environ.get("ENDPOINT_URL", "http://localhost:8000/v1"))
    add("--student-api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    add("--student-model",
        default=os.environ.get("AWM_EXAMPLE_AGENT_MODEL", "qwen3.5-4b"))

    add("--teacher-base-url",
        default=os.environ.get("TEACHER_BASE_URL",
                               "https://ark.cn-beijing.volces.com/api/v3"),
        help="Teacher endpoint. Defaults to $TEACHER_BASE_URL (.env), then the "
             "Volcano Ark OpenAI-compatible URL.")
    add("--teacher-api-key", default=os.environ.get("ARK_API_KEY"))
    parser.add_argument(
        "--teacher-model",
        default=None,
        help="Ark endpoint id. Defaults by dataset profile; TEACHER_MODEL can "
             "override it.",
    )
    add("--teacher-rpm", type=int, default=400)
    add("--teacher-tpm", type=int, default=800_000)

    add("--llm-judge", action=argparse.BooleanOptionalAction, default=None,
        help="LLM-as-judge decides the final reward: success is judge "
             "classification == 'complete', with the code verify result passed "
             "to the judge as evidence. On a failed judge call the code verdict "
             "is used (judge_fallback=True). --no-llm-judge restores "
             "code-verify-only scoring.")
    add("--judge-base-url", default=os.environ.get("JUDGE_BASE_URL"),
        help="Defaults to $JUDGE_BASE_URL (.env), then --teacher-base-url.")
    add("--judge-api-key", default=os.environ.get("JUDGE_API_KEY"),
        help="Defaults to $JUDGE_API_KEY (.env), then --teacher-api-key "
             "(ARK_API_KEY).")
    add("--judge-model", default="ep-20260609014859-6th5f",
        help="dpsk-v4-pro")
    add("--judge-temperature", type=float, default=1.0,
        help="1.0 matches the AWM server-side judge config (reasoning models want 1.0).")
    for option, default in (
        ("--judge-max-tokens", 8192),
        ("--judge-rpm", 400),
        ("--judge-tpm", 800_000),
    ):
        add(option, type=int, default=default)
    add("--judge-timeout", type=float, default=None,
        help="Per-call judge timeout in seconds; defaults to --llm-timeout.")

    add("--output-jsonl", default=None)
    add("--report", default=None)
    add("--resume", action="store_true")
    add("--limit", type=int, default=None,
        help="Optional cap on total (scenario,task) pairs.")
    return parser


def _resolve_backend_args(args: argparse.Namespace) -> BackendProfile:
    profile = _backend_profile(args.env_backend)
    backend_url_env = {
        "awm": "AWM_BASE_URL",
        "envscaler": "ENVSCALER_BASE_URL",
        "deepcoder-taco": "CODE_JUDGE_BASE_URL",
    }[profile.name]
    args.awm_base_url = (
        args.awm_base_url
        or os.environ.get("ENV_BASE_URL")
        or os.environ.get(backend_url_env)
        or profile.default_base_url
    )
    if args.tasks_per_scenario is None:
        args.tasks_per_scenario = profile.default_tasks_per_scenario
    if args.student_max_iterations is None:
        args.student_max_iterations = profile.default_student_iterations
    if args.student_max_tokens is None:
        args.student_max_tokens = profile.default_student_max_tokens
    if args.teacher_model is None:
        args.teacher_model = (
            os.environ.get("TEACHER_MODEL") or profile.default_teacher_model
        )
    if args.teacher_max_tokens is None:
        args.teacher_max_tokens = profile.default_teacher_max_tokens
    if args.teacher_max_tool_calls is None:
        args.teacher_max_tool_calls = profile.default_teacher_tool_calls
    if args.llm_judge is None:
        args.llm_judge = profile.default_llm_judge
    data_dir = "/mnt/storage/disk3/self_evolver/traj_data"
    args.output_jsonl = args.output_jsonl or f"{data_dir}/{profile.output_stem}.jsonl"
    args.report = args.report or f"{data_dir}/{profile.output_stem}.json"
    return profile


def _build_llm_clients(args: argparse.Namespace) -> tuple[RetryLLM, RetryLLM | None,
                                                           RetryLLM | None]:
    student_llm = RetryLLM(
        base_url=args.student_base_url,
        api_key=args.student_api_key,
        model=args.student_model,
        timeout=args.llm_timeout,
    )
    teacher_llm: RetryLLM | None = None
    if args.only_infer:
        print(f"[refine] student={args.student_model} teacher=<none> "
              f"(only_infer=True: advice module disabled)")
    else:
        if not args.teacher_api_key:
            raise SystemExit("teacher API key not set: export ARK_API_KEY or "
                             "pass --teacher-api-key (required when --no-only-infer).")
        teacher_llm = RetryLLM(
            base_url=args.teacher_base_url,
            api_key=args.teacher_api_key,
            model=args.teacher_model,
            timeout=args.teacher_timeout,
            limiter=RateLimiter(rpm=args.teacher_rpm, tpm=args.teacher_tpm),
        )
        print(f"[refine] student={args.student_model} teacher={args.teacher_model}")
        print(f"[refine] teacher rate limit: {args.teacher_rpm} RPM, "
              f"{args.teacher_tpm} TPM; timeout={args.teacher_timeout:g}s")

    judge_llm: RetryLLM | None = None
    if args.llm_judge:
        judge_base_url = args.judge_base_url or args.teacher_base_url
        judge_api_key = args.judge_api_key or args.teacher_api_key
        judge_model = args.judge_model or args.teacher_model
        if not judge_api_key:
            raise SystemExit(
                "judge API key not set: export ARK_API_KEY or pass "
                "--teacher-api-key/--judge-api-key (required when --llm-judge).")
        shares_teacher = (
            teacher_llm is not None
            and judge_base_url == args.teacher_base_url
            and judge_api_key == args.teacher_api_key
            and judge_model == args.teacher_model
        )
        judge_llm = teacher_llm if shares_teacher else RetryLLM(
            base_url=judge_base_url,
            api_key=judge_api_key,
            model=judge_model,
            timeout=args.llm_timeout,
            limiter=RateLimiter(rpm=args.judge_rpm, tpm=args.judge_tpm),
        )
        authority = ("final reward" if args.env_backend == "awm"
                     else "advisory only; code verifier remains authoritative")
        thinking = (", thinking=disabled" if args.env_backend == "awm" else "")
        print(f"[refine] judge={judge_model} @ {judge_base_url} "
              f"(llm_judge=True: {authority}"
              f"{thinking}"
              f"{', shares teacher client' if shares_teacher else ''})")
    else:
        print("[refine] judge=<none> (llm_judge=False: code verify decides success)")
    return student_llm, teacher_llm, judge_llm


async def _select_work(args: argparse.Namespace,
                       profile: BackendProfile | None = None) -> list[tuple[str, int]]:
    profile = profile or _backend_profile(args.env_backend)
    all_scenarios = await _discover_scenarios(profile, args.awm_base_url)
    print(f"Total scenarios on server: {len(all_scenarios)}")
    by_name = {item["name"]: item for item in all_scenarios}
    if args.scenarios:
        picked = [name.strip() for name in args.scenarios.split(",") if name.strip()]
        missing = [name for name in picked if name not in by_name]
        if missing:
            raise ValueError(f"unknown scenarios: {missing}")
    else:
        end = (args.end_scenario_idx if args.end_scenario_idx is not None
               else args.start_scenario_idx + args.num_scenarios)
        picked = [item["name"] for item in all_scenarios[args.start_scenario_idx:end]]
    print(f"Selected {len(picked)} scenarios (first 5: {picked[:5]})")

    todo = []
    for scenario in picked:
        advertised = by_name[scenario].get("num_tasks")
        if args.tasks_per_scenario is None:
            if advertised is None:
                raise ValueError(
                    f"scenario {scenario!r} does not advertise num_tasks; "
                    "pass --tasks-per-scenario explicitly")
            task_count = int(advertised)
        else:
            task_count = args.tasks_per_scenario
            if advertised is not None:
                task_count = min(task_count, int(advertised))
        todo.extend((scenario, task) for task in range(task_count))
    return todo[:args.limit] if args.limit is not None else todo


def _build_jobs(args: argparse.Namespace, profile: BackendProfile,
                todo: list[tuple[str, int]],
                student_llm: RetryLLM, teacher_llm: RetryLLM | None,
                judge_llm: RetryLLM | None) -> list[RefineJob]:
    judge_timeout = (args.judge_timeout if args.judge_timeout is not None
                     else args.llm_timeout)
    common = {
        "student_llm": student_llm,
        "teacher_llm": teacher_llm,
        "awm_base_url": args.awm_base_url,
        "data_source": profile.data_source,
        "student_system_prompt": profile.student_system_prompt,
        "student_final_system_prompt": profile.student_final_system_prompt,
        "teacher_system_prompt": profile.teacher_system_prompt,
        "teacher_finalize_system_prompt": profile.teacher_finalize_system_prompt,
        "include_verify_summary": profile.include_verify_summary,
        "only_infer": args.only_infer,
        "max_rounds": args.max_rounds,
        "student_max_iterations": args.student_max_iterations,
        "teacher_max_iterations": args.teacher_max_iterations,
        "teacher_max_tool_calls": args.teacher_max_tool_calls,
        "student_max_tokens": args.student_max_tokens,
        "teacher_max_tokens": args.teacher_max_tokens,
        "student_temperature": args.student_temperature,
        "student_extra_body": profile.student_extra_body,
        "reconnect_before_verify": profile.reconnect_before_verify,
        "teacher_temperature": args.teacher_temperature,
        "episode_timeout": args.episode_timeout,
        "judge_llm": judge_llm,
        "use_llm_judge": args.llm_judge,
        "judge_authoritative": profile.judge_authoritative,
        "judge_extra_body": profile.judge_extra_body,
        "judge_temperature": args.judge_temperature,
        "judge_max_tokens": args.judge_max_tokens,
        "judge_timeout": judge_timeout,
    }
    return [RefineJob(scenario=scenario, task_idx=task, **common)
            for scenario, task in todo]


async def main():
    args = _build_parser().parse_args()
    profile = _resolve_backend_args(args)
    student_llm, teacher_llm, judge_llm = _build_llm_clients(args)
    todo = await _select_work(args, profile)

    print(f"[refine] backend={profile.name} env={args.awm_base_url}")
    print(f"[refine] {len(todo)} (scenario, task) pairs, "
          f"max_rounds={args.max_rounds}, concurrency={args.concurrency}")
    jobs = _build_jobs(
        args, profile, todo, student_llm, teacher_llm, judge_llm)
    ckpt = JsonlCheckpoint(args.output_jsonl, key_fields=("scenario", "task_idx"))

    t0 = time.monotonic()
    await run_jobs(jobs, args.concurrency, ckpt,
                   progress_interval=args.progress_interval, resume=args.resume)
    _aggregate_and_print(ckpt, args, time.monotonic() - t0)


def _aggregate_and_print(ckpt: JsonlCheckpoint, args: argparse.Namespace,
                          elapsed_wall_sec: float) -> None:
    records = ckpt.read_all()
    n = len(records)
    n_success = sum(1 for r in records if r.get("success"))
    details = [detail for record in records
               for detail in (record.get("rounds_detail") or [])]
    n_with_advice = sum(
        1 for detail in details if detail.get("teacher_advice")
    )
    round_dist = dict(Counter(r.get("rounds") or 0 for r in records))
    success_by_round = dict(Counter(
        r.get("success_at_round") for r in records if r.get("success")))
    candidate_dist = dict(Counter(
        r.get("sample_candidate_type", "unknown") for r in records))
    score_deltas = [
        best - baseline
        for record in records
        if (baseline := _safe_score(record.get("baseline_score"))) is not None
        and (best := _safe_score(record.get("best_score"))) is not None
    ]

    judged = [detail for detail in details
              if detail.get("judge_classification") in _JUDGE_CLASSIFICATIONS]
    judge_dist = dict(Counter(
        detail["judge_classification"] for detail in judged))
    n_judge_runs = len(judged)
    n_judge_agree = sum(
        detail["judge_classification"] == detail.get("verify_reward_type")
        for detail in judged
    )
    n_judge_flips = sum(
        detail.get("verify_reward_type") is not None
        and detail["judge_classification"] == "complete"
        and detail["judge_classification"] != detail["verify_reward_type"]
        for detail in judged
    )
    n_judge_fallback = sum(
        detail.get("judge_classification") == "judge_error"
        for detail in details
    )

    report = {
        "config": vars(args),
        "num_records": n,
        "num_success": n_success,
        "success_rate": n_success / max(n, 1),
        "num_advice_lines": n_with_advice,
        "rounds_distribution": round_dist,
        "success_by_round": success_by_round,
        "sample_candidate_distribution": candidate_dist,
        "mean_best_score_delta": (
            sum(score_deltas) / len(score_deltas) if score_deltas else None
        ),
        "judge_classification_distribution": judge_dist,
        "judge_runs": n_judge_runs,
        "judge_code_agreement": (n_judge_agree / n_judge_runs) if n_judge_runs else None,
        "judge_complete_flips": n_judge_flips,
        "judge_fallbacks": n_judge_fallback,
        "elapsed_wall_sec": round(elapsed_wall_sec, 2),
    }
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print(f"{getattr(args, 'env_backend', 'AWM')} Refine Report")
    print("=" * 80)
    print(f"Records:               {n}")
    print(f"Success (any round):   {n_success}  ({report['success_rate']:.2%})")
    print(f"Advice lines total:    {n_with_advice}")
    print(f"Rounds distribution:   {round_dist}")
    print(f"Success by round:      {success_by_round}")
    print(f"Sample candidates:     {candidate_dist}")
    print(f"Judge classifications: {judge_dist}")
    if n_judge_runs:
        print(f"Judge vs code agree:   {n_judge_agree}/{n_judge_runs}"
              f" ({report['judge_code_agreement']:.2%})")
    else:
        print("Judge vs code agree:   n/a")
    print(f"Judge complete flips:  {n_judge_flips}  "
          f"(judge=complete where code!=complete)")
    print(f"Judge fallbacks:       {n_judge_fallback}  "
          f"(judge failed -> code verdict used)")
    print(f"Elapsed:               {elapsed_wall_sec:.1f}s")
    print(f"JSONL:                 {ckpt.path}")
    print(f"Report:                {args.report}")


if __name__ == "__main__":
    asyncio.run(main())
