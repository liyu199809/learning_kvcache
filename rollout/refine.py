"""
Iterative refinement pipeline for Self-OPD data collection.

Per (scenario, task_idx):

  1. Fresh AWMEnv session, reset(scenario, task_idx). This is the student's
     env. The env is RESET at the start of every round, so each round begins
     from a pristine DB state — a round that corrupts the DB cannot poison
     later rounds.

  2. For k = 1..K:
       a. If k > 1, the student env is reset first (fresh DB; conversation
          history — including teacher advice — carries over). Student
          (Qwen3.5-4B) continues the conversation, calls tools on the
          student env, produces `traj_k`. Each round the LLM has up
          to `student_max_iterations` LLM turns to converge.
       b. `verify` is invoked (code mode) — it's read-only so calling it
          multiple times is safe.
       c. If reward_type == "complete", success; break.
       d. If k == K, no more advice will be produced; break.
       e. Teacher (a stronger LLM) is invoked to produce ONE line of
          `# Advice: <one-liner>`. The teacher runs against its OWN
          separate AWMEnv session that has been synchronised to the
          student's current DB state by REPLAYING the student's successful
          tool_calls from the CURRENT round (earlier rounds' mutations were
          discarded by the per-round reset). Teacher can call_tool up to 3
          times to probe.
          Only the `# Advice:` line is extracted; the teacher's private
          conversation is discarded.
       f. The advice is appended to the student conversation as a single
          user message:   {"role":"user","content":"[Expert advice]\\n# Advice: ..."}

  3. Emit a JSONL record capturing:
       * student's full conversation (WITH the advice user turns — a
         downstream trainer can mask them off).
       * per-round detail: which turn range belongs to that round, the
         verify result at that point, and (for rounds 1..k-1) the teacher
         advice.
       * summary fields: rounds, success, success_at_round, elapsed.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import re
import time
from dataclasses import dataclass, field, asdict

from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction

from agent_world_model_env import AWMEnv

from rollout.common import (
    JsonlCheckpoint,
    RateLimiter,
    RetryLLM,
    execute_native_tool_call,
    format_tools,
    llm_turn_native,
    loads_lenient,
    run_jobs,
    tools_to_openai_schema,
)
from rollout.prompt import (
    STUDENT_NATIVE_SYSTEM_PROMPT,
    STUDENT_NATIVE_FINAL_SYSTEM_PROMPT,
    TEACHER_ADVICE_SYSTEM_PROMPT,
    TEACHER_FINALIZE_SYSTEM_PROMPT,
    build_teacher_advice_input,
)


# Relaxed: matches "# Advice:" / "Advice:" / "advice:" with optional leading
# whitespace, and captures everything until the next blank line or end of text.
# Format is not important — we just want the actionable content.
ADVICE_LINE_RE = re.compile(
    r"^\s*#?\s*[Aa]dvice\s*:\s*(.+?)(?:\n\s*\n|\Z)",
    re.MULTILINE | re.DOTALL,
)


# ---------------------------------------------------------------------------
# Student half-turn: continue an existing `messages` list on an existing env.
# Returns (updated_messages, per-turn trace list, final_answer_str,
#          made_tool_call, executed_tool_calls, error_or_None).
# The env may have been mutated by earlier tool calls in THIS round; we do
# NOT reset here — resetting between rounds is the caller's job.
# ---------------------------------------------------------------------------
async def _student_step(env, llm: RetryLLM, messages: list[dict],
                         tools_schema: list[dict],
                         max_iterations: int, max_tokens: int,
                         temperature: float,
                         tool_response_cap: int = 4000) -> dict:
    trace: list[dict] = []
    executed: list[dict] = []
    content = ""
    made_tool_call = False
    error: str | None = None
    step_used = 0

    for step in range(1, max_iterations + 1):
        step_used = step
        is_last = step == max_iterations
        tool_choice = "none" if is_last else "auto"
        system_override = STUDENT_NATIVE_FINAL_SYSTEM_PROMPT if is_last else None
        tools_schema = tools_schema if not is_last else None  # forbid tools on last turn
        try:
            turn = await llm_turn_native(llm, messages, tools=tools_schema,
                                         temperature=temperature,
                                         max_tokens=max_tokens,
                                         tool_choice=tool_choice,
                                         system_override=system_override)
        except Exception as e:
            error = f"student LLM error at step {step}: {e!r}"[:500]
            break

        content = turn.content
        trace.append({"step": step, "assistant": (content or "")[:2000]})

        tool_calls = turn.tool_calls
        if not tool_calls:
            break

        made_tool_call = True
        # Every native tool_call MUST be answered by a paired `role:tool`
        # message carrying the same tool_call_id, or the next request 400s
        # with MissingParameter. Iterate all of them in order.
        for tc in tool_calls[:3]:
            tc_id = tc.get("id", "")
            fn = tc.get("function") or {}
            name = fn.get("name", "")
            raw_args = fn.get("arguments", "{}")
            args = loads_lenient(raw_args) if isinstance(raw_args, str) else raw_args
            if not isinstance(args, dict):
                args = {}

            # Refuse verify/done from the student loop — those are pipeline-level.
            if name in ("verify", "done"):
                tool_response = (
                    "Error: 'verify' and 'done' are managed by the harness; "
                    "To finish, STOP calling tools and reply with your final answer as plain text."
                )
            else:
                try:
                    tool_response = (
                        await execute_native_tool_call(env, name, args)).text
                except Exception as e:
                    tool_response = f"Error executing tool: {e!r}"
            tool_response = (tool_response or "")[:tool_response_cap]
            trace.append({
                "step": step,
                "tool": name,
                "arguments": args,
                "response": tool_response[:800],
            })
            # Track successfully executed scenario tool calls for later replay
            # on the teacher's mirror env. Names are already real scenario tool
            # names in native mode, so record them directly.
            if name not in ("verify", "done") and not tool_response.startswith("Error"):
                executed.append({
                    "tool_name": name,
                    "arguments": args,
                })
            messages.append({
                "role": "tool",
                "tool_call_id": tc_id,
                "content": f"{tool_response}\n\nYou have {max_iterations-step-1} remaining opportunities for parallel tool calls. Once the count hits 0, you must respond to the user directly whether the task is completed or not.",
            })

    return {
        "messages": messages,
        "trace": trace,
        "final_answer": content[:2000],
        "made_tool_call": made_tool_call,
        "executed_tool_calls": executed,
        "steps": step_used,
        "error": error,
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
    }


# ---------------------------------------------------------------------------
# Teacher half-turn: open a fresh AWMEnv session, replay the student's
# executed tool_calls to synchronise DB state, then run a small tool-using
# agent loop to produce ONE `# Advice:` line.
# ---------------------------------------------------------------------------
async def _teacher_advise(awm_base_url: str, scenario: str, task_idx: int,
                          task: str, student_messages: list[dict],
                          past_tool_calls: list[dict],
                          verify_reward_type: str,
                          verify_error: str | None,
                          teacher_llm: RetryLLM,
                          max_iterations: int = 4,
                          max_tool_calls: int = 3,
                          max_tokens: int = 8192,
                          temperature: float = 0.4,
                          tool_response_cap: int = 4000) -> dict:
    """Return {'advice': str | None, 'rounds_used': int, 'tool_calls_made': int,
    'replay_ok': bool, 'error': str | None}. The teacher's private conversation
    is intentionally discarded — only the `# Advice:` line escapes."""
    env = AWMEnv(base_url=awm_base_url,
                 message_timeout_s=180.0, connect_timeout_s=60.0)
    advice: str | None = None
    rounds_used = 0
    tool_calls_made = 0
    replay_ok = True
    error: str | None = None
    final_text: str = ""
    try:
        await env.connect()
        await env.reset(scenario=scenario, task_idx=task_idx)
        list_res = await env.step(ListToolsAction())
        tools_text = format_tools(list_res.observation.tools)
        tools_schema = tools_to_openai_schema(list_res.observation.tools)

        # 1) Replay student's executed tool_calls in order so the DB state
        # matches. Any replay failure just means the teacher gets a slightly
        # stale view — we still let it try.
        for tc in past_tool_calls:
            try:
                await env.step(CallToolAction(
                    tool_name=tc["tool_name"], arguments=tc["arguments"]))
            except Exception:
                replay_ok = False

        # 2) Build initial teacher user message.
        user_text = build_teacher_advice_input(
            task=task,
            student_conversation=student_messages,
            verify_reward_type=verify_reward_type,
            verify_error=verify_error
        )
        messages: list[dict] = [
            {"role": "system", "content": TEACHER_ADVICE_SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ]

        # 3) Probing phase: teacher can issue up to `max_tool_calls` read-only
        # probes to inspect DB state. This phase produces NO advice — any
        # non-tool-call output is ignored. The loop exits early if the teacher
        # voluntarily stops calling tools.
        for step in range(1, max_tool_calls + 1):
            rounds_used = step
            try:
                turn = await llm_turn_native(teacher_llm, messages,
                                             tools=tools_schema,
                                             temperature=temperature,
                                             max_tokens=max_tokens,
                                             tool_choice="auto")
            except Exception as e:
                error = f"teacher LLM error at probe step {step}: {e!r}"[:500]
                break
            tool_calls = turn.tool_calls
            if not tool_calls:
                # Teacher chose to stop probing early — proceed to finalize.
                break

            # Answer every native tool_call with a paired `role:tool` message
            # (same tool_call_id) so the follow-up request stays valid.
            for tc in tool_calls[:3]:
                tc_id = tc.get("id", "")
                fn = tc.get("function") or {}
                name = fn.get("name", "")
                raw_args = fn.get("arguments", "{}")
                args = loads_lenient(raw_args) if isinstance(raw_args, str) else raw_args
                if not isinstance(args, dict):
                    args = {}

                tool_calls_made += 1
                # Force read-only: refuse anything that looks state-mutating.
                if name in ("verify", "done"):
                    tool_response = "Error: teacher may not call verify/done. To finish, STOP calling tools and reply with your final answer as plain text."
                else:
                    try:
                        tool_response = (
                            await execute_native_tool_call(env, name, args)).text
                    except Exception as e:
                        tool_response = f"Error executing tool: {e!r}"
                tool_response = (tool_response or "")[:tool_response_cap]
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": f"{tool_response}",
                })

        # 4) Finalize phase: swap system prompt so tools are forbidden and
        # advice is mandatory. This is a SEPARATE LLM call — its only job is
        # to emit the `# Advice: ...` block based on everything observed.
        # The trailing user turn is REQUIRED: without an explicit "emit advice
        # now" instruction the model tends to keep reasoning after the last
        # tool response and never produce `content` (advice leaks into the
        # reasoning channel / gets truncated) — the "finalize returned empty
        # text" failure.
        finalize_messages = (
            [{"role": "system", "content": TEACHER_FINALIZE_SYSTEM_PROMPT}]
            + messages[1:]  # drop the old probing system prompt
            + [{
                "role": "user",
                "content": (
                    "Probing phase is over. Do NOT call any tools. "
                    "Emit your final diagnosis now, starting with "
                    "`# Advice:` on its own line."
                ),
            }]
        )
        try:
            final_turn = await llm_turn_native(
                teacher_llm, finalize_messages,
                temperature=temperature, max_tokens=max_tokens)
            final_text = final_turn.content or ""
            # Safety net: with `--reasoning-parser` / Ark the advice can still
            # land in the reasoning channel while `content` stays empty (e.g.
            # response cut off by max_tokens). Fall back to reasoning so the
            # advice isn't lost.
            if not final_text.strip() and final_turn.reasoning:
                final_text = final_turn.reasoning
        except Exception as e:
            error = f"teacher finalize failed: {e!r}"[:500]

        raw = final_text.strip()
        # Strip any stray tool-call blocks the model might still emit.
        raw_clean = re.sub(r"<tool_call>.*?</tool_call>", "", raw,
                           flags=re.DOTALL).strip()
        m = ADVICE_LINE_RE.search(raw_clean)
        if m:
            advice = m.group(1).strip()
        elif raw_clean and len(raw_clean) > 20:
            # Fallback: no explicit marker but we have prose — take it whole.
            advice = raw_clean

        # If advice is still missing, classify why so the caller can log a
        # meaningful reason instead of a generic "no advice".
        if advice is None and error is None:
            if not raw:
                error = "finalize returned empty text"
            elif not raw_clean:
                error = "finalize returned only <tool_call> blocks"
            else:
                error = f"finalize text unparseable (len={len(raw_clean)})"
    except Exception as e:
        error = f"teacher run failed: {e!r}"[:500]
    finally:
        try:
            await env.close()
        except Exception:
            pass
    return {
        "advice": advice,
        "rounds_used": rounds_used,
        "tool_calls_made": tool_calls_made,
        "replay_ok": replay_ok,
        "error": error,
        "final_raw": (final_text or "")[:800],
    }


# ---------------------------------------------------------------------------
# RefineJob — one (scenario, task_idx). Emits one JSONL record per job.
# ---------------------------------------------------------------------------
@dataclass
class RefineJob:
    scenario: str
    task_idx: int
    student_llm: RetryLLM
    teacher_llm: RetryLLM | None
    awm_base_url: str
    only_infer: bool = True
    max_rounds: int = 3
    student_max_iterations: int = 10
    teacher_max_iterations: int = 4
    teacher_max_tool_calls: int = 3
    student_max_tokens: int = 2048
    teacher_max_tokens: int = 1024
    student_temperature: float = 1.0
    teacher_temperature: float = 0.4
    episode_timeout: float = 900.0

    def key(self) -> tuple:
        return (self.scenario, self.task_idx)

    async def run(self) -> dict:
        t0 = time.monotonic()
        env = AWMEnv(
            base_url=self.awm_base_url,
            message_timeout_s=self.episode_timeout,
            connect_timeout_s=60.0,
        )
        rounds_detail: list[dict] = []
        student_messages: list[dict] = []
        task_description: str = ""
        success = False
        success_at_round: int | None = None
        error: str | None = None

        try:
            await env.connect()
            reset_res = await env.reset(
                scenario=self.scenario, task_idx=self.task_idx)
            task_description = reset_res.observation.task
            list_res = await env.step(ListToolsAction())
            tools_text = format_tools(list_res.observation.tools)
            tools_schema = tools_to_openai_schema(list_res.observation.tools)

            # Native function-calling: the tool schemas are supplied via the
            # `tools` request field, so the student system prompt must NOT carry
            # the XML/call_tool protocol (that would mislead the model). The
            # tool signatures are still shown as text for the model's reference.
            student_messages = [
                {"role": "system", "content": STUDENT_NATIVE_SYSTEM_PROMPT},
                {"role": "user", "content": task_description},
            ]

            for k in range(1, self.max_rounds + 1):
                if k > 1:
                    # Fresh DB state every round: mutations made by earlier
                    # rounds are discarded so a corrupted state cannot cascade
                    # into every later round. The conversation (incl. teacher
                    # advice) carries over; only the env state is rebuilt.
                    rr = await env.reset(scenario=self.scenario,
                                         task_idx=self.task_idx)
                turn_start_idx = len(student_messages)
                step_result = await asyncio.wait_for(
                    _student_step(
                        env, self.student_llm, student_messages,
                        tools_schema,
                        max_iterations=self.student_max_iterations,
                        max_tokens=self.student_max_tokens,
                        temperature=self.student_temperature,
                    ),
                    timeout=self.episode_timeout,
                )
                student_messages = step_result["messages"]
                turn_end_idx = len(student_messages)
                new_executed = step_result["executed_tool_calls"]

                verify = await _verify_only(env, step_result["final_answer"])

                round_record = {
                    "round": k,
                    "message_range": [turn_start_idx, turn_end_idx],
                    "student_final_answer": step_result["final_answer"],
                    "student_steps": step_result["steps"],
                    "student_made_tool_call": step_result["made_tool_call"],
                    "student_error": step_result["error"],
                    "student_new_tool_calls": new_executed,
                    "verify_reward": verify["reward"],
                    "verify_reward_type": verify["reward_type"],
                    "teacher_advice": None,
                    "teacher_rounds_used": 0,
                    "teacher_tool_calls_made": 0,
                    "teacher_replay_ok": None,
                    "teacher_error": None,
                    "teacher_final_raw": "",
                }
                rounds_detail.append(round_record)

                if verify["reward_type"] == "complete":
                    success = True
                    success_at_round = k
                    break
                if k == self.max_rounds:
                    break

                # Inference-only mode: skip the advice module entirely. The
                # student attempts each task once (+ verify) with no teacher
                # feedback carried into a next round.
                if self.only_infer:
                    break

                # Ask the teacher for one line of advice. Uses a SEPARATE env
                # session that we replay THIS ROUND's executed tool calls
                # onto so its DB state matches the student's current state
                # (earlier rounds' mutations were discarded by the reset).
                verify_error = None
                vr = verify.get("verify_result") or {}
                if isinstance(vr, dict):
                    verify_error = vr.get("error") or vr.get("message")
                    if isinstance(verify_error, str):
                        verify_error = verify_error[:400]
                teacher = await _teacher_advise(
                    awm_base_url=self.awm_base_url,
                    scenario=self.scenario,
                    task_idx=self.task_idx,
                    task=task_description,
                    student_messages=student_messages,
                    past_tool_calls=new_executed,
                    verify_reward_type=verify["reward_type"],
                    verify_error=verify_error,
                    teacher_llm=self.teacher_llm,
                    max_iterations=self.teacher_max_iterations,
                    max_tool_calls=self.teacher_max_tool_calls,
                    max_tokens=self.teacher_max_tokens,
                    temperature=self.teacher_temperature,
                )
                round_record["teacher_advice"] = teacher["advice"]
                round_record["teacher_rounds_used"] = teacher["rounds_used"]
                round_record["teacher_tool_calls_made"] = teacher["tool_calls_made"]
                round_record["teacher_replay_ok"] = teacher["replay_ok"]
                round_record["teacher_error"] = teacher["error"]
                round_record["teacher_final_raw"] = teacher.get("final_raw", "")

                if not teacher["advice"]:
                    # Teacher failed to produce advice. Surface the specific
                    # reason (empty text / only tool_call / unparseable / etc.)
                    # so the top-level log is actionable.
                    reason = teacher.get("error") or "no advice; unknown reason"
                    error = (error or "") + f"[round {k}] teacher no-advice: {reason}; "
                    break

                student_messages.append({
                    "role": "user",
                    "content": f"[Expert advice]\n# Advice: {teacher['advice']}\n\nNotice: Your first tried failed and the environment has been reset to its initial state. please try again.",
                })

        except asyncio.TimeoutError:
            error = f"episode timeout > {self.episode_timeout}s"
        except Exception as e:
            import traceback
            error = f"refine run failed: {e!r} | tb: {traceback.format_exc()}"[:2000]
        finally:
            try:
                await env.close()
            except Exception:
                pass

        elapsed = round(time.monotonic() - t0, 3)
        return {
            "scenario": self.scenario,
            "task_idx": self.task_idx,
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


# ---------------------------------------------------------------------------
# Scenario listing (mirrors rollout.rollout.main).
# ---------------------------------------------------------------------------
async def _list_all_scenarios(awm_base_url: str) -> list[dict]:
    async with AWMEnv(base_url=awm_base_url) as env:
        list_res = await env.step(
            CallToolAction(tool_name="__list_scenarios__", arguments={}))
        return list_res.observation.scenarios


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
async def main():
    parser = argparse.ArgumentParser()
    # scenario selection
    parser.add_argument("--num-scenarios", type=int, default=1000)
    parser.add_argument("--tasks-per-scenario", type=int, default=10)
    parser.add_argument("--scenarios", default=None,
                        help="Comma-separated scenario names; overrides --num-scenarios.")
    parser.add_argument("--start-scenario-idx", type=int, default=0)
    parser.add_argument("--end-scenario-idx", type=int, default=None)

    # rollout / refinement hyperparams
    parser.add_argument("--only-infer", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Inference-only: run the student once per task and "
                             "skip the teacher advice module. Pass --no-only-infer "
                             "to enable the full refine (teacher-advice) loop.")
    parser.add_argument("--max-rounds", type=int, default=3,
                        help="Max number of student→verify→teacher_advice iterations.")
    parser.add_argument("--student-max-iterations", type=int, default=4,
                        help="Max LLM turns the student may take within ONE round.")
    parser.add_argument("--student-max-tokens", type=int, default=2048)
    parser.add_argument("--student-temperature", type=float, default=1.0)
    parser.add_argument("--teacher-max-iterations", type=int, default=4)
    parser.add_argument("--teacher-max-tool-calls", type=int, default=3)
    parser.add_argument("--teacher-max-tokens", type=int, default=8192)
    parser.add_argument("--teacher-temperature", type=float, default=0.4)

    # concurrency / timeouts
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--episode-timeout", type=float, default=900.0)
    parser.add_argument("--llm-timeout", type=float, default=180.0)
    parser.add_argument("--progress-interval", type=float, default=10.0)

    # student (vLLM local, OpenAI-compat)
    parser.add_argument("--awm-base-url",
                        default=os.environ.get("AWM_BASE_URL", "http://localhost:8899"))
    parser.add_argument("--student-base-url",
                        default=os.environ.get("ENDPOINT_URL", "http://localhost:8000/v1"))
    parser.add_argument("--student-api-key",
                        default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--student-model",
                        default=os.environ.get("AWM_EXAMPLE_AGENT_MODEL", "qwen3.5-4b"))

    # teacher (Ark by default; supports both seed and deepseek-v4-pro endpoints)
    parser.add_argument("--teacher-base-url",
                        default="https://ark.cn-beijing.volces.com/api/v3")
    parser.add_argument("--teacher-api-key",
                        default=os.environ.get("ARK_API_KEY"))
    parser.add_argument("--teacher-model",
                        default=os.environ.get("TEACHER_MODEL", "ep-20260707130305-26bjx"),
                        help="Ark endpoint id. e.g. seed-2.1-pro=ep-20260707130305-26bjx; "
                             "seed-2.0-lite / DeepSeek-v4-Pro have their own endpoints.")
    parser.add_argument("--teacher-rpm", type=int, default=400)
    parser.add_argument("--teacher-tpm", type=int, default=800_000)

    # I/O
    parser.add_argument("--output-jsonl",
                        default="/mnt/storage/disk3/self_evolver/traj_data/refine_epoch1.jsonl")
    parser.add_argument("--report",
                        default="/mnt/storage/disk3/self_evolver/traj_data/refine_epoch1.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional cap on total (scenario,task) pairs.")
    args = parser.parse_args()

    student_llm = RetryLLM(
        base_url=args.student_base_url,
        api_key=args.student_api_key,
        model=args.student_model,
        timeout=args.llm_timeout,
    )
    # The teacher is only needed for the full refine (advice) loop. In
    # inference-only mode we never build it, so no teacher API key is required.
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
            timeout=args.llm_timeout,
            limiter=RateLimiter(rpm=args.teacher_rpm, tpm=args.teacher_tpm),
        )
        print(f"[refine] student={args.student_model} teacher={args.teacher_model}")
        print(f"[refine] teacher rate limit: {args.teacher_rpm} RPM, {args.teacher_tpm} TPM")

    all_scenarios = await _list_all_scenarios(args.awm_base_url)
    print(f"Total scenarios on server: {len(all_scenarios)}")
    if args.scenarios:
        picked = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    else:
        end = args.end_scenario_idx if args.end_scenario_idx is not None \
              else args.start_scenario_idx + args.num_scenarios
        picked = [s["name"] for s in all_scenarios[args.start_scenario_idx:end]]
    print(f"Selected {len(picked)} scenarios (first 5: {picked[:5]})")

    todo = [(s, t) for s in picked for t in range(args.tasks_per_scenario)]
    if args.limit is not None:
        todo = todo[: args.limit]
    print(f"[refine] {len(todo)} (scenario, task) pairs, "
          f"max_rounds={args.max_rounds}, concurrency={args.concurrency}")

    jobs = [
        RefineJob(
            scenario=s,
            task_idx=t,
            student_llm=student_llm,
            teacher_llm=teacher_llm,
            awm_base_url=args.awm_base_url,
            only_infer=args.only_infer,
            max_rounds=args.max_rounds,
            student_max_iterations=args.student_max_iterations,
            teacher_max_iterations=args.teacher_max_iterations,
            teacher_max_tool_calls=args.teacher_max_tool_calls,
            student_max_tokens=args.student_max_tokens,
            teacher_max_tokens=args.teacher_max_tokens,
            student_temperature=args.student_temperature,
            teacher_temperature=args.teacher_temperature,
            episode_timeout=args.episode_timeout,
        )
        for (s, t) in todo
    ]

    ckpt = JsonlCheckpoint(args.output_jsonl,
                            key_fields=("scenario", "task_idx"))

    t0 = time.monotonic()
    await run_jobs(jobs, args.concurrency, ckpt,
                   progress_interval=args.progress_interval, resume=args.resume)
    elapsed = time.monotonic() - t0

    _aggregate_and_print(ckpt, args, elapsed)


def _aggregate_and_print(ckpt: JsonlCheckpoint, args: argparse.Namespace,
                          elapsed_wall_sec: float) -> None:
    records = ckpt.read_all()
    n = len(records)
    n_success = sum(1 for r in records if r.get("success"))
    n_with_advice = sum(
        1 for r in records
        for rd in (r.get("rounds_detail") or [])
        if rd.get("teacher_advice")
    )
    round_dist: dict = {}
    success_by_round: dict = {}
    for r in records:
        rd = r.get("rounds") or 0
        round_dist[rd] = round_dist.get(rd, 0) + 1
        if r.get("success"):
            k = r.get("success_at_round")
            success_by_round[k] = success_by_round.get(k, 0) + 1

    report = {
        "config": vars(args),
        "num_records": n,
        "num_success": n_success,
        "success_rate": n_success / max(n, 1),
        "num_advice_lines": n_with_advice,
        "rounds_distribution": round_dist,
        "success_by_round": success_by_round,
        "elapsed_wall_sec": round(elapsed_wall_sec, 2),
    }
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print("AWM Refine Report")
    print("=" * 80)
    print(f"Records:               {n}")
    print(f"Success (any round):   {n_success}  ({report['success_rate']:.2%})")
    print(f"Advice lines total:    {n_with_advice}")
    print(f"Rounds distribution:   {round_dist}")
    print(f"Success by round:      {success_by_round}")
    print(f"Elapsed:               {elapsed_wall_sec:.1f}s")
    print(f"JSONL:                 {ckpt.path}")
    print(f"Report:                {args.report}")


if __name__ == "__main__":
    asyncio.run(main())