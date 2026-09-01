"""Convert refine records into one Swift OPSD row per useful task.

Kept categories:
  * complete_first_try: clean task prefix + a results-free reference plan.
  * complete_after_refine: the immediately preceding attempt + its advice.
  * improved_partial: the same transition shape with its real partial score.

For multi-round records, the advice and trajectory always come from the round
immediately before the selected target round. Verifier/reset failures,
checker errors, missing advice, and non-improving partial attempts are dropped.

Run:
    cd /mnt/storage/disk3/self_evolver && python -m rollout.refine2swift \
        --refine-jsonl traj_data/refine_200.jsonl \
        --output-jsonl traj_data/swift_opsd.jsonl
    # then train with: --agent_template hermes
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional

from tqdm import tqdm

# Tells the student the env is being reset after a failed first round.
ENV_RESET_NOTE = "Notice: Your first tried failed and the environment has been reset to its initial state. please try again."

# Privileged advice block appended to the teacher's conversation only.
_ADVICE_USER_BLOCK = (
    "[Expert advice]\n# Advice: {advice}"
)

POSITIVE_FEEDBACK = (
    "This task is within your ability and can be solved directly. "
    "Work through it step by step and complete it with the appropriate tool calls."
)


_TRAJECTORY_HINT_BLOCK = (
    "[Reference plan]\n"
    "A working solution to this exact task uses the tool calls below, grouped "
    "into steps. Calls listed under the same step were made in parallel (order "
    "among them does not matter); steps run in the given order. Execute this "
    "plan yourself now!!\n\n"
    "{trajectory}"
)

_CODE_REFERENCE_BLOCK = (
    "[Reference solution]\n"
    "The following submission passed every hidden test. Use it as privileged "
    "evidence to derive a correct solution for the user question.\n\n"
    "{solution}"
)

# A tool result is a dead-end (drop its call from the plan) when it reports an
# error / non-2xx status. The plan should list only the CLEAN expert path, not
# the model's abandoned wrong guesses (e.g. a 422 int_parsing). We read results
# ONLY to filter these out — the results themselves are NOT rendered into the
# plan, so the teacher isn't handed the answers/intermediate values (that would
# let it conclude the task is finished and stop calling tools; empirically that
# collapses the student into a "think-and-declare-done, never act" pattern).
_TRAJ_ERROR_RE = re.compile(r"error|status code:\s*[45]\d\d", re.IGNORECASE)


def _is_error_result(text: str) -> bool:
    return bool(_TRAJ_ERROR_RE.search(text or ""))

def _render_success_trajectory(conv: List[Dict[str, Any]]) -> str:
    """Render the round-1 winning trajectory as an ordered TOOL-CALL PLAN for
    the teacher's privileged context, PRESERVING PARALLELISM.

    Each agent round (one assistant turn) becomes one Step. Tool calls the model
    issued together in that turn were dispatched in PARALLEL, so they are grouped
    under the same Step rather than flattened into a false serial order:

        Step 1: call <tool>(<args>)                     # single call
        Step 2 (parallel):                              # >1 call in one turn
            - call <toolA>(<args>)
            - call <toolB>(<args>)
        Step 3: call <tool>(<args>)

    Only the ORDER ACROSS steps is a real dependency (later rounds saw earlier
    results); calls WITHIN a step are order-independent.

    Deliberately OMITTED (see _TRAJECTORY_HINT_BLOCK):
      * tool RESULTS — handing the teacher the returned values / intermediate
        state lets it conclude the task is already solved and answer without
        acting; that distils the student into a "think-and-declare-done, never
        call a tool" pattern. Results are read ONLY to drop dead-end steps.
      * the FINAL ANSWER — an explicit "task is complete" turn is the strongest
        such completion signal, so it is never rendered.

    Dead-end calls are DROPPED: a call whose paired result reports an error /
    non-2xx status (e.g. a 422 from a wrong tool-name guess the model later
    abandoned) is skipped, so the plan lists only the clean expert path. Within
    a turn, call[k] pairs with the k-th following `tool` result (mirroring how
    the rollout appends them)."""
    # Group by assistant turn: each group = the calls issued together in one
    # turn (a parallel batch), each paired with its result for error-filtering.
    groups: List[List[Dict[str, Any]]] = []
    n = len(conv)
    i = 0
    while i < n:
        m = conv[i]
        tcs = m.get("tool_calls") or [] if m.get("role") == "assistant" else []
        if not tcs:
            i += 1
            continue
        calls = []
        for tc in tcs:
            fn = tc.get("function") or {}
            args = _lenient_json(fn.get("arguments"))
            if not isinstance(args, dict):
                args = {}
            calls.append({"name": fn.get("name", ""), "args": args})
        # The next len(calls) `tool` messages are this turn's results, in order.
        results: List[str] = []
        j = i + 1
        while j < n and conv[j].get("role") == "tool" and len(results) < len(calls):
            results.append((conv[j].get("content") or "").strip())
            j += 1
        # Keep only calls whose paired result is not an error / dead-end.
        kept = [c for idx, c in enumerate(calls)
                if not (idx < len(results) and _is_error_result(results[idx]))]
        if kept:
            groups.append(kept)
        i = j

    lines: List[str] = []
    step = 0
    for kept in groups:
        step += 1
        if len(kept) == 1:
            c = kept[0]
            lines.append(
                f"Step {step}: call {c['name']}"
                f"({json.dumps(c['args'], ensure_ascii=False)})")
        else:
            lines.append(f"Step {step} (parallel):")
            for c in kept:
                lines.append(
                    f"    - call {c['name']}"
                    f"({json.dumps(c['args'], ensure_ascii=False)})")
    return "\n".join(lines)

def _task_prefix(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return a CLEAN on-policy starting context: the system prompt (if any)
    plus the first user turn (the task). No tool history, no prior answer.

    Used for first-attempt-success cases: the student must re-solve the task
    from scratch on-policy, exactly as it did in round 1. Feeding the already-
    solved round-1 trajectory back as context makes the model "see" it has
    already finished and stop calling tools (it stalls / gives up), so we drop
    that history entirely and restart from [system, task]."""
    system_msg = next((m for m in messages if m.get("role") == "system"), None)
    user_msg = next((m for m in messages if m.get("role") == "user"), None)
    out: List[Dict[str, Any]] = []
    if system_msg is not None:
        out.append({k: v for k, v in system_msg.items()
                    if k not in ("reasoning", "tool_call_id")})
    if user_msg is not None:
        out.append({"role": "user", "content": user_msg.get("content", "")})
    return out


def _safe_score(value: Any) -> Optional[float]:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if math.isfinite(score) else None


def _checker_errors(detail: Dict[str, Any]) -> int:
    result = detail.get("verify_result")
    if not isinstance(result, dict):
        return 0
    try:
        return int(result.get("checker_errors") or 0)
    except (TypeError, ValueError):
        return 1


def _invalid_verification(detail: Dict[str, Any]) -> bool:
    return bool(detail.get("verify_error")) or _checker_errors(detail) != 0


def _is_complete(record: Dict[str, Any], detail: Dict[str, Any]) -> bool:
    if record.get("data_source") == "envscaler_rl":
        return detail.get("verify_reward_type") == "complete"
    return detail.get("final_reward_type", detail.get("verify_reward_type")) == "complete"


def _select_candidate(record: Dict[str, Any]) -> tuple[Optional[Dict[str, Any]], str]:
    """Choose at most one target transition, independently validating scores."""
    details = [d for d in (record.get("rounds_detail") or [])
               if isinstance(d, dict)]
    details.sort(key=lambda d: d.get("round", 0))
    if not details:
        return None, "no_rounds"

    complete_idx = next(
        (idx for idx, detail in enumerate(details)
         if _is_complete(record, detail)),
        None,
    )
    if complete_idx is not None:
        target = details[complete_idx]
        if _invalid_verification(target):
            return None, "target_verify_error"
        if complete_idx == 0:
            return {
                "sample_type": "complete_first_try",
                "target": target,
                "target_idx": complete_idx,
                "source": None,
                "baseline_score": _safe_score(details[0].get("verify_reward")),
                "source_score": None,
                "target_score": _safe_score(target.get("verify_reward")),
            }, "kept"
        source = details[complete_idx - 1]
        if _invalid_verification(source):
            return None, "source_verify_error"
        if not source.get("teacher_advice"):
            return None, "missing_advice"
        return {
            "sample_type": "complete_after_refine",
            "target": target,
            "target_idx": complete_idx,
            "source": source,
            "baseline_score": _safe_score(details[0].get("verify_reward")),
            "source_score": _safe_score(source.get("verify_reward")),
            "target_score": _safe_score(target.get("verify_reward")),
        }, "kept"

    baseline = details[0]
    baseline_score = _safe_score(baseline.get("verify_reward"))
    if baseline_score is None:
        return None, "missing_baseline_score"
    if _invalid_verification(baseline):
        return None, "baseline_verify_error"

    improved = []
    for idx, detail in enumerate(details[1:], start=1):
        score = _safe_score(detail.get("verify_reward"))
        if (score is not None and score > baseline_score + 1e-9
                and not _invalid_verification(detail)):
            improved.append((score, idx, detail))
    if not improved:
        return None, "no_score_improvement"

    best_score = max(score for score, _, _ in improved)
    _, target_idx, target = next(
        item for item in improved if abs(item[0] - best_score) <= 1e-12)
    source = details[target_idx - 1]
    if _invalid_verification(source):
        return None, "source_verify_error"
    if not source.get("teacher_advice"):
        return None, "missing_advice"
    return {
        "sample_type": "improved_partial",
        "target": target,
        "target_idx": target_idx,
        "source": source,
        "baseline_score": baseline_score,
        "source_score": _safe_score(source.get("verify_reward")),
        "target_score": best_score,
    }, "kept"


def _round_messages(record: Dict[str, Any], detail: Dict[str, Any]) -> List[Dict[str, Any]]:
    conv = record.get("student_conversation") or []
    bounds = detail.get("message_range") or []
    if (not isinstance(bounds, list) or len(bounds) != 2
            or not all(isinstance(value, int) for value in bounds)):
        return []
    start, end = bounds
    if start < 0 or end < start or end > len(conv):
        return []
    return copy.deepcopy(conv[start:end])


def _lenient_json(s: Any) -> Any:
    """Parse a JSON string into a Python object, tolerating minor malformation.
    Returns the input unchanged if it is already a dict/list; {} on failure."""
    if isinstance(s, (dict, list)):
        return s
    if not isinstance(s, str) or not s.strip():
        return {}
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        try:
            import json_repair
            return json_repair.loads(s)
        except Exception:
            return {}


def _to_swift_agent_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert an OpenAI-native message list (assistant carries a `tool_calls`
    list with string `arguments` + ids; tool results use role `tool` with
    `tool_call_id`) into swift's native agent format, which is what the
    `agent_template` (e.g. hermes) expects:

      * assistant text / reasoning  -> {"role":"assistant","content":"<think>..</think>text"}
      * each tool call              -> {"role":"tool_call",
                                        "content": '{"name":..,"arguments":{..}}'}
      * tool result                 -> {"role":"tool_response","content": ".."}

    Key normalisations (required — see swift Agent-support docs + Qwen3.5
    chat_template):
      * `arguments` is parsed from a JSON *string* into a *dict*; the Qwen3.5
        template renders `tool_call.arguments|items` and crashes on a string.
      * `id` / `tool_call_id` are dropped — swift pairs tool_call/tool_response
        by ORDER, not id, and the Qwen template never references them.
      * assistant `reasoning` is folded into content as a `<think>` block so it
        is not lost; if reasoning was already inlined in content it is kept.
    """
    out: List[Dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        if role == "assistant":
            reasoning = (m.get("reasoning") or "").strip()
            content = (m.get("content") or "").strip()
            if reasoning:
                text = f"<think>{reasoning}</think>{content}" if content \
                    else f"<think>{reasoning}</think>"
            else:
                text = content
            tool_calls = m.get("tool_calls") or []
            # Emit the assistant text turn (think + any prose) when present.
            if text:
                out.append({"role": "assistant", "content": text})
            # One swift `tool_call` message per native tool_call, arguments dict.
            for tc in tool_calls:
                fn = tc.get("function") or {}
                name = fn.get("name", "")
                args = _lenient_json(fn.get("arguments"))
                if not isinstance(args, dict):
                    args = {}
                out.append({
                    "role": "tool_call",
                    "content": json.dumps(
                        {"name": name, "arguments": args}, ensure_ascii=False),
                })
            # Assistant turn with neither text nor tool calls: keep a placeholder
            # so message alternation is preserved.
            if not text and not tool_calls:
                out.append({"role": "assistant", "content": ""})
        elif role == "tool":
            out.append({"role": "tool_response", "content": m.get("content") or ""})
        else:
            # system / user pass through; strip any stray bookkeeping key.
            out.append({k: v for k, v in m.items()
                        if k not in ("reasoning", "tool_call_id")})
    return out


async def _fetch_tools_by_scenario(scenarios: List[str],
                                   awm_base_url: str) -> Dict[str, str]:
    """Fetch each scenario's tool schemas once and return a mapping
    scenario -> tools JSON *string* (swift's `tools` field format). The schema
    is produced by the same `tools_to_openai_schema` used at rollout time so
    the training-time tool list matches what the student actually saw.

    Scenarios are fetched CONCURRENTLY via ``as_completed`` (rather than
    ``gather``) so a single slow / timing-out scenario does not block the
    rest; each one is merged into the result as soon as it finishes.
    Imports are done once in the outer scope so concurrent workers do not
    race on the import lock."""
    from agent_world_model_env import AWMEnv
    from openenv.core.env_server.mcp_types import ListToolsAction
    from rollout.common import tools_to_openai_schema

    sem = asyncio.Semaphore(500)

    async def _fetch_one(scenario: str) -> tuple[str, str]:
        async with sem:
            env = AWMEnv(base_url=awm_base_url,
                         message_timeout_s=60.0, connect_timeout_s=30.0)
            try:
                await env.connect()
                await env.reset(scenario=scenario, task_idx=0)
                list_res = await env.step(ListToolsAction())
                schemas = tools_to_openai_schema(list_res.observation.tools)
                return scenario, json.dumps(schemas, ensure_ascii=False)
            except Exception as e:  # noqa: BLE001 - best-effort; skip tools on failure
                print(f"[warn] failed to fetch tools for {scenario}: {e!r}")
                return scenario, ""
            finally:
                try:
                    await env.close()
                except Exception:
                    pass

    tasks = [asyncio.create_task(_fetch_one(s)) for s in scenarios]
    out: Dict[str, str] = {}
    for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks),
                    desc="Fetching tool schemas"):
        scenario, tools = await fut
        out[scenario] = tools
    return out


def convert(refine_jsonl: str, output_jsonl: str,
            awm_base_url: str = "http://localhost:8899",
            with_tools: bool = True) -> None:
    n_in = 0
    drop_reasons: Counter = Counter()
    rows_by_task: Dict[tuple, Dict[str, Any]] = {}

    with open(refine_jsonl, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                drop_reasons["invalid_json"] += 1
                continue
            candidate, reason = _select_candidate(rec)
            if candidate is None:
                drop_reasons[reason] += 1
                continue

            target = candidate["target"]
            source = candidate["source"]
            conv = rec.get("student_conversation") or []
            if candidate["sample_type"] == "complete_first_try":
                winning_round = _round_messages(rec, target)
                if not winning_round:
                    drop_reasons["invalid_target_range"] += 1
                    continue
                prefix = _task_prefix(copy.deepcopy(conv))
                task_text = prefix[-1]["content"] if prefix else ""
                trajectory = _render_success_trajectory(winning_round)
                if rec.get("data_source") == "deepcoder_taco":
                    solution = str(target.get("student_final_answer") or "").strip()
                    if not solution:
                        drop_reasons["missing_reference_solution"] += 1
                        continue
                    teacher_prompt = (
                        _CODE_REFERENCE_BLOCK.format(solution=solution)
                        + "\n\n[User Question]" + task_text
                    )
                elif trajectory.strip():
                    teacher_prompt = (
                        _TRAJECTORY_HINT_BLOCK.format(trajectory=trajectory)
                        + "\n\n[User Question]" + task_text
                    )
                else:
                    teacher_prompt = (
                        _ADVICE_USER_BLOCK.format(advice=POSITIVE_FEEDBACK)
                        + "\n\n[User Question]" + task_text
                    )
            else:
                source_round = _round_messages(rec, source)
                if not source_round:
                    drop_reasons["invalid_source_range"] += 1
                    continue
                prefix = _to_swift_agent_messages(
                    _task_prefix(copy.deepcopy(conv)) + source_round)
                prefix.append({"role": "user", "content": ENV_RESET_NOTE})
                teacher_prompt = (
                    _ADVICE_USER_BLOCK.format(
                        advice=str(source["teacher_advice"]).strip())
                    + "\n\n" + ENV_RESET_NOTE
                )

            baseline_score = candidate["baseline_score"]
            source_score = candidate["source_score"]
            target_score = candidate["target_score"]
            row = {
                "messages": prefix,
                "teacher_prompt": teacher_prompt,
                "data_source": rec.get("data_source", "awm"),
                "scenario": rec.get("scenario"),
                "task_idx": rec.get("task_idx"),
                "task_id": rec.get("task_id"),
                "env_config": {
                    "scenario": rec.get("scenario"),
                    "task_idx": rec.get("task_idx"),
                    "awm_base_url": awm_base_url,
                },
                "sample_type": candidate["sample_type"],
                "source_round": source.get("round") if source else None,
                "target_round": target.get("round"),
                "baseline_score": baseline_score,
                "source_score": source_score,
                "target_score": target_score,
                "score_delta": (
                    target_score - baseline_score
                    if target_score is not None and baseline_score is not None
                    else None
                ),
                "verify_reward": target_score,
                "verify_reward_type": target.get("verify_reward_type"),
            }
            key = (
                row["data_source"], row["scenario"],
                row["task_id"] if row["task_id"] is not None else row["task_idx"],
            )
            previous = rows_by_task.get(key)
            if previous is None:
                rows_by_task[key] = row
            else:
                drop_reasons["duplicate_task"] += 1
                priority = {"improved_partial": 1, "complete_after_refine": 2,
                            "complete_first_try": 3}
                row_score = (row["target_score"]
                             if row["target_score"] is not None else -math.inf)
                previous_score = (
                    previous["target_score"]
                    if previous["target_score"] is not None else -math.inf)
                if (priority[row["sample_type"]], row_score) > (
                        priority[previous["sample_type"]],
                        previous_score):
                    rows_by_task[key] = row

    rows = list(rows_by_task.values())

    # Pass 2: fetch each scenario's tool schemas once (swift `tools` field), so
    # the model sees the tool definitions during training.
    tools_by_scenario: Dict[str, str] = {}
    if with_tools and rows:
        scenarios = sorted({r["scenario"] for r in rows if r.get("scenario")})
        print(f"Fetching tool schemas for {len(scenarios)} scenarios...")
        tools_by_scenario = asyncio.run(
            _fetch_tools_by_scenario(scenarios, awm_base_url))

    # Pass 3: write out, attaching the tools string per row.
    with open(output_jsonl, "w", encoding="utf-8") as fout:
        for row in rows:
            tools = tools_by_scenario.get(row.get("scenario"), "")
            if tools:
                row = {"tools": tools, **row}
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    kept = Counter(row["sample_type"] for row in rows)
    print(f"Read {n_in} refine records, kept {len(rows)} cases "
          f"{dict(kept)}, dropped {dict(drop_reasons)} -> {output_jsonl}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refine-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--env-base-url", "--awm-base-url",
                        dest="awm_base_url", default="http://localhost:8899",
                        help="Environment server used to fetch tool schemas.")
    parser.add_argument("--no-tools", action="store_true",
                        help="Skip fetching/attaching the `tools` field.")
    args = parser.parse_args()
    convert(args.refine_jsonl, args.output_jsonl,
            awm_base_url=args.awm_base_url, with_tools=not args.no_tools)


if __name__ == "__main__":
    main()
