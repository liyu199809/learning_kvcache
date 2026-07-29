"""Convert a rollout.refine JSONL into a swift OPSD dataset (GKD online).

Student and teacher share the SAME prefix; the only difference is that the
teacher's conversation has the privileged info appended at the END (after the
trailing user note), while the student's does not. swift's OPSD
``build_teacher_view`` replaces the LAST user message of ``messages`` with the
``teacher_prompt`` string, so every row's ``messages`` must end in a user turn.

Two success cases are emitted, symmetric in shape:

  hard-success (round-1 failed, advice injected, eventual success):
      student messages:
          system(agent prompt)
          user(task / question)
          ... round-1 history ...
          user("[Round 1 failed] environment reset note")
      teacher_prompt:
          "[Expert advice]\n# Advice: <advice>\n\n" + env-reset note

  first-attempt success (solved round 1, no advice):
      student messages:
          system(agent prompt)
          user(task / question)              # CLEAN restart — no round-1 history
      teacher_prompt:
          "[Reference plan]\n<step-grouped tool-CALL plan (parallel-aware), no results/answer>\n\n" + task text

For first-attempt cases the student restarts from a clean [system, task] context
and re-solves on-policy (replaying the already-solved round-1 trajectory into the
student would make the model "see" it is done and stop calling tools). The
winning round-1 trajectory becomes the TEACHER's privileged plan, but rendered as
the ORDERED TOOL CALLS ONLY — no tool results, no final answer. Handing the
teacher the results/answer signals "task already complete", which distils the
student into a "think-and-declare-done, never act" collapse (observed: tool-call
rate fell to ~0% and num_turns→1 across training). A results-free plan instead
tells the teacher WHICH tools to call in WHAT order while forcing it (and thus
the student) to actually execute them and read the real responses. Both cases
carry ``verify_reward_type="complete"``.

The OpenAI-native tool-call turns produced by rollout.refine (assistant with a
`tool_calls` list, string `arguments`, ids; `role:tool` results) are converted
to swift's native agent format (`tool_call` / `tool_response` roles, arguments
as dicts, ids dropped) so swift's `agent_template` (e.g. hermes) can render
them. Each row also carries a `tools` JSON string with the scenario's tool
schemas, fetched live from the AWM env.

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
import re
from typing import Any, Dict, List, Optional

ADVICE_MARKER = "[Expert advice]"

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

_THINK_BLOCK_RE = re.compile(r'<think>.*?</think>', re.DOTALL)

def strip_think(text: str) -> str:
    text = _THINK_BLOCK_RE.sub('', text)      # closed <think>...</think>
    return text.strip()


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

def _find_advice_idx(messages: List[Dict[str, Any]]) -> Optional[int]:
    for i, m in enumerate(messages):
        if m.get("role") == "user" and ADVICE_MARKER in str(m.get("content", "")):
            return i
    return None

def drop_tool_before_user(messages):
      """删除紧邻在 user 消息之前的 tool 消息。

      遍历 messages,每当遇到 role=='user' 的消息,就把它前方连续的
      tool / tool_response 消息全部弹出,直到 user 前不再是 tool。
      返回新列表,不修改原列表;末尾 tool(后面没有 user 跟随)不删。
      """
      TOOL_ROLES = {'tool'}
      result = []
      for m in messages:
          if m.get('role') == 'user':
              while result and result[-1].get('role') in TOOL_ROLES:
                  result.pop()
          result.append(m)
      return result

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


def _last_advice(record: Dict[str, Any]) -> Optional[str]:
    advice = None
    for rd in record.get("rounds_detail") or []:
        a = rd.get("teacher_advice")
        if a:
            advice = a
    return advice


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
    the training-time tool list matches what the student actually saw."""
    from agent_world_model_env import AWMEnv
    from openenv.core.env_server.mcp_types import ListToolsAction
    from rollout.common import tools_to_openai_schema

    out: Dict[str, str] = {}
    for scenario in scenarios:
        env = AWMEnv(base_url=awm_base_url,
                     message_timeout_s=60.0, connect_timeout_s=30.0)
        try:
            await env.connect()
            await env.reset(scenario=scenario, task_idx=0)
            list_res = await env.step(ListToolsAction())
            schemas = tools_to_openai_schema(list_res.observation.tools)
            out[scenario] = json.dumps(schemas, ensure_ascii=False)
        except Exception as e:  # noqa: BLE001 — best-effort; skip tools on failure
            print(f"[warn] failed to fetch tools for {scenario}: {e!r}")
            out[scenario] = ""
        finally:
            try:
                await env.close()
            except Exception:
                pass
    return out


def convert(refine_jsonl: str, output_jsonl: str,
            awm_base_url: str = "http://localhost:8899",
            with_tools: bool = True) -> None:
    n_in = 0
    n_hard = 0    # hard-success: round-1 fail + advice + eventual success
    n_first = 0   # first-attempt success: solved round 1, no advice
    rows: List[Dict[str, Any]] = []

    # Pass 1: read refine records, build swift rows. Two success cases are
    # kept, symmetric in shape:
    #   * hard-success   - round-1 failed (advice injected) and the episode
    #                      ultimately succeeded; the real advice is the
    #                      teacher's privileged info.
    #   * first-attempt  - solved on round 1 (no advice); a fixed positive
    #                      feedback stands in as the privileged info.
    with open(refine_jsonl, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            if not rec.get("success"):
                continue

            conv = rec.get("student_conversation") or []
            advice_idx = _find_advice_idx(conv)

            if advice_idx is not None:
                # Hard-success: cut at the advice turn, append the env-reset
                # note; the teacher's privileged info is the real advice.
                advice = _last_advice(rec)
                if not advice:
                    continue
                prefix = _to_swift_agent_messages(
                    copy.deepcopy(conv[:advice_idx]))
                # prefix.append({"role": "assistant", "content": strip_think(conv[advice_idx-1].get("content", ""))})
                prefix.append({"role": "user", "content": ENV_RESET_NOTE})
                teacher_prompt = (
                    _ADVICE_USER_BLOCK.format(advice=advice.strip())
                    + "\n\n" + ENV_RESET_NOTE
                )
                n_hard += 1
            else:
                # First-attempt success: no advice was injected. Start the
                # on-policy rollout from a CLEAN [system, task] context and let
                # the student re-solve from scratch (exactly as it did in round

                prefix = _task_prefix(copy.deepcopy(conv))
                task_text = prefix[-1]["content"] if prefix else ""
                trajectory = _render_success_trajectory(conv)
                if trajectory.strip():
                    teacher_prompt = (
                        _TRAJECTORY_HINT_BLOCK.format(trajectory=trajectory)
                        + "\n\n" + "[User Question]"+task_text
                    )
                else:
                    # Degenerate trajectory (no tool calls captured): fall back
                    # to the generic positive nudge so the row is still usable.
                    teacher_prompt = (
                        _ADVICE_USER_BLOCK.format(advice=POSITIVE_FEEDBACK)
                        + "\n\n[User Question]" + task_text
                    )
                n_first += 1

            rows.append({
                "messages": prefix,
                "teacher_prompt": teacher_prompt,
                "scenario": rec.get("scenario"),
                "task_idx": rec.get("task_idx"),
                # Consumed by awm_scheduler (req.data_dict['env_config']) to
                # reset the AWM env for on-policy rollout during training.
                "env_config": {
                    "scenario": rec.get("scenario"),
                    "task_idx": rec.get("task_idx"),
                    "awm_base_url": awm_base_url,
                },
                # Positive reward signal for GRPO/OPD-RL: the privileged
                # trajectory is a known success.
                "verify_reward_type": "complete",
            })

    # Pass 2: fetch each scenario's tool schemas once (swift `tools` field), so
    # the model sees the tool definitions during training.
    tools_by_scenario: Dict[str, str] = {}
    if with_tools and rows:
        scenarios = sorted({r["scenario"] for r in rows if r.get("scenario")})
        print(f"Fetching tool schemas for {len(scenarios)} scenarios...")
        tools_by_scenario = asyncio.run(
            _fetch_tools_by_scenario(scenarios, awm_base_url))

    # Pass 3: write out, attaching the tools string per row.
    n_kept = 0
    with open(output_jsonl, "w", encoding="utf-8") as fout:
        for row in rows:
            tools = tools_by_scenario.get(row.get("scenario"), "")
            if tools:
                # swift expects `tools` as a JSON *string* of the tool list.
                row = {"tools": tools, **row}
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_kept += 1

    print(f"Read {n_in} refine records, kept {n_kept} cases "
          f"(hard-success={n_hard}, first-attempt={n_first}) -> {output_jsonl}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refine-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--awm-base-url", default="http://localhost:8899",
                        help="AWM env server used to fetch per-scenario tool schemas.")
    parser.add_argument("--no-tools", action="store_true",
                        help="Skip fetching/attaching the `tools` field.")
    args = parser.parse_args()
    convert(args.refine_jsonl, args.output_jsonl,
            awm_base_url=args.awm_base_url, with_tools=not args.no_tools)


if __name__ == "__main__":
    main()
