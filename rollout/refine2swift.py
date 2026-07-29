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
          user(task / question)
          ... round-1 history (incl. the successful final answer) ...
          user("[First attempt succeeded] task-done note")
      teacher_prompt:
          "[Expert advice]\n# Advice: <positive feedback>\n\n" + task-done note

For first-attempt cases no real advice exists, so a fixed positive feedback
stands in as the teacher's privileged info, affirming the workflow that just
worked. Both cases carry ``verify_reward_type="complete"``.

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

# Tells the student the task was solved on the FIRST attempt - no failure,
# no reset. Mirrors ENV_RESET_NOTE's role for first-attempt-success cases.
TASK_DONE_NOTE = (
    "Notice: You completed the task on your first attempt. "
    "The environment has been reset to its initial state. please try again."
)

# Privileged advice block appended to the teacher's conversation only.
_ADVICE_USER_BLOCK = (
    "[Expert advice]\n# Advice: {advice}"
)

POSITIVE_FEEDBACK = (
    "Your approach was correct and the task is complete. "
)

_THINK_BLOCK_RE = re.compile(r'<think>.*?</think>', re.DOTALL)

def strip_think(text: str) -> str:
    text = _THINK_BLOCK_RE.sub('', text)      # closed <think>...</think>
    return text.strip()

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
                # First-attempt success: no advice was injected. The whole
                # round-1 conversation is the prefix; a positive feedback is
                # the teacher's privileged info (mirrors the advice block).
                prefix = _to_swift_agent_messages(copy.deepcopy(conv))
                prefix.append({"role": "user", "content": TASK_DONE_NOTE})
                teacher_prompt = (
                    _ADVICE_USER_BLOCK.format(advice=POSITIVE_FEEDBACK)
                    + "\n\n" + TASK_DONE_NOTE
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
