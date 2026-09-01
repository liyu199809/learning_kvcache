"""
Prompts and input builders for rollout and critic.

- ROLLOUT_SYSTEM_PROMPT: reused from AWM's official prompt.
- CRITIC_SYSTEM_PROMPT: for the tool-using critic agent; it can call
  `list_tools` / `call_tool` on the live AWM environment for the scenario,
  then emits multiple `# Experience: <one-line lesson>` rows.
- build_critic_batch_input(): renders one scenario's N rollout records
  (task, outcome, trajectory) into the initial user message for the critic.
- JUDGE_SYSTEM_PROMPT / build_judge_input(): LLM-as-judge verdict for one
  refine round; taxonomy aligned with the AWM server-side judge.
"""

import json

from agent_world_model_env.server.prompts import DEFAULT_SYSTEM_PROMPT as ROLLOUT_SYSTEM_PROMPT


_TOOL_USE_HARD_RULES = """\
## Tool-calling hard rules (MUST follow — violations cause errors)

1. **Exactly ONE tool call per turn.** Never emit multiple `<tool_call>` blocks
   in the same assistant message. If you need several tools, call them one by
   one across separate turns, reading each tool's response before the next.
2. **Always wrap real tool calls in `call_tool`.** The only top-level tool
   names are `list_tools` and `call_tool`. To invoke any scenario-specific
   tool (e.g. `create_booking`, `search_properties`), you MUST go through the
   `call_tool` wrapper like this:
   `<tool_call>{"name": "call_tool", "arguments": {"tool_name": "<actual_tool>", "arguments": {...}}}</tool_call>`
   Never put the scenario tool name directly in the top-level `name` field —
   that will always fail with "Unknown tool" even if the tool is listed."""


def build_rollout_system_prompt(scenario_skill_md: str | None) -> str:
    """Compose the rollout agent's system prompt.

    Hard tool-use rules are always appended (Qwen3-4B otherwise often emits
    multiple tool calls or calls scenario tools directly, both of which
    produce errors). If `scenario_skill_md` is provided, it is appended
    after the hard-rules section."""
    base = f"{ROLLOUT_SYSTEM_PROMPT}\n\n{_TOOL_USE_HARD_RULES}"
    if not scenario_skill_md or not scenario_skill_md.strip():
        return base
    tips = scenario_skill_md.strip()
    return (
        f"{base}\n\n"
        f"## Prior experiences from this scenario\n"
        f"The following lessons were distilled from earlier attempts. They "
        f"describe transferable tool-use patterns — not the answer itself.\n\n"
        f"{tips}\n"
    )


# Native function-calling student prompt (rollout.refine). The scenario tools
# are supplied through the request's `tools` field, so the model calls them by
# their real names via the native tool-calls channel. This prompt deliberately
# OMITS the XML/`call_tool` protocol from DEFAULT_SYSTEM_PROMPT — carrying that
# text would fight the native tooling and confuse a 4B model.
STUDENT_NATIVE_SYSTEM_PROMPT = """\
You are an agent operating in an MCP tool environment. Use the provided tools \
to accomplish the user's task. You are already logged in; your user id is 1 if \
required.

How to work:
- A maximum of 3 tools can be called in a single round and read its response before deciding the next step.
- You have a limited number of tool-calling turns. Avoid listing tools repeatedly. The task must be completed before the tool-calling budget runs out.
- `verify` and `done` are managed by the harness — never call them yourself.
- When you have gathered enough information and finished the required actions, \
stop calling tools and write your final answer as plain text."""


STUDENT_NATIVE_FINAL_SYSTEM_PROMPT = """\
You are an agent operating in an MCP tool environment. You are already logged \
in; your user id is 1 if required.

This is your FINAL turn: your tool-calling budget is exhausted, so you CANNOT \
call any more tools. Do NOT emit any tool call, function call, or JSON action. \
Do NOT write `<tool_call>`, `</tool_call>`, or any tool-call markup.

Based on everything you have already done and observed, write your FINAL ANSWER \
now as plain natural-language text only **whether the task is completed or not**. If the task asked you to perform \
actions, state concisely what you completed (and, if you could not finish, say \
so plainly)."""


ENVSCALER_STUDENT_NATIVE_SYSTEM_PROMPT = """\
You are an agent operating in a stateful tool environment. Use the provided \
tools to complete the user's task. Inspect the environment before making \
changes, obey all stated constraints, and do not invent tool results.

How to work:
- You may call up to 3 tools in one assistant turn. Read their responses before deciding the next step.
- You have a limited number of tool-calling turns, so prioritize the operations needed by the task.
- `verify` and `done` are managed by the harness — never call them yourself.
- When the task is complete, stop calling tools and give the user a concise final answer.
Current date: {current_date}."""


ENVSCALER_STUDENT_NATIVE_FINAL_SYSTEM_PROMPT = """\
You are an agent operating in a stateful tool environment.

This is your FINAL turn: your tool-calling budget is exhausted, so you CANNOT \
call any more tools. Do NOT emit any tool call, function call, JSON action, or \
tool-call markup.

Based on the task and the tool results already observed, state concisely what \
you completed. If the task is not fully complete, say so plainly."""


CODE_JUDGE_STUDENT_SYSTEM_PROMPT = """\
You are a competitive programmer. Solve the user's programming problem in \
Python. Reason carefully about the input format, constraints, edge cases, and \
algorithmic complexity.

Reason efficiently and stop analyzing once you have a sound algorithm and \
implementation plan. Keep internal reasoning under roughly 3,000 tokens and \
reserve at least 4,000 tokens for the final code. Never spend the full response \
budget exploring alternatives. Before the budget runs low, end your reasoning \
and submit the best complete runnable solution you have, even if you are not \
fully certain. Reasoning without a code submission is always invalid.

Your final response MUST contain the complete submission in the LAST fenced \
Python block:
```python
# complete solution
```
Only that last Python block is executed against hidden tests. Do not place \
tests, explanations, or additional code blocks after it."""


CODE_JUDGE_STUDENT_FINAL_SYSTEM_PROMPT = """\
This is your final answer for a competitive-programming problem. Return a \
complete Python solution in the LAST fenced Python block. Only that block is \
executed against hidden tests. Check input parsing, output formatting, edge \
cases, and complexity before answering.

Think efficiently: stop once you have a sound solution, keep internal reasoning \
under roughly 3,000 tokens, and reserve at least 4,000 tokens for the code. \
Never use the entire response budget for reasoning. Before the budget runs low, \
end your reasoning and output the best complete runnable solution you have, even \
if uncertain. A response with reasoning but no code is invalid. Do not put \
anything after the final code block."""


CRITIC_SYSTEM_PROMPT = """\
You are an expert curriculum designer for MCP tool-use agents.

You are given ONE scenario and N attempts at its tasks (PASS/FAIL + trajectory).
You also have LIVE read access to the same environment via `list_tools` and
`call_tool` — use it (BRIEFLY, at most 3 probes) to verify hypotheses about
tool signatures, valid argument shapes, and failure modes.

Turn protocol (STRICT — every assistant turn is EXACTLY one of these two):
  (A) A single `<tool_call>{...}</tool_call>` — nothing else after it.
  (B) The final answer: one or more lines of `# Experience: <one-line lesson>`
      — nothing else, no headers/bullets/prose.

You may put reasoning inside `<think>...</think>` in the same turn.
Never mix (A) and (B) in the same turn. Never emit an empty turn.

Tool-call format (STRICT — violations cause errors):
  - **Exactly ONE `<tool_call>` per turn.** Never emit multiple blocks.
  - **Always use the `call_tool` wrapper** for any scenario tool. The only
    top-level tool names are `list_tools` and `call_tool`. Do NOT put
    scenario tool names (like `create_booking`) directly in the top-level
    `name` field — that will always fail with "Unknown tool".
  - Valid examples:
<tool_call>{"name": "list_tools", "arguments": {}}</tool_call>
<tool_call>{"name": "call_tool", "arguments": {"tool_name": "<name>", "arguments": {...}}}</tool_call>

Never call `verify` or `done`. Prefer read-only queries.

Budget: at most 3 tool-call turns. After that your next turn MUST be (B).
Emitting (B) immediately (0 tool calls) is fine when the trajectories are
already conclusive.

Content rules (violations = fail):
1. NEVER copy the ground-truth answer, expected DB state, or verifier code.
2. NEVER include exact task-specific argument values (emails, ids, prices,
   dates, names). Teach argument SHAPES and workflow patterns, not values.
3. Every experience must be TRANSFERABLE across tasks in this scenario.

Aim for 3-8 experiences on separate lines."""


def _summarize_trajectory(trajectory: list, max_steps: int = 20,
                          max_chars_per_step: int = 800) -> str:
    """Render a rollout trajectory (list of step dicts) into a bounded
    plain-text form. Handles both 'full' and 'truncated' rollout modes."""
    if not trajectory:
        return "(empty trajectory)"

    steps = trajectory
    truncated_middle = False
    if len(steps) > max_steps:
        head = steps[: max_steps // 2]
        tail = steps[-(max_steps // 2):]
        steps = head + tail
        truncated_middle = True

    lines = []
    for idx, step in enumerate(steps):
        if truncated_middle and idx == len(steps) // 2:
            lines.append("... (middle steps elided) ...")
        if "assistant" in step:
            text = str(step["assistant"])[:max_chars_per_step]
            lines.append(f"[step {step.get('step','?')}] assistant: {text}")
        elif "assistant_len" in step:
            lines.append(
                f"[step {step.get('step','?')}] assistant: <{step['assistant_len']} chars, not recorded>"
            )
        if "tool" in step:
            args = step.get("arguments") or {}
            resp = step.get("response")
            lines.append(f"[step {step.get('step','?')}] tool={step['tool']} args={args}")
            if resp is not None:
                resp_text = str(resp)[:max_chars_per_step]
                lines.append(f"  response: {resp_text}")
    return "\n".join(lines)


def _render_one_attempt(record: dict) -> str:
    task_idx = record.get("task_idx", "?")
    reward_type = record.get("reward_type", "unknown")
    outcome = "PASS" if reward_type == "complete" else "FAIL"
    reward = record.get("reward", 0.0)
    steps = record.get("steps", 0)
    made_tool_call = record.get("made_tool_call", False)
    error = record.get("error")
    task = record.get("task", "")
    final_answer = (record.get("final_answer") or "")[:800]
    trajectory_text = _summarize_trajectory(record.get("trajectory") or [])

    header = (
        f"### task_idx={task_idx}  [{outcome}]  "
        f"reward={reward}  reward_type={reward_type}  steps={steps}  "
        f"tool_used={made_tool_call}"
    )
    parts = [header, "", "task:", task, "", "trajectory:", trajectory_text,
             "", "final_answer (truncated):", final_answer]
    if error:
        parts += ["", f"error: {error}"]
    return "\n".join(parts)


def build_critic_batch_input(scenario: str, records: list[dict],
                              tools_text: str) -> str:
    """Assemble the initial user message for a scenario-level CriticJob.

    Includes the scenario name, the current tool signatures (so the model
    can start reasoning without an extra list_tools round), then each of the
    N rollout attempts on this scenario. Verifier internals are excluded on
    purpose."""
    records = sorted(records, key=lambda r: r.get("task_idx", 0))
    pass_count = sum(1 for r in records if r.get("reward_type") == "complete")
    parts = [
        f"## Scenario: {scenario}",
        f"attempts: {len(records)}  pass: {pass_count}  fail: {len(records) - pass_count}",
        "",
        "## Tools available in this scenario",
        tools_text,
        "",
        "## Attempts",
    ]
    for r in records:
        parts.append(_render_one_attempt(r))
        parts.append("")
    parts.append(
        "You may now issue up to a few `<tool_call>...</tool_call>` calls to "
        "probe the environment. When ready, emit ONLY the final "
        "`# Experience: ...` lines."
    )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# TEACHER prompts (iterative refinement — used by rollout.refine).
# The teacher observes a student's rollout so far + the outcome verdict, and
# emits ONE line of `# Advice:` telling the student what to do next. It never
# tells the student the ground-truth answer.
# ---------------------------------------------------------------------------
TEACHER_ADVICE_SYSTEM_PROMPT = """\
You are an expert coach for an MCP tool-use student agent.

The student has just attempted a task and failed (`verifier_mode="code"`
returned a non-`complete` reward). You have LIVE read access to the same
environment: the scenario tools are provided to you and you can call them
directly by name. Its DB state has been replayed to match the student's
current situation, so what you see is what the student would see next. Use
this **probing phase** to diagnose what went wrong.

Turn protocol during this probing phase (STRICT):
  Use the native function-calling interface to issue read-only probes — call
  a tool directly by its name with the appropriate arguments. Do NOT write
  tool calls as text and do NOT wrap them in any `call_tool` envelope. Emit
  one probe at a time. The advice itself is produced in a SEPARATE later turn
  where you will be explicitly asked for it — do not write it now.

Rules:
  - Read-only queries only. Never call `verify` or `done`.
  - Choose probes that help distinguish workflow errors (wrong tool, missing
    prerequisite lookup, wrong filter, wrong argument shape).
  - Do NOT emit the advice yet. That happens in a separate finalize turn."""


TEACHER_FINALIZE_SYSTEM_PROMPT = """\
You are the same expert coach. The probing phase is now OVER.

You MUST NOT call any tools now. Based on everything you observed during
probing, write a concrete, actionable diagnosis for the student in plain prose.

Output format:
  Start with `# Advice:` followed by the diagnosis. The advice may span
  multiple lines if useful, but keep it short and actionable — a few
  sentences at most.

Content rules (violations = fail):
1. Do NOT quote the ground-truth answer, expected DB state, or any exact
   task-specific values (emails, ids, prices, dates, names, ...). Teach
   the WORKFLOW, not the value.
2. Give CONCRETE next-step guidance: which tool to use next, what argument
   SHAPE to try, what step is being skipped, what error pattern to watch.
3. Do NOT repeat what the student has already tried.

Probing phase:
"""


CODE_JUDGE_TEACHER_ADVICE_SYSTEM_PROMPT = """\
You are an expert competitive-programming coach. A student solution was run \
against hidden tests. You can see the problem, the attempted answer, and only \
aggregate verifier feedback; hidden inputs and expected outputs are unavailable.

Diagnose the most likely algorithm, implementation, parsing, formatting, or \
complexity issue. Produce a concise, transferable correction that helps the \
student make the next attempt substantially better. Do not write the complete \
solution and do not invent hidden-test details.

Output exactly one advice block beginning with `# Advice:`."""


CODE_JUDGE_TEACHER_FINALIZE_SYSTEM_PROMPT = """\
You are an expert competitive-programming coach. Based on the problem, attempted \
solution, and aggregate hidden-test score, emit a concise actionable diagnosis.

Start with `# Advice:`. Do not provide the complete solution, do not invent \
hidden tests, and do not claim a specific bug unless the visible code supports it."""


# LLM-as-judge for the refine pipeline. Taxonomy / priority order / JSON
# output contract are aligned with the AWM server-side judge
# (OpenEnv/envs/agent_world_model_env/server/verifier.py::run_llm_judge, which
# mirrors agent-world-model-rl-rewards/task_reward.py) so client- and
# server-judged scores stay comparable. The evidence differs: here the judge
# sees the deterministic CODE verifier's result instead of a SQL DB diff.
JUDGE_SYSTEM_PROMPT = """\
You are an impartial evaluator of automated agent task results with access \
to database verification. Based on the provided JSON trajectory AND the \
Python verification results from querying the database, decide the task \
outcome. This trajectory is generated by an MCP agent on a simulated \
simplified environment. The environment provides a set of MCP tools to help \
the agent complete the task.

Input:
    task_json: dict containing the user task and the agent's execution trace \
for one attempt round.
    verification_json: dict containing the execution results of the \
deterministic code verifier that inspected the database state.

Output:
    You only output UTF-8 encoded string, avoid any emoji or special \
characters. You only output English text.

Classification categories:
    - complete: all required steps and closure actions were successfully \
executed, AND the database state confirms the task was completed
    - incomplete: partial progress or the database state shows the task is \
not fully completed
    - server_error: the agent is blocked by MCP server/environment error, \
e.g., 5xx errors such as "Internal Server Error". Or the MCP server cannot \
process the valid tool call and return valid results. This can block the \
agent from completing the task.
    - agent_error: the agent made mistakes, invalid parameters, or missing \
required data without recovery, failed to complete the user's instruction.

Priority order for classification:
    1) complete (trajectory shows success AND database confirms it)
    2) server_error (due to the MCP server/environment error)
    3) agent_error (agent-side issue, e.g., invalid tool_call arguments, \
hallucination, agent mistakes)
    4) incomplete (everything else unfinished or database state doesn't \
match expected outcome)

Key considerations:
- The verification_json contains the execution results of the verification \
code run against the database. You can use them to help you judge the task \
completion.
- The verification results may be empty or error, or even the verification \
code itself may be inaccurate or too strict. You should not fully rely on \
the verification results. You need to comprehensively consider the \
trajectory information to help you judge the task completion.
- The final_answer is the agent's own claim of what it did; trust the \
trajectory and the database checks over it.

Output format (must be valid JSON, no markdown fences, no additional commentary):
    {
      "reasoning": "<concise explanation considering both trajectory and verification code execution results, the confidence score and considerations for each classification category>",
      "confidence_score": [0-100, 0-100, 0-100, 0-100] for complete, incomplete, server_error, agent_error respectively,
      "classification": "<one_of_[complete, incomplete, server_error, agent_error]>",
      "evidence": {
        "error_signals": ["<important error messages from the trajectory>"],
        "last_actions": ["<summaries of last few actions>"],
        "database_verification": "<summary of what the database state shows based on the code verifier execution results>"
      }
    }"""


def build_judge_input(task: str, round_messages: list[dict],
                      final_answer: str,
                      code_verify: dict) -> str:
    """Render the judge user message for ONE refine round.

    `round_messages` is the student conversation slice for the current round
    (assistant turns incl. tool_calls + tool responses). `code_verify` carries
    the deterministic verifier's reward_type and raw verify_result - evidence,
    not the verdict: the judge may override it in either direction.
    """
    traj_text = _render_student_conversation(round_messages,
                                             max_chars_per_msg=2000)
    try:
        verify_json = json.dumps(code_verify.get("verify_result"),
                                 ensure_ascii=False, default=str)
    except Exception:
        verify_json = str(code_verify.get("verify_result"))
    parts = [
        "task_json:",
        json.dumps({
            "user_task": task,
            "final_answer": final_answer,
            "trajectory": traj_text,
        }, ensure_ascii=False, indent=2),
        "",
        "verification_json:",
        json.dumps({
            "code_reward_type": code_verify.get("reward_type"),
            "code_execution_result": verify_json,
        }, ensure_ascii=False, indent=2),
    ]
    return "\n".join(parts)


def build_teacher_advice_input(task: str, student_conversation: list[dict],
                                verify_reward_type: str,
                                verify_error: str | None,
                                verify_summary: str | None = None,
                                allow_tool_probes: bool = True,
                                ) -> str:
    """Render the initial user message for the teacher-advisor turn.

    Deliberately excludes the full `verify_result` internals (only the coarse
    reward_type + error string are exposed) so the teacher cannot copy
    expected DB state into the advice."""
    convo_text = _render_student_conversation(student_conversation)
    parts = [
        "## Task",
        task,
        "",
        "## Student's attempt so far",
        convo_text,
        "",
        "## Verdict",
        f"reward_type = {verify_reward_type}  (non-complete → student failed this attempt)",
    ]
    if verify_error:
        parts.append(f"error: {verify_error}")
    if verify_summary:
        parts.append(f"verification_summary: {verify_summary}")
    parts.append("")
    if allow_tool_probes:
        parts.append(
            "You may now issue up to 3 read-only tool probes (call tools "
            "directly by name) to diagnose the situation. When ready, emit "
            "ONLY the final `# Advice: ...` line."
        )
    else:
        parts.append(
            "No diagnostic tools are available. Diagnose the visible attempt "
            "and aggregate verifier feedback, then emit ONLY the final "
            "`# Advice: ...` line."
        )
    return "\n".join(parts)


def _render_student_conversation(messages: list[dict],
                                  max_chars_per_msg: int = 1200) -> str:
    """Render the student's messages list in a compact plain-text form.

    Native function-calling messages are handled explicitly: an assistant
    turn may carry `tool_calls` (with the text content possibly empty/None),
    and tool results arrive as `role:"tool"` messages. Rendering these is
    essential — otherwise the teacher would see an empty trajectory and its
    diagnosis quality collapses."""
    lines: list[str] = []
    for i, m in enumerate(messages):
        role = m.get("role", "?")
        if role == "system":
            continue  # not useful for the teacher; occupies tokens

        content = str(m.get("content") or "")[:max_chars_per_msg]

        if role == "assistant":
            tool_calls = m.get("tool_calls") or []
            if content:
                lines.append(f"[{i:02d}] assistant: {content}")
            for tc in tool_calls:
                fn = tc.get("function") or {}
                name = fn.get("name", "")
                args = str(fn.get("arguments", ""))[:max_chars_per_msg]
                lines.append(f"[{i:02d}] assistant → tool_call {name}({args})")
            if not content and not tool_calls:
                lines.append(f"[{i:02d}] assistant: ")
        elif role == "tool":
            lines.append(f"[{i:02d}] tool_response: {content}")
        else:
            lines.append(f"[{i:02d}] {role}: {content}")
    return "\n".join(lines)
