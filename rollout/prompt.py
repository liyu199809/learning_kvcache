"""
Prompts and input builders for rollout and critic.

- ROLLOUT_SYSTEM_PROMPT: reused from AWM's official prompt.
- CRITIC_SYSTEM_PROMPT: for the tool-using critic agent; it can call
  `list_tools` / `call_tool` on the live AWM environment for the scenario,
  then emits multiple `# Experience: <one-line lesson>` rows.
- build_critic_batch_input(): renders one scenario's N rollout records
  (task, outcome, trajectory) into the initial user message for the critic.
"""

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
- `verify` and `done` are managed by the harness — never call them yourself.
- When you have gathered enough information and finished the required actions, \
stop calling tools and write your final answer as plain text."""


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


def build_teacher_advice_input(task: str, student_conversation: list[dict],
                                verify_reward_type: str,
                                verify_error: str | None,
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
    parts += [
        "",
        "You may now issue up to 3 read-only tool probes (call tools directly "
        "by name) to diagnose the situation. When ready, emit ONLY the final "
        "`# Advice: ...` line.",
    ]
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
