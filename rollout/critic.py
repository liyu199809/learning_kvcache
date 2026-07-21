"""
Scenario-level tool-using critic.

For each scenario in the rollout JSONL:
  1. Open a fresh AWMEnv session, reset it (task_idx=0) to get the tool list.
  2. Feed seed-2.1-pro (chat.completions, XML tool-call format) all N rollout
     attempts on this scenario.
  3. Let it call `list_tools` / `call_tool` up to a small budget (default 8)
     to probe the environment.
  4. Read its final text output — expected to be one or more lines of
     `# Experience: <one-line lesson>`.
  5. Write one markdown per scenario at <skills-dir>/<scenario>.md whose
     entire body is just those `# Experience:` lines.

Run:
    export ARK_API_KEY=...
    cd /mnt/storage/disk3/self_evolver && python -m rollout.critic \
        --rollout-jsonl awm_smoke_full.jsonl \
        --output-jsonl critic_smoke.jsonl \
        --skills-dir skills_smoke/ \
        --concurrency 4
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, field

from agent_world_model_env import AWMEnv
from openenv.core.env_server.mcp_types import ListToolsAction

from rollout.common import (
    JsonlCheckpoint,
    RateLimiter,
    RetryLLM,
    execute_tool_call,
    format_tools,
    llm_turn,
    run_jobs,
)
from rollout.prompt import CRITIC_SYSTEM_PROMPT, build_critic_batch_input


EXPERIENCE_LINE_RE = re.compile(r"^\s*#\s*Experience:\s*(.+?)\s*$", re.MULTILINE)


# ---------------------------------------------------------------------------
# CriticJob — one scenario, up to N tool-use rounds, produces one skill md.
# ---------------------------------------------------------------------------
@dataclass
class CriticJob:
    scenario: str
    records: list[dict]
    ark: RetryLLM
    awm_base_url: str
    max_iterations: int = 4    # = max_tool_calls + 1 (each turn either a
                               # tool call or the final `# Experience:` block)
    max_tool_calls: int = 3
    temperature: float = 0.4
    max_tokens: int = 2048
    tool_response_cap: int = 4000
    task_idx: int = 0  # used only for env reset (fresh state)
    trace: list[dict] = field(default_factory=list)

    def key(self) -> tuple:
        return (self.scenario,)

    async def run(self) -> dict:
        t0 = time.monotonic()
        env = AWMEnv(base_url=self.awm_base_url,
                     message_timeout_s=180.0, connect_timeout_s=60.0)
        experiences: list[str] = []
        final_text: str = ""
        error: str | None = None
        rounds_used = 0
        tool_calls_made = 0

        try:
            await env.connect()
            await env.reset(scenario=self.scenario, task_idx=self.task_idx)
            list_res = await env.step(ListToolsAction())
            tools_text = format_tools(list_res.observation.tools)

            user_text = build_critic_batch_input(
                self.scenario, self.records, tools_text
            )
            messages: list[dict] = [
                {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ]

            for step in range(1, self.max_iterations + 1):
                rounds_used = step
                try:
                    turn = await llm_turn(
                        self.ark, messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    )
                except Exception as e:
                    error = f"ark chat failed at step {step}: {e!r}"[:500]
                    break

                content = turn.content
                tc = turn.tool_call

                self.trace.append({
                    "step": step,
                    "assistant": content[:1500],
                    "native_tool_call": tc if not content and tc else None,
                })
                if content:
                    final_text = content

                # Turn protocol: (B) no tool call → treat this turn as the
                # final answer and exit. (A) tool call → execute it and loop.
                if tc is None:
                    break

                tool_calls_made += 1
                if tool_calls_made > self.max_tool_calls:
                    # Over-budget tool call: refuse to execute and terminate.
                    # The model violated the protocol; downstream we'll flag
                    # this via `budget_violation=True`.
                    error = (
                        f"budget violation: model attempted tool call #{tool_calls_made} "
                        f"after {self.max_tool_calls} allowed"
                    )
                    break

                try:
                    tool_response = (await execute_tool_call(env, tc)).text
                except Exception as e:
                    tool_response = f"Error executing tool: {e!r}"
                tool_response = (tool_response or "")[: self.tool_response_cap]
                self.trace.append({
                    "step": step,
                    "tool": tc.get("name"),
                    "arguments": tc.get("arguments"),
                    "response": tool_response[:800],
                })
                messages.append(
                    {"role": "user", "content": f"Tool response:\n{tool_response}"}
                )

            experiences = EXPERIENCE_LINE_RE.findall(final_text or "")
        except Exception as e:
            error = f"critic run failed: {e!r}"[:500]
        finally:
            try:
                await env.close()
            except Exception:
                pass

        elapsed = round(time.monotonic() - t0, 3)
        return {
            "scenario": self.scenario,
            "num_attempts": len(self.records),
            "num_experiences": len(experiences),
            "experiences": experiences,
            "rounds_used": rounds_used,
            "tool_calls_made": tool_calls_made,
            "final_text": final_text[:4000],
            "elapsed_sec": elapsed,
            "error": error,
        }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def _read_jsonl(path: str) -> list[dict]:
    out: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def _write_skills_dir(critic_jsonl_path: str, skills_dir: str) -> int:
    """One markdown per scenario. File body is ONLY the '# Experience: ...'
    lines — no headers, no separators."""
    os.makedirs(skills_dir, exist_ok=True)
    records = _read_jsonl(critic_jsonl_path)
    for r in records:
        experiences: list[str] = r.get("experiences") or []
        scen = r.get("scenario", "unknown")
        if not experiences:
            body = "# Experience: (no experiences produced)"
        else:
            body = "\n".join(f"# Experience: {e}" for e in experiences)
        with open(os.path.join(skills_dir, f"{scen}.md"), "w", encoding="utf-8") as f:
            f.write(body + "\n")
    return len(records)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-jsonl",
                        default="/mnt/storage/disk3/self_evolver/awm_report.jsonl")
    parser.add_argument("--output-jsonl",
                        default="/mnt/storage/disk3/self_evolver/critic_report.jsonl")
    parser.add_argument("--skills-dir",
                        default="/mnt/storage/disk3/self_evolver/skills/")
    parser.add_argument("--awm-base-url",
                        default=os.environ.get("AWM_BASE_URL", "http://localhost:8899"))
    parser.add_argument("--ark-base-url",
                        default="https://ark.cn-beijing.volces.com/api/v3")
    parser.add_argument("--ark-api-key",
                        default=os.environ.get("ARK_API_KEY"))
    parser.add_argument("--ark-model", default="ep-20260707130305-26bjx")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--rpm", type=int, default=400,
                        help="Client-side RPM cap (headroom vs the account "
                             "limit; e.g. 400 for a 500 RPM budget).")
    parser.add_argument("--tpm", type=int, default=800_000,
                        help="Client-side TPM cap (headroom vs the account "
                             "limit; e.g. 800000 for a 1M TPM budget).")
    parser.add_argument("--llm-timeout", type=float, default=180.0)
    parser.add_argument("--max-iterations", type=int, default=4,
                        help="Max critic agent turns per scenario. Each turn "
                             "is either a tool call or the final "
                             "`# Experience:` output, so this equals "
                             "max_tool_calls + 1.")
    parser.add_argument("--max-tool-calls", type=int, default=3,
                        help="Cap on tool-exploration calls before the final "
                             "answer turn is required.")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-interval", type=float, default=10.0)
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional cap on number of scenarios to process.")
    args = parser.parse_args()

    if not args.ark_api_key:
        raise SystemExit("ARK_API_KEY not set. Export it or pass --ark-api-key.")
    if not os.path.exists(args.rollout_jsonl):
        raise SystemExit(f"rollout JSONL not found: {args.rollout_jsonl}")

    ark = RetryLLM(
        base_url=args.ark_base_url,
        api_key=args.ark_api_key,
        model=args.ark_model,
        timeout=args.llm_timeout,
        limiter=RateLimiter(rpm=args.rpm, tpm=args.tpm),
    )
    print(f"[critic] rate limit: {args.rpm} RPM, {args.tpm} TPM (client-side)")

    records = _read_jsonl(args.rollout_jsonl)
    by_scenario: dict[str, list[dict]] = {}
    for r in records:
        by_scenario.setdefault(r["scenario"], []).append(r)

    scenarios = list(by_scenario.keys())
    if args.limit is not None:
        scenarios = scenarios[: args.limit]
    print(f"[critic] {len(scenarios)} scenarios, "
          f"{sum(len(by_scenario[s]) for s in scenarios)} rollout records total")

    jobs = [
        CriticJob(
            scenario=s,
            records=by_scenario[s],
            ark=ark,
            awm_base_url=args.awm_base_url,
            max_iterations=args.max_iterations,
            max_tool_calls=args.max_tool_calls,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        for s in scenarios
    ]

    ckpt = JsonlCheckpoint(args.output_jsonl, key_fields=("scenario",))

    t0 = time.monotonic()
    await run_jobs(jobs, args.concurrency, ckpt,
                   progress_interval=args.progress_interval, resume=args.resume)
    elapsed = time.monotonic() - t0

    n_scenarios = _write_skills_dir(args.output_jsonl, args.skills_dir)

    out_records = ckpt.read_all()
    total = len(out_records)
    with_exp = sum(1 for r in out_records if r.get("experiences"))
    exp_total = sum(len(r.get("experiences") or []) for r in out_records)
    err = sum(1 for r in out_records if r.get("error"))
    print("=" * 80)
    print("AWM Critic Report")
    print("=" * 80)
    print(f"Ark model:            {args.ark_model}")
    print(f"Scenarios processed:  {total}")
    print(f"With experiences:     {with_exp}")
    print(f"Total experiences:    {exp_total} (avg {exp_total/max(total,1):.1f} per scenario)")
    print(f"With error:           {err}")
    print(f"Markdown files:       {n_scenarios} -> {args.skills_dir}")
    print(f"Critic JSONL:         {args.output_jsonl}")
    print(f"Elapsed:              {elapsed:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
