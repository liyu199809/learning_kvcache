"""
Resample student rollouts from the point just BEFORE teacher advice.

For each record in a refine JSONL, we:

  1. Locate the first `[Expert advice]` user turn in `student_conversation`
     and cut the conversation there — everything from that turn onward
     (advice + all later rounds) is discarded. The surviving prefix is the
     student's own round-1 interaction: system prompt, task, and the
     assistant/tool exchange up to the point where advice would have been
     injected.

  2. Open a FRESH AWMEnv (reset → clean DB). The environment starts from a
     pristine state — matching refine's per-round reset behaviour. The
     conversation prefix is kept as context, but the DB is NOT replayed.

  3. Feed the cut prefix back to the student (native function-calling, tool
     schemas re-registered from the live env) and let it continue for up to
     `student_max_iterations` turns — WITHOUT any privileged advice.

The output JSONL mirrors the shape of refine's records so downstream OPD
training can consume it directly: `student_conversation` holds the full
resampled conversation, plus per-job verify / error fields.

Run:
    cd /mnt/storage/disk3/self_evolver && python -m rollout.resample \
        --refine-jsonl traj_data/refine_200.jsonl \
        --output-jsonl traj_data/resample_epoch1.jsonl \
        --concurrency 16
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import time
from dataclasses import dataclass, field, asdict

from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction

from agent_world_model_env import AWMEnv

from rollout.common import (
    JsonlCheckpoint,
    RetryLLM,
    format_tools,
    run_jobs,
    tools_to_openai_schema,
)
from rollout.refine import _student_step, _verify_only
from rollout.prompt import STUDENT_NATIVE_SYSTEM_PROMPT


ADVICE_MARKER = "[Expert advice]"

# Appended after the cut prefix so the student knows the DB it will query is
# pristine — earlier tool responses in the prefix reflect a prior episode and
# their concrete values (ids, prices, timestamps) may no longer exist.
ENV_RESET_NOTE = (
    "[Environment reset]\n"
    "The environment has been reset to its initial state. Tool responses in "
    "the conversation above were obtained in a previous episode; any concrete "
    "values they returned (ids, prices, timestamps, record contents) may no "
    "longer exist or may differ now. Re-query the environment to get fresh "
    "values before acting."
)


# ---------------------------------------------------------------------------
# Refine record parsing
# ---------------------------------------------------------------------------
def _find_advice_idx(messages: list[dict]) -> int | None:
    """Return the index of the first `[Expert advice]` user turn, or None."""
    for i, m in enumerate(messages):
        if m.get("role") == "user" and ADVICE_MARKER in str(m.get("content", "")):
            return i
    return None


def _cut_prefix(messages: list[dict]) -> list[dict]:
    """Return the conversation prefix before the first advice turn, plus a
    trailing user note telling the student the environment has been reset.

    Deep-copied so callers can mutate without touching the source record."""
    idx = _find_advice_idx(messages)
    cut = messages if idx is None else messages[:idx]
    prefix = copy.deepcopy(cut)
    prefix.append({"role": "user", "content": ENV_RESET_NOTE})
    return prefix


# ---------------------------------------------------------------------------
# ResampleJob — one (scenario, task_idx, group_idx) sample
# ---------------------------------------------------------------------------
@dataclass
class ResampleJob:
    scenario: str
    task_idx: int
    group_idx: int
    group_size: int
    prefix_messages: list[dict]
    task_description: str
    llm: RetryLLM
    awm_base_url: str
    student_max_iterations: int = 4
    student_max_tokens: int = 2048
    student_temperature: float = 1.0
    episode_timeout: float = 900.0

    def key(self) -> tuple:
        return (self.scenario, self.task_idx, self.group_idx)

    async def run(self) -> dict:
        t0 = time.monotonic()
        env = AWMEnv(
            base_url=self.awm_base_url,
            message_timeout_s=self.episode_timeout,
            connect_timeout_s=60.0,
        )
        error: str | None = None
        step_result: dict = {}
        verify: dict = {}
        final_messages: list[dict] = list(self.prefix_messages)

        try:
            await env.connect()
            reset_res = await env.reset(scenario=self.scenario,
                                        task_idx=self.task_idx)
            if reset_res.observation.reward_type == "reset_error":
                error = f"reset_error: {reset_res.observation.error}"
            else:
                list_res = await env.step(ListToolsAction())
                tools_schema = tools_to_openai_schema(list_res.observation.tools)

                # Continue the cut conversation on a FRESH env state (mirrors
                # refine's per-round reset). Conversation prefix is kept as
                # context; the DB is NOT replayed.
                messages = copy.deepcopy(self.prefix_messages)
                step_result = await asyncio.wait_for(
                    _student_step(
                        env, self.llm, messages, tools_schema,
                        max_iterations=self.student_max_iterations,
                        max_tokens=self.student_max_tokens,
                        temperature=self.student_temperature,
                    ),
                    timeout=self.episode_timeout,
                )
                final_messages = step_result["messages"]
                verify = await _verify_only(env, step_result["final_answer"])
                if step_result.get("error"):
                    error = step_result["error"]

        except asyncio.TimeoutError:
            error = f"episode timeout > {self.episode_timeout}s"
        except Exception as e:
            import traceback
            error = f"resample run failed: {e!r} | tb: {traceback.format_exc()}"[:2000]
        finally:
            try:
                await env.close()
            except Exception:
                pass

        elapsed = round(time.monotonic() - t0, 3)
        return {
            "scenario": self.scenario,
            "task_idx": self.task_idx,
            "group_idx": self.group_idx,
            "group_size": self.group_size,
            "task": self.task_description,
            "num_prefix_messages": len(self.prefix_messages),
            "student_conversation": final_messages,
            "student_final_answer": step_result.get("final_answer", ""),
            "student_steps": step_result.get("steps", 0),
            "student_made_tool_call": step_result.get("made_tool_call", False),
            "student_new_tool_calls": step_result.get("executed_tool_calls", []),
            "verify_reward": verify.get("reward"),
            "verify_reward_type": verify.get("reward_type"),
            "verify_result": verify.get("verify_result"),
            "success": verify.get("reward_type") == "complete",
            "elapsed_sec": elapsed,
            "error": error,
        }


# ---------------------------------------------------------------------------
# Load jobs from a refine JSONL
# ---------------------------------------------------------------------------
def _load_jobs_from_refine(path: str, llm: RetryLLM, awm_base_url: str,
                            args: argparse.Namespace) -> list[ResampleJob]:
    jobs: list[ResampleJob] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            conv = rec.get("student_conversation") or []
            if not conv:
                continue
            prefix = _cut_prefix(conv)
            # Skip records whose round-1 prefix is trivially short (system+task
            # only, no interaction) — nothing meaningful to resample from.
            if len(prefix) < 3:
                continue
            # Keep only HARD cases that ultimately succeeded: the student
            # failed round 1 (which is why advice was injected at all) AND the
            # refine episode ended in success. These are the cases where the
            # teacher advice demonstrably worked, so the pre-advice prefix is
            # a valuable on-policy state for OPD.
            has_advice = _find_advice_idx(conv) is not None
            if not (has_advice and rec.get("success")):
                continue
            for g in range(args.group_size):
                jobs.append(ResampleJob(
                    scenario=rec["scenario"],
                    task_idx=rec["task_idx"],
                    group_idx=g,
                    group_size=args.group_size,
                    prefix_messages=prefix,
                    task_description=rec.get("task", ""),
                    llm=llm,
                    awm_base_url=awm_base_url,
                    student_max_iterations=args.student_max_iterations,
                    student_max_tokens=args.student_max_tokens,
                    student_temperature=args.student_temperature,
                    episode_timeout=args.episode_timeout,
                ))
    return jobs


# ---------------------------------------------------------------------------
# Aggregation + report
# ---------------------------------------------------------------------------
def _aggregate_and_print(ckpt: JsonlCheckpoint, args: argparse.Namespace,
                          elapsed_wall_sec: float) -> None:
    records = ckpt.read_all()
    n = len(records)
    n_success = sum(1 for r in records if r.get("success"))
    type_dist: dict = {}
    for r in records:
        rt = r.get("verify_reward_type", "?")
        type_dist[rt] = type_dist.get(rt, 0) + 1

    # Per-group (scenario, task_idx) success distribution: how many of the
    # group_size samples succeeded. Useful for GRPO-style filtering (drop
    # all-pass / all-fail groups).
    from collections import defaultdict
    groups: dict = defaultdict(list)
    for r in records:
        groups[(r["scenario"], r["task_idx"])].append(bool(r.get("success")))
    group_pass_counts: dict = {}
    for key, succs in groups.items():
        k = sum(succs)
        group_pass_counts[k] = group_pass_counts.get(k, 0) + 1
    n_groups = len(groups)
    n_all_pass = sum(1 for succs in groups.values() if all(succs))
    n_all_fail = sum(1 for succs in groups.values() if not any(succs))
    n_mixed = n_groups - n_all_pass - n_all_fail

    report = {
        "config": vars(args),
        "num_records": n,
        "num_success": n_success,
        "success_rate": n_success / max(n, 1),
        "num_groups": n_groups,
        "group_all_pass": n_all_pass,
        "group_all_fail": n_all_fail,
        "group_mixed": n_mixed,
        "group_pass_count_distribution": group_pass_counts,
        "verify_reward_type_distribution": type_dist,
        "elapsed_wall_sec": round(elapsed_wall_sec, 2),
    }
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print("AWM Resample Report (pre-advice continuation, no privileged info)")
    print("=" * 80)
    print(f"Records:               {n}  ({n_groups} groups × group_size={args.group_size})")
    print(f"Success:               {n_success}  ({report['success_rate']:.2%})")
    print(f"Group all-pass:        {n_all_pass}")
    print(f"Group all-fail:        {n_all_fail}")
    print(f"Group mixed:           {n_mixed}  (GRPO-useful)")
    print(f"Group pass-count dist: {dict(sorted(group_pass_counts.items()))}")
    print(f"Reward types:          {type_dist}")
    print(f"Elapsed:               {elapsed_wall_sec:.1f}s")
    print(f"JSONL:                 {ckpt.path}")
    print(f"Report:                {args.report}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refine-jsonl", required=True,
                        help="Path to a rollout.refine output JSONL whose "
                             "student_conversation / rounds_detail are used to "
                             "reconstruct the pre-advice state.")
    parser.add_argument("--output-jsonl",
                        default="/mnt/storage/disk3/self_evolver/traj_data/resample_epoch1.jsonl")
    parser.add_argument("--report",
                        default="/mnt/storage/disk3/self_evolver/traj_data/resample_epoch1.json")
    parser.add_argument("--student-max-iterations", type=int, default=4,
                        help="Max LLM turns for the continuation (matches refine).")
    parser.add_argument("--student-max-tokens", type=int, default=2048)
    parser.add_argument("--student-temperature", type=float, default=1.0)
    parser.add_argument("--group-size", type=int, default=1,
                        help="Number of independent samples per (scenario, task_idx). "
                             "Each sample runs on its own fresh env. group_size>1 produces "
                             "GRPO-style groups for downstream OPD-RL training.")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--episode-timeout", type=float, default=900.0)
    parser.add_argument("--llm-timeout", type=float, default=180.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress-interval", type=float, default=10.0)
    parser.add_argument("--awm-base-url",
                        default=os.environ.get("AWM_BASE_URL", "http://localhost:8899"))
    parser.add_argument("--student-base-url",
                        default=os.environ.get("ENDPOINT_URL", "http://localhost:8000/v1"))
    parser.add_argument("--student-api-key",
                        default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--student-model",
                        default=os.environ.get("AWM_EXAMPLE_AGENT_MODEL", "qwen3.5-4b"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    llm = RetryLLM(base_url=args.student_base_url,
                   api_key=args.student_api_key,
                   model=args.student_model,
                   timeout=args.llm_timeout)
    print(f"[resample] student={args.student_model}")

    jobs = _load_jobs_from_refine(args.refine_jsonl, llm, args.awm_base_url, args)
    if args.limit is not None:
        jobs = jobs[: args.limit]
    n_pairs = len({(j.scenario, j.task_idx) for j in jobs})
    print(f"[resample] {n_pairs} (scenario, task) pairs × group_size={args.group_size} "
          f"= {len(jobs)} samples to resample, concurrency={args.concurrency}")

    ckpt = JsonlCheckpoint(args.output_jsonl,
                            key_fields=("scenario", "task_idx", "group_idx"))

    t0 = time.monotonic()
    await run_jobs(jobs, args.concurrency, ckpt,
                   progress_interval=args.progress_interval, resume=args.resume)
    elapsed = time.monotonic() - t0

    _aggregate_and_print(ckpt, args, elapsed)


if __name__ == "__main__":
    asyncio.run(main())
