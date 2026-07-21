"""
AWM concurrent rollout (Qwen3.5-4B via vLLM) — migrated to use rollout.common.

  - Each episode is a RolloutJob(scenario, task_idx).
  - Uses shared RetryLLM + JsonlCheckpoint + run_jobs from rollout.common.
  - Default trajectory-mode is 'full' so the critic pipeline has enough context.

Run:
    cd /mnt/storage/disk3/self_evolver && python -m rollout.rollout \
        --num-scenarios 5 --tasks-per-scenario 2 \
        --concurrency 4 --reset-concurrency 2 \
        --checkpoint-path awm_smoke_full.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from dataclasses import dataclass, field, asdict

from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction

from agent_world_model_env import AWMEnv

from rollout.common import (
    JsonlCheckpoint,
    RetryLLM,
    execute_tool_call,
    format_tools,
    llm_turn,
    run_jobs,
)
from rollout.prompt import ROLLOUT_SYSTEM_PROMPT, build_rollout_system_prompt


# ---------------------------------------------------------------------------
# EpisodeResult — same schema as awm_smoke_eval.py so downstream tooling
# (critic, aggregators) keeps working.
# ---------------------------------------------------------------------------
@dataclass
class EpisodeResult:
    scenario: str
    task_idx: int
    task: str = ""
    steps: int = 0
    final_answer: str = ""
    reward: float = 0.0
    reward_type: str = ""
    verify_result: dict | None = None
    error: str | None = None
    trajectory: list = field(default_factory=list)
    trajectory_path: str | None = None
    step_reward_types: list = field(default_factory=list)
    made_tool_call: bool = False
    episode_sec: float = 0.0
    skill_used: bool = False


# ---------------------------------------------------------------------------
# Trajectory recording (mode-aware) — 'full' includes tool responses, which
# is what the critic pipeline needs.
# ---------------------------------------------------------------------------
def _record_assistant(traj: list, step: int, content: str, mode: str) -> None:
    if mode == "off":
        return
    if mode == "truncated":
        traj.append({"step": step, "assistant_len": len(content)})
    else:
        traj.append({"step": step, "assistant": content[:2000]})


def _record_tool(traj: list, step: int, name: str, arguments: dict,
                 response: str | None, mode: str) -> None:
    if mode == "off":
        return
    if mode == "truncated":
        traj.append({"step": step, "tool": name, "arguments": arguments})
    else:
        traj.append({"step": step, "tool": name, "arguments": arguments,
                     "response": (response or "")[:2000]})


# ---------------------------------------------------------------------------
# Agent loop (per-episode). Assumes env is connected + reset done already.
# ---------------------------------------------------------------------------
async def _agent_loop(env, llm: RetryLLM, task_description: str,
                       tools_text: str, max_iterations: int, max_tokens: int,
                       temperature: float, trajectory_mode: str,
                       system_prompt: str = ROLLOUT_SYSTEM_PROMPT) -> dict:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task_description},
        {"role": "user", "content": f"Available tools:\n{tools_text}"},
    ]
    traj: list = []
    content = ""
    error: str | None = None
    step_used = 0
    step_reward_types: list = []
    made_tool_call = False

    for step in range(1, max_iterations + 1):
        step_used = step
        try:
            turn = await llm_turn(llm, messages, temperature=temperature,
                                  max_tokens=max_tokens)
        except Exception as e:
            error = f"LLM error at step {step}: {e}"
            break

        content = turn.content
        _record_assistant(traj, step, content, trajectory_mode)

        tc = turn.tool_call
        if not tc:
            break

        name = tc["name"]
        arguments = tc.get("arguments") or {}
        made_tool_call = True
        try:
            tool_response, r = await execute_tool_call(env, tc)
        except Exception as e:
            error = f"Env step error at step {step}: {e}"
            break

        if r is not None:
            step_reward_types.append(getattr(r.observation, "reward_type", "unknown"))

        tool_response = tool_response[:6000]
        _record_tool(traj, step, name, arguments, tool_response, trajectory_mode)
        messages.append({"role": "user", "content": f"Tool response:\n{tool_response}"})

    verify = await env.step(
        CallToolAction(
            tool_name="verify",
            arguments={"verifier_mode": "code", "final_answer": content},
        )
    )
    reward = verify.reward
    reward_type = verify.observation.reward_type
    verify_result = verify.observation.verify_result

    done_res = await env.step(CallToolAction(tool_name="done", arguments={"keep_session": True}))
    trajectory_path = getattr(done_res.observation, "trajectory_path", None)

    return {
        "task": task_description,
        "steps": step_used,
        "final_answer": content[:2000],
        "reward": reward,
        "reward_type": reward_type,
        "verify_result": verify_result,
        "error": error,
        "trajectory": traj,
        "trajectory_path": trajectory_path,
        "step_reward_types": step_reward_types,
        "made_tool_call": made_tool_call,
    }


# ---------------------------------------------------------------------------
# Skill loading (per-scenario markdown of `# Experience:` lines).
# ---------------------------------------------------------------------------
_SKILL_CACHE: dict[tuple[str, str], str | None] = {}


def _load_scenario_skill(skills_dir: str | None, scenario: str) -> str | None:
    """Read <skills_dir>/<scenario>.md if it exists; cached per (dir, scenario)."""
    if not skills_dir:
        return None
    cache_key = (skills_dir, scenario)
    if cache_key in _SKILL_CACHE:
        return _SKILL_CACHE[cache_key]
    path = os.path.join(skills_dir, f"{scenario}.md")
    if not os.path.exists(path):
        _SKILL_CACHE[cache_key] = None
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError:
        text = None
    _SKILL_CACHE[cache_key] = text
    return text


# ---------------------------------------------------------------------------
# RolloutJob — one episode; implements the Job protocol.
# ---------------------------------------------------------------------------
@dataclass
class RolloutJob:
    scenario: str
    task_idx: int
    args: argparse.Namespace
    llm: RetryLLM
    reset_sem: asyncio.Semaphore

    def key(self) -> tuple:
        return (self.scenario, self.task_idx)

    async def run(self) -> dict:
        args = self.args
        t0 = time.monotonic()
        env = AWMEnv(
            base_url=args.awm_base_url,
            message_timeout_s=args.episode_timeout,
            connect_timeout_s=60.0,
        )
        ep = EpisodeResult(scenario=self.scenario, task_idx=self.task_idx)
        skill_md = _load_scenario_skill(args.skills_dir, self.scenario)
        system_prompt = build_rollout_system_prompt(skill_md)
        ep.skill_used = bool(skill_md)
        try:
            async with self.reset_sem:
                await env.connect()
                reset_res = await env.reset(scenario=self.scenario, task_idx=self.task_idx)
                task_description = reset_res.observation.task
                list_res = await env.step(ListToolsAction())
                tools_text = format_tools(list_res.observation.tools)

            result = await asyncio.wait_for(
                _agent_loop(
                    env, self.llm,
                    task_description, tools_text,
                    args.max_iterations, args.max_tokens, args.temperature,
                    args.trajectory_mode,
                    system_prompt=system_prompt,
                ),
                timeout=args.episode_timeout,
            )
            for k, v in result.items():
                setattr(ep, k, v)
        except asyncio.TimeoutError:
            ep.reward_type = "timeout"
            ep.error = f"episode timeout > {args.episode_timeout}s"
        except Exception as e:
            ep.reward_type = "client_error"
            ep.error = repr(e)[:500]
        finally:
            try:
                await env.close()
            except Exception:
                pass
            ep.episode_sec = round(time.monotonic() - t0, 3)

        return asdict(ep)


# ---------------------------------------------------------------------------
# Aggregation + report
# ---------------------------------------------------------------------------
def _aggregate_and_print(ckpt: JsonlCheckpoint, args: argparse.Namespace,
                          elapsed_wall_sec: float, report_path: str) -> None:
    episodes = ckpt.read_all()
    n = len(episodes)
    reward_sum = sum(e.get("reward", 0.0) for e in episodes)
    pass_count = sum(1 for e in episodes if e.get("reward_type") == "complete")
    early_exit = sum(1 for e in episodes if not e.get("made_tool_call", False))
    sum_episode_sec = sum(e.get("episode_sec", 0.0) for e in episodes)

    type_dist: dict = {}
    for e in episodes:
        type_dist[e.get("reward_type", "?")] = type_dist.get(e.get("reward_type", "?"), 0) + 1

    step_type_dist: dict = {}
    for e in episodes:
        for t in e.get("step_reward_types", []):
            step_type_dist[t] = step_type_dist.get(t, 0) + 1

    per_scenario: dict = {}
    for e in episodes:
        d = per_scenario.setdefault(e["scenario"], {"n": 0, "reward_sum": 0.0, "pass": 0})
        d["n"] += 1
        d["reward_sum"] += e.get("reward", 0.0)
        d["pass"] += int(e.get("reward_type") == "complete")

    with_skill = [e for e in episodes if e.get("skill_used")]
    without_skill = [e for e in episodes if not e.get("skill_used")]

    def _rate(eps):
        if not eps:
            return None
        return {
            "n": len(eps),
            "pass_rate": sum(1 for e in eps if e.get("reward_type") == "complete") / len(eps),
            "avg_reward": sum(e.get("reward", 0.0) for e in eps) / len(eps),
        }

    report = {
        "config": vars(args),
        "num_episodes": n,
        "elapsed_wall_sec": round(elapsed_wall_sec, 2),
        "sum_episode_sec": round(sum_episode_sec, 2),
        "concurrency_speedup": round(sum_episode_sec / max(elapsed_wall_sec, 1e-6), 2),
        "pass_rate": pass_count / max(n, 1),
        "avg_reward": reward_sum / max(n, 1),
        "early_exit_count": early_exit,
        "verifier_reward_type_distribution": type_dist,
        "step_reward_type_distribution": step_type_dist,
        "with_skill": _rate(with_skill),
        "without_skill": _rate(without_skill),
        "per_scenario": per_scenario,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print("AWM Rollout Report")
    print("=" * 80)
    print(f"Model:                    {args.llm_model}")
    print(f"Episodes:                 {n}")
    print(f"Elapsed (wall):           {report['elapsed_wall_sec']:.1f}s")
    print(f"Sum episode sec:          {report['sum_episode_sec']:.1f}s")
    print(f"Concurrency speedup:      {report['concurrency_speedup']:.1f}x")
    print(f"Pass@1 (complete):        {report['pass_rate']:.2%}")
    print(f"Avg reward:               {report['avg_reward']:.4f}")
    print(f"Early-exit (no tool):     {early_exit}/{n}")
    print(f"Reward types:             {type_dist}")
    print(f"Step reward types:        {step_type_dist}")
    if report["with_skill"] and report["without_skill"]:
        ws = report["with_skill"]; ns = report["without_skill"]
        print(f"With    skill (n={ws['n']}):   pass={ws['pass_rate']:.2%}  avg_r={ws['avg_reward']:.4f}")
        print(f"Without skill (n={ns['n']}):   pass={ns['pass_rate']:.2%}  avg_r={ns['avg_reward']:.4f}")
    elif report["with_skill"]:
        ws = report["with_skill"]
        print(f"With skill only (n={ws['n']}): pass={ws['pass_rate']:.2%}  avg_r={ws['avg_reward']:.4f}")
    print(f"Checkpoint:               {ckpt.path}")
    print(f"Report:                   {report_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-scenarios", type=int, default=1000)
    parser.add_argument("--tasks-per-scenario", type=int, default=10)
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--concurrency", type=int, default=128)
    parser.add_argument("--reset-concurrency", type=int, default=32)
    parser.add_argument("--episode-timeout", type=float, default=600.0)
    parser.add_argument("--llm-timeout", type=float, default=120.0)
    parser.add_argument("--awm-base-url", default=os.environ.get("AWM_BASE_URL", "http://localhost:8899"))
    parser.add_argument("--llm-base-url", default=os.environ.get("ENDPOINT_URL", "http://localhost:8000/v1"))
    parser.add_argument("--llm-api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--llm-model", default=os.environ.get("AWM_EXAMPLE_AGENT_MODEL", "qwen3.5-4b"))
    parser.add_argument("--scenarios", default=None,
                        help="Comma-separated scenario names; overrides --num-scenarios.")
    parser.add_argument("--start-scenario-idx", type=int, default=0)
    parser.add_argument("--end-scenario-idx", type=int, default=None,
                        help="Exclusive end index into the scenario list.")
    parser.add_argument("--checkpoint-path", default="/mnt/storage/disk3/self_evolver/awm_report.jsonl")
    parser.add_argument("--report", default="/mnt/storage/disk3/self_evolver/awm_report.json")
    parser.add_argument("--skills-dir", default=None,
                        help="Optional directory containing <scenario>.md files "
                             "produced by rollout.critic. When set, the rollout "
                             "agent's system prompt is augmented with the "
                             "matching scenario's `# Experience:` lines.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--trajectory-mode", choices=["full", "truncated", "off"],
                        default="full",
                        help="Trajectory verbosity. Default 'full' keeps tool responses "
                             "so the critic pipeline has enough context.")
    parser.add_argument("--progress-interval", type=float, default=10.0)
    args = parser.parse_args()

    llm = RetryLLM(base_url=args.llm_base_url, api_key=args.llm_api_key,
                   model=args.llm_model, timeout=args.llm_timeout)

    # Discover scenario list from AWM
    async with AWMEnv(base_url=args.awm_base_url) as env:
        list_res = await env.step(CallToolAction(tool_name="__list_scenarios__", arguments={}))
        all_scenarios = list_res.observation.scenarios
    print(f"Total scenarios on server: {len(all_scenarios)}")

    if args.scenarios:
        picked = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    else:
        end = args.end_scenario_idx if args.end_scenario_idx is not None \
              else args.start_scenario_idx + args.num_scenarios
        picked = [s["name"] for s in all_scenarios[args.start_scenario_idx:end]]
    print(f"Selected {len(picked)} scenarios (first 5: {picked[:5]})")

    if args.skills_dir:
        hits = sum(1 for s in picked
                   if os.path.exists(os.path.join(args.skills_dir, f"{s}.md")))
        print(f"[skill] using skills_dir={args.skills_dir}: "
              f"{hits}/{len(picked)} scenarios have a skill md")
    else:
        print("[skill] no skills_dir (vanilla rollout, no experience injection)")

    todo = [(s, t) for s in picked for t in range(args.tasks_per_scenario)]

    reset_sem = asyncio.Semaphore(args.reset_concurrency)
    jobs = [RolloutJob(s, t, args, llm, reset_sem) for (s, t) in todo]

    ckpt = JsonlCheckpoint(args.checkpoint_path)

    print(f"Starting {len(jobs)} episodes: concurrency={args.concurrency}, "
          f"reset_concurrency={args.reset_concurrency}, "
          f"episode_timeout={args.episode_timeout}s, llm_timeout={args.llm_timeout}s, "
          f"trajectory_mode={args.trajectory_mode}")

    t0 = time.monotonic()
    await run_jobs(jobs, args.concurrency, ckpt,
                   progress_interval=args.progress_interval, resume=args.resume)
    elapsed = time.monotonic() - t0

    _aggregate_and_print(ckpt, args, elapsed, args.report)


if __name__ == "__main__":
    asyncio.run(main())
