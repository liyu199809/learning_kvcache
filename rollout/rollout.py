"""
AWM rollout with advice-augmented system prompt.

Pipeline:
  1. Run rollout.refine on the target (scenario, task_idx) pairs to harvest
     teacher advice lines into a refine JSONL.
  2. For each pair, take the advice from refine's `rounds_detail` and inject
     it into the student's system prompt (advice mode) or leave the prompt
     untouched (vanilla mode).
  3. Run exactly ONE round of the refine student loop — fresh env, native
     function-calling, per-round `verify` — and report pass@1.

The agent loop, prompts, tool handling and verify call are imported from
rollout.refine so the only difference between vanilla and advice modes is
the system prompt the student starts with.

Run:
    cd /mnt/storage/disk3/self_evolver && python -m rollout.rollout \
        --mode advice --refine-jsonl traj_data/refine_epoch1.jsonl \
        --num-scenarios 5 --tasks-per-scenario 2 --concurrency 16
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
    RateLimiter,
    RetryLLM,
    format_tools,
    tools_to_openai_schema,
)
from rollout.refine import (
    _student_step,
    _teacher_advise,
    _verify_only,
    run_jobs,
)
from rollout.prompt import STUDENT_NATIVE_SYSTEM_PROMPT


_ADVICE_PROMPT_SUFFIX = """\

## Expert advice for this exact task
A stronger model attempted this task before you. Its advice (quoted below) \
points out the likely pitfall and the workflow that works. Read it first, \
then attempt the task. You may ignore parts of it if your own tool results \
contradict it.

# Advice: {advice}"""


def build_student_prompt(advice: str | None) -> str:
    """Student system prompt, optionally carrying the harvested advice."""
    if not advice:
        return STUDENT_NATIVE_SYSTEM_PROMPT
    return STUDENT_NATIVE_SYSTEM_PROMPT + _ADVICE_PROMPT_SUFFIX.format(
        advice=advice.strip())


# ---------------------------------------------------------------------------
# EpisodeResult
# ---------------------------------------------------------------------------
@dataclass
class EpisodeResult:
    scenario: str
    task_idx: int
    mode: str = "vanilla"
    task: str = ""
    steps: int = 0
    final_answer: str = ""
    reward: float = 0.0
    reward_type: str = ""
    verify_result: dict | None = None
    error: str | None = None
    trace: list = field(default_factory=list)
    made_tool_call: bool = False
    executed_tool_calls: list = field(default_factory=list)
    advice: str | None = None
    advice_source: str = "none"
    episode_sec: float = 0.0


# ---------------------------------------------------------------------------
# Single-round agent episode. Mirrors RefineJob.run for exactly ONE round:
# fresh env, native student step, verify. The student conversation and the
# env both start from scratch — advice, when present, lives in the system
# prompt only.
# ---------------------------------------------------------------------------
async def run_episode(env, llm: RetryLLM, args: argparse.Namespace,
                      task_description: str, tools_schema: list[dict],
                      advice: str | None) -> dict:
    system_prompt = STUDENT_NATIVE_SYSTEM_PROMPT if args.mode == "vanilla" \
        else build_student_prompt(advice)
    student_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task_description},
    ]

    step_result = await _student_step(
        env, llm, student_messages, tools_schema,
        max_iterations=args.max_iterations,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    verify = await _verify_only(env, step_result["final_answer"])

    return {
        "task": task_description,
        "steps": step_result["steps"],
        "final_answer": step_result["final_answer"],
        "reward": verify["reward"],
        "reward_type": verify["reward_type"],
        "verify_result": verify["verify_result"],
        "error": step_result["error"],
        "trace": step_result["trace"],
        "made_tool_call": step_result["made_tool_call"],
        "executed_tool_calls": step_result["executed_tool_calls"],
    }


# ---------------------------------------------------------------------------
# RolloutJob — one episode; implements the Job protocol.
# ---------------------------------------------------------------------------
@dataclass
class RolloutJob:
    scenario: str
    task_idx: int
    args: argparse.Namespace
    llm: RetryLLM
    advice_map: dict | None = None

    def key(self) -> tuple:
        return (self.scenario, self.task_idx)

    async def _run_once(self, llm: RetryLLM, advice: str | None) -> tuple:
        env = AWMEnv(
            base_url=self.args.awm_base_url,
            message_timeout_s=self.args.episode_timeout,
            connect_timeout_s=60.0,
        )
        try:
            await env.connect()
            reset_res = await env.reset(scenario=self.scenario,
                                        task_idx=self.task_idx)
            if reset_res.observation.reward_type == "reset_error":
                return None, f"reset_error: {reset_res.observation.error}"
            task_description = reset_res.observation.task
            list_res = await env.step(ListToolsAction())
            tools_schema = tools_to_openai_schema(list_res.observation.tools)
            result = await asyncio.wait_for(
                run_episode(env, llm, self.args, task_description,
                            tools_schema, advice),
                timeout=self.args.episode_timeout,
            )
            return result, None
        finally:
            try:
                await env.close()
            except Exception:
                pass

    def _final_ep(self, ep: EpisodeResult, t0: float,
                  advice: str | None, source: str) -> dict:
        ep.advice = advice
        ep.advice_source = source
        ep.mode = self.args.mode
        ep.episode_sec = round(time.monotonic() - t0, 3)
        return asdict(ep)

    async def run(self) -> dict:
        args = self.args
        t0 = time.monotonic()
        ep = EpisodeResult(scenario=self.scenario, task_idx=self.task_idx)

        # --- vanilla / advice: one pass, no teacher --------------------------
        if args.mode in ("vanilla", "advice"):
            advice = None
            source = "none"
            if args.mode == "advice":
                advice = (self.advice_map or {}).get(self.key())
                source = "refine_jsonl" if advice else "missing"
            try:
                result, err = await self._run_once(self.llm, advice)
                if err:
                    ep.error = err
                else:
                    for k, v in result.items():
                        setattr(ep, k, v)
            except asyncio.TimeoutError:
                ep.reward_type = "timeout"
                ep.error = f"episode timeout > {args.episode_timeout}s"
            except Exception as e:
                ep.reward_type = "client_error"
                ep.error = repr(e)[:500]
            return self._final_ep(ep, t0, advice, source)

        # --- refine_collect: round 1 student attempt → teacher advice --------
        advice = None
        err: str | None = None
        try:
            result, err = await self._run_once(self.llm, advice=None)
            if err:
                ep.error = err
            else:
                for k, v in result.items():
                    setattr(ep, k, v)
        except asyncio.TimeoutError:
            ep.reward_type = "timeout"
            ep.error = f"episode timeout > {args.episode_timeout}s"
            return self._final_ep(ep, t0, None, "none")
        except Exception as e:
            ep.reward_type = "client_error"
            ep.error = repr(e)[:500]
            return self._final_ep(ep, t0, None, "none")

        if ep.reward_type == "complete":
            # Already solved without help; no advice needed.
            return self._final_ep(ep, t0, None, "none")

        verify_error = None
        vr = ep.verify_result or {}
        if isinstance(vr, dict):
            verify_error = vr.get("error") or vr.get("message")
            if isinstance(verify_error, str):
                verify_error = verify_error[:400]
        student_conversation = [
            {"role": "system", "content": STUDENT_NATIVE_SYSTEM_PROMPT},
            {"role": "user", "content": ep.task},
        ]
        teacher = await _teacher_advise(
            awm_base_url=args.awm_base_url,
            scenario=self.scenario,
            task_idx=self.task_idx,
            task=ep.task,
            student_messages=student_conversation,
            past_tool_calls=ep.executed_tool_calls,
            verify_reward_type=ep.reward_type,
            verify_error=verify_error,
            teacher_llm=args._teacher_llm,
            max_iterations=args.teacher_max_iterations,
            max_tool_calls=args.teacher_max_tool_calls,
            max_tokens=args.teacher_max_tokens,
            temperature=args.teacher_temperature,
        )
        advice = teacher.get("advice")
        if advice:
            ep.advice = advice
            ep.advice_source = "online"
            ep.mode = "refine_collect"
            ep.episode_sec = round(time.monotonic() - t0, 3)
            return asdict(ep)
        ep.error = (ep.error or "") + (
            f" teacher no-advice: {teacher.get('error') or 'unknown'};")
        return self._final_ep(ep, t0, None, "none")


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
        rt = e.get("reward_type", "?")
        type_dist[rt] = type_dist.get(rt, 0) + 1

    per_scenario: dict = {}
    for e in episodes:
        d = per_scenario.setdefault(
            e["scenario"], {"n": 0, "reward_sum": 0.0, "pass": 0})
        d["n"] += 1
        d["reward_sum"] += e.get("reward", 0.0)
        d["pass"] += int(e.get("reward_type") == "complete")

    def _rate(eps):
        if not eps:
            return None
        return {
            "n": len(eps),
            "pass_rate": sum(1 for e in eps if e.get("reward_type") == "complete") / len(eps),
            "avg_reward": sum(e.get("reward", 0.0) for e in eps) / len(eps),
        }

    with_advice = [e for e in episodes if e.get("advice")]
    without_advice = [e for e in episodes if not e.get("advice")]
    n_advice = len(with_advice)
    n_missing = sum(1 for e in episodes if e.get("advice_source") == "missing")

    report = {
        "config": {k: v for k, v in vars(args).items()
                   if not k.startswith("_")},
        "mode": args.mode,
        "num_episodes": n,
        "elapsed_wall_sec": round(elapsed_wall_sec, 2),
        "sum_episode_sec": round(sum_episode_sec, 2),
        "concurrency_speedup": round(sum_episode_sec / max(elapsed_wall_sec, 1e-6), 2),
        "pass_rate": pass_count / max(n, 1),
        "avg_reward": reward_sum / max(n, 1),
        "early_exit_count": early_exit,
        "verifier_reward_type_distribution": type_dist,
        "num_with_advice": n_advice,
        "num_missing_advice": n_missing,
        "with_advice": _rate(with_advice),
        "without_advice": _rate(without_advice),
        "per_scenario": per_scenario,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print(f"AWM Rollout Report  (mode={args.mode})")
    print("=" * 80)
    print(f"Model:                    {args.llm_model}")
    print(f"Episodes:                 {n}")
    print(f"Elapsed (wall):           {report['elapsed_wall_sec']:.1f}s")
    print(f"Concurrency speedup:      {report['concurrency_speedup']:.1f}x")
    print(f"Pass@1 (complete):        {report['pass_rate']:.2%}")
    print(f"Avg reward:               {report['avg_reward']:.4f}")
    print(f"Early-exit (no tool):     {early_exit}/{n}")
    print(f"Reward types:             {type_dist}")
    if args.mode == "advice":
        print(f"Advice coverage:          {n_advice}/{n} "
              f"(missing={n_missing})")
        wa = report["with_advice"]
        wo = report["without_advice"]
        if wa:
            print(f"With    advice (n={wa['n']}):  pass={wa['pass_rate']:.2%}  avg_r={wa['avg_reward']:.4f}")
        if wo:
            print(f"Without advice (n={wo['n']}):  pass={wo['pass_rate']:.2%}  avg_r={wo['avg_reward']:.4f}")
    if args.mode == "refine_collect":
        print(f"Advice collected:         {n_advice}/{n}")
    print(f"Checkpoint:               {ckpt.path}")
    print(f"Report:                   {report_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def _load_advice_map(path: str) -> dict:
    """Parse a rollout.refine JSONL and return {(scenario, task_idx): advice}.

    Uses the LAST non-null `teacher_advice` in `rounds_detail` — that is the
    advice produced after the final failed attempt and reflects the most
    diagnosis."""
    advice_map: dict = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            advice = None
            for rd in rec.get("rounds_detail") or []:
                a = rd.get("teacher_advice")
                if a:
                    advice = a
            if advice:
                advice_map[(rec["scenario"], rec["task_idx"])] = advice
    return advice_map


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["vanilla", "advice", "refine_collect"],
                        default="advice",
                        help="vanilla: no advice. advice: inject advice from "
                             "--refine-jsonl into the system prompt. "
                             "refine_collect: run the student once, ask the "
                             "teacher for advice, and store it (no retry).")
    parser.add_argument("--refine-jsonl", default=None,
                        help="Path to a rollout.refine output JSONL used to "
                             "harvest per-(scenario,task_idx) advice.")
    parser.add_argument("--num-scenarios", type=int, default=1000)
    parser.add_argument("--tasks-per-scenario", type=int, default=10)
    parser.add_argument("--max-iterations", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--episode-timeout", type=float, default=900.0)
    parser.add_argument("--llm-timeout", type=float, default=180.0)
    parser.add_argument("--awm-base-url", default=os.environ.get("AWM_BASE_URL", "http://localhost:8899"))
    parser.add_argument("--llm-base-url", default=os.environ.get("ENDPOINT_URL", "http://localhost:8000/v1"))
    parser.add_argument("--llm-api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--llm-model", default=os.environ.get("AWM_EXAMPLE_AGENT_MODEL", "qwen3.5-4b"))
    parser.add_argument("--scenarios", default=None,
                        help="Comma-separated scenario names; overrides --num-scenarios.")
    parser.add_argument("--start-scenario-idx", type=int, default=0)
    parser.add_argument("--end-scenario-idx", type=int, default=None,
                        help="Exclusive end index into the scenario list.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional cap on total (scenario,task) pairs.")
    parser.add_argument("--checkpoint-path", default="/mnt/storage/disk3/self_evolver/traj_data/rollout_advice.jsonl")
    parser.add_argument("--report", default="/mnt/storage/disk3/self_evolver/traj_data/rollout_advice.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-interval", type=float, default=10.0)
    # teacher (only used in refine_collect mode)
    parser.add_argument("--teacher-base-url", default="https://ark.cn-beijing.volces.com/api/v3")
    parser.add_argument("--teacher-api-key", default=os.environ.get("ARK_API_KEY"))
    parser.add_argument("--teacher-model", default=os.environ.get("TEACHER_MODEL", "ep-20260707130305-26bjx"))
    parser.add_argument("--teacher-rpm", type=int, default=400)
    parser.add_argument("--teacher-tpm", type=int, default=800_000)
    parser.add_argument("--teacher-max-iterations", type=int, default=4)
    parser.add_argument("--teacher-max-tool-calls", type=int, default=3)
    parser.add_argument("--teacher-max-tokens", type=int, default=8192)
    parser.add_argument("--teacher-temperature", type=float, default=0.4)
    args = parser.parse_args()

    if args.mode == "advice" and not args.refine_jsonl:
        raise SystemExit("--mode advice requires --refine-jsonl.")
    if args.mode == "refine_collect" and not args.teacher_api_key:
        raise SystemExit("--mode refine_collect requires a teacher API key "
                         "(export ARK_API_KEY or pass --teacher-api-key).")

    llm = RetryLLM(base_url=args.llm_base_url, api_key=args.llm_api_key,
                   model=args.llm_model, timeout=args.llm_timeout)

    if args.mode == "refine_collect":
        args._teacher_llm = RetryLLM(
            base_url=args.teacher_base_url,
            api_key=args.teacher_api_key,
            model=args.teacher_model,
            timeout=args.llm_timeout,
            limiter=RateLimiter(rpm=args.teacher_rpm, tpm=args.teacher_tpm),
        )

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

    advice_map: dict = {}
    if args.mode == "advice":
        advice_map = _load_advice_map(args.refine_jsonl)
        print(f"[advice] loaded {len(advice_map)} advice entries from "
              f"{args.refine_jsonl}")

    todo = [(s, t) for s in picked for t in range(args.tasks_per_scenario)]
    if args.limit is not None:
        todo = todo[: args.limit]

    jobs = [RolloutJob(s, t, args, llm, advice_map) for (s, t) in todo]

    ckpt = JsonlCheckpoint(args.checkpoint_path)

    print(f"Starting {len(jobs)} episodes: mode={args.mode}, "
          f"concurrency={args.concurrency}, "
          f"episode_timeout={args.episode_timeout}s, llm_timeout={args.llm_timeout}s")

    t0 = time.monotonic()
    await run_jobs(jobs, args.concurrency, ckpt,
                   progress_interval=args.progress_interval, resume=args.resume)
    elapsed = time.monotonic() - t0

    _aggregate_and_print(ckpt, args, elapsed, args.report)


if __name__ == "__main__":
    asyncio.run(main())
