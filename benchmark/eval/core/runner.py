"""通用评测 runner：单 episode 环路 + 并发调度 + 结果落盘 + pass@1 汇总。

与具体 benchmark 无关，只依赖 Task / TaskSuite / Scorer / ReActAgent 抽象。
环路逻辑移植并瘦身自 opd_evolver/runners/task_runner.py::SimpleTaskRunner.run。
"""
from __future__ import annotations

import asyncio
import csv
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .agent import ReActAgent
from .llm_client import AsyncLLM
from .scorer import Scorer
from .types import EpisodeResult, StepRecord
from ..benchmarks.base import Task, TaskSuite


class EvalRunner:
    def __init__(
        self,
        llm: AsyncLLM,
        suite: TaskSuite,
        *,
        max_steps: int,
        step_timeout: Optional[float],
        prompt_type: str,
        output_dir: Path,
        concurrency: int = 4,
    ):
        self.llm = llm
        self.suite = suite
        self.max_steps = max_steps
        self.step_timeout = step_timeout
        self.prompt_type = prompt_type
        self.output_dir = Path(output_dir)
        self.concurrency = concurrency
        self.scorer: Scorer = suite.scorer()
        self.trajectory_dir = self.output_dir / "trajectories"
        self.csv_path = self.output_dir / "summary.csv"
        self._csv_lock = asyncio.Lock()

    async def _run_episode(self, index: int) -> EpisodeResult:
        task: Task = self.suite.make_task(index)
        info = task.get_basic_info()
        agent = ReActAgent(self.llm, prompt_type=info.prompt_type or self.prompt_type)
        agent.reset(info)

        trace: List[StepRecord] = []
        total_reward = 0.0
        done = False
        error: Optional[str] = None
        try:
            obs = await task.reset()
            for t in range(self.max_steps):
                current_step = t + 1
                try:
                    if self.step_timeout:
                        step_result = await asyncio.wait_for(
                            agent.step(obs, trace, current_step, self.max_steps),
                            timeout=self.step_timeout,
                        )
                    else:
                        step_result = await agent.step(obs, trace, current_step, self.max_steps)
                except asyncio.TimeoutError:
                    trace.append(
                        StepRecord(obs, {"error": "step_timeout"}, 0.0, "step timeout", True,
                                   {"error": "step_timeout"})
                    )
                    break
                action, raw_response, raw_input = step_result
                obs_before = obs
                obs_next, reward, step_done, step_info = await task.step(action)
                trace.append(
                    StepRecord(obs_before, action, reward, raw_response, step_done, step_info,
                               raw_input=raw_input, observation_after=obs_next)
                )
                total_reward += reward
                obs = obs_next
                if step_done:
                    done = True
                    break
            if not done:
                # max_steps 到达，强制 submit 结算。
                try:
                    obs_before = obs
                    obs, reward, _, step_info = await task.step({"action": "submit", "params": {}})
                    total_reward = float(reward)
                    trace.append(
                        StepRecord(obs_before, {"action": "submit", "params": {}}, reward,
                                   "forced_submit", True, step_info, observation_after=obs)
                    )
                    done = True
                except Exception as e:  # noqa
                    error = f"forced_submit_failed: {e}"
                    done = True
        except Exception as e:  # noqa
            error = f"{type(e).__name__}: {e}"
        finally:
            try:
                await task.close()
            except Exception:  # noqa
                pass

        result = EpisodeResult(
            task_id=info.env_id,
            benchmark=self.suite.name,
            success=False,
            total_reward=total_reward,
            steps=len(trace),
            trace=trace,
            model=self.llm.model,
            error=error,
        )
        result = await self.scorer.score(result)
        self._save_trajectory(info, result)
        await self._save_csv(result)
        return result

    async def run(self, indices: List[int]) -> List[EpisodeResult]:
        self.trajectory_dir.mkdir(parents=True, exist_ok=True)
        semaphore = asyncio.Semaphore(self.concurrency)

        async def _guarded(idx: int) -> EpisodeResult:
            async with semaphore:
                try:
                    return await self._run_episode(idx)
                except Exception as e:  # noqa
                    ids = self.suite.list_task_ids()
                    tid = ids[idx] if idx < len(ids) else str(idx)
                    return EpisodeResult(
                        task_id=tid, benchmark=self.suite.name, success=False,
                        total_reward=0.0, steps=0, model=self.llm.model,
                        error=f"{type(e).__name__}: {e}",
                    )

        return await asyncio.gather(*[_guarded(i) for i in indices])

    def _save_trajectory(self, info, result: EpisodeResult) -> None:
        try:
            self.trajectory_dir.mkdir(parents=True, exist_ok=True)
            steps = []
            for i, r in enumerate(result.trace):
                steps.append({
                    "step": i + 1,
                    "action": r.action,
                    "observation": r.observation,
                    "observation_after": r.observation_after,
                    "reward": r.reward,
                    "done": r.done,
                    "info": r.info,
                    "raw_response": r.raw_response,
                })
            payload = {
                "task_id": info.env_id,
                "benchmark": result.benchmark,
                "instruction": info.instruction,
                "model": result.model,
                "success": result.success,
                "total_reward": result.total_reward,
                "steps": result.steps,
                "error": result.error,
                "timestamp": result.timestamp,
                "meta_data": info.meta_data,
                "execution_trace": steps,
            }
            safe_id = str(info.env_id).replace("/", "_").replace(":", "_")
            fname = f"{safe_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.json"
            with (self.trajectory_dir / fname).open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:  # noqa
            print(f"[EvalRunner] save trajectory failed: {e}")

    async def _save_csv(self, result: EpisodeResult) -> None:
        async with self._csv_lock:
            try:
                self.csv_path.parent.mkdir(parents=True, exist_ok=True)
                fieldnames = ["task_id", "benchmark", "success", "reward", "steps", "error", "timestamp"]
                need_header = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
                mode = "a"
                with self.csv_path.open(mode, newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    if need_header:
                        writer.writeheader()
                    writer.writerow({
                        "task_id": result.task_id,
                        "benchmark": result.benchmark,
                        "success": result.success,
                        "reward": f"{result.total_reward:.4f}",
                        "steps": result.steps,
                        "error": result.error or "",
                        "timestamp": result.timestamp,
                    })
            except Exception as e:  # noqa
                print(f"[EvalRunner] save csv failed: {e}")
