"""Scorer 抽象：程序化打分 vs LLM-as-judge。

- ProgrammaticScorer：reward 已由 env 在 step() 内算好（db / intercode），
  Scorer 只从 episode trace 里读取，判定 success。
- JudgeScorer：judge 家族（AMA-Bench / MemoryArena）用；持有独立 judge LLM，
  对 rollout 结果后处理打分。首期仅定义接口，不接具体 judge。
"""
from __future__ import annotations

from typing import List, Optional

from .llm_client import AsyncLLM
from .types import EpisodeResult, StepRecord


class Scorer:
    """打分器基类。runner 在 episode 交互结束后调用。"""

    async def score(self, result: EpisodeResult) -> EpisodeResult:
        raise NotImplementedError


class ProgrammaticScorer(Scorer):
    """通用程序化打分：以 trace 中 submit 步的 reward/info 判定成功。

    适用于 reward 已由环境算好的 benchmark（db、intercode）。
    """

    def __init__(self, reward_threshold: float = 1.0):
        self.reward_threshold = reward_threshold

    def _determine_success(self, trace: List[StepRecord], total_reward: float) -> bool:
        if total_reward >= self.reward_threshold:
            return True
        for record in reversed(trace):
            action = record.action if isinstance(record.action, dict) else {}
            if str(action.get("action", "")).lower() == "submit":
                info = record.info or {}
                if info.get("success") or float(info.get("reward", 0) or 0) >= self.reward_threshold:
                    return True
        return False

    async def score(self, result: EpisodeResult) -> EpisodeResult:
        result.success = self._determine_success(result.trace, result.total_reward)
        return result


class JudgeScorer(Scorer):
    """LLM-as-judge 打分器（judge 家族 benchmark 用）。

    首期占位：保存 judge LLM 与 judge prompt 构造钩子，具体 benchmark 在
    子类里实现 ``build_judge_prompt`` 与结果解析。
    """

    def __init__(self, judge_llm: Optional[AsyncLLM] = None):
        self.judge_llm = judge_llm

    async def score(self, result: EpisodeResult) -> EpisodeResult:
        raise NotImplementedError(
            "JudgeScorer 需由具体 benchmark 子类实现（AMA-Bench / MemoryArena）。"
        )
