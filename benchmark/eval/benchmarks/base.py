"""所有 benchmark 适配层遵循的抽象接口。

一个 benchmark = 一个 TaskSuite（数据集封装）+ 若干 Task（单样本环境）+ 一个 Scorer。
runner 只依赖这些抽象，不感知具体 benchmark。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..core.scorer import Scorer
from ..core.types import Action, BasicInfo, Observation, StepReturn


class Task:
    """一条待评测样本对应的环境。

    交互式任务（db/intercode）实现 reset/step/close；
    judge 式任务（ama/memoryarena）可在 step 内累积 answer，done 时不给 reward，
    交由 JudgeScorer 后处理。
    """

    task_id: str

    def get_basic_info(self) -> BasicInfo:
        raise NotImplementedError

    async def reset(self) -> Observation:
        raise NotImplementedError

    async def step(self, action: Action) -> StepReturn:
        raise NotImplementedError

    async def close(self) -> None:
        return None


class TaskSuite:
    """某个 benchmark 的数据集封装。"""

    name: str

    def list_task_ids(self) -> List[str]:
        """返回全部 task_id（顺序即索引顺序）。"""
        raise NotImplementedError

    def select(
        self,
        *,
        tasks: Optional[str] = None,
        max_tasks: Optional[int] = None,
        task_ids: Optional[str] = None,
    ) -> List[int]:
        """根据 CLI 选择参数返回要跑的索引列表。默认全跑。"""
        total = len(self.list_task_ids())
        if task_ids:
            wanted = {t.strip() for t in str(task_ids).replace(",", " ").split() if t.strip()}
            all_ids = self.list_task_ids()
            id_to_idx = {tid: i for i, tid in enumerate(all_ids)}
            return sorted({id_to_idx[t] for t in wanted if t in id_to_idx})
        if tasks:
            return _parse_indices(str(tasks), total)
        if max_tasks:
            return list(range(min(max_tasks, total)))
        return list(range(total))

    def make_task(self, index: int) -> Task:
        raise NotImplementedError

    def scorer(self) -> Scorer:
        raise NotImplementedError

    def default_max_steps(self) -> int:
        return 10


def _parse_indices(spec: str, total: int) -> List[int]:
    """解析 "0,3,8" 或 "0-99" 或混合形式为索引列表。"""
    out: List[int] = []
    for part in spec.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            lo_i = int(lo) if lo else 0
            hi_i = int(hi) if hi else total - 1
            out.extend(range(lo_i, min(hi_i, total - 1) + 1))
        else:
            idx = int(part)
            if 0 <= idx < total:
                out.append(idx)
    seen: Dict[int, None] = {}
    for i in out:
        seen.setdefault(i, None)
    return list(seen.keys())
