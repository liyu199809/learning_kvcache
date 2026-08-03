"""通用数据类型与接口，供内核与各 benchmark 适配层共享。"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# 与 opd_evolver 保持一致的松散类型
Observation = Dict[str, Any]
Action = Dict[str, Any]


@dataclass
class BasicInfo:
    """一条任务的静态信息，供 agent 构造 prompt。"""

    env_id: str
    instruction: str
    action_space: str
    max_steps: int
    # ReAct agent 用哪套 prompt 模板（如 "db" / "sql" / "intercode"）
    prompt_type: str = "db"
    meta_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StepRecord:
    observation: Observation
    action: Action
    reward: float
    raw_response: str
    done: bool
    info: Dict[str, Any]
    raw_input: Optional[str] = None
    observation_after: Optional[Observation] = None
    debug: Optional[Dict[str, Any]] = None


@dataclass
class EpisodeResult:
    task_id: str
    benchmark: str
    success: bool
    total_reward: float
    steps: int
    trace: List[StepRecord] = field(default_factory=list)
    cost: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    extra: Dict[str, Any] = field(default_factory=dict)


# step() 的返回约定：(observation, reward, done, info)
StepReturn = Tuple[Observation, float, bool, Dict[str, Any]]
