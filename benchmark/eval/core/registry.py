"""benchmark 名 -> TaskSuite 构造器 的注册表。

新增 benchmark 时，在此登记一个构造函数即可被 CLI 选择。
构造函数签名：``(args) -> TaskSuite``（args 为 argparse.Namespace）。
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict

if TYPE_CHECKING:
    from ..benchmarks.base import TaskSuite

SuiteFactory = Callable[[Any], "TaskSuite"]

_REGISTRY: Dict[str, SuiteFactory] = {}


def register(name: str, factory: SuiteFactory) -> None:
    _REGISTRY[name] = factory


def get_suite(name: str, args: Any) -> "TaskSuite":
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise SystemExit(
            f"benchmark '{name}' 尚未实现或未注册。已注册: {available}"
        )
    return _REGISTRY[name](args)


def available() -> list[str]:
    return sorted(_REGISTRY)
