"""benchmark.eval.benchmarks — 各数据集适配层。

首期仅 lifelong_db 落地。导入本包会触发已实现 benchmark 的自动注册。
"""
from . import lifelong_db  # noqa: F401  (触发注册)
