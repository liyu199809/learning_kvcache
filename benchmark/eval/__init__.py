"""benchmark.eval — 通用评测 harness（不依赖 opd_evolver）。

内核（core）与 benchmark 无关，各 benchmark 只需在 benchmarks/ 下实现自己的
TaskSuite / Task / Scorer 并在 core.registry 注册即可扩展。
首期仅落地 LifelongAgentBench/db。
"""
