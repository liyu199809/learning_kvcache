#!/usr/bin/env python3
"""统一评测入口：用已部署的 vLLM 服务 eval 指定 benchmark。

首期仅支持 --benchmark lifelong_db（LifelongAgentBench/db）；
其余 benchmark（intercode / ama / memoryarena）架构已预留，后续在
benchmarks/ 下新增适配文件并注册即可启用。

用法示例：
  python -m benchmark.eval.run_eval --benchmark lifelong_db \
      --openai-base-url http://127.0.0.1:8000/v1 --model qwen3.5-4b \
      --data-dir benchmark/LifelongAgentBench --split test \
      --tasks 0-9 --concurrency 2 --max-steps 6
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime
from pathlib import Path

# 允许以脚本或模块方式运行。
if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from benchmark.eval.core import registry
    from benchmark.eval.core.llm_client import AsyncLLM
    from benchmark.eval.core.runner import EvalRunner
    import benchmark.eval.benchmarks  # noqa: F401  触发注册
else:
    from .core import registry
    from .core.llm_client import AsyncLLM
    from .core.runner import EvalRunner
    from . import benchmarks  # noqa: F401  触发注册


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", default="lifelong_db",
                    help=f"要评测的 benchmark。已注册: {', '.join(registry.available())}")
    # LLM / vLLM
    ap.add_argument("--openai-base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="qwen3.5-4b")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--llm-max-completion-tokens", type=int, default=None)
    # 数据 / 选择
    ap.add_argument("--data-dir", default="benchmark/LifelongAgentBench")
    ap.add_argument("--split", default="test")
    ap.add_argument("--tasks", default=None, help="索引，如 0,3,8 或 0-99")
    ap.add_argument("--max-tasks", type=int, default=None)
    ap.add_argument("--task-ids", default=None, help="按 task_id 选择，如 db_0,db_3")
    # 运行
    ap.add_argument("--max-steps", type=int, default=6)
    ap.add_argument("--step-timeout", type=float, default=180.0)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--output-dir", default=None)
    # lifelong_db 专用：MySQL 镜像（9.x 缺 MD5/SHA1 内置函数，须用 8.0）
    ap.add_argument("--mysql-image", default="mysql:8.0")
    # judge 家族预留（本期未使用）
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--judge-base-url", default=None)
    return ap.parse_args()


async def main() -> int:
    args = parse_args()

    suite = registry.get_suite(args.benchmark, args)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = (
            Path("workspace") / "eval_logs" / args.benchmark / args.split / stamp
        )

    llm = AsyncLLM(
        model=args.model,
        base_url=args.openai_base_url,
        api_key=args.api_key,
        temperature=args.temperature,
        top_p=args.top_p,
        max_completion_tokens=args.llm_max_completion_tokens,
    )

    indices = suite.select(tasks=args.tasks, max_tasks=args.max_tasks, task_ids=args.task_ids)
    if not indices:
        raise SystemExit("没有选中任何任务，请检查 --tasks/--max-tasks/--task-ids。")

    runner = EvalRunner(
        llm=llm,
        suite=suite,
        max_steps=args.max_steps,
        step_timeout=args.step_timeout,
        prompt_type="db",
        output_dir=output_dir,
        concurrency=args.concurrency,
    )

    results = await runner.run(indices)

    total = len(results)
    successes = sum(1 for r in results if r.success)
    errors = sum(1 for r in results if r.error)
    print("\n" + "=" * 72)
    print(f"EVAL RESULTS — {args.benchmark}")
    print("=" * 72)
    print(f"Split:        {args.split}")
    print(f"Model:        {args.model}  @ {args.openai_base_url}")
    print(f"Tasks:        {total}")
    print(f"Successes:    {successes}")
    print(f"Errors:       {errors}")
    print(f"pass@1:       {(successes / total * 100) if total else 0:.1f}%")
    print(f"Output:       {output_dir}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
