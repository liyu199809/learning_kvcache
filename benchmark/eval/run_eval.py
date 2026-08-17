#!/usr/bin/env python3
"""统一评测入口：用已部署的 vLLM 服务 eval 指定 benchmark。

交互式 benchmark（lifelong_db / lifelong_os）走通用 EvalRunner 逐 episode 环路；
BFCL v3 因判分逻辑必须严格对齐官方 leaderboard，走官方批处理管线（见 bfcl_v3.py），
仍复用同一入口与同一套 endpoint / model / output-dir 配置。

用法示例：
  # 交互式 benchmark
  python -m benchmark.eval.run_eval --benchmark lifelong_db \
      --openai-base-url http://127.0.0.1:8000/v1 --model qwen3.5-4b \
      --data-dir benchmark/LifelongAgentBench --split test \
      --tasks 0-9 --concurrency 2 --max-steps 6

  # BFCL v3（官方管线，判分口径与 leaderboard 一致）
  python -m benchmark.eval.run_eval --benchmark bfcl_v3 \
      --openai-base-url http://127.0.0.1:8000/v1 --model qwen3.5-4b \
      --bfcl-model-path /path/to/checkpoint --bfcl-categories non_live \
      --concurrency 100
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
    from benchmark.eval.bfcl_v3 import run_bfcl
    from benchmark.eval.tau2 import run_tau2
    import benchmark.eval.benchmarks  # noqa: F401  触发注册
else:
    from .core import registry
    from .core.llm_client import AsyncLLM
    from .core.runner import EvalRunner
    from .bfcl_v3 import run_bfcl
    from .tau2 import run_tau2
    from . import benchmarks  # noqa: F401  触发注册


BFCL_BENCHMARK = "bfcl_v3"
TAU2_BENCHMARK = "tau2"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", default="lifelong_db",
                    help=f"要评测的 benchmark。交互式已注册: {', '.join(registry.available())}；"
                         f"官方管线: {BFCL_BENCHMARK}, {TAU2_BENCHMARK}")
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
    # lifelong_os 专用：单条 bash 命令超时（秒）
    ap.add_argument("--os-timeout", type=int, default=20)
    # judge 家族预留（本期未使用）
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--judge-base-url", default=None)
    # bfcl_v3 专用（走官方批处理管线）：
    ap.add_argument("--bfcl-model-path",
                    default="/mnt/storage/disk1/verl_data/base_model/Qwen3.5-4B",
                    help="本地权重目录；仅作官方 handler 的 tokenizer/config 来源，"
                         "API model id 由 --model（BFCL_API_MODEL_ID）独立指定。")
    ap.add_argument("--bfcl-model-key", default="Qwen/Qwen3-4B-FC",
                    help="BFCL 官方 model handler 注册名（决定 prompt 模板与解析）。")
    ap.add_argument("--bfcl-categories", default="all",
                    help="官方 test-category 或 collection，如 all/non_live/live/multi_turn。")
    ap.add_argument("--bfcl-include-input-log", action="store_true", default=True)
    ap.add_argument("--bfcl-allow-overwrite", action="store_true", default=False)
    ap.add_argument("--bfcl-run-ids-file", default=None,
                    help="官方 test_case_ids_to_generate.json，用于定向生成部分样本。")
    # tau2-bench 专用（走官方批处理管线，独立 Python 3.12 venv）：
    ap.add_argument("--tau2-domain", default="airline",
                    help="tau2 domain: airline/retail/telecom/mock/banking_knowledge。")
    ap.add_argument("--tau2-split", default="base",
                    help="任务 split（默认 base，与原 tau-bench 对齐）。")
    ap.add_argument("--tau2-num-tasks", type=int, default=None,
                    help="限制任务数（省略=该 split 全跑）。")
    ap.add_argument("--tau2-num-trials", type=int, default=1,
                    help="每任务重复次数（pass^k 需要 >1）。")
    ap.add_argument("--tau2-max-steps", type=int, default=None,
                    help="覆盖 tau2 单次对话最大步数（省略=官方默认 200）。")
    ap.add_argument("--tau2-agent-thinking", action="store_true",
                    help="启用 tau2 agent 的 Qwen thinking（默认关闭以提速）。")
    ap.add_argument("--tau2-user-thinking", action="store_true",
                    help="启用 tau2 user simulator 的 Qwen thinking（默认关闭）。")
    return ap.parse_args()


async def main() -> int:
    args = parse_args()

    # BFCL v3 走官方批处理管线（判分对齐 leaderboard），不进入 EvalRunner 环路。
    if args.benchmark == BFCL_BENCHMARK:
        if not args.output_dir:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.output_dir = str(
                Path("workspace") / "eval_logs" / args.benchmark / args.bfcl_categories / stamp
            )
        return run_bfcl(args)

    # tau2-bench 走官方管线（独立 3.12 venv + litellm 路由到本地 vLLM）。
    if args.benchmark == TAU2_BENCHMARK:
        if not args.output_dir:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.output_dir = str(
                Path("workspace") / "eval_logs" / args.benchmark / args.tau2_domain / stamp
            )
        return run_tau2(args)

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
        llm_max_tokens=args.llm_max_completion_tokens,
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
