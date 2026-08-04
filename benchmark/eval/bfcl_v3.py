"""BFCL v3 驱动：把统一评测入口对接到官方 BFCL 批处理管线。

与 lifelong_db/os 不同，BFCL v3 是一个「全量 generate → 全量 evaluate → 聚合打分」
的离线批处理流程，判分（AST checker / multi-turn 有状态后端 / relevance 检测 /
category 聚合 / scoreboard）全部属于官方代码，必须原样运行才能与官方 leaderboard
口径一致。因此本模块不复用 EvalRunner 的逐 episode 环路，而是以子进程方式跑固定
commit 的官方 ``bfcl_eval``，只做「入口 + 配置 + 产物目录」的统一：

  - 复用统一 CLI 的 --openai-base-url / --model / --output-dir / --concurrency 等；
  - 官方源码通过 PYTHONPATH 挂载（不注册发行版元数据，避免 NumPy 1.x 等回退）；
  - 官方 result/score 落到统一的 output-dir，与其它 benchmark 同一棵产物树；
  - 官方 OSS handler 把 --local-model-path 同时当作本地 tokenizer 路径与 vLLM 的
    API model id，故以 model 名建软链指向权重目录，并在该目录下运行；
  - 绕开官方 ``bfcl scores`` 的 'Non-Live Exec Acc' 列崩溃（v3 已删 exec 类别，
    但 scores 子命令仍硬编码该列），改为直接读 data_overall.csv 打印安全列，
    不修改官方源码。
"""
from __future__ import annotations

import csv
import os
import subprocess
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse


def _bfcl_root() -> Path:
    """benchmark/bfcl_v3 目录（官方源码、软链、setup.sh 所在处）。"""
    return Path(__file__).resolve().parents[1] / "bfcl_v3"


def _official_root(bfcl_root: Path) -> Path:
    return Path(os.getenv("BFCL_OFFICIAL_ROOT", str(bfcl_root / "official")))


def _ensure_official(bfcl_root: Path, official_root: Path) -> None:
    """官方源码缺失时运行 setup.sh 导出固定 commit 子树。"""
    if (official_root / "bfcl_eval" / "__main__.py").exists():
        return
    setup = bfcl_root / "setup.sh"
    if not setup.exists():
        raise SystemExit(f"BFCL 官方源码缺失且未找到 setup.sh: {setup}")
    subprocess.run([str(setup)], check=True)


def _split_endpoint(base_url: str) -> tuple[str, str]:
    """从 http://host:port/v1 解析出 (host, port) 供官方 handler 使用。"""
    parsed = urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"
    port = str(parsed.port or 8000)
    return host, port


def _prepare_model_alias(bfcl_root: Path, model_alias: str, model_path: str) -> None:
    """建软链 <bfcl_root>/<model_alias> -> model_path。

    官方 OSS handler 用 --local-model-path 同时作为本地 tokenizer 路径与 API
    model id；软链名必须与 vLLM --served-model-name 一致，才能连上现有服务。
    """
    if "/" in model_alias or model_alias in (".", ".."):
        raise SystemExit(f"--model 必须是纯模型名（用作 vLLM model id）: {model_alias}")
    alias_path = bfcl_root / model_alias
    target = Path(model_path).resolve()
    if alias_path.exists() or alias_path.is_symlink():
        if alias_path.resolve() != target:
            raise SystemExit(f"模型软链已被占用: {alias_path} -> {alias_path.resolve()}")
        return
    alias_path.symlink_to(target)


def _check_server(base_url: str, model_alias: str) -> None:
    import requests

    # base_url 形如 http://host:port/v1
    url = f"{base_url.rstrip('/')}/models"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    model_ids = [item["id"] for item in resp.json().get("data", [])]
    if model_alias not in model_ids:
        raise SystemExit(
            f"vLLM model 不匹配：期望 {model_alias!r}，实际 {model_ids!r}。"
            " 请先用 ./start_services.sh start vllm 启动对应模型。"
        )
    print(f"[bfcl_v3] Using vLLM model: {model_alias}")


def _bfcl_cmd(
    official_root: Path,
    project_python: str,
    subcmd_args: List[str],
) -> List[str]:
    return [project_python, "-m", "bfcl_eval", *subcmd_args]


def _run_bfcl(
    *,
    official_root: Path,
    bfcl_root: Path,
    project_python: str,
    project_root_env: Path,
    host: str,
    port: str,
    args: List[str],
) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        f"{official_root}{os.pathsep}{env['PYTHONPATH']}"
        if env.get("PYTHONPATH")
        else str(official_root)
    )
    env["BFCL_PROJECT_ROOT"] = str(project_root_env)
    env["VLLM_ENDPOINT"] = host
    env["VLLM_PORT"] = port
    subprocess.run(
        _bfcl_cmd(official_root, project_python, args),
        cwd=str(bfcl_root),
        env=env,
        check=True,
    )


def _print_scores(score_dir: Path) -> None:
    """绕开官方 `bfcl scores`（硬编码 'Non-Live Exec Acc' 列会崩溃），
    直接读 data_overall.csv 打印一组存在的安全列。"""
    overall = score_dir / "data_overall.csv"
    if not overall.exists():
        print(f"[bfcl_v3] 未找到打分文件: {overall}")
        return
    wanted = [
        "Rank",
        "Model",
        "Overall Acc",
        "Non-Live AST Acc",
        "Live Acc",
        "Multi Turn Acc",
        "Relevance Detection",
        "Irrelevance Detection",
    ]
    with overall.open(newline="") as f:
        reader = csv.reader(f)
        headers = next(reader, [])
        idx = [(c, headers.index(c)) for c in wanted if c in headers]
        cols = [c for c, _ in idx]
        rows = [[row[i] for _, i in idx] for row in reader]
    print("\n" + " | ".join(cols))
    print("-" * (len(" | ".join(cols))))
    for row in rows:
        print(" | ".join(row))
    print(f"\n[bfcl_v3] 明细见: {score_dir}")


def run_bfcl(args) -> int:
    """统一入口的 bfcl_v3 分发目标。args 为 run_eval 的 argparse.Namespace。"""
    bfcl_root = _bfcl_root()
    official_root = _official_root(bfcl_root)
    project_python = os.getenv(
        "PROJECT_PYTHON",
        str(Path(__file__).resolve().parents[2] / ".venv" / "bin" / "python"),
    )

    _ensure_official(bfcl_root, official_root)

    host, port = _split_endpoint(args.openai_base_url)
    model_alias = args.model
    model_path = args.bfcl_model_path

    _prepare_model_alias(bfcl_root, model_alias, model_path)
    _check_server(args.openai_base_url, model_alias)

    # 官方 result/score 落到统一 output-dir（作为 BFCL_PROJECT_ROOT）。
    project_root_env = Path(args.output_dir).resolve()
    project_root_env.mkdir(parents=True, exist_ok=True)
    result_dir = "result"
    score_dir = "score"

    categories = args.bfcl_categories
    model_key = args.bfcl_model_key

    print("=" * 72)
    print("BFCL v3 (official pipeline)")
    print("=" * 72)
    print(f"Model key:    {model_key}")
    print(f"vLLM model:   {model_alias} @ {args.openai_base_url}")
    print(f"Categories:   {categories}")
    print(f"Output:       {project_root_env}")
    print("=" * 72)

    # 1) generate
    gen_args = [
        "generate",
        "--model", model_key,
        "--test-category", categories,
        "--temperature", str(args.temperature),
        "--num-threads", str(args.concurrency),
        "--skip-server-setup",
        "--local-model-path", model_alias,
        "--result-dir", result_dir,
    ]
    if getattr(args, "bfcl_include_input_log", True):
        gen_args.append("--include-input-log")
    if getattr(args, "bfcl_allow_overwrite", False):
        gen_args.append("--allow-overwrite")
    run_ids_file = getattr(args, "bfcl_run_ids_file", None)
    if run_ids_file:
        src = Path(run_ids_file)
        if not src.exists():
            raise SystemExit(f"--bfcl-run-ids-file 不存在: {src}")
        (project_root_env / "test_case_ids_to_generate.json").write_bytes(src.read_bytes())
        gen_args.append("--run-ids")

    _run_bfcl(
        official_root=official_root, bfcl_root=bfcl_root, project_python=project_python,
        project_root_env=project_root_env, host=host, port=port, args=gen_args,
    )

    # 2) evaluate
    eval_args = [
        "evaluate",
        "--model", model_key,
        "--test-category", categories,
        "--result-dir", result_dir,
        "--score-dir", score_dir,
    ]
    _run_bfcl(
        official_root=official_root, bfcl_root=bfcl_root, project_python=project_python,
        project_root_env=project_root_env, host=host, port=port, args=eval_args,
    )

    # 3) scores（绕开官方 scores 子命令的列崩溃）
    _print_scores(project_root_env / score_dir)
    return 0
