#!/usr/bin/env python3
"""Merge a training checkpoint, serve it with vLLM, and run evaluations.

The merged Hugging Face model is an inference-only temporary artifact.  By
default it is deleted after every requested benchmark finishes successfully.
The original checkpoint and all evaluation outputs are always preserved.

Example:

    .venv/bin/python eval_main.py \
        --checkpoint checkpoints/self_evolver_opsd/experiment/global_step_200 \
        --model-name experiment-step200 \
        --benchmarks bfcl,tau2,lifelong,coding
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable
from urllib.request import urlopen


PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
SERVICE_SCRIPT = PROJECT_ROOT / "start_services.sh"
TEMP_BASE = PROJECT_ROOT / "workspace" / "eval_tmp"
OUTPUT_BASE = PROJECT_ROOT / "workspace" / "eval_logs" / "eval_main"
OWNERSHIP_MARKER = ".eval_main_owned"

ALL_BENCHMARKS = (
    "bfcl_v3",
    "tau2_airline",
    "tau2_retail",
    "lifelong_db",
    "lifelong_os",
    "humaneval_plus",
    "mbpp_plus",
    "livecodebench_v5",
    "livecodebench_v6",
)

BENCHMARK_ALIASES = {
    "all": ALL_BENCHMARKS,
    "bfcl": ("bfcl_v3",),
    "bfcl_v3": ("bfcl_v3",),
    "tau2": ("tau2_airline", "tau2_retail"),
    "tau2_airline": ("tau2_airline",),
    "tau2_retail": ("tau2_retail",),
    "lifelong": ("lifelong_db", "lifelong_os"),
    "lifelong_db": ("lifelong_db",),
    "lifelong_os": ("lifelong_os",),
    "coding": ("humaneval_plus", "mbpp_plus", "livecodebench_v5", "livecodebench_v6"),
    "humaneval+": ("humaneval_plus",),
    "humaneval_plus": ("humaneval_plus",),
    "mbpp+": ("mbpp_plus",),
    "mbpp_plus": ("mbpp_plus",),
    "livecodebench": ("livecodebench_v5", "livecodebench_v6"),
    "lcb": ("livecodebench_v5", "livecodebench_v6"),
    "lcb_v5": ("livecodebench_v5",),
    "lcb_v6": ("livecodebench_v6",),
    "livecodebench_v5": ("livecodebench_v5",),
    "livecodebench_v6": ("livecodebench_v6",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="global_step_* 或其 actor 目录；相对路径以项目根目录为准。",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help="LoRA base model。默认从导出的 adapter_config.json 自动读取。",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="vLLM served model name。默认根据实验目录和 global step 生成。",
    )
    parser.add_argument(
        "--benchmarks",
        default="all",
        help="逗号分隔：bfcl,tau2,lifelong,coding；coding 包含 "
             "humaneval+ / mbpp+ / LCB v5 / LCB v6。",
    )
    parser.add_argument("--run-name", default=None, help="输出目录名；默认自动生成。")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--ready-timeout", type=int, default=600)
    parser.add_argument("--bfcl-concurrency", type=int, default=64)
    parser.add_argument("--tau2-concurrency", type=int, default=64)
    parser.add_argument("--lifelong-concurrency", type=int, default=64)
    parser.add_argument("--coding-concurrency", type=int, default=8)
    parser.add_argument("--coding-eval-workers", type=int, default=4)
    parser.add_argument("--coding-n-samples", type=int, default=1)
    parser.add_argument("--coding-max-completion-tokens", type=int, default=16384)
    parser.add_argument("--tau2-trials", type=int, default=1)
    parser.add_argument("--tau2-max-steps", type=int, default=200)
    parser.add_argument("--max-completion-tokens", type=int, default=4096)
    parser.add_argument(
        "--tau2-judge-model",
        default=None,
        help="可选：传给 run_eval 的 tau2 Retail NL judge 模型。",
    )
    parser.add_argument("--tau2-judge-base-url", default=None)
    parser.add_argument("--tau2-judge-api-key-env", default="ARK_API_KEY")
    parser.add_argument(
        "--keep-merged",
        action="store_true",
        help="评测成功后仍保留临时 merged model（默认删除）。",
    )
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def actor_dir_from_checkpoint(checkpoint: Path) -> Path:
    if checkpoint.name == "actor":
        actor_dir = checkpoint
    else:
        actor_dir = checkpoint / "actor"
    if not actor_dir.is_dir():
        raise FileNotFoundError(f"未找到 actor checkpoint 目录: {actor_dir}")
    return actor_dir


def slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-._")
    return value or "eval"


def default_identity(actor_dir: Path) -> tuple[str, str]:
    step_dir = actor_dir.parent
    experiment_dir = step_dir.parent
    identity = slug(f"{experiment_dir.name}-{step_dir.name}")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return identity, f"{identity}-{stamp}"


def expand_benchmarks(raw: str) -> list[str]:
    selected: list[str] = []
    for item in raw.replace(" ", ",").split(","):
        item = item.strip().lower()
        if not item:
            continue
        if item not in BENCHMARK_ALIASES:
            choices = ", ".join(sorted(BENCHMARK_ALIASES))
            raise ValueError(f"未知 benchmark {item!r}；可选: {choices}")
        for benchmark in BENCHMARK_ALIASES[item]:
            if benchmark not in selected:
                selected.append(benchmark)
    if not selected:
        raise ValueError("--benchmarks 不能为空")
    return selected


def run_command(command: Iterable[str], *, env: dict[str, str] | None = None) -> None:
    command = [str(part) for part in command]
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)


def is_huggingface_model(path: Path) -> bool:
    has_weights = any(path.glob("*.safetensors")) or any(path.glob("*.bin"))
    return (path / "config.json").is_file() and has_weights


def infer_base_model(adapter_dir: Path, explicit_base_model: str | None) -> Path:
    if explicit_base_model:
        base_model = resolve_path(explicit_base_model)
    else:
        adapter_config = json.loads((adapter_dir / "adapter_config.json").read_text())
        raw_base_model = adapter_config.get("base_model_name_or_path")
        if not raw_base_model:
            raise ValueError(
                "adapter_config.json 没有 base_model_name_or_path，请传 --base-model"
            )
        base_model = resolve_path(raw_base_model)
    if not base_model.is_dir():
        raise FileNotFoundError(f"LoRA base model 不存在: {base_model}")
    return base_model


def merge_lora(adapter_dir: Path, base_model: Path, output_dir: Path) -> None:
    import torch
    from peft import PeftModel
    from transformers import (
        AutoConfig,
        AutoModelForImageTextToText,
        AutoProcessor,
        AutoTokenizer,
    )

    print(f"[merge] base model: {base_model}", flush=True)
    print(f"[merge] adapter:    {adapter_dir}", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        base_model,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(model, adapter_dir)
    model = model.merge_and_unload(safe_merge=True)
    output_dir.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size="5GB",
    )

    # Save tokenizer/processor assets without copying the base model weights.
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.save_pretrained(output_dir)
    try:
        processor = AutoProcessor.from_pretrained(base_model)
        processor.save_pretrained(output_dir)
    except (OSError, ValueError) as exc:
        print(f"[merge] processor 未单独保存（模型可能是纯文本模型）: {exc}")

    AutoConfig.from_pretrained(output_dir)
    del tokenizer, model
    if "processor" in locals():
        del processor
    gc.collect()


def prepare_model(
    actor_dir: Path,
    temp_root: Path,
    base_model_arg: str | None,
) -> tuple[Path, bool]:
    """Return (model_path, model_is_owned_temporary_artifact)."""
    if is_huggingface_model(actor_dir):
        print(f"[merge] actor 已是 Hugging Face 模型，直接使用: {actor_dir}")
        return actor_dir, False

    temp_root.mkdir(parents=True, exist_ok=False)
    marker = temp_root / OWNERSHIP_MARKER
    marker.write_text(
        json.dumps({"actor_dir": str(actor_dir)}, ensure_ascii=False, indent=2)
    )
    export_dir = temp_root / "export"
    run_command(
        [
            PYTHON,
            "-m",
            "verl.model_merger",
            "merge",
            "--backend",
            "fsdp",
            "--local_dir",
            actor_dir,
            "--target_dir",
            export_dir,
        ]
    )

    adapter_dir = export_dir / "lora_adapter"
    if (adapter_dir / "adapter_config.json").is_file():
        base_model = infer_base_model(adapter_dir, base_model_arg)
        merged_dir = temp_root / "merged_model"
        merge_lora(adapter_dir, base_model, merged_dir)
        shutil.rmtree(export_dir)
        if not is_huggingface_model(merged_dir):
            raise RuntimeError(f"LoRA merge 产物不完整: {merged_dir}")
        return merged_dir, True

    if not is_huggingface_model(export_dir):
        raise RuntimeError(
            f"VERL merger 既未生成完整模型，也未生成 LoRA adapter: {export_dir}"
        )
    return export_dir, True


def service_environment(model_path: Path, model_name: str, port: int, timeout: int) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "PROJECT_ROOT": str(PROJECT_ROOT),
            "MODEL_PATH": str(model_path),
            "MODEL_NAME": model_name,
            "VLLM_PORT": str(port),
            "VLLM_READY_TIMEOUT": str(timeout),
        }
    )
    return env


def verify_service(model_path: Path, model_name: str, port: int) -> None:
    with urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=10) as response:
        payload = json.loads(response.read().decode())
    models = {item.get("id"): item for item in payload.get("data", [])}
    if model_name not in models:
        raise RuntimeError(
            f"vLLM model id 不匹配：期望 {model_name!r}，实际 {list(models)}"
        )
    served_root = models[model_name].get("root")
    if served_root and resolve_path(served_root) != model_path.resolve():
        raise RuntimeError(
            f"vLLM model root 不匹配：期望 {model_path}，实际 {served_root}"
        )
    print(f"[service] verified: {model_name} -> {model_path}")


def wait_service_down(port: int, timeout: int = 30) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=1):
                pass
        except OSError:
            return True
        time.sleep(1)
    return False


def common_eval_args(args: argparse.Namespace, model_name: str) -> list[str]:
    return [
        str(PYTHON),
        "-m",
        "benchmark.eval.run_eval",
        "--openai-base-url",
        f"http://127.0.0.1:{args.port}/v1",
        "--model",
        model_name,
        "--api-key",
        args.api_key,
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--llm-max-completion-tokens",
        str(args.max_completion_tokens),
    ]


def tau2_judge_args(args: argparse.Namespace) -> list[str]:
    result = ["--tau2-judge-api-key-env", args.tau2_judge_api_key_env]
    if args.tau2_judge_model:
        result += ["--tau2-judge-model", args.tau2_judge_model]
    if args.tau2_judge_base_url:
        result += ["--tau2-judge-base-url", args.tau2_judge_base_url]
    return result


def benchmark_command(
    benchmark: str,
    args: argparse.Namespace,
    model_name: str,
    model_path: Path,
    output_root: Path,
) -> list[str]:
    command = common_eval_args(args, model_name)
    if benchmark == "bfcl_v3":
        return command + [
            "--benchmark",
            "bfcl_v3",
            "--bfcl-model-path",
            str(model_path),
            "--bfcl-categories",
            "all",
            "--concurrency",
            str(args.bfcl_concurrency),
            "--output-dir",
            str(output_root / "bfcl_v3" / "all"),
        ]
    if benchmark.startswith("tau2_"):
        domain = benchmark.removeprefix("tau2_")
        return command + [
            "--benchmark",
            "tau2",
            "--tau2-domain",
            domain,
            "--tau2-split",
            "base",
            "--tau2-num-trials",
            str(args.tau2_trials),
            "--tau2-max-steps",
            str(args.tau2_max_steps),
            "--concurrency",
            str(args.tau2_concurrency),
            "--output-dir",
            str(output_root / "tau2" / domain),
        ] + tau2_judge_args(args)
    if benchmark == "lifelong_db":
        return command + [
            "--benchmark",
            "lifelong_db",
            "--data-dir",
            "benchmark/LifelongAgentBench",
            "--split",
            "test",
            "--max-steps",
            "6",
            "--step-timeout",
            "180",
            "--concurrency",
            str(args.lifelong_concurrency),
            "--mysql-image",
            "mysql:8.0",
            "--output-dir",
            str(output_root / "lifelong_db" / "test"),
        ]
    if benchmark == "lifelong_os":
        return command + [
            "--benchmark",
            "lifelong_os",
            "--data-dir",
            "benchmark/LifelongAgentBench",
            "--split",
            "test",
            "--max-steps",
            "8",
            "--step-timeout",
            "180",
            "--os-timeout",
            "20",
            "--concurrency",
            str(args.lifelong_concurrency),
            "--output-dir",
            str(output_root / "lifelong_os" / "test"),
        ]
    if benchmark in {"humaneval_plus", "mbpp_plus"}:
        command[command.index("--llm-max-completion-tokens") + 1] = str(
            args.coding_max_completion_tokens
        )
        public_name = "humaneval+" if benchmark == "humaneval_plus" else "mbpp+"
        return command + [
            "--benchmark",
            public_name,
            "--concurrency",
            str(args.coding_concurrency),
            "--code-eval-workers",
            str(args.coding_eval_workers),
            "--code-n-samples",
            str(args.coding_n_samples),
            "--output-dir",
            str(output_root / public_name),
        ]
    if benchmark in {"livecodebench_v5", "livecodebench_v6"}:
        command[command.index("--llm-max-completion-tokens") + 1] = str(
            args.coding_max_completion_tokens
        )
        release = benchmark.removeprefix("livecodebench_")
        return command + [
            "--benchmark",
            "livecodebench",
            "--lcb-release",
            release,
            "--concurrency",
            str(args.coding_concurrency),
            "--code-eval-workers",
            str(args.coding_eval_workers),
            "--code-n-samples",
            str(args.coding_n_samples),
            "--output-dir",
            str(output_root / "livecodebench" / release),
        ]
    raise AssertionError(f"未实现 benchmark: {benchmark}")


def safe_cleanup(temp_root: Path) -> None:
    resolved_base = TEMP_BASE.resolve()
    resolved_root = temp_root.resolve()
    if resolved_root.parent != resolved_base:
        raise RuntimeError(f"拒绝删除 TEMP_BASE 直属目录以外的路径: {resolved_root}")
    if not (resolved_root / OWNERSHIP_MARKER).is_file():
        raise RuntimeError(f"拒绝删除没有 ownership marker 的目录: {resolved_root}")
    shutil.rmtree(resolved_root)
    print(f"[cleanup] 已删除推理临时 merged model: {resolved_root}")


def write_run_config(
    output_root: Path,
    *,
    args: argparse.Namespace,
    actor_dir: Path,
    model_path: Path,
    model_name: str,
    benchmarks: list[str],
) -> None:
    output_root.mkdir(parents=True, exist_ok=False)
    payload = {
        "created_at": datetime.now().isoformat(),
        "actor_dir": str(actor_dir),
        "inference_model_path": str(model_path),
        "served_model_name": model_name,
        "benchmarks": benchmarks,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "tau2_trials": args.tau2_trials,
        "coding_n_samples": args.coding_n_samples,
        "coding_concurrency": args.coding_concurrency,
        "coding_eval_workers": args.coding_eval_workers,
        "coding_max_completion_tokens": args.coding_max_completion_tokens,
        "tau2_agent_thinking": False,
        "tau2_user_thinking": False,
    }
    (output_root / "eval_config.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    )


def main() -> int:
    args = parse_args()
    if not PYTHON.is_file():
        raise FileNotFoundError(f"项目 Python 不存在: {PYTHON}")
    if not SERVICE_SCRIPT.is_file():
        raise FileNotFoundError(f"服务脚本不存在: {SERVICE_SCRIPT}")

    checkpoint = resolve_path(args.checkpoint)
    actor_dir = actor_dir_from_checkpoint(checkpoint)
    default_model_name, default_run_name = default_identity(actor_dir)
    model_name = slug(args.model_name or default_model_name)
    run_name = slug(args.run_name or default_run_name)
    benchmarks = expand_benchmarks(args.benchmarks)
    temp_root = TEMP_BASE / run_name
    output_root = OUTPUT_BASE / run_name

    if temp_root.exists():
        raise FileExistsError(f"临时目录已存在，请更换 --run-name: {temp_root}")
    if output_root.exists():
        raise FileExistsError(f"评测输出目录已存在，请更换 --run-name: {output_root}")

    print(f"[eval] actor:      {actor_dir}")
    print(f"[eval] model name: {model_name}")
    print(f"[eval] benchmarks: {', '.join(benchmarks)}")
    print(f"[eval] outputs:    {output_root}")

    model_path, owned_temporary_model = prepare_model(
        actor_dir,
        temp_root,
        args.base_model,
    )
    write_run_config(
        output_root,
        args=args,
        actor_dir=actor_dir,
        model_path=model_path,
        model_name=model_name,
        benchmarks=benchmarks,
    )

    service_env = service_environment(
        model_path,
        model_name,
        args.port,
        args.ready_timeout,
    )
    service_attempted = False
    service_stopped = True
    succeeded = False
    try:
        service_attempted = True
        run_command([SERVICE_SCRIPT, "restart", "vllm"], env=service_env)
        verify_service(model_path, model_name, args.port)

        for index, benchmark in enumerate(benchmarks, start=1):
            print(f"\n[eval] ({index}/{len(benchmarks)}) running {benchmark}")
            run_command(
                benchmark_command(
                    benchmark,
                    args,
                    model_name,
                    model_path,
                    output_root,
                )
            )
        succeeded = True
    finally:
        # vLLM must release all model files before the temporary directory is removed.
        if service_attempted:
            stop_result = subprocess.run(
                [str(SERVICE_SCRIPT), "stop", "vllm"],
                cwd=PROJECT_ROOT,
                env=service_env,
                check=False,
            )
            service_stopped = stop_result.returncode == 0 and wait_service_down(args.port)

    if succeeded and owned_temporary_model and not args.keep_merged:
        if not service_stopped:
            raise RuntimeError(
                f"vLLM 未完全停止，为避免删除仍在使用的模型，临时目录已保留: {temp_root}"
            )
        safe_cleanup(temp_root)
    elif owned_temporary_model:
        print(f"[cleanup] 临时 merged model 已保留: {model_path}")

    print(f"\n[done] evaluation outputs: {output_root}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[error] 用户中断；评测产物与临时 merged model 已保留。", file=sys.stderr)
        raise SystemExit(130)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"\n[error] {exc}", file=sys.stderr)
        raise SystemExit(1)
