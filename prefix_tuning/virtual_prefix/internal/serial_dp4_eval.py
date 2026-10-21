#!/usr/bin/env python3
"""Evaluate the composed three-domain prefix serially with one four-way DP vLLM.

This orchestrator starts only after the training queue reaches
``awaiting_evaluation``.  It owns all four visible GPUs, evaluates the ordered
concatenation of the three final domain prefixes, stops the service, and
verifies GPU release.  Single-domain results from the previous serial run are
reused for comparison; only the composed model is evaluated here.
No symlinks are created.
"""
from __future__ import annotations

import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[3]
EXP = Path(
    os.environ.get(
        "EXPERIMENT_ROOT",
        ROOT / "outputs/prefix_composition_3domain_20260922_4gpu",
    )
).resolve()
PYTHON = Path(os.environ.get("KVCACHE_PYTHON", "/usr/local/miniconda3/envs/kvcache/bin/python"))
BENCHMARK = ROOT / "benchmark/run.sh"
EVAL_ROOT = EXP / "eval"
STATE = EVAL_ROOT / "serial_dp4_state.json"
LOCK = EVAL_ROOT / "serial_dp4.lock"

DOMAINS = ("awm", "envscaler", "taco")
MODELS = (
    {
        "name": "awm_envscaler_taco",
        "alias": "prefix-awm-envscaler-taco",
        "temperature": 0.6,
        "composed": True,
    },
)
CODE_SUITES = (
    ("humaneval+", 164),
    ("mbpp+", 378),
    ("livecodebench", 1055),
)
PORT = 18140
DP_RPC_PORT = 18150
CODE_API_CONCURRENCY = 16
CODE_EVAL_WORKERS = 8
BFCL_CONCURRENCY = 24
TAU2_CONCURRENCY = 2


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


class RunState:
    def __init__(self, run_root: Path):
        self.value = {
            "state": "running",
            "pid": os.getpid(),
            "started_at": now(),
            "updated_at": now(),
            "run_root": str(run_root),
            "models": [item["name"] for item in MODELS],
            "temperatures": {item["name"]: item["temperature"] for item in MODELS},
            "completed_models": [],
            "completed_jobs": [],
            "current_model": None,
            "current_job": None,
            "service": None,
        }
        self.flush()

    def update(self, **changes) -> None:
        self.value.update(changes)
        self.value["updated_at"] = now()
        self.flush()

    def flush(self) -> None:
        atomic_json(STATE, self.value)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def validate_training_complete() -> None:
    status_path = EXP / "queue_status.json"
    if not status_path.is_file():
        raise RuntimeError(f"Missing training status: {status_path}")
    status = read_json(status_path)
    if status.get("state") != "awaiting_evaluation":
        raise RuntimeError(f"Training is not ready for evaluation: state={status.get('state')!r}")
    if status.get("failed"):
        raise RuntimeError(f"Training queue recorded failures: {status['failed']}")
    if set(status.get("completed", [])) != set(DOMAINS):
        raise RuntimeError(f"Unexpected completed domains: {status.get('completed')}")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_composed_prefix(model: dict) -> Path:
    """Validate the ordered-concatenation model against the source prefixes."""
    path = EXP / "models" / model["name"]
    for filename in ("config.json", "prefix_composition_manifest.json"):
        if not (path / filename).is_file():
            raise FileNotFoundError(path / filename)
    manifest = read_json(path / "prefix_composition_manifest.json")
    if manifest.get("method") != "ordered_token_concatenation":
        raise RuntimeError(f"Unexpected composition method: {manifest.get('method')!r}")
    expected_inputs = [str((EXP / "prefixes" / domain).resolve()) for domain in DOMAINS]
    ordered_inputs = [str(Path(p).resolve()) for p in manifest.get("ordered_inputs", [])]
    if ordered_inputs != expected_inputs:
        raise RuntimeError(
            f"Composed model input order mismatch: {ordered_inputs} != {expected_inputs}"
        )
    hashes = manifest.get("prefix_sha256")
    if not isinstance(hashes, list) or len(hashes) != len(DOMAINS):
        raise RuntimeError(f"Invalid prefix_sha256 manifest: {hashes!r}")
    for domain, expected in zip(DOMAINS, hashes):
        for filename, digest in expected.items():
            source = EXP / "prefixes" / domain / filename
            if not source.is_file():
                raise FileNotFoundError(source)
            if sha256_of(source) != digest:
                raise RuntimeError(
                    f"{model['name']} is stale: {source} no longer matches the "
                    f"composed hash {digest}"
                )
    config = read_json(path / "config.json")
    hybrid = config.get("hybrid_prefix", {})
    domain_tokens = [
        read_json(EXP / "prefixes" / domain / "config.json").get("hybrid_prefix", {})
        for domain in DOMAINS
    ]
    expected_delta = sum(
        item.get("delta_prefix_num_virtual_tokens", 0) for item in domain_tokens
    )
    expected_attention = sum(
        item.get("attention_prefix_num_virtual_tokens", 0) for item in domain_tokens
    )
    if hybrid.get("delta_prefix_num_virtual_tokens") != expected_delta or hybrid.get(
        "attention_prefix_num_virtual_tokens"
    ) != expected_attention:
        raise RuntimeError(
            f"Composed token counts mismatch: {hybrid} "
            f"(expected delta={expected_delta}, attention={expected_attention})"
        )
    return path


def validate_prefix(model: dict) -> Path:
    if model.get("composed"):
        return validate_composed_prefix(model)
    path = EXP / "prefixes" / model["name"]
    for filename in ("config.json", "prefix_merge_manifest.json"):
        if not (path / filename).is_file():
            raise FileNotFoundError(path / filename)
    manifest = read_json(path / "prefix_merge_manifest.json")
    source = Path(manifest.get("source_checkpoint", ""))
    expected = f"global_step_{model['step']}"
    checkpoint_name = source.parent.name if source.name == "actor" else source.name
    if checkpoint_name != expected:
        raise RuntimeError(
            f"{model['name']} prefix came from {checkpoint_name!r}, expected {expected!r}"
        )
    metadata = source / "trainable_only_meta.json" if source.name == "actor" else source / "actor/trainable_only_meta.json"
    if not metadata.is_file():
        raise FileNotFoundError(metadata)
    return path


def validate_benchmark_runtime() -> None:
    required = (
        BENCHMARK,
        ROOT / "benchmark/.venv/bin/python",
        ROOT / "benchmark/coding/.venv/bin/python",
        ROOT / "benchmark/tau2/.venv/bin/python",
        ROOT / "benchmark/runtime/bin/nsjail",
    )
    for path in required:
        if not path.is_file() or not os.access(path, os.X_OK):
            raise RuntimeError(f"Benchmark runtime is not executable: {path}")
    if not (ROOT / ".env").is_file():
        raise RuntimeError(f"Missing {ROOT / '.env'}; tau2 external user/judge credentials required")


def service_environment() -> dict[str, str]:
    no_proxy = os.environ.get("NO_PROXY", "")
    no_proxy = ",".join(item for item in (no_proxy, "localhost", "127.0.0.1") if item)
    return {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "0,1,2,3",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
        "NUMEXPR_NUM_THREADS": "2",
        "VECLIB_MAXIMUM_THREADS": "2",
        "HF_HUB_OFFLINE": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "PYTHONPATH": f"{ROOT}:{os.environ.get('PYTHONPATH', '')}",
        "NO_PROXY": no_proxy,
        "no_proxy": no_proxy,
        "LD_PRELOAD": "/usr/lib/x86_64-linux-gnu/libstdc++.so.6",
    }


def benchmark_environment() -> dict[str, str]:
    """Keep benchmark/tool workers from starving tool calls of CPU."""
    return {
        **os.environ,
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    }


def service_command(model: dict, model_path: Path) -> list[str]:
    return [
        str(PYTHON),
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(model_path),
        "--served-model-name",
        model["alias"],
        "--host",
        "127.0.0.1",
        "--port",
        str(PORT),
        "--trust-remote-code",
        "--model-impl",
        "vllm",
        "--dtype",
        "bfloat16",
        "--tensor-parallel-size",
        "1",
        "--data-parallel-size",
        "4",
        "--data-parallel-size-local",
        "4",
        "--data-parallel-backend",
        "mp",
        "--data-parallel-rpc-port",
        str(DP_RPC_PORT),
        "--gpu-memory-utilization",
        "0.80",
        "--max-model-len",
        "65536",
        "--max-num-seqs",
        "512",
        "--max-num-batched-tokens",
        "131072",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "qwen3_coder",
        "--reasoning-parser",
        "qwen3",
        "--language-model-only",
        "--generation-config",
        "vllm",
        "--default-chat-template-kwargs",
        '{"enable_thinking":false}',
    ]


def api_request(path: str, payload: dict | None = None, timeout: float = 10) -> dict:
    headers = {"Content-Type": "application/json", "Authorization": "Bearer EMPTY"}
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{PORT}{path}", data=data, headers=headers
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        return json.load(response)


def wait_for_service(process: subprocess.Popen, alias: str, timeout: int = 900) -> None:
    deadline = time.monotonic() + timeout
    last_error = "not queried"
    while time.monotonic() < deadline:
        code = process.poll()
        if code is not None:
            raise RuntimeError(f"vLLM exited during startup with code {code}")
        try:
            payload = api_request("/v1/models", timeout=3)
            ids = [item["id"] for item in payload.get("data", [])]
            if alias in ids:
                smoke = api_request(
                    "/v1/chat/completions",
                    {
                        "model": alias,
                        "messages": [{"role": "user", "content": "Reply with OK."}],
                        "temperature": 0,
                        "max_tokens": 16,
                    },
                    timeout=120,
                )
                if smoke.get("choices"):
                    return
                last_error = f"empty smoke response: {smoke}"
        except (OSError, ValueError, urllib.error.URLError) as exc:
            last_error = repr(exc)
        time.sleep(3)
    raise TimeoutError(f"vLLM did not become healthy: {last_error}")


def stop_service(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30)


def wait_for_gpu_release(timeout: int = 180) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
            check=True,
        )
        memory = [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]
        if len(memory) >= 4 and all(value < 1024 for value in memory[:4]):
            return
        time.sleep(3)
    raise TimeoutError("GPU memory was not released after stopping vLLM")


def run_logged(state: RunState, model: dict, name: str, command: list[str], log: Path) -> None:
    state.update(current_model=model["name"], current_job=name)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as handle:
        handle.write(f"\n[{now()}] command={json.dumps(command)}\n")
        handle.flush()
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=benchmark_environment(),
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    if result.returncode:
        raise RuntimeError(f"{model['name']} {name} failed with code {result.returncode}; see {log}")
    state.value["completed_jobs"].append(f"{model['name']}:{name}")
    state.update(current_job=None)


def benchmark_base(model: dict, output: Path) -> list[str]:
    return [
        "bash",
        str(BENCHMARK),
        "--openai-base-url",
        f"http://127.0.0.1:{PORT}/v1",
        "--model",
        model["alias"],
        "--temperature",
        str(model["temperature"]),
        "--top-p",
        "1.0",
        "--output-dir",
        str(output),
    ]


def validate_code_output(output: Path, expected: int, model: dict) -> None:
    summary = read_json(output / "summary.json")
    manifest = read_json(output / "manifest.json")
    if summary.get("tasks") != expected or summary.get("samples") != expected:
        raise RuntimeError(f"Incomplete coding output in {output}: {summary}")
    generation_errors = int(summary.get("generation_errors") or 0)
    if generation_errors:
        # A per-sample generation failure (for example, exhausting the token
        # budget before producing a final answer) is a valid benchmark outcome.
        # The evaluator already counts it as an incorrect answer, so retain the
        # result and continue the serial suite instead of treating it as an
        # infrastructure failure.
        print(
            f"WARNING: retaining {generation_errors} generation errors in {output}; "
            "they are included as failed samples in the benchmark score",
            flush=True,
        )
    if float(manifest.get("temperature")) != float(model["temperature"]):
        raise RuntimeError(f"Temperature mismatch in {output / 'manifest.json'}")


def evaluate_model(state: RunState, run_root: Path, model: dict, model_path: Path) -> None:
    model_root = run_root / model["name"]
    logs = run_root / "logs"
    service_log = logs / f"{model['name']}_service.log"
    command = service_command(model, model_path)
    service_log.parent.mkdir(parents=True, exist_ok=True)
    with service_log.open("a") as handle:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=service_environment(),
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    state.update(
        current_model=model["name"],
        current_job="service_start",
        service={
            "pid": process.pid,
            "port": PORT,
            "gpus": "0,1,2,3",
            "tensor_parallel_size": 1,
            "data_parallel_size": 4,
            "model": str(model_path),
            "alias": model["alias"],
            "log": str(service_log),
            "command": command,
        },
    )
    try:
        wait_for_service(process, model["alias"])
        state.update(current_job=None)

        for suite, expected in CODE_SUITES:
            output = model_root / suite
            cmd = benchmark_base(model, output)
            cmd[2:2] = ["--benchmark", suite]
            if suite == "livecodebench":
                cmd += ["--lcb-release", "v6"]
            cmd += [
                "--code-execution-backend",
                "nsjail",
                "--code-thinking",
                "on",
                "--code-pass-k",
                "1",
                "--code-n-samples",
                "1",
                "--code-seed",
                "42",
                "--code-eval-workers",
                str(CODE_EVAL_WORKERS),
                "--code-memory",
                "64g",
                "--code-eval-timeout",
                "7200",
                "--code-api-timeout",
                "1800",
                "--llm-max-completion-tokens",
                "16384",
                "--concurrency",
                str(CODE_API_CONCURRENCY),
            ]
            run_logged(state, model, suite, cmd, logs / f"{model['name']}_{suite}.log")
            validate_code_output(output, expected, model)

        bfcl_output = model_root / "bfcl_v3"
        cmd = benchmark_base(model, bfcl_output)
        cmd[2:2] = ["--benchmark", "bfcl_v3"]
        cmd += [
            "--bfcl-model-path",
            str(model_path),
            "--bfcl-categories",
            "all",
            "--concurrency",
            str(BFCL_CONCURRENCY),
        ]
        run_logged(state, model, "bfcl_v3", cmd, logs / f"{model['name']}_bfcl_v3.log")
        if not (bfcl_output / "score/data_overall.csv").is_file():
            raise RuntimeError(f"Missing BFCL aggregate score: {bfcl_output / 'score/data_overall.csv'}")

        for domain, expected in (("airline", 50), ("retail", 114)):
            output = model_root / f"tau2_{domain}"
            cmd = benchmark_base(model, output)
            cmd[2:2] = ["--benchmark", "tau2"]
            cmd += [
                "--tau2-domain",
                domain,
                "--tau2-split",
                "base",
                "--tau2-num-trials",
                "1",
                "--tau2-max-steps",
                "200",
                "--tau2-require-external-user",
                "--tau2-require-external-judge",
                "--llm-max-completion-tokens",
                "16384",
                "--concurrency",
                str(TAU2_CONCURRENCY),
            ]
            run_logged(state, model, f"tau2_{domain}", cmd, logs / f"{model['name']}_tau2_{domain}.log")
            results = read_json(output / "results.json")
            tasks = len(results.get("tasks", []))
            simulations = len(results.get("simulations", []))
            if tasks != expected or simulations != expected:
                raise RuntimeError(
                    f"Incomplete tau2 {domain}: tasks={tasks}, simulations={simulations}, expected={expected}"
                )

        state.value["completed_models"].append(model["name"])
        state.update(current_model=None, current_job=None)
    finally:
        stop_service(process)
        state.update(service=None)
        wait_for_gpu_release()


def run() -> int:
    EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    with LOCK.open("w") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"Serial DP=4 evaluation is already running; lock={LOCK}")
            return 0

        validate_training_complete()
        paths = {model["name"]: validate_prefix(model) for model in MODELS}
        validate_benchmark_runtime()

        stamp = time.strftime("%Y%m%d_%H%M%S")
        run_root = EVAL_ROOT / f"serial_dp4_{stamp}"
        run_root.mkdir(parents=True)
        manifest = {
            "started_at": now(),
            "experiment_root": str(EXP),
            "models": MODELS,
            "service": {
                "tensor_parallel_size": 1,
                "data_parallel_size": 4,
                "gpus": "0,1,2,3",
            },
            "scope": {
                "humaneval+": 164,
                "mbpp+": 378,
                "livecodebench_v6": 1055,
                "bfcl_v3": 4441,
                "tau2": {"airline": 50, "retail": 114},
            },
            "sampling": {
                **{item["name"]: {"temperature": item["temperature"], "top_p": 1.0} for item in MODELS},
                "code_seed": 42,
                "code_pass_k": 1,
            },
            "cpu_safety": {
                "visible_cpus_at_launch": os.cpu_count(),
                "service_threads_per_process": 2,
                "benchmark_threads_per_process": 1,
                "code_api_concurrency": CODE_API_CONCURRENCY,
                "code_eval_workers": CODE_EVAL_WORKERS,
                "bfcl_concurrency": BFCL_CONCURRENCY,
                "tau2_concurrency": TAU2_CONCURRENCY,
            },
            "no_symlinks_created": True,
        }
        atomic_json(run_root / "run_manifest.json", manifest)
        state = RunState(run_root)
        try:
            for model in MODELS:
                evaluate_model(state, run_root, model, paths[model["name"]])
            state.update(state="complete", completed_at=now(), current_model=None, current_job=None)
            return 0
        except BaseException as exc:
            state.update(
                state="failed",
                failed_at=now(),
                error=repr(exc),
                traceback=traceback.format_exc(),
            )
            raise


if __name__ == "__main__":
    raise SystemExit(run())
