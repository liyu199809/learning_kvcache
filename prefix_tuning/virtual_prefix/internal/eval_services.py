#!/usr/bin/env python3
"""Start four two-GPU local-DP vLLM services for the formal rerun."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
EXP = Path(os.environ.get(
    "EXPERIMENT_ROOT", ROOT / "outputs/prefix_composition_3domain_20260917"
)).resolve()
STATE = EXP / "eval/active_prefix_eval_services.json"
SPECS = (
    ("awm", "0,1", 18140, 18150, "prefix-awm", EXP / "prefixes/awm"),
    ("envscaler", "2,3", 18142, 18152, "prefix-envscaler", EXP / "prefixes/envscaler"),
    ("taco", "4,5", 18144, 18154, "prefix-taco", EXP / "prefixes/taco"),
    ("base", "6,7", 18146, 18156, "base-qwen3.5-4b", Path("/home/liyu.468/models/Qwen3.5-4B")),
)


def live(pid):
    return bool(pid and Path(f"/proc/{pid}/cmdline").exists())


def stop(records):
    for record in records:
        pid = record["pid"]
        if live(pid):
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
            if "vllm.entrypoints.openai.api_server" not in cmd:
                raise RuntimeError(f"PID {pid} no longer belongs to vLLM")
            os.killpg(pid, signal.SIGTERM)


def start_services():
    if STATE.exists():
        previous = json.loads(STATE.read_text())
        if any(live(item.get("pid")) for item in previous.get("services", [])):
            raise SystemExit(f"Live prefix evaluation services already recorded in {STATE}")
    for name, _, _, _, _, model in SPECS:
        required = ["config.json"]
        if name != "base":
            required.append("prefix_merge_manifest.json")
        for filename in required:
            if not (model / filename).is_file():
                raise FileNotFoundError(model / filename)
    log_dir = EXP / "eval/service_logs_rerun"
    log_dir.mkdir(parents=True, exist_ok=True)
    services = []
    try:
        for name, gpus, port, dp_rpc_port, alias, model in SPECS:
            command = [
                "/usr/local/miniconda3/envs/kvcache/bin/python", "-m",
                "vllm.entrypoints.openai.api_server", "--model", str(model),
                "--served-model-name", alias, "--host", "127.0.0.1", "--port", str(port),
                "--trust-remote-code", "--model-impl", "vllm", "--dtype", "bfloat16",
                "--tensor-parallel-size", "1", "--data-parallel-size", "2",
                "--data-parallel-size-local", "2", "--data-parallel-backend", "mp",
                "--data-parallel-rpc-port", str(dp_rpc_port),
                "--gpu-memory-utilization", "0.80", "--max-model-len", "65536",
                "--max-num-seqs", "512", "--max-num-batched-tokens", "131072",
                "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder",
                "--reasoning-parser", "qwen3", "--language-model-only",
                "--generation-config", "vllm",
                "--default-chat-template-kwargs", '{"enable_thinking":false}',
            ]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus, TOKENIZERS_PARALLELISM="false",
                       HF_HUB_OFFLINE="1", VLLM_WORKER_MULTIPROC_METHOD="spawn",
                       PYTHONPATH=f"{ROOT}:{os.environ.get('PYTHONPATH', '')}",
                       NO_PROXY="localhost,127.0.0.1,6.5.85.147", no_proxy="localhost,127.0.0.1,6.5.85.147",
                       LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6")
            log_path = log_dir / f"{name}.log"
            handle = log_path.open("a")
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                       stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            handle.close()
            services.append(dict(name=name, gpus=gpus, tensor_parallel_size=1,
                                 data_parallel_size=2, data_parallel_rpc_port=dp_rpc_port,
                                 enforce_eager=False, max_num_seqs=512,
                                 max_model_len=65536,
                                 max_num_batched_tokens=131072,
                                 port=port, alias=alias, model=str(model),
                                 pid=process.pid, log=str(log_path), command=command))
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(json.dumps({"started_at": time.time(), "services": services}, indent=2) + "\n")
        deadline = time.monotonic() + 900
        pending = {item["name"] for item in services}
        while pending and time.monotonic() < deadline:
            for item in services:
                if item["name"] not in pending:
                    continue
                if not live(item["pid"]):
                    raise RuntimeError(f"{item['name']} exited during startup; see {item['log']}")
                try:
                    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                    with opener.open(f"http://127.0.0.1:{item['port']}/v1/models", timeout=3) as response:
                        payload = json.load(response)
                    if not any(model["id"] == item["alias"] for model in payload["data"]):
                        raise RuntimeError(f"Unexpected model list for {item['name']}: {payload}")
                    pending.remove(item["name"])
                except OSError:
                    pass
            if pending:
                time.sleep(3)
        if pending:
            raise TimeoutError(f"Services not healthy after 900 seconds: {sorted(pending)}")
        print(json.dumps({"state": str(STATE), "healthy": sorted(set(x["name"] for x in services))}, indent=2))
    except BaseException:
        stop(services)
        raise


def read_services():
    if not STATE.exists():
        return []
    return json.loads(STATE.read_text()).get("services", [])


def stop_services():
    services = read_services()
    stop(services)
    deadline = time.monotonic() + 60
    while any(live(item.get("pid")) for item in services) and time.monotonic() < deadline:
        time.sleep(1)
    remaining = [item for item in services if live(item.get("pid"))]
    for item in remaining:
        os.killpg(item["pid"], signal.SIGKILL)
    STATE.write_text(json.dumps({"stopped_at": time.time(), "services": services}, indent=2) + "\n")
    print(json.dumps({"stopped": [item["name"] for item in services]}, indent=2))


def status_services():
    rows = []
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for item in read_services():
        healthy = False
        if live(item.get("pid")):
            try:
                with opener.open(f"http://127.0.0.1:{item['port']}/v1/models", timeout=2) as response:
                    payload = json.load(response)
                healthy = any(model["id"] == item["alias"] for model in payload["data"])
            except OSError:
                pass
        rows.append({"name": item["name"], "pid": item["pid"], "gpus": item["gpus"],
                     "port": item["port"], "alive": live(item.get("pid")), "healthy": healthy})
    print(json.dumps(rows, indent=2))
    return 0 if rows and all(item["healthy"] for item in rows) else 1


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "status"
    if action == "start":
        start_services()
    elif action == "stop":
        stop_services()
    elif action == "restart":
        stop_services()
        start_services()
    elif action == "status":
        raise SystemExit(status_services())
    else:
        raise SystemExit("usage: eval_services.py {start|stop|restart|status}")
