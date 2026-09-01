#!/usr/bin/env bash
set -uo pipefail

project_root=/mnt/storage/disk3/self_evolver
model_path=/mnt/storage/disk1/verl_data/base_model/Qwen3.5-4B-Base
model_name=qwen3.5-4b-base
output_root="$project_root/workspace/eval_logs/qwen3_5_4b_base_20260831"
python="$project_root/.venv/bin/python"
overall_rc=0

mkdir -p "$output_root"
cd "$project_root"

run_eval() {
  "$python" -m benchmark.eval.run_eval \
    --openai-base-url http://127.0.0.1:8000/v1 \
    --model "$model_name" \
    --api-key EMPTY \
    --temperature 0 \
    --top-p 1 \
    "$@"
}

run_logged() {
  local log_file=$1
  shift
  if ! "$@" 2>&1 | tee "$log_file"; then
    overall_rc=1
    printf 'FAILED: %s\n' "$log_file" >&2
  fi
}

"$python" - "$model_name" "$model_path" "$output_root/eval_config.json" <<'PY'
import json
import subprocess
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

name, expected_path, config_path = sys.argv[1:]
with urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=10) as response:
    payload = json.load(response)
models = {item["id"]: item for item in payload["data"]}
assert name in models, (name, list(models))
actual_path = models[name].get("root")
assert actual_path == expected_path, (expected_path, actual_path)
config = {
    "created_at": datetime.now().astimezone().isoformat(),
    "model_repo": "Qwen/Qwen3.5-4B-Base",
    "inference_model_path": expected_path,
    "served_model_name": name,
    "vllm": {
        "version": "0.18.0",
        "data_parallel_size": 8,
        "tensor_parallel_size": 1,
        "dtype": "bfloat16",
        "max_model_len": 262144,
        "gpu_memory_utilization": 0.85,
    },
    "benchmarks": ["bfcl_v3", "tau2_airline", "tau2_retail", "lifelong_db", "lifelong_os"],
    "temperature": 0,
    "top_p": 1,
    "tau2_trials": 1,
    "tau2_agent_thinking": False,
    "tau2_user_thinking": False,
}
try:
    config["project_git_head"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
except Exception:
    pass
Path(config_path).write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
print(f"verified endpoint: {name} -> {actual_path}")
PY

if [[ ! -f "$output_root/bfcl_v3/all/score/data_overall.csv" ]]; then
  mkdir -p "$output_root/bfcl_v3"
  run_logged "$output_root/bfcl_v3_runner.log" run_eval \
    --benchmark bfcl_v3 \
    --bfcl-model-path "$model_path" \
    --bfcl-categories all \
    --llm-max-completion-tokens 4096 \
    --concurrency 64 \
    --output-dir "$output_root/bfcl_v3/all"
fi

for domain in airline retail; do
  if [[ ! -f "$output_root/tau2/$domain/results.json" ]]; then
    mkdir -p "$output_root/tau2/$domain"
    run_logged "$output_root/tau2_${domain}_runner.log" run_eval \
      --benchmark tau2 \
      --tau2-domain "$domain" \
      --tau2-split base \
      --tau2-num-trials 1 \
      --tau2-max-steps 50 \
      --tau2-judge-model openai/deepseek-v4-pro-ga-260813 \
      --tau2-judge-base-url https://ark.cn-beijing.volces.com/api/v3 \
      --tau2-judge-api-key-env ARK_API_KEY \
      --llm-max-completion-tokens 4096 \
      --concurrency 32 \
      --output-dir "$output_root/tau2/$domain"
  fi
done

if [[ ! -f "$output_root/lifelong_db/test/summary.csv" ]]; then
  mkdir -p "$output_root/lifelong_db"
  run_logged "$output_root/lifelong_db_runner.log" run_eval \
    --benchmark lifelong_db \
    --data-dir benchmark/LifelongAgentBench \
    --split test \
    --max-steps 6 \
    --step-timeout 180 \
    --llm-max-completion-tokens 2048 \
    --concurrency 16 \
    --mysql-image mysql:8.0 \
    --output-dir "$output_root/lifelong_db/test"
fi

if [[ ! -f "$output_root/lifelong_os/test/summary.csv" ]]; then
  mkdir -p "$output_root/lifelong_os"
  run_logged "$output_root/lifelong_os_runner.log" run_eval \
    --benchmark lifelong_os \
    --data-dir benchmark/LifelongAgentBench \
    --split test \
    --max-steps 8 \
    --step-timeout 180 \
    --os-timeout 20 \
    --llm-max-completion-tokens 2048 \
    --concurrency 16 \
    --output-dir "$output_root/lifelong_os/test"
fi

exit "$overall_rc"
