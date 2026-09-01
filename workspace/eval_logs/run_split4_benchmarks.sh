#!/usr/bin/env bash
set -Eeuo pipefail

project_root=/mnt/storage/disk3/self_evolver
output_root="$project_root/workspace/eval_logs/split4_cluster_merge_20260828"
report="$project_root/workspace/eval_logs/benchmark_results_split4_cluster_merge_20260828.md"
summarizer="$project_root/workspace/eval_logs/summarize_split4_benchmarks.py"
python="$project_root/.venv/bin/python"
current_log_dir=""
current_model_path=""

models=(
  "cluster0-step35|split4-cluster0-step35|/mnt/storage/disk1/verl_data/base_model/Qwen3.5-4B-HybridDeltaResidualAttentionPrefix-split4-trained-eval/cluster0-step35"
  "cluster1-step44|split4-cluster1-step44|/mnt/storage/disk1/verl_data/base_model/Qwen3.5-4B-HybridDeltaResidualAttentionPrefix-split4-trained-eval/cluster1-step44"
  "cluster2-step66|split4-cluster2-step66|/mnt/storage/disk1/verl_data/base_model/Qwen3.5-4B-HybridDeltaResidualAttentionPrefix-split4-trained-eval/cluster2-step66"
  "cluster3-step57|split4-cluster3-step57|/mnt/storage/disk1/verl_data/base_model/Qwen3.5-4B-HybridDeltaResidualAttentionPrefix-split4-trained-eval/cluster3-step57"
  "merged-split4|split4-merged-g2048-a256|/mnt/storage/disk1/verl_data/base_model/Qwen3.5-4B-HybridDeltaResidualAttentionPrefix-G2048-A256-step203-split4-trained-merged"
)

summarize() {
  "$python" "$summarizer" "$output_root" "$report"
}

stop_vllm() {
  if [[ -n "$current_log_dir" && -n "$current_model_path" ]]; then
    LOG_DIR="$current_log_dir" MODEL_PATH="$current_model_path" \
      "$project_root/start_services.sh" stop vllm || true
  fi
}

on_exit() {
  rc=$?
  stop_vllm
  summarize || true
  exit "$rc"
}
trap on_exit EXIT INT TERM

run_eval() {
  local model_name=$1
  shift
  "$python" -m benchmark.eval.run_eval \
    --openai-base-url http://127.0.0.1:8000/v1 \
    --model "$model_name" \
    --api-key EMPTY \
    --temperature 0 \
    --top-p 1 \
    "$@"
}

cd "$project_root"
mkdir -p "$output_root"
summarize

for spec in "${models[@]}"; do
  IFS='|' read -r slug model_name model_path <<<"$spec"
  model_root="$output_root/$slug"
  current_log_dir="$model_root/vllm_server"
  current_model_path="$model_path"
  mkdir -p "$current_log_dir"

  if [[ ! -f "$model_path/config.json" || ! -f "$model_path/model.safetensors.index.json" ]]; then
    echo "incomplete model directory: $model_path" >&2
    exit 1
  fi

  MODEL_PATH="$model_path" MODEL_NAME="$model_name" LOG_DIR="$current_log_dir" \
    VLLM_READY_TIMEOUT=900 "$project_root/start_services.sh" restart vllm

  "$python" - "$model_name" "$model_path" <<'PY'
import json, sys, urllib.request
name, expected_path = sys.argv[1:]
with urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=10) as response:
    payload = json.load(response)
models = {item["id"]: item for item in payload["data"]}
assert name in models, (name, list(models))
actual_path = models[name].get("root")
assert actual_path == expected_path, (expected_path, actual_path)
print(f"verified endpoint: {name} -> {actual_path}")
PY

  if [[ ! -f "$model_root/bfcl_v3/all/score/data_overall.csv" ]]; then
    run_eval "$model_name" \
      --benchmark bfcl_v3 \
      --bfcl-model-path "$model_path" \
      --bfcl-categories all \
      --concurrency 64 \
      --output-dir "$model_root/bfcl_v3/all" \
      2>&1 | tee "$model_root/bfcl_v3_runner.log"
  fi

  for domain in airline retail; do
    if [[ ! -f "$model_root/tau2/$domain/results.json" ]]; then
      run_eval "$model_name" \
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
        --output-dir "$model_root/tau2/$domain" \
        2>&1 | tee "$model_root/tau2_${domain}_runner.log"
    fi
  done

  if [[ ! -f "$model_root/lifelong_db/test/summary.csv" ]]; then
    run_eval "$model_name" \
      --benchmark lifelong_db \
      --data-dir benchmark/LifelongAgentBench \
      --split test \
      --max-steps 6 \
      --step-timeout 180 \
      --llm-max-completion-tokens 2048 \
      --concurrency 16 \
      --mysql-image mysql:8.0 \
      --output-dir "$model_root/lifelong_db/test" \
      2>&1 | tee "$model_root/lifelong_db_runner.log"
  fi

  if [[ ! -f "$model_root/lifelong_os/test/summary.csv" ]]; then
    run_eval "$model_name" \
      --benchmark lifelong_os \
      --data-dir benchmark/LifelongAgentBench \
      --split test \
      --max-steps 8 \
      --step-timeout 180 \
      --os-timeout 20 \
      --llm-max-completion-tokens 2048 \
      --concurrency 16 \
      --output-dir "$model_root/lifelong_os/test" \
      2>&1 | tee "$model_root/lifelong_os_runner.log"
  fi

  stop_vllm
  current_log_dir=""
  current_model_path=""
  summarize
done

trap - EXIT INT TERM
summarize
