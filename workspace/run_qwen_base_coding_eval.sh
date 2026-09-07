#!/usr/bin/env bash
set -euo pipefail

ROOT=/disk3/self_evolver
MODEL_PATH=/mnt/storage/disk1/verl_data/base_model/Qwen3.5-4B
MODEL_NAME=qwen3.5-4b-base
OUTPUT_ROOT=$ROOT/workspace/eval_logs/eval_main/qwen3.5-4b-base
RUN_LOG_DIR=$ROOT/logs/eval-qwen-base
PORTS=(8400 8500 8600 8700 8800 9500 9000 9100)

mkdir -p "$RUN_LOG_DIR"
if [[ -e "$OUTPUT_ROOT" ]]; then
    echo "Refusing to overwrite existing output: $OUTPUT_ROOT" >&2
    exit 1
fi
mkdir -p "$OUTPUT_ROOT"

service_env() {
    local gpu=$1
    shift
    env PROJECT_ROOT="$ROOT" VENV_PATH="$ROOT/.venv" \
      MODEL_PATH="$MODEL_PATH" MODEL_NAME="$MODEL_NAME" \
      LOG_DIR="/tmp/self-evolver-eval-qwen-base-$gpu" VLLM_PORT="${PORTS[$gpu]}" \
      VLLM_GPU="$gpu" VLLM_DP=1 VLLM_TP=1 VLLM_API_SERVER_COUNT=1 \
      VLLM_MAX_LEN=262144 VLLM_MAX_NUM_SEQS=64 VLLM_GPU_UTIL=0.85 "$@"
}

stop_servers() {
    local gpu
    for gpu in {0..7}; do
      service_env "$gpu" bash "$ROOT/start_services.sh" stop vllm || true
    done
}
trap stop_servers EXIT

server_pids=()
for gpu in {0..7}; do
  service_env "$gpu" bash "$ROOT/start_services.sh" start vllm \
    >"$RUN_LOG_DIR/server_$gpu.log" 2>&1 &
  server_pids+=("$!")
done
for pid in "${server_pids[@]}"; do
  wait "$pid"
done

common=(
  --model "$MODEL_NAME" --api-key EMPTY
  --temperature 0 --top-p 1
  --llm-max-completion-tokens 16384
  --step-timeout 1200 --concurrency 128
  --code-n-samples 1 --code-pass-k 1
)

cd "$ROOT"
"$ROOT/.venv/bin/python" -m benchmark.eval.run_eval "${common[@]}" \
  --openai-base-url "http://127.0.0.1:${PORTS[0]}/v1" \
  --benchmark humaneval+ --code-eval-workers 24 --code-memory 32g \
  --output-dir "$OUTPUT_ROOT/humaneval+" \
  >"$RUN_LOG_DIR/humaneval.log" 2>&1

"$ROOT/.venv/bin/python" -m benchmark.eval.run_eval "${common[@]}" \
  --openai-base-url "http://127.0.0.1:${PORTS[1]}/v1" \
  --benchmark mbpp+ --code-eval-workers 24 --code-memory 32g \
  --output-dir "$OUTPUT_ROOT/mbpp+" \
  >"$RUN_LOG_DIR/mbpp.log" 2>&1

generate_shard() {
  local gpu=$1 release=$2 tasks=$3 output=$4 log=$5 rc
  set +e
  "$ROOT/.venv/bin/python" -m benchmark.eval.run_eval "${common[@]}" \
    --openai-base-url "http://127.0.0.1:${PORTS[$gpu]}/v1" \
    --benchmark livecodebench --lcb-release "$release" --tasks "$tasks" \
    --code-generate-only --concurrency 64 --output-dir "$output" >"$log" 2>&1
  rc=$?
  set -e
  (( rc <= 1 ))
}

shard_pids=()
for gpu in {0..7}; do
  lo=$((gpu * 110)); hi=$((lo + 109))
  generate_shard "$gpu" v5 "$lo-$hi" \
    "$OUTPUT_ROOT/livecodebench/v5_shard_$gpu" \
    "$RUN_LOG_DIR/lcb_v5_shard_$gpu.log" &
  shard_pids+=("$!")
done
for pid in "${shard_pids[@]}"; do wait "$pid"; done

shard_pids=()
for gpu in {0..7}; do
  lo=$((880 + gpu * 22)); hi=$((lo + 21))
  (( gpu == 7 )) && hi=1054
  generate_shard "$gpu" v6 "$lo-$hi" \
    "$OUTPUT_ROOT/livecodebench/v6_new_shard_$gpu" \
    "$RUN_LOG_DIR/lcb_v6_new_shard_$gpu.log" &
  shard_pids+=("$!")
done
for pid in "${shard_pids[@]}"; do wait "$pid"; done

stop_servers
trap - EXIT

"$ROOT/.venv/bin/python" "$ROOT/workspace/merge_qwen_base_lcb.py" "$OUTPUT_ROOT"

for release in v5 v6; do
  "$ROOT/.venv/bin/python" -m benchmark.eval.run_eval \
    --openai-base-url http://127.0.0.1:1/v1 --model "$MODEL_NAME" --api-key EMPTY \
    --temperature 0 --top-p 1 --llm-max-completion-tokens 16384 \
    --benchmark livecodebench --lcb-release "$release" \
    --code-samples "$OUTPUT_ROOT/coding_merged/livecodebench/$release.samples.jsonl" \
    --code-n-samples 1 --code-pass-k 1 --code-eval-workers 8 \
    --code-memory 32g --code-eval-timeout 7200 \
    --output-dir "$OUTPUT_ROOT/livecodebench/${release}_scored" \
    >"$RUN_LOG_DIR/lcb_${release}_score.log" 2>&1
done

echo "QWEN_BASE_CODING_EVAL_COMPLETE"
