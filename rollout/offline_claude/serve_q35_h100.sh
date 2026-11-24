#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${PYTHONPATH:-/tmp/prefix-vllm-env:/tmp/kvcache-review.0BeCXi}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1

MODEL_PATH="${MODEL_PATH:-/tmp/prefix-vllm-4b/base}"
SERVED_NAME="${SERVED_NAME:-qwen3.5-4b}"
PORT="${PORT:-18764}"
DP="${DP:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
GPU_MEM="${GPU_MEM:-0.85}"

exec /usr/local/bin/python -S -m vllm.entrypoints.openai.api_server \
  --host 127.0.0.1 --port "$PORT" \
  --model "$MODEL_PATH" --served-model-name "$SERVED_NAME" --dtype bfloat16 \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --tensor-parallel-size 1 --data-parallel-size "$DP" \
  --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS" \
  --gpu-memory-utilization "$GPU_MEM" \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking":true}'
