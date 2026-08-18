#!/usr/bin/env bash
set -euo pipefail

REPO=/mnt/storage/disk3/self_evolver
cd "$REPO"
source .venv/bin/activate

MODEL="${MODEL:-/mnt/storage/disk1/verl_data/base_model/Qwen3.5-4B-DeltaVirtualPrefix-256}"
DATASET="${DATASET:-$REPO/traj_data/task1/swift_opsd.jsonl}"
TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3,4,5,6,7}"
NPROC="${NPROC:-8}"
PREFIX_LR="${PREFIX_LR:-1e-3}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO/output/opsd_delta_virtual_prefix_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUTPUT_DIR"

echo "[opsd-delta-virtual-prefix] model=$MODEL GPUs=$TRAIN_GPUS lr=$PREFIX_LR output=$OUTPUT_DIR"

python - <<'PY'
import importlib.metadata

plugins = {entry.name for entry in importlib.metadata.entry_points(group="vllm.general_plugins")}
if "qwen35_delta_virtual_prefix" not in plugins:
    raise SystemExit(
        "Missing vLLM plugin. Run: /root/.local/bin/uv pip install "
        "--python .venv/bin/python -e prefix_tuning"
    )
PY

PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" \
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" \
NPROC_PER_NODE="$NPROC" \
swift rlhf \
    --rlhf_type grpo \
    --model "$MODEL" \
    --model_type qwen3_5_delta_virtual_prefix \
    --trust_remote_code true \
    --external_plugins \
        prefix_tuning/virtual_prefix/plugin.py \
        rollout/awm_opsd_plugin.py \
        rollout/awm_scheduler_plugin.py \
    --callbacks delta_virtual_prefix_only \
    --multi_turn_scheduler awm_scheduler \
    --max_turns 6 \
    --reward_funcs awm_verify \
    --agent_template qwen3_coder \
    --enable_thinking false \
    --dataset "$DATASET" \
    --dataset_shuffle true \
    --tuner_type full \
    --weight_decay 0.0 \
    --padding_free false \
    --packing false \
    --torch_dtype bfloat16 \
    --attn_impl flash_attn \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.7 \
    --vllm_max_model_len 32768 \
    --vllm_tensor_parallel_size 1 \
    --sleep_level 1 \
    --num_generations 8 \
    --temperature 1.0 \
    --beta 0.04 \
    --max_length 32768 \
    --max_completion_length 2048 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate "$PREFIX_LR" \
    --num_train_epochs 2 \
    --warmup_ratio 0.05 \
    --logging_steps 1 \
    --save_steps 50 \
    --save_total_limit 20 \
    --save_only_model true \
    --output_dir "$OUTPUT_DIR" \
    --deepspeed zero2 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --log_completions true \
    --log_rollout_offpolicy_metrics true \
    --report_to wandb
