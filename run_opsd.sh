#!/usr/bin/env bash
# =============================================================================
# OPSD (On-Policy Self-Distillation) training for AWM / Qwen3.5-4B via ms-swift.
#
# Pipeline recap
# --------------
#   student rolls out on-policy  ->  teacher scores the SAME token sequence but
#   with privileged `advice` in its context  ->  student is distilled toward the
#   teacher's distribution. Teacher == student weights (self-distillation); the
#   ONLY difference is the extra advice the teacher sees, driven by the dataset's
#   `teacher_prompt` column (swift/rl_core/data.py: teacher_prompt replaces the
#   last user msg content for the teacher forward).
#
# Two plugins (in rollout/):
#   * awm_opsd_plugin.py      -> dataset preprocessor (student `messages` +
#                                privileged `teacher_prompt`) + `awm_verify` ORM.
#   * awm_scheduler_plugin.py -> `awm_scheduler` multi-turn scheduler that resets
#                                the AWM env per (scenario, task_idx) and drives
#                                native function-calling rollout.
#
# Mode: GRPO + colocate + LoRA (single command; vLLM runs inside the training
# process on the same GPUs — no separate `swift rollout` server needed).
#   * multi_turn_scheduler requires --rlhf_type grpo (not gkd; rlhf_args.py:667).
#   * LoRA self-distillation: teacher == student with disable_adapter() (the
#     frozen base is the teacher, the LoRA-adapted model is the student), so we
#     do NOT pass --teacher_model.
#   * colocate needs --sleep_level 1 so vLLM releases VRAM during the optim step.
#
# Topology (8x A800-80G): GPU 0 currently holds the standing inference server
# (start_services.sh, ~70G). Default here uses GPUs 1-7. Once you free GPU 0,
# set TRAIN_GPUS=0,1,2,3,4,5,6,7 NPROC=8.
#
# Prereqs:
#   1) AWM env server up on :8899   ->  ./start_services.sh status
#   2) OPSD dataset built from expert-refine trajectories:
#        # refine.py already produced traj_data/refine_200.jsonl (expert advice)
#        python -m rollout.refine2swift \
#          --refine-jsonl traj_data/refine_200.jsonl \
#          --output-jsonl traj_data/swift_opsd.jsonl
#      This emits messages / teacher_prompt(str) / tools / env_config /
#      verify_reward_type. Training itself does the on-policy sampling via the
#      scheduler — no separate rollout step is needed.
#
# DATASET CONTRACT (emitted by rollout.refine2swift, consumed by the ONLINE
# grpo+scheduler path — verified against swift 4.5.0.dev0 source):
#   * messages        : student's on-policy starting context. The LAST message
#                       MUST be role=user — swift OPSD replaces its *content*
#                       with teacher_prompt (swift/rl_core/data.py:build_teacher_view).
#   * teacher_prompt  : a STRING (NOT a message list) = the privileged version
#                       of that last user turn (reset-note + "# Advice: ...").
#   * env_config      : {"scenario":..., "task_idx":..., "awm_base_url":...} —
#                       read by awm_scheduler (req.data_dict['env_config']).
#   * verify_reward_type : passthrough column -> bound as the awm_verify kwarg.
#   Extra columns survive into req.data_dict / sample.extra automatically
#   (swift/rl_core/data.py:288-292).
#
# Usage:
#   ./run_opsd.sh          # start OPSD training (single command)
# =============================================================================
set -euo pipefail

REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$REPO"
source .venv/bin/activate

MODEL=${MODEL:-/mnt/storage/disk1/verl_data/base_model/Qwen3.5-4B}
DATASET="$REPO/traj_data/task1/swift_opsd.jsonl"

# GPUs. Default excludes GPU 0 (standing server). Free GPU 0 later, then use
#   TRAIN_GPUS=0,1,2,3,4,5,6,7 NPROC=8 ./run_opsd.sh
#
# NB (GRPO batch rule): generation_batch_size = per_device_train_batch_size ×
# NPROC × gradient_accumulation_steps, and it MUST be divisible by
# num_generations. Here 1 × NPROC × 8 must be a multiple of num_generations(8),
# which holds for NPROC=7 (56) and NPROC=8 (64). If you change NPROC / batch /
# grad_accum, keep that product divisible by --num_generations.
TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3,4,5,6,7}"
NPROC="${NPROC:-8}"

OUTPUT_DIR="${OUTPUT_DIR:-$REPO/output/opsd_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUTPUT_DIR"

echo "[opsd] GRPO+colocate+LoRA on GPUs=$TRAIN_GPUS -> $OUTPUT_DIR"

# GRPO/OPSD training, colocate vLLM, LoRA tuner.
#   --reward_funcs awm_verify  : reward from the verify_reward_type column.
#   --enable_thinking true     : student rollout 走思考模式(加 <think> 前缀,
#                                生成带 <think>..</think>);GRPO 多轮下默认推导不可靠,显式钉死。
#   OPSD teacher KD signal comes from the teacher_prompt column (no --teacher_model).
# PYTHONPATH=$REPO so the plugins' `from rollout.common import ...` resolves
# (swift's worker subprocess doesn't add the repo root to sys.path itself).
PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" \
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" \
NPROC_PER_NODE="$NPROC" \
swift rlhf \
    --rlhf_type grpo \
    --model "$MODEL" \
    --external_plugins rollout/awm_opsd_plugin.py rollout/awm_scheduler_plugin.py \
    --multi_turn_scheduler awm_scheduler \
    --max_turns 6 \
    --reward_funcs awm_verify \
    --agent_template qwen3_coder \
    --enable_thinking false \
    --dataset "$DATASET" \
    --dataset_shuffle true \
    --tuner_type lora \
    --lora_rank 16 \
    --lora_alpha 32 \
    --target_modules all-linear \
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
    --learning_rate 5e-6 \
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
