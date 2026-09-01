#!/usr/bin/env bash
# One real GRPO optimizer step through AWMAgentLoop and EnvScaler OpenEnv.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
VERL_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
REPO_ROOT=$(cd -- "$VERL_ROOT/.." && pwd)

source "$REPO_ROOT/.venv/bin/activate"
cd "$REPO_ROOT"

MODEL=${MODEL:-/mnt/storage/disk1/verl_data/base_model/Qwen3.5-4B}
ENVSCALER_BASE_URL=${ENVSCALER_BASE_URL:-http://127.0.0.1:8900}
DATA_DIR=${DATA_DIR:-$REPO_ROOT/traj_data/envscaler_rl_smoke}
TASK_ID=${TASK_ID:-env_144_rl-task_37}
RUN_NAME=${RUN_NAME:-qwen3_5_4b_envscaler_rl_smoke}
ROLLOUT_N=${ROLLOUT_N:-4}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-8192}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-12289}
PPO_BATCH_SIZE=${PPO_BATCH_SIZE:-1}

if ! curl --fail --silent --show-error "$ENVSCALER_BASE_URL/health" >/dev/null; then
    echo "EnvScaler OpenEnv is not healthy at $ENVSCALER_BASE_URL" >&2
    exit 1
fi
if [[ ! -f "$MODEL/config.json" ]]; then
    echo "Model checkpoint is missing: $MODEL" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
if [[ $(awk -F, '{print NF}' <<<"$CUDA_VISIBLE_DEVICES") -ne 2 ]]; then
    echo "This smoke launcher requires exactly two visible GPUs" >&2
    exit 1
fi
export PYTHONPATH="$VERL_ROOT:$REPO_ROOT:$REPO_ROOT/OpenEnv/src:$REPO_ROOT/OpenEnv/envs${PYTHONPATH:+:$PYTHONPATH}"
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

python "$SCRIPT_DIR/prepare_envscaler_rl_smoke_data.py" \
    --data-dir "$REPO_ROOT/envscaler_data" \
    --output-dir "$DATA_DIR" \
    --task-id "$TASK_ID" \
    --base-url "$ENVSCALER_BASE_URL" \
    --model "$MODEL" \
    --max-prompt-tokens "$MAX_PROMPT_LENGTH"

mkdir -p "$REPO_ROOT/logs" "$REPO_ROOT/rollout_trajs/$RUN_NAME"
LOG_FILE="$REPO_ROOT/logs/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"
echo "EnvScaler RL smoke: task=$TASK_ID, rollouts=$ROLLOUT_N, GPUs=$CUDA_VISIBLE_DEVICES"
echo "Log: $LOG_FILE"

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/val.parquet" \
    data.train_batch_size=1 \
    data.max_prompt_length="$MAX_PROMPT_LENGTH" \
    data.max_response_length="$MAX_RESPONSE_LENGTH" \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=1 \
    data.truncation=error \
    data.shuffle=False \
    data.dataloader_num_workers=0 \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path="$MODEL" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_BATCH_SIZE" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$MAX_MODEL_LEN" \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.n="$ROLLOUT_N" \
    actor_rollout_ref.rollout.do_sample=True \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN" \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$MAX_MODEL_LEN" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.format=qwen3_coder \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=8 \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=7 \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=3 \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=1200 \
    actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side=right \
    actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode=ignore_strippable \
    actor_rollout_ref.rollout.agent.default_agent_loop=awm_agent \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="$REPO_ROOT/rollout/awm_agent_loop.yaml" \
    actor_rollout_ref.rollout.agent.num_workers="$ROLLOUT_N" \
    trainer.logger=console \
    trainer.project_name=self_evolver_envscaler \
    trainer.experiment_name="$RUN_NAME" \
    trainer.default_local_dir="$REPO_ROOT/checkpoints/self_evolver_envscaler/$RUN_NAME" \
    trainer.rollout_data_dir="$REPO_ROOT/rollout_trajs/$RUN_NAME" \
    trainer.balance_batch=True \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=1 \
    trainer.resume_mode=disable \
    trainer.use_v1=False \
    distillation.enabled=False \
    "$@" 2>&1 | tee "$LOG_FILE"
