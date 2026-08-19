#!/usr/bin/env bash
# Minimal native-verl OPSD smoke test: two actor/FSDP GPUs, one frozen-teacher GPU,
# one optimizer step, no AWM environment and no checkpoint/W&B side effects.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
VERL_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
REPO_ROOT=$(cd -- "$VERL_ROOT/.." && pwd)

source "$REPO_ROOT/.venv/bin/activate"
cd "$REPO_ROOT"

MODEL=${MODEL:-/mnt/storage/disk1/verl_data/base_model/Qwen3.5-4B}
DATA_DIR=${DATA_DIR:-$REPO_ROOT/traj_data/opsd_smoke}
TRAIN_FILE="$DATA_DIR/train.parquet"
VAL_FILE="$DATA_DIR/val.parquet"

if [[ ! -f "$TRAIN_FILE" || ! -f "$VAL_FILE" ]]; then
    PYTHONPATH="$VERL_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
        python "$SCRIPT_DIR/prepare_opsd_smoke_data.py" --output-dir "$DATA_DIR"
fi

# FSDP offload is enabled below, but FSDP with world-size 1 cannot shard the
# actor's parameters/gradients. Use two actor GPUs plus one teacher GPU so the
# optimizer update is actually sharded.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2}
export PYTHONPATH="$VERL_ROOT:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
    data.train_batch_size=2 \
    data.max_prompt_length=1024 \
    data.max_response_length=256 \
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
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.30 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.do_sample=False \
    actor_rollout_ref.rollout.temperature=0 \
    actor_rollout_ref.rollout.max_model_len=1281 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    trainer.logger=console \
    trainer.project_name=verl_opsd_smoke \
    trainer.experiment_name=qwen3_5_4b_self_distill_debug \
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
    distillation.enabled=True \
    distillation.self_distillation=True \
    distillation.n_gpus_per_node=1 \
    distillation.nnodes=1 \
    distillation.teacher_models.teacher_model.model_path="$MODEL" \
    distillation.teacher_models.teacher_model.inference.name=vllm \
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=1 \
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=0.30 \
    distillation.teacher_models.teacher_model.inference.enforce_eager=True \
    distillation.teacher_models.teacher_model.inference.max_model_len=1281 \
    distillation.privileged_mode=chat_turn \
    distillation.privileged_solution_key=reward_model.ground_truth \
    distillation.privileged_problem_key=extra_info.problem \
    distillation.privileged_enable_thinking=True \
    distillation.distillation_loss.loss_mode=k1 \
    distillation.distillation_loss.topk=64 \
    distillation.distillation_loss.use_policy_gradient=True \
    distillation.distillation_loss.use_task_rewards=False \
    distillation.distillation_loss.loss_max_clamp=10.0 \
    distillation.distillation_loss.log_prob_min_clamp=-10.0 \
    "$@"
