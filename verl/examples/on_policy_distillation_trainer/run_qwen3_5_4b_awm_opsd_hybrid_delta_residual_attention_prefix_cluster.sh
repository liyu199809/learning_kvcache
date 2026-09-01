#!/usr/bin/env bash
# Train one contiguous hybrid-prefix partition on one clustered dataset.
# Uses 6 GPUs for actor/rollout and two teacher GPUs for privileged distillation.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
VERL_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
REPO_ROOT=$(cd -- "$VERL_ROOT/.." && pwd)

source "$REPO_ROOT/.venv/bin/activate"
cd "$REPO_ROOT"

CLUSTER_ID=${CLUSTER_ID:?Set CLUSTER_ID to one of 0, 1, 2, or 3}
if [[ ! "$CLUSTER_ID" =~ ^[0-3]$ ]]; then
    echo "CLUSTER_ID must be one of 0, 1, 2, or 3; got $CLUSTER_ID" >&2
    exit 1
fi

ACTOR_ROOT=${ACTOR_ROOT:-/mnt/storage/disk1/verl_data/base_model/Qwen3.5-4B-HybridDeltaResidualAttentionPrefix-G2048-A256-step203-split4}
ACTOR_MODEL=${ACTOR_MODEL:-$ACTOR_ROOT/cluster$CLUSTER_ID}
TEACHER_MODEL=${TEACHER_MODEL:-/mnt/storage/disk1/verl_data/base_model/Qwen3.5-4B}
AWM_BASE_URL=${AWM_BASE_URL:-http://localhost:8899}
DATA_DIR=${DATA_DIR:-$REPO_ROOT/traj_data/awm_opsd_full_task1}
CLUSTER_DATA_DIR=${CLUSTER_DATA_DIR:-$DATA_DIR/cluster_split}
TRAIN_FILE=${TRAIN_FILE:-$CLUSTER_DATA_DIR/train_cluster$CLUSTER_ID.parquet}
VAL_FILE=${VAL_FILE:-$DATA_DIR/val.parquet}
DATA_MANIFEST=${DATA_MANIFEST:-$DATA_DIR/split_manifest.json}

RUN_NAME=${RUN_NAME:-qwen3_5_4b_awm_opsd_hybrid_prefix_split4_cluster$CLUSTER_ID}
PROJECT_NAME=${PROJECT_NAME:-self_evolver_opsd}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-$REPO_ROOT/checkpoints/$PROJECT_NAME/$RUN_NAME}
ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-$REPO_ROOT/rollout_trajs/$RUN_NAME}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-24}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-20}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
ACTOR_LR=${ACTOR_LR:-5e-6}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-28672}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8192}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-40960}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-40960}
DISTILLATION_TOPK=${DISTILLATION_TOPK:-64}

if ! curl --fail --silent --show-error "$AWM_BASE_URL/health" >/dev/null; then
    echo "AWM OpenEnv is not healthy at $AWM_BASE_URL. Start it with ./start_services.sh first." >&2
    exit 1
fi
if [[ ! -f "$ACTOR_MODEL/config.json" ]]; then
    echo "Actor model checkpoint is missing or invalid: $ACTOR_MODEL" >&2
    exit 1
fi
if [[ ! -f "$TEACHER_MODEL/config.json" ]]; then
    echo "Teacher model checkpoint is missing or invalid: $TEACHER_MODEL" >&2
    exit 1
fi
if [[ ! -f "$TRAIN_FILE" || ! -f "$VAL_FILE" || ! -f "$DATA_MANIFEST" ]]; then
    echo "Cluster train, validation, or data manifest file is missing" >&2
    echo "TRAIN_FILE=$TRAIN_FILE" >&2
    echo "VAL_FILE=$VAL_FILE" >&2
    echo "DATA_MANIFEST=$DATA_MANIFEST" >&2
    exit 1
fi
if ! python -c 'import wandb' >/dev/null 2>&1; then
    echo "wandb is not installed in $REPO_ROOT/.venv" >&2
    exit 1
fi
if ! grep -q 'machine api.wandb.ai' "$HOME/.netrc" 2>/dev/null && [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "No W&B credentials found. Run wandb login or set WANDB_API_KEY." >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
if [[ $(awk -F, '{print NF}' <<<"$CUDA_VISIBLE_DEVICES") -ne 8 ]]; then
    echo "This launcher requires exactly 8 visible GPUs; got CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2
    exit 1
fi
export PYTHONPATH="$VERL_ROOT:$REPO_ROOT:$REPO_ROOT/OpenEnv/src:$REPO_ROOT/OpenEnv/envs${PYTHONPATH:+:$PYTHONPATH}"
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export WANDB_PROJECT="$PROJECT_NAME"

MODEL_FACTS=$(python - "$ACTOR_MODEL" "$CLUSTER_ID" <<'PY'
import importlib.metadata
import json
import sys
from pathlib import Path

from safetensors import safe_open

model = Path(sys.argv[1])
cluster_id = int(sys.argv[2])
config = json.loads((model / "config.json").read_text())
delta_config = config.get("independent_delta_kv_prefix", {})
attention_config = config.get("residual_attention_prefix", {})
if config.get("model_type") != "qwen3_5":
    raise SystemExit(f"Expected model_type=qwen3_5, got {config.get('model_type')!r}")
if config.get("architectures") != ["Qwen3_5HybridDeltaResidualAttentionPrefixForConditionalGeneration"]:
    raise SystemExit(f"Unexpected actor architecture: {config.get('architectures')!r}")
delta_tokens = delta_config.get("num_virtual_tokens")
attention_tokens = attention_config.get("num_virtual_tokens")
if not isinstance(delta_tokens, int) or delta_tokens <= 0:
    raise SystemExit(f"Invalid Delta virtual-token count: {delta_tokens!r}")
if not isinstance(attention_tokens, int) or attention_tokens <= 0:
    raise SystemExit(f"Invalid attention virtual-token count: {attention_tokens!r}")
split_manifest = json.loads((model / "prefix_split_manifest.json").read_text())
if split_manifest.get("split_index") != cluster_id:
    raise SystemExit(
        f"Actor split index {split_manifest.get('split_index')!r} does not match cluster {cluster_id}"
    )
weights_file = model / "independent_delta_kv_prefix.safetensors"
with safe_open(weights_file, framework="pt", device="cpu") as handle:
    shapes = {name: tuple(handle.get_slice(name).get_shape()) for name in handle.keys()}
expected_suffix_shapes = {
    ".prefix_k": (delta_tokens, 2048),
    ".prefix_v": (delta_tokens, 4096),
    ".prefix_beta_logits": (delta_tokens, 32),
    ".prefix_a": (delta_tokens, 32),
}
if len(shapes) != 96:
    raise SystemExit(f"Expected 96 prefix tensors, got {len(shapes)}")
for name, shape in shapes.items():
    matches = [expected for suffix, expected in expected_suffix_shapes.items() if name.endswith(suffix)]
    if len(matches) != 1 or shape != matches[0]:
        raise SystemExit(f"Unexpected prefix tensor {name} shaped {shape}")
delta_numel = sum(shape[0] * shape[1] for shape in shapes.values())
attention_weights_file = model / "residual_attention_prefix.safetensors"
with safe_open(attention_weights_file, framework="pt", device="cpu") as handle:
    attention_shapes = {
        name: tuple(handle.get_slice(name).get_shape()) for name in handle.keys()
    }
if len(attention_shapes) != 16:
    raise SystemExit(f"Expected 16 attention prefix tensors, got {len(attention_shapes)}")
if any(shape != (attention_tokens, 2560) for shape in attention_shapes.values()):
    raise SystemExit(f"Unexpected attention prefix shapes: {attention_shapes}")
attention_numel = sum(shape[0] * shape[1] for shape in attention_shapes.values())
plugins = {entry.name for entry in importlib.metadata.entry_points(group="vllm.general_plugins")}
if "qwen35_delta_virtual_prefix" not in plugins:
    raise SystemExit(
        "Missing vLLM plugin. Run: /root/.local/bin/uv pip install "
        "--python .venv/bin/python -e prefix_tuning"
    )
print(delta_tokens, attention_tokens, len(shapes) + len(attention_shapes), delta_numel + attention_numel)
PY
)
read -r DELTA_TOKENS ATTENTION_TOKENS EXPECTED_PARAM_COUNT EXPECTED_NUMEL <<<"$MODEL_FACTS"

MAX_DATA_PROMPT=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["prompt_tokens"]["max"])' "$DATA_MANIFEST")
if (( MAX_DATA_PROMPT > MAX_PROMPT_LENGTH )); then
    echo "Dataset max prompt length $MAX_DATA_PROMPT exceeds MAX_PROMPT_LENGTH=$MAX_PROMPT_LENGTH" >&2
    exit 1
fi
if (( MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 1 > MAX_MODEL_LEN )); then
    echo "MAX_MODEL_LEN=$MAX_MODEL_LEN is too small for prompt+response+1=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 1))" >&2
    exit 1
fi

mkdir -p "$CHECKPOINT_DIR" "$ROLLOUT_DATA_DIR" "$REPO_ROOT/logs"
START_TIME=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$REPO_ROOT/logs/${RUN_NAME}_${START_TIME}.log"
TRAIN_ROWS=$(python -c 'import pyarrow.parquet as pq,sys; print(pq.read_metadata(sys.argv[1]).num_rows)' "$TRAIN_FILE")
echo "Training: $TRAIN_ROWS rows, batch $TRAIN_BATCH_SIZE, $TOTAL_EPOCHS epochs; actor GPUs=6, teacher GPUs=2"
echo "Cluster: $CLUSTER_ID; train=$TRAIN_FILE; validation=$VAL_FILE"
echo "Actor: $ACTOR_MODEL (G=$DELTA_TOKENS/A=$ATTENTION_TOKENS; $EXPECTED_PARAM_COUNT tensors; $EXPECTED_NUMEL parameters)"
echo "Teacher: $TEACHER_MODEL"
echo "Checkpoints: $CHECKPOINT_DIR"
echo "Rollout trajectories: $ROLLOUT_DATA_DIR"
echo "Log: $LOG_FILE"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "Dry run passed; trainer was not started"
    exit 0
fi

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.val_batch_size="$VAL_BATCH_SIZE" \
    data.max_prompt_length="$MAX_PROMPT_LENGTH" \
    data.max_response_length="$MAX_RESPONSE_LENGTH" \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=4 \
    data.truncation=error \
    data.shuffle=True \
    data.seed=42 \
    data.dataloader_num_workers=4 \
    data.trust_remote_code=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path="$ACTOR_MODEL" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.use_remove_padding=True \
    'actor_rollout_ref.model.trainable_param_patterns=[".*[.](linear_attn[.](prefix_k|prefix_v|prefix_beta_logits|prefix_a)|self_attn[.](prefix_key_tokens|prefix_value_tokens))$"]' \
    actor_rollout_ref.model.expected_trainable_param_count="$EXPECTED_PARAM_COUNT" \
    actor_rollout_ref.model.expected_trainable_numel="$EXPECTED_NUMEL" \
    actor_rollout_ref.model.rollout_sync_trainable_only=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr="$ACTOR_LR" \
    actor_rollout_ref.actor.optim.weight_decay=0.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size="$TRAIN_BATCH_SIZE" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$PPO_MAX_TOKEN_LEN_PER_GPU" \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True \
    actor_rollout_ref.actor.fsdp_config.entropy_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.actor.fsdp_config.offload_policy=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.use_torch_compile=False \
    actor_rollout_ref.actor.checkpoint.save_trainable_only=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.65 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.do_sample=True \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN" \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=False \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$PPO_MAX_TOKEN_LEN_PER_GPU" \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.format=qwen3_coder \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=20 \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=19 \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=3 \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=800 \
    actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side=right \
    actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode=ignore_strippable \
    actor_rollout_ref.rollout.agent.default_agent_loop=awm_agent \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="$REPO_ROOT/rollout/awm_agent_loop.yaml" \
    actor_rollout_ref.rollout.agent.num_workers=24 \
    'trainer.logger=["console","wandb"]' \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$RUN_NAME" \
    trainer.default_local_dir="$CHECKPOINT_DIR" \
    trainer.rollout_data_dir="$ROLLOUT_DATA_DIR" \
    trainer.log_val_generations=10 \
    trainer.balance_batch=True \
    trainer.n_gpus_per_node=6 \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.test_freq=20 \
    trainer.save_freq=20 \
    trainer.max_actor_ckpt_to_keep=3 \
    trainer.max_critic_ckpt_to_keep=3 \
    trainer.total_epochs="$TOTAL_EPOCHS" \
    trainer.resume_mode=auto \
    trainer.use_v1=False \
    distillation.enabled=True \
    distillation.self_distillation=True \
    distillation.n_gpus_per_node=2 \
    distillation.nnodes=1 \
    distillation.teacher_models.teacher_model.model_path="$TEACHER_MODEL" \
    distillation.teacher_models.teacher_model.inference.name=vllm \
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=1 \
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=0.65 \
    distillation.teacher_models.teacher_model.inference.enforce_eager=True \
    distillation.teacher_models.teacher_model.inference.max_model_len="$MAX_MODEL_LEN" \
    distillation.privileged_mode=append \
    distillation.privileged_insert_before=$'"<|im_end|>\n<|im_start|>assistant\n"' \
    distillation.privileged_solution_key=reward_model.ground_truth \
    distillation.distillation_loss.loss_mode=forward_kl_topk \
    distillation.distillation_loss.topk="$DISTILLATION_TOPK" \
    distillation.distillation_loss.use_policy_gradient=False \
    +distillation.distillation_loss.use_chunked_topk=True \
    distillation.distillation_loss.use_task_rewards=False \
    distillation.distillation_loss.loss_max_clamp=10.0 \
    distillation.distillation_loss.log_prob_min_clamp=-10.0 \
    "$@" 2>&1 | tee "$LOG_FILE"
