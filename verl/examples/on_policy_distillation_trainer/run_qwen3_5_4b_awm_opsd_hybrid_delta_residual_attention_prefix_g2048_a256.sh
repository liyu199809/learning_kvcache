#!/usr/bin/env bash
# Hybrid Delta K/V/beta/a + residual attention soft-prefix 8-GPU AWM OPSD run:
# 6 GPUs for actor/rollout and two teacher GPUs for privileged distillation.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
VERL_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
REPO_ROOT=$(cd -- "$VERL_ROOT/.." && pwd)

source "$REPO_ROOT/.venv/bin/activate"
cd "$REPO_ROOT"

ACTOR_MODEL=${ACTOR_MODEL:-/mnt/storage/disk1/verl_data/base_model/Qwen3.5-4B-HybridDeltaResidualAttentionPrefix-G2048-A256}
TEACHER_MODEL=${TEACHER_MODEL:-/mnt/storage/disk1/verl_data/base_model/Qwen3.5-4B}
AWM_BASE_URL=${AWM_BASE_URL:-http://localhost:8899}
SOURCE_JSONL=${SOURCE_JSONL:-$REPO_ROOT/traj_data/task1/swift_opsd.jsonl}
DATA_DIR=${DATA_DIR:-$REPO_ROOT/traj_data/awm_opsd_full_task1}
TRAIN_FILE="$DATA_DIR/train.parquet"
VAL_FILE="$DATA_DIR/val.parquet"

RUN_NAME=${RUN_NAME:-qwen3_5_4b_awm_opsd_hybrid_delta_residual_attention_prefix_g2048_a256}
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
if [[ ! -f "$SOURCE_JSONL" ]]; then
    echo "Source training JSONL is missing: $SOURCE_JSONL" >&2
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

python - "$ACTOR_MODEL" <<'PY'
import importlib.metadata
import json
import sys
from pathlib import Path

from safetensors import safe_open

model = Path(sys.argv[1])
config = json.loads((model / "config.json").read_text())
delta_config = config.get("independent_delta_kv_prefix", {})
attention_config = config.get("residual_attention_prefix", {})
if config.get("model_type") != "qwen3_5":
    raise SystemExit(f"Expected model_type=qwen3_5, got {config.get('model_type')!r}")
if config.get("architectures") != ["Qwen3_5HybridDeltaResidualAttentionPrefixForConditionalGeneration"]:
    raise SystemExit(f"Unexpected actor architecture: {config.get('architectures')!r}")
if delta_config.get("num_virtual_tokens") != 2048:
    raise SystemExit(f"Expected 2048 Delta virtual tokens, got {delta_config.get('num_virtual_tokens')!r}")
if attention_config.get("num_virtual_tokens") != 256:
    raise SystemExit(f"Expected 256 attention virtual tokens, got {attention_config.get('num_virtual_tokens')!r}")
weights_file = model / "independent_delta_kv_prefix.safetensors"
with safe_open(weights_file, framework="pt", device="cpu") as handle:
    shapes = {name: tuple(handle.get_slice(name).get_shape()) for name in handle.keys()}
expected_suffix_shapes = {
    ".prefix_k": (2048, 2048),
    ".prefix_v": (2048, 4096),
    ".prefix_beta_logits": (2048, 32),
    ".prefix_a": (2048, 32),
}
if len(shapes) != 96:
    raise SystemExit(f"Expected 96 prefix tensors, got {len(shapes)}")
for name, shape in shapes.items():
    matches = [expected for suffix, expected in expected_suffix_shapes.items() if name.endswith(suffix)]
    if len(matches) != 1 or shape != matches[0]:
        raise SystemExit(f"Unexpected prefix tensor {name} shaped {shape}")
if sum(shape[0] * shape[1] for shape in shapes.values()) != 305135616:
    raise SystemExit("Unexpected Independent Delta prefix parameter count")
attention_weights_file = model / "residual_attention_prefix.safetensors"
with safe_open(attention_weights_file, framework="pt", device="cpu") as handle:
    attention_shapes = {
        name: tuple(handle.get_slice(name).get_shape()) for name in handle.keys()
    }
if len(attention_shapes) != 16:
    raise SystemExit(f"Expected 16 attention prefix tensors, got {len(attention_shapes)}")
if any(shape != (256, 2560) for shape in attention_shapes.values()):
    raise SystemExit(f"Unexpected attention prefix shapes: {attention_shapes}")
if sum(shape[0] * shape[1] for shape in attention_shapes.values()) != 10485760:
    raise SystemExit("Unexpected residual attention prefix parameter count")
plugins = {entry.name for entry in importlib.metadata.entry_points(group="vllm.general_plugins")}
if "qwen35_delta_virtual_prefix" not in plugins:
    raise SystemExit(
        "Missing vLLM plugin. Run: /root/.local/bin/uv pip install "
        "--python .venv/bin/python -e prefix_tuning"
    )
print("Validated G=2048/A=256 hybrid prefix actor and native vLLM plugin")
PY

if [[ ! -f "$TRAIN_FILE" || ! -f "$VAL_FILE" || ! -f "$DATA_DIR/split_manifest.json" ]]; then
    python "$SCRIPT_DIR/prepare_awm_opsd_smoke_data.py" \
        --input-jsonl "$SOURCE_JSONL" \
        --output-dir "$DATA_DIR" \
        --all-rows \
        --val-size 100 \
        --val-seed 42 \
        --progress-every 100 \
        --model "$TEACHER_MODEL" \
        --awm-base-url "$AWM_BASE_URL" \
        --max-prompt-tokens "$MAX_PROMPT_LENGTH"
fi

MAX_DATA_PROMPT=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["prompt_tokens"]["max"])' "$DATA_DIR/split_manifest.json")
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
echo "Actor: $ACTOR_MODEL (112 hybrid prefix tensors; 315,621,376 parameters)"
echo "Teacher: $TEACHER_MODEL"
echo "Checkpoints: $CHECKPOINT_DIR"
echo "Rollout trajectories: $ROLLOUT_DATA_DIR"
echo "Log: $LOG_FILE"

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
    actor_rollout_ref.model.expected_trainable_param_count=112 \
    actor_rollout_ref.model.expected_trainable_numel=315621376 \
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
