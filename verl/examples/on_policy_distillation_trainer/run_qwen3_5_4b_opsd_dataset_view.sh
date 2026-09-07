#!/usr/bin/env bash
# Train one OPSD data view (AWM, EnvScaler, TACO, or their balanced mixture).
# RUN_MODE=smoke runs one real optimizer step on 2 actor GPUs + 1 teacher GPU.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
VERL_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
REPO_ROOT=$(cd -- "$VERL_ROOT/.." && pwd)

source "$REPO_ROOT/.venv/bin/activate"
cd "$REPO_ROOT"

VIEW=${1:-${DATASET_VIEW:-}}
if [[ -z "$VIEW" ]]; then
    echo "Usage: $0 {awm|envscaler|taco|mixed} [Hydra overrides ...]" >&2
    exit 2
fi
shift || true

DATA_ROOT=${DATA_ROOT:-$REPO_ROOT/traj_data/opsd_mixed_1625_v1}
MODEL=${MODEL:-/mnt/storage/disk1/verl_data/base_model/Qwen3.5-4B}
TEACHER_MODEL=${TEACHER_MODEL:-$MODEL}
RUN_MODE=${RUN_MODE:-full}

case "$VIEW" in
    awm)
        TRAIN_FILE="$DATA_ROOT/sources/awm/train.parquet"
        EXPECTED_SOURCE=awm
        ENABLE_THINKING=${ENABLE_THINKING:-True}
        MAX_ASSISTANT_TURNS=${MAX_ASSISTANT_TURNS:-20}
        MAX_USER_TURNS=${MAX_USER_TURNS:-19}
        MAX_TOOL_RESPONSE_LENGTH=${MAX_TOOL_RESPONSE_LENGTH:-800}
        ;;
    envscaler)
        TRAIN_FILE="$DATA_ROOT/sources/envscaler/train.parquet"
        EXPECTED_SOURCE=envscaler_rl
        ENABLE_THINKING=${ENABLE_THINKING:-True}
        # Sixteen user/tool-response turns correspond to the requested
        # sixteen-round EnvScaler tool budget.
        MAX_ASSISTANT_TURNS=${MAX_ASSISTANT_TURNS:-17}
        MAX_USER_TURNS=${MAX_USER_TURNS:-16}
        MAX_TOOL_RESPONSE_LENGTH=${MAX_TOOL_RESPONSE_LENGTH:-4000}
        ;;
    taco)
        TRAIN_FILE="$DATA_ROOT/sources/taco/train.parquet"
        EXPECTED_SOURCE=deepcoder_taco
        ENABLE_THINKING=${ENABLE_THINKING:-True}
        MAX_ASSISTANT_TURNS=${MAX_ASSISTANT_TURNS:-1}
        MAX_USER_TURNS=${MAX_USER_TURNS:-1}
        MAX_TOOL_RESPONSE_LENGTH=${MAX_TOOL_RESPONSE_LENGTH:-800}
        ;;
    mixed)
        TRAIN_FILE="$DATA_ROOT/train.parquet"
        EXPECTED_SOURCE=mixed
        # Thinking stays enabled in the mixed run so TACO isn't forced into
        # direct-answer mode. Tool tasks are bounded by their system prompts.
        ENABLE_THINKING=${ENABLE_THINKING:-True}
        MAX_ASSISTANT_TURNS=${MAX_ASSISTANT_TURNS:-20}
        MAX_USER_TURNS=${MAX_USER_TURNS:-19}
        MAX_TOOL_RESPONSE_LENGTH=${MAX_TOOL_RESPONSE_LENGTH:-4000}
        ;;
    *)
        echo "Unknown data view '$VIEW'; expected awm, envscaler, taco, or mixed." >&2
        exit 2
        ;;
esac

VAL_FILE=${VAL_FILE:-$DATA_ROOT/val.parquet}
PROJECT_NAME=${PROJECT_NAME:-self_evolver_opsd_3way}

if [[ "$RUN_MODE" == smoke ]]; then
    ACTOR_GPUS=${ACTOR_GPUS:-2}
    TEACHER_GPUS=${TEACHER_GPUS:-1}
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2}
    TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-2}
    VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-2}
    TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
    TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-1}
    MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}
    MAX_MODEL_LEN=${MAX_MODEL_LEN:-40960}
    PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-40960}
    GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.40}
    NUM_WORKERS=${NUM_WORKERS:-2}
    FILTER_WORKERS=${FILTER_WORKERS:-1}
    DATALOADER_WORKERS=${DATALOADER_WORKERS:-0}
    VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
    TEST_FREQ=${TEST_FREQ:--1}
    SAVE_FREQ=${SAVE_FREQ:--1}
    RESUME_MODE=${RESUME_MODE:-disable}
    LOGGER_OVERRIDE=trainer.logger=console
else
    if [[ "$RUN_MODE" != full ]]; then
        echo "RUN_MODE must be 'full' or 'smoke'; got '$RUN_MODE'." >&2
        exit 2
    fi
    ACTOR_GPUS=${ACTOR_GPUS:-6}
    TEACHER_GPUS=${TEACHER_GPUS:-2}
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
    TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-24}
    # Evaluate the complete 100-row validation set in one request wave. The
    # agent-loop dispatcher selects an exact worker divisor, so no samples are
    # padded or regenerated.
    VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-100}
    TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
    TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-}
    # All four runs share a validation set containing TACO. These settings are
    # global to a trainer process, so every run must use the TACO-safe budget
    # and thinking mode for comparable validation metrics.
    MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-16384}
    MAX_MODEL_LEN=${MAX_MODEL_LEN:-49152}
    PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-49152}
    GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.65}
    NUM_WORKERS=${NUM_WORKERS:-24}
    FILTER_WORKERS=${FILTER_WORKERS:-4}
    DATALOADER_WORKERS=${DATALOADER_WORKERS:-4}
    VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}
    TEST_FREQ=${TEST_FREQ:-20}
    SAVE_FREQ=${SAVE_FREQ:-20}
    RESUME_MODE=${RESUME_MODE:-auto}
    LOGGER_OVERRIDE='trainer.logger=["console","wandb"]'
fi

if [[ "$RUN_MODE" == smoke ]]; then
    DEFAULT_RUN_NAME=qwen3_5_4b_opsd_${VIEW}_smoke
else
    DEFAULT_RUN_NAME=qwen3_5_4b_opsd_${VIEW}_1625
fi
RUN_NAME=${RUN_NAME:-$DEFAULT_RUN_NAME}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-$REPO_ROOT/checkpoints/$PROJECT_NAME/$RUN_NAME}
ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-$REPO_ROOT/rollout_trajs/$RUN_NAME}

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-28672}
ACTOR_LR=${ACTOR_LR:-1e-6}
DISTILLATION_TOPK=${DISTILLATION_TOPK:-64}
# TACO can produce 16K-token responses. Keep the mathematically equivalent
# full-vocabulary logsumexp in small token chunks so its fp32 scratch buffer
# cannot consume several GiB during the distillation update.
DISTILLATION_TOPK_CHUNK_SIZE=${DISTILLATION_TOPK_CHUNK_SIZE:-512}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-$TRAIN_BATCH_SIZE}
ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-False}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-49152}

if [[ ! -f "$MODEL/config.json" ]]; then
    echo "Model checkpoint is missing or invalid: $MODEL" >&2
    exit 1
fi
if [[ ! -f "$TEACHER_MODEL/config.json" ]]; then
    echo "Teacher model checkpoint is missing or invalid: $TEACHER_MODEL" >&2
    exit 1
fi
if [[ ! -f "$TRAIN_FILE" || ! -f "$VAL_FILE" ]]; then
    echo "Dataset files are missing under $DATA_ROOT" >&2
    exit 1
fi
if (( MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 1 > MAX_MODEL_LEN )); then
    echo "MAX_MODEL_LEN=$MAX_MODEL_LEN is smaller than prompt+response+1=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 1))." >&2
    exit 1
fi

# Validate the selected data view and discover every OpenEnv endpoint needed by
# train and validation. This catches stale/misrouted parquet before GPUs start.
mapfile -t DATA_INFO < <(python - "$TRAIN_FILE" "$VAL_FILE" "$EXPECTED_SOURCE" <<'PY'
import sys
from urllib.parse import urlsplit

import pyarrow.parquet as pq

train_path, val_path, expected = sys.argv[1:]
train = pq.read_table(train_path, columns=["data_source", "env_config"])
val = pq.read_table(val_path, columns=["data_source", "env_config"])
train_sources = sorted(set(train["data_source"].to_pylist()))
if expected != "mixed" and train_sources != [expected]:
    raise SystemExit(f"unexpected train data_source values: {train_sources}; expected {expected}")
if expected == "mixed" and train_sources != ["awm", "deepcoder_taco", "envscaler_rl"]:
    raise SystemExit(f"unexpected mixed data_source values: {train_sources}")
val_sources = sorted(set(val["data_source"].to_pylist()))
if val_sources != ["deepcoder_taco", "envscaler_rl"]:
    raise SystemExit(f"validation must contain only EnvScaler and TACO; got {val_sources}")
print(f"TRAIN_ROWS={train.num_rows}")
print(f"VAL_ROWS={val.num_rows}")
urls = set()
for table in (train, val):
    for config in table["env_config"].to_pylist():
        url = str(config["awm_base_url"]).rstrip("/")
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise SystemExit(f"invalid awm_base_url in parquet: {url!r}")
        urls.add(url)
for url in sorted(urls):
    print(f"ENV_URL={url}")
PY
)

TRAIN_ROWS=0
VAL_ROWS=0
ENV_URLS=()
for item in "${DATA_INFO[@]}"; do
    case "$item" in
        TRAIN_ROWS=*) TRAIN_ROWS=${item#TRAIN_ROWS=} ;;
        VAL_ROWS=*) VAL_ROWS=${item#VAL_ROWS=} ;;
        ENV_URL=*) ENV_URLS+=("${item#ENV_URL=}") ;;
    esac
done
for url in "${ENV_URLS[@]}"; do
    if ! curl --fail --silent --show-error --max-time 5 "$url/health" >/dev/null; then
        echo "OpenEnv is not healthy at $url" >&2
        exit 1
    fi
done

if [[ "$RUN_MODE" == full ]]; then
    if ! python -c 'import wandb' >/dev/null 2>&1; then
        echo "wandb is not installed in $REPO_ROOT/.venv" >&2
        exit 1
    fi
    if ! grep -q 'machine api.wandb.ai' "$HOME/.netrc" 2>/dev/null && [[ -z "${WANDB_API_KEY:-}" ]]; then
        echo "No W&B credentials found. Run wandb login or set WANDB_API_KEY." >&2
        exit 1
    fi
fi

EXPECTED_GPUS=$((ACTOR_GPUS + TEACHER_GPUS))
if [[ $(awk -F, '{print NF}' <<<"$CUDA_VISIBLE_DEVICES") -ne "$EXPECTED_GPUS" ]]; then
    echo "Expected $EXPECTED_GPUS visible GPUs; got CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES
export PYTHONPATH="$VERL_ROOT:$REPO_ROOT:$REPO_ROOT/OpenEnv/src:$REPO_ROOT/OpenEnv/envs${PYTHONPATH:+:$PYTHONPATH}"
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export WANDB_PROJECT="$PROJECT_NAME"

mkdir -p "$CHECKPOINT_DIR" "$ROLLOUT_DATA_DIR" "$REPO_ROOT/logs"
START_TIME=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$REPO_ROOT/logs/${RUN_NAME}_${RUN_MODE}_${START_TIME}.log"

echo "OPSD view=$VIEW mode=$RUN_MODE train=$TRAIN_ROWS val=$VAL_ROWS"
echo "thinking=$ENABLE_THINKING turns=$MAX_ASSISTANT_TURNS/$MAX_USER_TURNS repetition_penalty=1.0"
echo "tokens: prompt=$MAX_PROMPT_LENGTH response=$MAX_RESPONSE_LENGTH model=$MAX_MODEL_LEN"
echo "actor GPUs=$ACTOR_GPUS teacher GPUs=$TEACHER_GPUS batch=$TRAIN_BATCH_SIZE"
echo "student model=$MODEL teacher model=$TEACHER_MODEL"
echo "Log: $LOG_FILE"

if [[ "${PREFLIGHT_ONLY:-0}" == 1 ]]; then
    echo "Preflight passed; training was not started."
    exit 0
fi

STEP_OVERRIDE=()
if [[ -n "$TOTAL_TRAINING_STEPS" ]]; then
    STEP_OVERRIDE+=(trainer.total_training_steps="$TOTAL_TRAINING_STEPS")
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
    data.filter_overlong_prompts_workers="$FILTER_WORKERS" \
    data.truncation=error \
    data.shuffle="$([[ "$RUN_MODE" == smoke ]] && echo False || echo True)" \
    data.seed=42 \
    data.dataloader_num_workers="$DATALOADER_WORKERS" \
    +data.apply_chat_template_kwargs.enable_thinking="$ENABLE_THINKING" \
    actor_rollout_ref.model.path="$MODEL" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr="$ACTOR_LR" \
    actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$PPO_MAX_TOKEN_LEN_PER_GPU" \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True \
    actor_rollout_ref.actor.fsdp_config.entropy_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.entropy_from_logits_with_chunking=False \
    actor_rollout_ref.actor.fsdp_config.offload_policy=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEMORY_UTILIZATION" \
    actor_rollout_ref.rollout.enforce_eager="$ROLLOUT_ENFORCE_EAGER" \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.do_sample=True \
    actor_rollout_ref.rollout.temperature=1.0 \
    +actor_rollout_ref.rollout.repetition_penalty=1.0 \
    actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN" \
    actor_rollout_ref.rollout.max_num_batched_tokens="$MAX_NUM_BATCHED_TOKENS" \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$PPO_MAX_TOKEN_LEN_PER_GPU" \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.format=qwen3_coder \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns="$MAX_ASSISTANT_TURNS" \
    actor_rollout_ref.rollout.multi_turn.max_user_turns="$MAX_USER_TURNS" \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=3 \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length="$MAX_TOOL_RESPONSE_LENGTH" \
    actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side=right \
    actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode=ignore_strippable \
    actor_rollout_ref.rollout.agent.default_agent_loop=awm_agent \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="$REPO_ROOT/rollout/awm_agent_loop.yaml" \
    actor_rollout_ref.rollout.agent.num_workers="$NUM_WORKERS" \
    "$LOGGER_OVERRIDE" \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$RUN_NAME" \
    trainer.default_local_dir="$CHECKPOINT_DIR" \
    trainer.rollout_data_dir="$ROLLOUT_DATA_DIR" \
    trainer.log_val_generations=10 \
    trainer.balance_batch=True \
    trainer.n_gpus_per_node="$ACTOR_GPUS" \
    trainer.nnodes=1 \
    trainer.val_before_train="$VAL_BEFORE_TRAIN" \
    trainer.test_freq="$TEST_FREQ" \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.max_actor_ckpt_to_keep=3 \
    trainer.max_critic_ckpt_to_keep=3 \
    trainer.total_epochs="$TOTAL_EPOCHS" \
    trainer.resume_mode="$RESUME_MODE" \
    trainer.use_v1=False \
    distillation.enabled=True \
    distillation.self_distillation=True \
    distillation.n_gpus_per_node="$TEACHER_GPUS" \
    distillation.nnodes=1 \
    distillation.teacher_models.teacher_model.model_path="$TEACHER_MODEL" \
    distillation.teacher_models.teacher_model.inference.name=vllm \
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=1 \
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization="$GPU_MEMORY_UTILIZATION" \
    distillation.teacher_models.teacher_model.inference.enforce_eager=True \
    distillation.teacher_models.teacher_model.inference.max_model_len="$MAX_MODEL_LEN" \
    distillation.privileged_mode=append \
    distillation.privileged_insert_before=$'"<|im_end|>\n<|im_start|>assistant\n"' \
    distillation.privileged_solution_key=reward_model.ground_truth \
    distillation.distillation_loss.loss_mode=forward_kl_topk \
    distillation.distillation_loss.topk="$DISTILLATION_TOPK" \
    distillation.distillation_loss.use_policy_gradient=False \
    +distillation.distillation_loss.use_chunked_topk=True \
    +distillation.distillation_loss.chunked_topk_chunk_size="$DISTILLATION_TOPK_CHUNK_SIZE" \
    distillation.distillation_loss.use_task_rewards=False \
    distillation.distillation_loss.loss_max_clamp=10.0 \
    distillation.distillation_loss.log_prob_min_clamp=-10.0 \
    "${STEP_OVERRIDE[@]}" \
    "$@" 2>&1 | tee "$LOG_FILE"
