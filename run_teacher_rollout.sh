#!/usr/bin/env bash
# Run rollout with seed-2.1-pro (Ark) as the acting model, to measure the
# teacher's pass rate on AWM. Env-server and everything else identical to
# a normal rollout — we just swap the OpenAI-compatible endpoint.
#
# Rate-limit budget on Ark:
#   500 RPM, 1M TPM. seed-2.1-pro uses ~5-7k token/round, ~60k/episode.
#   Concurrency 8 -> ~480k TPM (safe). Concurrency 16 hits the ceiling.
#
# Usage:
#   ./run_teacher_rollout.sh            # 100-scenario smoke (default)
#   MODE=smoke ./run_teacher_rollout.sh # 5 scenarios × 2 tasks
#   MODE=full  ./run_teacher_rollout.sh # full 1000 scenarios × 10 tasks
#   USE_SKILL=1 MODE=smoke ./run_teacher_rollout.sh  # add skill injection
#
# Requires ARK_API_KEY in env (or exported inline before invocation).

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/storage/disk3/self_evolver}"
VENV_PATH="${VENV_PATH:-$PROJECT_ROOT/.venv}"
OPENENV_ROOT="${OPENENV_ROOT:-$PROJECT_ROOT/OpenEnv}"

# ---- teacher endpoint (Ark, OpenAI-compatible) -----------------------------
ARK_BASE_URL="${ARK_BASE_URL:-https://ark.cn-beijing.volces.com/api/v3}"
ARK_MODEL="${ARK_MODEL:-ep-20260609014859-6th5f}"        # seed-2.1-pro
: "${ARK_API_KEY:?ARK_API_KEY must be set in env}"

# ---- rollout knobs (safe defaults for teacher rate limits) ----------------
MODE="${MODE:-medium}"          # smoke | medium | full
CONCURRENCY="${CONCURRENCY:-8}" # keep TPM below 1M/min
RESET_CONCURRENCY="${RESET_CONCURRENCY:-8}"
MAX_ITERATIONS="${MAX_ITERATIONS:-5}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
LLM_TIMEOUT="${LLM_TIMEOUT:-180}"
EPISODE_TIMEOUT="${EPISODE_TIMEOUT:-900}"
TRAJECTORY_MODE="${TRAJECTORY_MODE:-full}"

case "$MODE" in
    smoke)  NUM_SCEN=5   TASKS=2  ;;
    medium) NUM_SCEN=100 TASKS=1  ;;
    full)   NUM_SCEN=1000 TASKS=10 ;;
    *) echo "unknown MODE=$MODE (want smoke|medium|full)" >&2; exit 2 ;;
esac

# ---- skill injection (default off; teacher = baseline) --------------------
USE_SKILL="${USE_SKILL:-0}"
SKILLS_DIR="${SKILLS_DIR:-$PROJECT_ROOT/skills}"
SKILL_ARG=""
TAG="teacher"
if [[ "$USE_SKILL" == "1" ]]; then
    SKILL_ARG="--skills-dir $SKILLS_DIR"
    TAG="teacher_with_skill"
fi

# ---- output paths ---------------------------------------------------------
OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/traj_data}"
mkdir -p "$OUT_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
CKPT="$OUT_DIR/${TAG}_${MODE}_${STAMP}.jsonl"
REPORT="$OUT_DIR/${TAG}_${MODE}_${STAMP}.json"

echo "==== teacher rollout ===================================="
echo "  mode:            $MODE  ($NUM_SCEN scenarios × $TASKS tasks)"
echo "  model:           $ARK_MODEL"
echo "  base_url:        $ARK_BASE_URL"
echo "  concurrency:     $CONCURRENCY  (reset=$RESET_CONCURRENCY)"
echo "  max_iterations:  $MAX_ITERATIONS  llm_timeout=$LLM_TIMEOUT"
echo "  skill inject:    $USE_SKILL  ($SKILL_ARG)"
echo "  checkpoint:      $CKPT"
echo "  report:          $REPORT"
echo "========================================================="

# ---- activate + run -------------------------------------------------------
# shellcheck disable=SC1091
source "$VENV_PATH/bin/activate"

cd "$PROJECT_ROOT"
export PYTHONPATH="$OPENENV_ROOT/src:$OPENENV_ROOT/envs${PYTHONPATH:+:$PYTHONPATH}"

exec python -m rollout.rollout \
    --num-scenarios "$NUM_SCEN" \
    --tasks-per-scenario "$TASKS" \
    --concurrency "$CONCURRENCY" \
    --reset-concurrency "$RESET_CONCURRENCY" \
    --max-iterations "$MAX_ITERATIONS" \
    --max-tokens "$MAX_TOKENS" \
    --llm-timeout "$LLM_TIMEOUT" \
    --episode-timeout "$EPISODE_TIMEOUT" \
    --trajectory-mode "$TRAJECTORY_MODE" \
    --llm-base-url "$ARK_BASE_URL" \
    --llm-api-key "$ARK_API_KEY" \
    --llm-model "$ARK_MODEL" \
    --checkpoint-path "$CKPT" \
    --report "$REPORT" \
    $SKILL_ARG
