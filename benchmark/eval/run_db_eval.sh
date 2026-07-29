#!/usr/bin/env bash
# 用已部署的 vLLM 服务，对 LifelongAgentBench/db 跑评测（pass@1）。
#
# 三档规模：
#   MODE=smoke   -> 10 条         (--tasks 0-9)
#   MODE=medium  -> 100 条        (--max-tasks 100)
#   MODE=full    -> 全量 500 条   (test split 全跑)
#
# 用法：
#   ./benchmark/eval/run_db_eval.sh                 # 默认 smoke
#   MODE=full ./benchmark/eval/run_db_eval.sh       # 全量
#   MODE=full CONCURRENCY=8 ./benchmark/eval/run_db_eval.sh
#   # 后台跑全量（推荐）：
#   MODE=full nohup ./benchmark/eval/run_db_eval.sh > /tmp/db_eval_full.log 2>&1 &
#
# 每个 db 任务都会起一个独立的 MySQL 容器，CONCURRENCY 不宜过高。

set -euo pipefail

# ---- paths ----------------------------------------------------------------
PROJECT_ROOT="${PROJECT_ROOT:-/mnt/storage/disk3/self_evolver}"
VENV_PATH="${VENV_PATH:-$PROJECT_ROOT/.venv}"
DATA_DIR="${DATA_DIR:-benchmark/LifelongAgentBench}"

# ---- vLLM endpoint --------------------------------------------------------
OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8000/v1}"
MODEL="${MODEL:-qwen3.5-4b}"
API_KEY="${API_KEY:-EMPTY}"

# ---- run knobs ------------------------------------------------------------
MODE="${MODE:-smoke}"                 # smoke | medium | full
CONCURRENCY="${CONCURRENCY:-8}"       # 每任务一个 MySQL 容器，别开太高
MAX_STEPS="${MAX_STEPS:-6}"
STEP_TIMEOUT="${STEP_TIMEOUT:-180}"
SPLIT="${SPLIT:-test}"
MYSQL_IMAGE="${MYSQL_IMAGE:-mysql:8.0}"   # MySQL 9.x 缺 MD5/SHA1，必须 8.0
LLM_MAX_TOKENS="${LLM_MAX_TOKENS:-2048}"

case "$MODE" in
    smoke)  SELECT_ARGS=(--tasks 0-9) ;;
    medium) SELECT_ARGS=(--max-tasks 100) ;;
    full)   SELECT_ARGS=() ;;          # 全量：不加选择参数，跑整个 split
    *) echo "unknown MODE=$MODE (want smoke|medium|full)" >&2; exit 2 ;;
esac

# ---- output ---------------------------------------------------------------
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/workspace/eval_logs/lifelong_db/${SPLIT}/${MODE}_${STAMP}}"
mkdir -p "$OUT_DIR"

echo "==== lifelong_db eval ==================================="
echo "  mode:          $MODE"
echo "  model:         $MODEL @ $OPENAI_BASE_URL"
echo "  split:         $SPLIT"
echo "  concurrency:   $CONCURRENCY   max_steps=$MAX_STEPS  step_timeout=$STEP_TIMEOUT"
echo "  mysql_image:   $MYSQL_IMAGE"
echo "  output:        $OUT_DIR"
echo "========================================================="

# ---- preflight ------------------------------------------------------------
# 1) vLLM 在线？
if ! curl -sf "${OPENAI_BASE_URL%/v1}/v1/models" >/dev/null 2>&1; then
    echo "[preflight] WARN: vLLM 端点不可达: ${OPENAI_BASE_URL} (继续尝试，失败会体现在结果里)" >&2
fi
# 2) MySQL 镜像存在？
if ! docker image inspect "$MYSQL_IMAGE" >/dev/null 2>&1; then
    echo "[preflight] 拉取 $MYSQL_IMAGE ..." >&2
    docker pull "$MYSQL_IMAGE"
fi

# ---- activate + run -------------------------------------------------------
# shellcheck disable=SC1091
source "$VENV_PATH/bin/activate"
cd "$PROJECT_ROOT"

exec python -m benchmark.eval.run_eval \
    --benchmark lifelong_db \
    --openai-base-url "$OPENAI_BASE_URL" \
    --model "$MODEL" \
    --api-key "$API_KEY" \
    --data-dir "$DATA_DIR" \
    --split "$SPLIT" \
    "${SELECT_ARGS[@]}" \
    --max-steps "$MAX_STEPS" \
    --step-timeout "$STEP_TIMEOUT" \
    --concurrency "$CONCURRENCY" \
    --mysql-image "$MYSQL_IMAGE" \
    --llm-max-completion-tokens "$LLM_MAX_TOKENS" \
    --output-dir "$OUT_DIR"
