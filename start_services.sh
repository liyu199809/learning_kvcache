#!/usr/bin/env bash
# Start / stop / check the two backend services for AWM eval:
#   1) vLLM OpenAI-compatible server (Qwen3.5-4B) on :$VLLM_PORT
#   2) AWM environment server (FastAPI + MCP) on :$AWM_PORT
#
# Usage:
#   ./start_services.sh start        # start both (or only what's missing)
#   ./start_services.sh stop         # stop both
#   ./start_services.sh restart      # stop then start
#   ./start_services.sh status       # report PIDs + health
#   ./start_services.sh logs vllm    # tail -f vllm log
#   ./start_services.sh logs awm     # tail -f awm log

set -euo pipefail

# ---------------------------------------------------------------------------
# Config (override via env vars)
# ---------------------------------------------------------------------------
PROJECT_ROOT="${PROJECT_ROOT:-/mnt/storage/disk3/self_evolver}"
VENV_PATH="${VENV_PATH:-$PROJECT_ROOT/.venv}"
OPENENV_ROOT="${OPENENV_ROOT:-$PROJECT_ROOT/OpenEnv}"

MODEL_PATH="${MODEL_PATH:-/mnt/storage/disk1/verl_data/base_model/Qwen3.5-4B}"
MODEL_NAME="${MODEL_NAME:-qwen3.5-4b}"

VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_GPU="${VLLM_GPU:-0}"
VLLM_MAX_LEN="${VLLM_MAX_LEN:-262144}"
VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.85}"
VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
VLLM_DP="${VLLM_DP:-1}"                        # data-parallel replicas
VLLM_TP="${VLLM_TP:-1}"                        # tensor-parallel size
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-256}"  # per-replica batch concurrency

AWM_PORT="${AWM_PORT:-8899}"
AWM_DATA_DIR="${AWM_DATA_DIR:-$PROJECT_ROOT/awm_data}"

LOG_DIR="${LOG_DIR:-/tmp}"
VLLM_LOG="$LOG_DIR/vllm.log"
AWM_LOG="$LOG_DIR/awm.log"
VLLM_PID_FILE="$LOG_DIR/vllm.pid"
AWM_PID_FILE="$LOG_DIR/awm.pid"

VLLM_READY_TIMEOUT="${VLLM_READY_TIMEOUT:-600}"
AWM_READY_TIMEOUT="${AWM_READY_TIMEOUT:-120}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
c_reset='\033[0m'; c_red='\033[31m'; c_green='\033[32m'; c_yellow='\033[33m'; c_cyan='\033[36m'
info()  { printf "${c_cyan}[info]${c_reset} %s\n" "$*"; }
ok()    { printf "${c_green}[ ok ]${c_reset} %s\n" "$*"; }
warn()  { printf "${c_yellow}[warn]${c_reset} %s\n" "$*"; }
err()   { printf "${c_red}[err ]${c_reset} %s\n" "$*" >&2; }

is_alive() {
    local pid=$1
    [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null
}

port_in_use() {
    local port=$1
    ss -ltn "sport = :$port" 2>/dev/null | grep -q ":$port" \
        || lsof -iTCP:"$port" -sTCP:LISTEN -Pn >/dev/null 2>&1
}

wait_http() {
    # wait_http <url> <timeout_sec> <name>
    local url=$1 timeout=$2 name=$3
    local start=$(date +%s)
    while true; do
        if curl -sf -o /dev/null "$url"; then
            ok "$name ready at $url"
            return 0
        fi
        local now=$(date +%s)
        if (( now - start > timeout )); then
            err "$name did not become ready within ${timeout}s at $url"
            return 1
        fi
        sleep 2
    done
}

# ---------------------------------------------------------------------------
# vLLM
# ---------------------------------------------------------------------------
start_vllm() {
    if [[ -f "$VLLM_PID_FILE" ]] && is_alive "$(cat "$VLLM_PID_FILE")"; then
        warn "vLLM already running (pid $(cat "$VLLM_PID_FILE")); skipping."
        return 0
    fi
    if port_in_use "$VLLM_PORT"; then
        warn "port $VLLM_PORT already in use; skipping vLLM start."
        return 0
    fi

    info "Starting vLLM on GPU=$VLLM_GPU DP=$VLLM_DP TP=$VLLM_TP port=$VLLM_PORT model=$MODEL_PATH"
    nohup bash -c "
        source '$VENV_PATH/bin/activate' && \
        CUDA_VISIBLE_DEVICES=$VLLM_GPU vllm serve '$MODEL_PATH' \
            --served-model-name '$MODEL_NAME' \
            --host 0.0.0.0 --port $VLLM_PORT \
            --max-model-len $VLLM_MAX_LEN \
            --gpu-memory-utilization $VLLM_GPU_UTIL \
            --dtype $VLLM_DTYPE \
            --data-parallel-size $VLLM_DP \
            --tensor-parallel-size $VLLM_TP \
            --max-num-seqs $VLLM_MAX_NUM_SEQS \
            --enable-auto-tool-choice \
            --reasoning-parser qwen3 \
            --tool-call-parser qwen3_coder 
    " >"$VLLM_LOG" 2>&1 </dev/null &
    local pid=$!
    disown "$pid" 2>/dev/null || true
    echo "$pid" > "$VLLM_PID_FILE"
    info "vLLM launched (pid $pid); log: $VLLM_LOG"

    if wait_http "http://127.0.0.1:$VLLM_PORT/v1/models" "$VLLM_READY_TIMEOUT" "vLLM"; then
        return 0
    else
        err "vLLM failed to come up. Tail of log:"
        tail -30 "$VLLM_LOG" >&2 || true
        return 1
    fi
}

stop_vllm() {
    if [[ -f "$VLLM_PID_FILE" ]]; then
        local pid; pid=$(cat "$VLLM_PID_FILE")
        if is_alive "$pid"; then
            info "Killing vLLM tree (pid $pid)..."
            pkill -TERM -P "$pid" 2>/dev/null || true
            kill -TERM "$pid" 2>/dev/null || true
            sleep 2
            kill -0 "$pid" 2>/dev/null && kill -KILL "$pid" 2>/dev/null || true
            ok "vLLM stopped."
        fi
        rm -f "$VLLM_PID_FILE"
    fi
    # extra sweep: any lingering vllm serve owned by us
    pkill -f "vllm serve $MODEL_PATH" 2>/dev/null || true
}

# ---------------------------------------------------------------------------
# AWM env
# ---------------------------------------------------------------------------
start_awm() {
    if [[ -f "$AWM_PID_FILE" ]] && is_alive "$(cat "$AWM_PID_FILE")"; then
        warn "AWM already running (pid $(cat "$AWM_PID_FILE")); skipping."
        return 0
    fi
    if port_in_use "$AWM_PORT"; then
        warn "port $AWM_PORT already in use; skipping AWM start."
        return 0
    fi

    info "Starting AWM env server on port $AWM_PORT, data=$AWM_DATA_DIR"
    nohup bash -c "
        source '$VENV_PATH/bin/activate' && \
        cd '$OPENENV_ROOT' && \
        AWM_DATA_DIR='$AWM_DATA_DIR' PYTHONPATH=src:envs \
            '$VENV_PATH/bin/uvicorn' envs.agent_world_model_env.server.app:app \
                --host 0.0.0.0 --port $AWM_PORT
    " >"$AWM_LOG" 2>&1 </dev/null &
    local pid=$!
    disown "$pid" 2>/dev/null || true
    echo "$pid" > "$AWM_PID_FILE"
    info "AWM launched (pid $pid); log: $AWM_LOG"

    if wait_http "http://127.0.0.1:$AWM_PORT/health" "$AWM_READY_TIMEOUT" "AWM"; then
        return 0
    else
        err "AWM failed to come up. Tail of log:"
        tail -30 "$AWM_LOG" >&2 || true
        return 1
    fi
}

stop_awm() {
    if [[ -f "$AWM_PID_FILE" ]]; then
        local pid; pid=$(cat "$AWM_PID_FILE")
        if is_alive "$pid"; then
            info "Killing AWM tree (pid $pid)..."
            pkill -TERM -P "$pid" 2>/dev/null || true
            kill -TERM "$pid" 2>/dev/null || true
            sleep 2
            kill -0 "$pid" 2>/dev/null && kill -KILL "$pid" 2>/dev/null || true
            ok "AWM stopped."
        fi
        rm -f "$AWM_PID_FILE"
    fi
    pkill -f "agent_world_model_env.server.app:app" 2>/dev/null || true
}

# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------
status() {
    for svc in vllm awm; do
        local pid_file port name url
        if [[ "$svc" == "vllm" ]]; then
            pid_file="$VLLM_PID_FILE"; port="$VLLM_PORT"; name="vLLM"
            url="http://127.0.0.1:$VLLM_PORT/v1/models"
        else
            pid_file="$AWM_PID_FILE"; port="$AWM_PORT"; name="AWM"
            url="http://127.0.0.1:$AWM_PORT/health"
        fi

        local pid=""
        [[ -f "$pid_file" ]] && pid=$(cat "$pid_file")

        if [[ -n "$pid" ]] && is_alive "$pid"; then
            local healthy="?"
            if curl -sf -o /dev/null "$url"; then healthy="ok"; else healthy="unhealthy"; fi
            printf "  %-6s pid=%-7s port=%-5s status=running  health=%s\n" "$name" "$pid" "$port" "$healthy"
        elif curl -sf -o /dev/null "$url"; then
            # Service is answering on the port but we don't own its pid.
            local ext_pid
            ext_pid=$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null | head -1)
            printf "  %-6s pid=%-7s port=%-5s status=external  health=ok\n" "$name" "${ext_pid:-?}" "$port"
        else
            printf "  %-6s pid=-       port=%-5s status=stopped\n" "$name" "$port"
        fi
    done
}

# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------
cmd="${1:-start}"

case "$cmd" in
    start)
        mkdir -p "$LOG_DIR"
        start_vllm
        start_awm
        echo
        status
        echo
        ok "All services up. To run smoke eval:"
        echo "    cd $OPENENV_ROOT && source $VENV_PATH/bin/activate && \\"
        echo "    PYTHONPATH=src:envs python examples/agent_world_model/awm_smoke_eval.py \\"
        echo "        --num-scenarios 3 --tasks-per-scenario 1"
        echo
        echo "For 8-GPU DP throughput mode, restart with:"
        echo "    VLLM_GPU=0,1,2,3,4,5,6,7 VLLM_DP=8 VLLM_MAX_LEN=32768 \\"
        echo "        $0 restart"
        ;;
    stop)
        stop_vllm
        stop_awm
        ;;
    restart)
        stop_vllm; stop_awm
        sleep 2
        start_vllm; start_awm
        status
        ;;
    status)
        status
        ;;
    logs)
        case "${2:-}" in
            vllm) exec tail -n 200 -f "$VLLM_LOG" ;;
            awm)  exec tail -n 200 -f "$AWM_LOG" ;;
            *)    err "usage: $0 logs {vllm|awm}"; exit 2 ;;
        esac
        ;;
    *)
        cat <<EOF >&2
Usage: $0 {start|stop|restart|status|logs vllm|logs awm}

Environment overrides:
  MODEL_PATH, MODEL_NAME, VLLM_PORT, VLLM_GPU, VLLM_MAX_LEN,
  VLLM_GPU_UTIL, VLLM_DTYPE, VLLM_DP, VLLM_TP, VLLM_MAX_NUM_SEQS,
  AWM_PORT, AWM_DATA_DIR,
  PROJECT_ROOT, VENV_PATH, OPENENV_ROOT, LOG_DIR,
  VLLM_READY_TIMEOUT, AWM_READY_TIMEOUT
EOF
        exit 2
        ;;
esac
