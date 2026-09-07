#!/usr/bin/env bash
# Start / stop / check the two backend services for AWM eval:
#   1) vLLM OpenAI-compatible server (Qwen3.5-4B) on :$VLLM_PORT
#   2) AWM environment server (FastAPI + MCP) on :$AWM_PORT
#
# Usage:
#   ./start_services.sh start               # start both (or only what's missing)
#   ./start_services.sh start vllm          # start only vLLM
#   ./start_services.sh start tool          # start only the AWM tool/env server
#   ./start_services.sh stop                # stop both
#   ./start_services.sh stop vllm           # stop only vLLM
#   ./start_services.sh stop tool           # stop only the AWM tool/env server
#   ./start_services.sh restart [vllm|tool] # stop then start (optionally one)
#   ./start_services.sh status              # report PIDs + health
#   ./start_services.sh logs vllm           # tail -f vllm log
#   ./start_services.sh logs tool           # tail -f awm log

set -euo pipefail

# ---------------------------------------------------------------------------
# Config (override via env vars)
# ---------------------------------------------------------------------------
PROJECT_ROOT="${PROJECT_ROOT:-/mnt/storage/disk3/self_evolver}"
VENV_PATH="${VENV_PATH:-$PROJECT_ROOT/.venv}"
OPENENV_ROOT="${OPENENV_ROOT:-$PROJECT_ROOT/OpenEnv}"

MODEL_PATH="${MODEL_PATH:-/mnt/storage/disk1/verl_data/base_model/Qwen3.5-4B}" # /mnt/storage/disk1/verl_data/base_model/Qwen3.5-4B
MODEL_NAME="${MODEL_NAME:-qwen3.5-4b}"

VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_GPU="${VLLM_GPU:-0,1,2,3,4,5,6,7}"
VLLM_MAX_LEN="${VLLM_MAX_LEN:-262144}"
VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.85}"
VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
VLLM_DP="${VLLM_DP:-8}"                        # data-parallel replicas
VLLM_TP="${VLLM_TP:-1}"                        # tensor-parallel size
VLLM_EXECUTOR_BACKEND="${VLLM_EXECUTOR_BACKEND:-uni}"  # TP=1: avoid needless worker subprocesses
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-256}"  # per-replica batch concurrency
VLLM_API_SERVER_COUNT="${VLLM_API_SERVER_COUNT:-$VLLM_DP}"  # one frontend per local DP replica

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

terminate_tree() {
    # vLLM DP creates API server, engine-core, worker and resource-tracker
    # descendants. Killing only the immediate child can leave a rendezvous
    # listener behind and make the next model restart fail with EADDRINUSE.
    local pid=$1 signal=${2:-TERM} child
    while read -r child; do
        [[ -n "$child" ]] && terminate_tree "$child" "$signal"
    done < <(pgrep -P "$pid" 2>/dev/null || true)
    kill -"$signal" "$pid" 2>/dev/null || true
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

# Resolve CLI service selector(s) (vllm|tool) into the internal service ids
# used by this script (vllm|awm). No selector => all services.
# Fills the global array SERVICES.
SERVICES=()
parse_services() {
    SERVICES=()
    local arg
    for arg in "$@"; do
        case "$arg" in
            vllm) SERVICES+=("vllm") ;;
            tool) SERVICES+=("awm") ;;
            *) err "unknown service '$arg' (expected: vllm|tool)"; exit 2 ;;
        esac
    done
    if [[ ${#SERVICES[@]} -eq 0 ]]; then
        SERVICES=("vllm" "awm")
    fi
}

# ---------------------------------------------------------------------------
# vLLM
# ---------------------------------------------------------------------------
start_vllm() {
    mkdir -p "$LOG_DIR"
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
            --api-server-count $VLLM_API_SERVER_COUNT \
            --tensor-parallel-size $VLLM_TP \
            --distributed-executor-backend $VLLM_EXECUTOR_BACKEND \
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
            terminate_tree "$pid" TERM
            sleep 2
            kill -0 "$pid" 2>/dev/null && terminate_tree "$pid" KILL || true
            ok "vLLM stopped."
        fi
        rm -f "$VLLM_PID_FILE"
    fi
    # Extra sweep for this service instance only.  Multiple evaluators may
    # serve the same model path on different GPUs/ports, so matching only the
    # model path would terminate peer shards.
    pkill -f "vllm serve $MODEL_PATH.*--port $VLLM_PORT" 2>/dev/null || true
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
                --host 0.0.0.0 --port $AWM_PORT \
                --ws-ping-interval 20 --ws-ping-timeout 300
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
        parse_services "${@:2}"
        for svc in "${SERVICES[@]}"; do
            if [[ "$svc" == "vllm" ]]; then start_vllm; else start_awm; fi
        done
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
        parse_services "${@:2}"
        for svc in "${SERVICES[@]}"; do
            if [[ "$svc" == "vllm" ]]; then stop_vllm; else stop_awm; fi
        done
        ;;
    restart)
        parse_services "${@:2}"
        for svc in "${SERVICES[@]}"; do
            if [[ "$svc" == "vllm" ]]; then stop_vllm; else stop_awm; fi
        done
        sleep 2
        for svc in "${SERVICES[@]}"; do
            if [[ "$svc" == "vllm" ]]; then start_vllm; else start_awm; fi
        done
        status
        ;;
    status)
        status
        ;;
    logs)
        case "${2:-}" in
            vllm)     exec tail -n 200 -f "$VLLM_LOG" ;;
            tool|awm) exec tail -n 200 -f "$AWM_LOG" ;;
            *)        err "usage: $0 logs {vllm|tool}"; exit 2 ;;
        esac
        ;;
    *)
        cat <<EOF >&2
Usage: $0 {start|stop|restart} [vllm|tool]
       $0 status
       $0 logs {vllm|tool}

  start/stop/restart with no service selector act on BOTH services.
  vllm = vLLM OpenAI server;  tool = AWM tool/env server.

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
