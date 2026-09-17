#!/usr/bin/env bash
# Manage the CPU-only DeepCoder/TACO OpenEnv judge on port 8901.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-$SCRIPT_DIR}
OPENENV_ROOT=${OPENENV_ROOT:-$PROJECT_ROOT/OpenEnv}
PYTHON_BIN=${PYTHON_BIN:-$(command -v python)}
CODE_JUDGE_PORT=${CODE_JUDGE_PORT:-8901}
DEEPCODER_TACO_DATA_DIR=${DEEPCODER_TACO_DATA_DIR:-$PROJECT_ROOT/data/deepcoder_taco/177913a7bd43791646ef6a43645caa3c871ab3db}
CODE_JUDGE_LOG=${CODE_JUDGE_LOG:-/tmp/code_judge_openenv.log}
CODE_JUDGE_PID_FILE=${CODE_JUDGE_PID_FILE:-/tmp/code_judge_openenv.pid}
READY_TIMEOUT=${CODE_JUDGE_READY_TIMEOUT:-180}

is_alive() {
    local pid=$1
    [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null
}

is_service_pid() {
    local pid=$1 cmdline
    is_alive "$pid" || return 1
    [[ -r "/proc/$pid/cmdline" ]] || return 1
    cmdline=$(tr '\0' ' ' <"/proc/$pid/cmdline")
    [[ "$cmdline" == *"envs.code_judge_env.server.app:app"* ]] &&
        [[ "$cmdline" == *"--port $CODE_JUDGE_PORT"* ]]
}

find_service_pid() {
    local candidate
    while read -r candidate; do
        if is_service_pid "$candidate"; then
            echo "$candidate"
            return 0
        fi
    done < <(pgrep -f "envs.code_judge_env.server.app:app" 2>/dev/null || true)
    return 1
}

wait_ready() {
    local started now
    started=$(date +%s)
    while true; do
        if curl --fail --silent --show-error \
            "http://127.0.0.1:$CODE_JUDGE_PORT/dataset" >/dev/null; then
            return 0
        fi
        now=$(date +%s)
        if (( now - started >= READY_TIMEOUT )); then
            return 1
        fi
        sleep 1
    done
}

start_service() {
    local existing_pid
    existing_pid=$(find_service_pid || true)
    if [[ -n "$existing_pid" ]] && wait_ready; then
        echo "$existing_pid" >"$CODE_JUDGE_PID_FILE"
        echo "CodeJudge OpenEnv is already running (pid $existing_pid)."
        return 0
    fi
    if [[ ! -d "$DEEPCODER_TACO_DATA_DIR" ]]; then
        echo "DeepCoder/TACO data directory does not exist: $DEEPCODER_TACO_DATA_DIR" >&2
        return 1
    fi
    (
        cd "$OPENENV_ROOT"
        exec env DEEPCODER_TACO_DATA_DIR="$DEEPCODER_TACO_DATA_DIR" \
            PYTHONPATH="$OPENENV_ROOT/src:$OPENENV_ROOT/envs:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
            "$PYTHON_BIN" -m uvicorn envs.code_judge_env.server.app:app \
                --host 0.0.0.0 --port "$CODE_JUDGE_PORT" \
                --ws-ping-interval 20 --ws-ping-timeout 300
    ) >"$CODE_JUDGE_LOG" 2>&1 </dev/null &
    local pid=$!
    echo "$pid" >"$CODE_JUDGE_PID_FILE"
    if wait_ready; then
        echo "CodeJudge OpenEnv ready at http://127.0.0.1:$CODE_JUDGE_PORT (pid $pid)."
        return 0
    fi
    echo "CodeJudge OpenEnv failed to become ready; log: $CODE_JUDGE_LOG" >&2
    tail -40 "$CODE_JUDGE_LOG" >&2 || true
    return 1
}

stop_service() {
    local pid=""
    if [[ -f "$CODE_JUDGE_PID_FILE" ]] && is_service_pid "$(<"$CODE_JUDGE_PID_FILE")"; then
        pid=$(<"$CODE_JUDGE_PID_FILE")
    else
        pid=$(find_service_pid || true)
    fi
    if [[ -z "$pid" ]]; then
        rm -f "$CODE_JUDGE_PID_FILE"
        echo "CodeJudge OpenEnv is not running."
        return 0
    fi
    kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 20); do
        is_alive "$pid" || break
        sleep 0.25
    done
    is_alive "$pid" && kill -KILL "$pid" 2>/dev/null || true
    rm -f "$CODE_JUDGE_PID_FILE"
    echo "CodeJudge OpenEnv stopped."
}

status_service() {
    if curl --fail --silent --show-error \
        "http://127.0.0.1:$CODE_JUDGE_PORT/dataset"; then
        echo
        return 0
    fi
    echo "CodeJudge OpenEnv is unavailable on port $CODE_JUDGE_PORT."
    return 1
}

case "${1:-status}" in
    start) start_service ;;
    stop) stop_service ;;
    restart) stop_service; start_service ;;
    status) status_service ;;
    logs) tail -f "$CODE_JUDGE_LOG" ;;
    *) echo "Usage: $0 {start|stop|restart|status|logs}" >&2; exit 2 ;;
esac
