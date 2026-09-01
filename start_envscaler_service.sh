#!/usr/bin/env bash
# Manage the CPU-only EnvScaler OpenEnv tool service on port 8900.

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/mnt/storage/disk3/self_evolver}
VENV_PATH=${VENV_PATH:-$PROJECT_ROOT/.venv}
OPENENV_ROOT=${OPENENV_ROOT:-$PROJECT_ROOT/OpenEnv}
ENVSCALER_PORT=${ENVSCALER_PORT:-8900}
ENVSCALER_DATA_DIR=${ENVSCALER_DATA_DIR:-$PROJECT_ROOT/envscaler_data}
ENVSCALER_LOG=${ENVSCALER_LOG:-/tmp/envscaler_openenv.log}
ENVSCALER_PID_FILE=${ENVSCALER_PID_FILE:-/tmp/envscaler_openenv.pid}
READY_TIMEOUT=${ENVSCALER_READY_TIMEOUT:-180}

is_alive() {
    local pid=$1
    [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null
}

is_service_pid() {
    local pid=$1 cmdline
    is_alive "$pid" || return 1
    [[ -r "/proc/$pid/cmdline" ]] || return 1
    cmdline=$(tr '\0' ' ' <"/proc/$pid/cmdline")
    [[ "$cmdline" == *"envs.env_scaler_env.server.app:app"* ]] &&
        [[ "$cmdline" == *"--port $ENVSCALER_PORT"* ]]
}

find_service_pid() {
    local candidate
    while read -r candidate; do
        if is_service_pid "$candidate"; then
            echo "$candidate"
            return 0
        fi
    done < <(pgrep -f "envs.env_scaler_env.server.app:app" 2>/dev/null || true)
    return 1
}

wait_ready() {
    local started now
    started=$(date +%s)
    while true; do
        if curl --fail --silent --show-error \
            "http://127.0.0.1:$ENVSCALER_PORT/dataset" >/dev/null; then
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
        echo "$existing_pid" >"$ENVSCALER_PID_FILE"
        echo "EnvScaler OpenEnv is already running (pid $existing_pid)."
        return 0
    fi
    if [[ ! -d "$ENVSCALER_DATA_DIR" ]]; then
        echo "EnvScaler data directory does not exist: $ENVSCALER_DATA_DIR" >&2
        return 1
    fi
    nohup bash -c "
        cd '$OPENENV_ROOT' &&
        exec env ENVSCALER_DATA_DIR='$ENVSCALER_DATA_DIR' PYTHONPATH=src:envs \
            '$VENV_PATH/bin/uvicorn' envs.env_scaler_env.server.app:app \
                --host 0.0.0.0 --port '$ENVSCALER_PORT' \
                --ws-ping-interval 20 --ws-ping-timeout 300
    " >"$ENVSCALER_LOG" 2>&1 </dev/null &
    local pid=$!
    echo "$pid" >"$ENVSCALER_PID_FILE"
    if wait_ready; then
        echo "EnvScaler OpenEnv ready at http://127.0.0.1:$ENVSCALER_PORT (pid $pid)."
        return 0
    fi
    echo "EnvScaler OpenEnv failed to become ready; log: $ENVSCALER_LOG" >&2
    tail -40 "$ENVSCALER_LOG" >&2 || true
    return 1
}

stop_service() {
    local pid=""
    if [[ -f "$ENVSCALER_PID_FILE" ]] && is_service_pid "$(<"$ENVSCALER_PID_FILE")"; then
        pid=$(<"$ENVSCALER_PID_FILE")
    else
        pid=$(find_service_pid || true)
    fi
    if [[ -z "$pid" ]]; then
        rm -f "$ENVSCALER_PID_FILE"
        echo "EnvScaler OpenEnv is not running."
        return 0
    fi
    kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 20); do
        is_alive "$pid" || break
        sleep 0.25
    done
    is_alive "$pid" && kill -KILL "$pid" 2>/dev/null || true
    rm -f "$ENVSCALER_PID_FILE"
    echo "EnvScaler OpenEnv stopped."
}

status_service() {
    if curl --fail --silent --show-error \
        "http://127.0.0.1:$ENVSCALER_PORT/dataset"; then
        echo
        return 0
    fi
    echo "EnvScaler OpenEnv is unavailable on port $ENVSCALER_PORT."
    return 1
}

case "${1:-status}" in
    start) start_service ;;
    stop) stop_service ;;
    restart) stop_service; start_service ;;
    status) status_service ;;
    logs) tail -f "$ENVSCALER_LOG" ;;
    *) echo "Usage: $0 {start|stop|restart|status|logs}" >&2; exit 2 ;;
esac
