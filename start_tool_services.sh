#!/usr/bin/env bash
# Manage all three OpenEnv tool backends: AWM, EnvScaler, and CodeJudge.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PROJECT_ROOT=${PROJECT_ROOT:-$SCRIPT_DIR}

# Use the project-specific clone of base by default. Callers can still
# override PYTHON_BIN explicitly when testing another interpreter.
KVCACHE_PYTHON=${KVCACHE_PYTHON:-/usr/local/miniconda3/envs/kvcache/bin/python}
if [[ ! -x "$KVCACHE_PYTHON" ]]; then
    echo "kvcache Python not found or not executable: $KVCACHE_PYTHON" >&2
    exit 1
fi
export PYTHON_BIN=${PYTHON_BIN:-$KVCACHE_PYTHON}

# Proxy settings are useful for package/data downloads, but local service
# health checks must always connect directly to the loopback interface.
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost"
export no_proxy="${no_proxy:+$no_proxy,}127.0.0.1,localhost"

run_one() {
    local action=$1 service=$2
    case "$service" in
        awm) "$SCRIPT_DIR/start_services.sh" "$action" tool ;;
        envscaler) "$SCRIPT_DIR/start_envscaler_service.sh" "$action" ;;
        codejudge) "$SCRIPT_DIR/start_code_judge_service.sh" "$action" ;;
        *) echo "Unknown service: $service" >&2; return 2 ;;
    esac
}

action=${1:-status}
shift || true
case "$action" in
    start|stop|restart|status) ;;
    *) echo "Usage: $0 {start|stop|restart|status} [awm|envscaler|codejudge ...]" >&2; exit 2 ;;
esac

if [[ $# -eq 0 ]]; then
    services=(awm envscaler codejudge)
else
    services=("$@")
fi

failed=0
for service in "${services[@]}"; do
    printf '\n[%s] %s\n' "$service" "$action"
    run_one "$action" "$service" || failed=1
done
exit "$failed"
