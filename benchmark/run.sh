#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
cd "$BENCHMARK_PROJECT_ROOT"
exec "$PROJECT_PYTHON" -m benchmark.eval.run_eval "$@"
