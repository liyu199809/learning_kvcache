#!/usr/bin/env bash
# Recreate the isolated, pinned evaluation environments without changing kvcache.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
cd "$BENCHMARK_PROJECT_ROOT"
command -v uv >/dev/null
if [[ ! -x benchmark/.venv/bin/python ]]; then
    uv python install 3.11
    uv venv --python 3.11 benchmark/.venv
fi
uv pip install --python benchmark/.venv/bin/python -r benchmark/requirements-b200.lock
UV=$(command -v uv) PYTHON="$PROJECT_PYTHON" bash benchmark/coding/setup.sh
TAU2_PYTHON_VERSION=${TAU2_PYTHON_VERSION:-/usr/local/miniconda3/envs/kvcache/bin/python} bash benchmark/tau2/setup.sh
uv pip install --python benchmark/tau2/.venv/bin/python -r benchmark/tau2/requirements-b200.lock
bash benchmark/bfcl_v3/setup.sh
uv pip check --python benchmark/.venv/bin/python
benchmark/coding/.venv/bin/python -c "from evalplus.evaluate import check_correctness; from evalplus.sanitize import sanitize"
uv pip check --python benchmark/tau2/.venv/bin/python
printf 'Python environments ready. Dataset and container status: benchmark/README_B200.md\n'
