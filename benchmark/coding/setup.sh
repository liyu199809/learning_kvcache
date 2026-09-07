#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
UV=${UV:-$HOME/.local/bin/uv}
PYTHON=${PYTHON:-$ROOT/../../.venv/bin/python}
LCB_COMMIT=28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24
if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
    "$UV" venv "$ROOT/.venv" --python "$PYTHON"
fi
# Only the official evaluator/data/sanitizer dependencies are needed; model
# inference is via OpenAI API, so no torch/vLLM/provider stacks are installed.
"$UV" pip install --python "$ROOT/.venv/bin/python" --no-deps -r "$ROOT/requirements.lock"
if [[ ! -f "$ROOT/official/lcb_runner/evaluation/testing_util.py" ]]; then
    mkdir -p "$ROOT/official"
    curl --fail --location --retry 3 \
        "https://codeload.github.com/LiveCodeBench/LiveCodeBench/tar.gz/$LCB_COMMIT" \
        | tar -xz --strip-components=1 -C "$ROOT/official" "LiveCodeBench-$LCB_COMMIT/lcb_runner"
fi
"$ROOT/.venv/bin/python" -c 'from evalplus.evaluate import check_correctness; from evalplus.sanitize import sanitize'
"$ROOT/.venv/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 11), "Coding sandbox requires Python 3.11"'
echo "Ready. Scoring requires Docker; use --code-docker-image to select a local Linux image."
