#!/usr/bin/env bash
# Start the DeepCoder/TACO OpenEnv judge on port 8901.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/.venv/bin/activate"

export PYTHONPATH="$SCRIPT_DIR/OpenEnv/src:$SCRIPT_DIR/OpenEnv/envs${PYTHONPATH:+:$PYTHONPATH}"
export DEEPCODER_TACO_DATA_DIR=${DEEPCODER_TACO_DATA_DIR:-$SCRIPT_DIR/data/deepcoder_taco/177913a7bd43791646ef6a43645caa3c871ab3db}

exec python -m code_judge_env.server.app
