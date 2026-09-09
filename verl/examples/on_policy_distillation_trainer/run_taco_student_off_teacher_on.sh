#!/usr/bin/env bash
# Controlled ablation against taco_condensed_v2_nothink: change teacher mode only.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
export DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/traj_data/opsd_taco_condensed_v2}"
export RUN_NAME="${RUN_NAME:-qwen3_5_4b_taco_condensed_v2_soff_ton}"
export RUN_MODE="${RUN_MODE:-full}"
export RESUME_MODE="${RESUME_MODE:-disable}"
export ENABLE_THINKING=False
echo "Independent OPSD modes: Student=False; Teacher=True (scoring only)"
exec bash "$SCRIPT_DIR/run_qwen3_5_4b_opsd_dataset_view.sh" taco \
  "$@" \
  $'distillation.privileged_prefix="\n\n"' \
  'distillation.privileged_suffix=""' \
  distillation.privileged_append_enable_thinking=True
