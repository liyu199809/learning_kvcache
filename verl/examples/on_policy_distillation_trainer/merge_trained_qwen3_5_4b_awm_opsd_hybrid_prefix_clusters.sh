#!/usr/bin/env bash
# Concatenate the latest trained cluster checkpoints back to G=2048/A=256.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
VERL_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
REPO_ROOT=$(cd -- "$VERL_ROOT/.." && pwd)

source "$REPO_ROOT/.venv/bin/activate"
cd "$REPO_ROOT"

PROJECT_NAME=${PROJECT_NAME:-self_evolver_opsd}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-$REPO_ROOT/checkpoints/$PROJECT_NAME}
RUN_PREFIX=${RUN_PREFIX:-qwen3_5_4b_awm_opsd_hybrid_prefix_split4_cluster}
MERGED_OUTPUT=${MERGED_OUTPUT:-/mnt/storage/disk1/verl_data/base_model/Qwen3.5-4B-HybridDeltaResidualAttentionPrefix-G2048-A256-step203-split4-trained-merged}

merge_args=()
for cluster_id in 0 1 2 3; do
    run_dir="$CHECKPOINT_ROOT/${RUN_PREFIX}${cluster_id}"
    tracker="$run_dir/latest_checkpointed_iteration.txt"
    if [[ ! -f "$tracker" ]]; then
        echo "Missing checkpoint tracker for cluster $cluster_id: $tracker" >&2
        exit 1
    fi
    step=$(<"$tracker")
    if [[ ! "$step" =~ ^[0-9]+$ ]]; then
        echo "Invalid checkpoint step for cluster $cluster_id: $step" >&2
        exit 1
    fi
    actor="$run_dir/global_step_$step/actor"
    if [[ ! -f "$actor/trainable_only_meta.json" ]]; then
        echo "Missing actor checkpoint for cluster $cluster_id: $actor" >&2
        exit 1
    fi
    echo "cluster$cluster_id: step $step ($actor)"
    merge_args+=(--input "$actor")
done

python "$REPO_ROOT/prefix_tuning/virtual_prefix/merge_hybrid_prefix_partitions.py" \
    "${merge_args[@]}" \
    --output "$MERGED_OUTPUT" \
    --dtype bfloat16 \
    --copy-mode hardlink

echo "Merged model: $MERGED_OUTPUT"
