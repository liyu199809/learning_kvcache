#!/usr/bin/env bash
# Sequentially train all four prefix partitions. Existing checkpoints resume automatically.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
START_CLUSTER=${START_CLUSTER:-0}
END_CLUSTER=${END_CLUSTER:-3}

if (( START_CLUSTER < 0 || END_CLUSTER > 3 || START_CLUSTER > END_CLUSTER )); then
    echo "Expected 0 <= START_CLUSTER <= END_CLUSTER <= 3" >&2
    exit 1
fi

for ((cluster_id = START_CLUSTER; cluster_id <= END_CLUSTER; cluster_id++)); do
    echo "Starting hybrid-prefix cluster $cluster_id"
    CLUSTER_ID=$cluster_id \
        "$SCRIPT_DIR/run_qwen3_5_4b_awm_opsd_hybrid_delta_residual_attention_prefix_cluster.sh" \
        "$@"
done

if (( START_CLUSTER == 0 && END_CLUSTER == 3 )) && [[ "${MERGE_AFTER_TRAINING:-1}" == "1" ]]; then
    "$SCRIPT_DIR/merge_trained_qwen3_5_4b_awm_opsd_hybrid_prefix_clusters.sh"
fi
