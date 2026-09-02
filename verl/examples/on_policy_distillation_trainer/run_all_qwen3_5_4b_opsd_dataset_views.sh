#!/usr/bin/env bash
# Sequentially train the three single-source OPSD datasets and their mixture.
# For preflight validation use: RUN_MODE=smoke VIEWS="envscaler taco" ./this-script

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RUN_ONE="$SCRIPT_DIR/run_qwen3_5_4b_opsd_dataset_view.sh"

RUN_MODE=${RUN_MODE:-full}
VIEWS=${VIEWS:-"awm envscaler taco mixed"}

for view in $VIEWS; do
    echo "Starting OPSD data view: $view (mode=$RUN_MODE)"
    RUN_MODE="$RUN_MODE" "$RUN_ONE" "$view" "$@"
done

echo "Completed OPSD views: $VIEWS"
