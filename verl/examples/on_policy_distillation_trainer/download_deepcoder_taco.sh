#!/usr/bin/env bash
# Download the pinned DeepCoder Preview TACO subset through a reachable mirror.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
REVISION=${REVISION:-177913a7bd43791646ef6a43645caa3c871ab3db}
MIRROR=${HF_ENDPOINT:-https://hf-mirror.com}
OUTPUT_DIR=${OUTPUT_DIR:-$REPO_ROOT/data/deepcoder_taco/$REVISION}

if [[ "$REVISION" != "177913a7bd43791646ef6a43645caa3c871ab3db" ]]; then
    echo "No checksums are registered for revision $REVISION" >&2
    exit 2
fi

mkdir -p "$OUTPUT_DIR"
for index in 00000 00001 00002 00003; do
    file="train-${index}-of-00004.parquet"
    curl --fail --location --retry 5 --retry-delay 2 --continue-at - \
        --output "$OUTPUT_DIR/$file" \
        "$MIRROR/datasets/agentica-org/DeepCoder-Preview-Dataset/resolve/$REVISION/taco/$file"
done

(
    cd "$OUTPUT_DIR"
    sha256sum --check <<'CHECKSUMS'
6e666b446c9feb6232f29a5c843883caba108398981ea25c3c72f02ebdda7eb6  train-00000-of-00004.parquet
594adb3eaea2b813e1b42ca88890e5bf9e70a64b43d11d27d56554ce4a0bd889  train-00001-of-00004.parquet
b9c8fb5788bc3b80ce31a78b9ef7e57ee1231d1d4c423d157b7a3b9a9901e481  train-00002-of-00004.parquet
920c612796225d328e4291a8c825bea516c170eb16d97c8bab043026686af14b  train-00003-of-00004.parquet
CHECKSUMS
)

echo "DeepCoder TACO revision $REVISION is ready in $OUTPUT_DIR"
