#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
run=${1:-"$PWD/envscaler-joint-prefix"}
mode=${2:-joint}
gdn_mode=${3:-state_rows_joint_softmax}
repo=/host-ssd/learning-kvcache-agent-world-pr
mkdir -p "$run/train"
exec 8>"$run/train.lock"
flock -n 8
if [[ -e "$run/train/loss.jsonl" ]]; then
    echo "Training output already exists at $run/train" >&2
    exit 1
fi
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2
export PYTHONPATH="$PWD/runtime:/tmp/prefix-hf5:/tmp/prefix-vllm-env:$repo"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-"$run/inductor-cache"}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-"$run/triton-cache"}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
test -s "$run/initial-model/config.json"
until [[ -s "$run/encoded.pt" ]]; do sleep 2; done
exec /usr/local/bin/python -u -S -m torch.distributed.run --standalone --nproc-per-node=8 \
    --no-python /usr/local/bin/python -u -S train_envscaler_joint_prefix.py --run "$run" \
    --attention-mode "$mode" --gdn-mode "$gdn_mode"
