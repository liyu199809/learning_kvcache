#!/usr/bin/env bash
# Stop on any failed check; never release unrelated GPU processes here.
set -euo pipefail
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
EXPERIMENT_ROOT="$REPO_ROOT/outputs/prefix_composition_3domain_20260917"
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate kvcache
cd "$REPO_ROOT"
export EXPERIMENT_ROOT PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT/verl:$REPO_ROOT:$REPO_ROOT/OpenEnv/src:$REPO_ROOT/OpenEnv/envs${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$EXPERIMENT_ROOT/logs" "$EXPERIMENT_ROOT/configs"
exec 9>"$EXPERIMENT_ROOT/training_queue.lock"
flock -n 9 || { echo 'This experiment queue is already running'; exit 1; }
PHASE=starting
write_status() {
    python - "$EXPERIMENT_ROOT/queue_status.json" "$PHASE" "${1:-running}" "$$" <<'PY'
import datetime, json, os, sys
path, phase, state, pid = sys.argv[1:]
temp = path + '.tmp'
with open(temp, 'w') as f:
    json.dump(dict(phase=phase, state=state, pid=int(pid),
                   evaluation_status="pending_dataset",
                   updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat()), f, indent=2)
os.replace(temp, path)
PY
}
trap 'write_status failed' ERR
read -r -a smoke_domains <<< "${SMOKE_DOMAINS-awm envscaler taco}"
for domain in "${smoke_domains[@]}"; do
    PHASE="smoke_$domain"; write_status
    RUN_MODE=smoke RUN_NAME="smoke_$domain" TOTAL_TRAINING_STEPS=2 \
      SAVE_FREQ=1 MAX_RESPONSE_LENGTH=2048 GPU_MEMORY_UTILIZATION=0.40 \
      RESUME_MODE=disable bash prefix_tuning/virtual_prefix/run_three_domain_prefix.sh "$domain"
done
for domain in awm envscaler taco; do
    PHASE="train_$domain"; write_status
    RUN_MODE=full RUN_NAME="$domain" RESUME_MODE=disable VAL_BEFORE_TRAIN=False TEST_FREQ=-1 \
      bash prefix_tuning/virtual_prefix/run_three_domain_prefix.sh "$domain"
    PHASE="export_$domain"; write_status
    checkpoint=$(python - "$EXPERIMENT_ROOT/checkpoints/$domain" <<'PY'
from pathlib import Path
import sys
paths = [p for p in Path(sys.argv[1]).glob('global_step_*/actor') if (p/'trainable_only_meta.json').exists()]
assert paths, 'No completed prefix checkpoint found'
print(max(paths, key=lambda p: int(p.parent.name.rsplit('_', 1)[1])))
PY
)
    CUDA_VISIBLE_DEVICES="" python \
      prefix_tuning/virtual_prefix/export_hybrid_delta_residual_attention_prefix_checkpoint.py \
      --checkpoint "$checkpoint" \
      --prepared-model "$EXPERIMENT_ROOT/prepared/hybrid-G512-A64" \
      --output "$EXPERIMENT_ROOT/prefixes/$domain"
done
PHASE=compose_models; write_status
for combination in awm_envscaler awm_taco envscaler_taco awm_envscaler_taco; do
    IFS=_ read -r -a domains <<< "$combination"
    inputs=()
    for domain in "${domains[@]}"; do inputs+=(--input "$EXPERIMENT_ROOT/prefixes/$domain"); done
    CUDA_VISIBLE_DEVICES="" python -m prefix_tuning.virtual_prefix.compose_hybrid_prefixes \
      "${inputs[@]}" --output "$EXPERIMENT_ROOT/models/$combination"
done
PHASE=training_and_export_complete; write_status awaiting_evaluation
