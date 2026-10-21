#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == -m && "${2:-}" == verl.trainer.main_ppo ]]; then
    /usr/local/miniconda3/envs/kvcache/bin/python - "$@" <<'PY'
import json
from pathlib import Path
import sys
args = sys.argv[1:]
for required in ['data.train_batch_size=24', 'actor_rollout_ref.actor.ppo_mini_batch_size=24',
                 'trainer.v1.colocate_async.rollout_target_size=32', 'trainer.use_v1=True',
                 'trainer.v1.trainer_mode=colocate_async', '+ray_kwargs.ray_init.address=local']:
    assert required in args, required
path = Path('/home/liyu.468/learning_kvcache/outputs/partial_rollout_32_h200_20260928/launcher_arguments.json')
path.write_text(json.dumps(args, indent=2) + '\n')
print('Launcher invocation validated; diagnostic stub did not initialize GPUs.')
PY
    exit 0
fi
exec /usr/local/miniconda3/envs/kvcache/bin/python "$@"
