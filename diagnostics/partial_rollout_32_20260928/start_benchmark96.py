"""Benchmark three accumulated-gradient updates: effective batch 96, rollout pool 116."""
import os
from pathlib import Path
import subprocess

root = Path('/home/liyu.468/learning_kvcache')
exp = root / 'outputs/partial_rollout_96_h200_20260928'
exp.mkdir(parents=True, exist_ok=True)
env = os.environ.copy()
env.update(
    KVCACHE_PYTHON='/usr/local/miniconda3/envs/kvcache/bin/python',
    PYTHON_BIN='/usr/local/miniconda3/envs/kvcache/bin/python',
    USE_WANDB='0',
    DATA_ROOT=str(root / 'outputs/prefix_composition_3domain_20260922_4gpu/data'),
    MODEL=str(root / 'outputs/prefix_composition_3domain_20260917/prepared/hybrid-G512-A64'),
    EXPERIMENT_ROOT=str(exp),
    RAY_TMPDIR='/tmp/kvcache-partial-pool116', RAY_ADDRESS='local',
    RUN_NAME='envscaler_96train_116rollout',
    TOTAL_TRAINING_STEPS='3', SAVE_FREQ='3', TEST_FREQ='-1',
    RESUME_MODE='disable', VAL_BEFORE_TRAIN='False',
    VAL_FILE=str(root / 'diagnostics/partial_rollout_20260928/smoke_val.parquet'),
    ACTOR_GPUS='3', TEACHER_GPUS='1', CUDA_VISIBLE_DEVICES='0,1,2,3',
    TRAIN_BATCH_SIZE='96', PPO_MINI_BATCH_SIZE='96', PARTIAL_ROLLOUT_TARGET='116',
    MAX_RESPONSE_LENGTH='16384', MAX_MODEL_LEN='49152', PPO_MAX_TOKEN_LEN_PER_GPU='49152',
    NUM_WORKERS='24', GPU_MEMORY_UTILIZATION='0.65',
)
args = ['bash', './experiment.sh', 'train', 'envscaler',
        '+ray_kwargs.ray_init.address=local', 'ray_kwargs.ray_init.num_cpus=64',
        '+ray_kwargs.ray_init.num_gpus=4', '+ray_kwargs.ray_init.include_dashboard=False',
        '+ray_kwargs.ray_init.object_store_memory=8589934592']
import json
import sys
import time

if '--worker' in sys.argv:
    started = time.time()
    result = subprocess.run(args, cwd=root, env=env)
    (exp / 'launcher_result.json').write_text(json.dumps({
        'exit_code': result.returncode, 'elapsed_wall_seconds': time.time() - started,
        'train_batch': 96, 'mini_batch': 96, 'rollout_target': 116,
    }, indent=2) + '\n')
    sys.exit(result.returncode)
elif '--preflight' in sys.argv:
    subprocess.run(args, cwd=root, env={**env, 'PREFLIGHT_ONLY': '1'}, check=True)
else:
    if (exp / 'launcher.log').exists():
        raise RuntimeError('Refusing to overwrite an existing benchmark log')
    with (exp / 'launcher.log').open('w') as handle:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker'],
                                   cwd=root, stdout=handle, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
    (exp / 'launcher.pid').write_text(str(process.pid) + '\n')
    print(process.pid)
