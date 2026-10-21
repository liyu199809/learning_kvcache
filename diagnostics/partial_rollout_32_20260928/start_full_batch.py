"""Run three real updates with train batch 24, bounded rollout pool 32."""
import os
from pathlib import Path
import subprocess

root = Path('/home/liyu.468/learning_kvcache')
exp = root / 'outputs/partial_rollout_32_h200_20260928'
exp.mkdir(parents=True, exist_ok=True)
env = os.environ.copy()
env.update(
    KVCACHE_PYTHON='/usr/local/miniconda3/envs/kvcache/bin/python',
    PYTHON_BIN='/usr/local/miniconda3/envs/kvcache/bin/python',
    USE_WANDB='0',
    DATA_ROOT=str(root / 'outputs/prefix_composition_3domain_20260922_4gpu/data'),
    MODEL=str(root / 'outputs/prefix_composition_3domain_20260917/prepared/hybrid-G512-A64'),
    EXPERIMENT_ROOT=str(exp),
    RAY_TMPDIR='/tmp/kvcache-partial-pool32', RAY_ADDRESS='local',
    RUN_NAME='envscaler_24train_32rollout',
    TOTAL_TRAINING_STEPS='3', SAVE_FREQ='3', TEST_FREQ='-1',
    RESUME_MODE='disable', VAL_BEFORE_TRAIN='False',
    VAL_FILE=str(root / 'diagnostics/partial_rollout_20260928/smoke_val.parquet'),
    ACTOR_GPUS='3', TEACHER_GPUS='1', CUDA_VISIBLE_DEVICES='0,1,2,3',
    TRAIN_BATCH_SIZE='24', PPO_MINI_BATCH_SIZE='24', PARTIAL_ROLLOUT_TARGET='32',
    MAX_RESPONSE_LENGTH='16384', MAX_MODEL_LEN='49152', PPO_MAX_TOKEN_LEN_PER_GPU='49152',
    NUM_WORKERS='24', GPU_MEMORY_UTILIZATION='0.65',
)
args = ['bash', './experiment.sh', 'train', 'envscaler',
        '+ray_kwargs.ray_init.address=local', 'ray_kwargs.ray_init.num_cpus=64',
        '+ray_kwargs.ray_init.num_gpus=4', '+ray_kwargs.ray_init.include_dashboard=False',
        '+ray_kwargs.ray_init.object_store_memory=8589934592']
if os.environ.get('POOL_PREFLIGHT_ONLY') == '1':
    subprocess.run(args, cwd=root, env={**env, 'PREFLIGHT_ONLY': '1'}, check=True)
elif os.environ.get('POOL_LAUNCHER_CHECK') == '1':
    stub = str(root / 'diagnostics/partial_rollout_32_20260928/launcher_python_stub.bash')
    with (exp / 'launcher_check.log').open('w') as handle:
        subprocess.run(args, cwd=root, env={**env, 'KVCACHE_PYTHON': stub, 'PYTHON_BIN': stub},
                       stdout=handle, stderr=subprocess.STDOUT, check=True)
    print('Complete shell launcher exited successfully with the GPU entrypoint stubbed.')
else:
    with (exp / 'launcher.log').open('w') as handle:
        process = subprocess.Popen(args, cwd=root, env=env, stdout=handle,
                                   stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                   start_new_session=True)
    (exp / 'launcher.pid').write_text(str(process.pid) + '\n')
    print(process.pid)
