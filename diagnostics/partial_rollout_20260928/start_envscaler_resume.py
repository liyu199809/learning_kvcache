"""Run on gpu-h200 from the project root to resume the four-GPU checkpoint for one final drained update."""
import os
from pathlib import Path
import subprocess

root = Path('/home/liyu.468/learning_kvcache')
exp = root / 'outputs/partial_rollout_h200_20260928'
env = os.environ.copy()
env.update(
    KVCACHE_PYTHON='/usr/local/miniconda3/envs/kvcache/bin/python',
    PYTHON_BIN='/usr/local/miniconda3/envs/kvcache/bin/python',
    USE_WANDB='0',
    DATA_ROOT=str(root / 'outputs/prefix_composition_3domain_20260922_4gpu/data'),
    MODEL=str(root / 'outputs/prefix_composition_3domain_20260917/prepared/hybrid-G512-A64'),
    EXPERIMENT_ROOT=str(exp),
    RAY_TMPDIR='/tmp/kvcache-partial-env-resume', RAY_ADDRESS='local',
    RUN_NAME='envscaler_partial_3actor1teacher_smoke',
    TOTAL_TRAINING_STEPS='4', SAVE_FREQ='4', TEST_FREQ='-1',
    RESUME_MODE='resume_path', VAL_BEFORE_TRAIN='False',
    VAL_FILE=str(root / 'diagnostics/partial_rollout_20260928/smoke_val.parquet'),
    ACTOR_GPUS='3', TEACHER_GPUS='1', CUDA_VISIBLE_DEVICES='0,1,2,3',
    TRAIN_BATCH_SIZE='6', PPO_MINI_BATCH_SIZE='6', NUM_WORKERS='6',
)
args = ['bash', './experiment.sh', 'train-smoke', 'envscaler',
        '+ray_kwargs.ray_init.address=local', 'ray_kwargs.ray_init.num_cpus=32',
        '+ray_kwargs.ray_init.num_gpus=4', '+ray_kwargs.ray_init.include_dashboard=False',
        '+ray_kwargs.ray_init.object_store_memory=8589934592',
        'trainer.resume_from_path=' + str(exp / 'checkpoints/envscaler_partial_3actor1teacher_smoke/global_step_3')]
with (exp / 'envscaler_resume_launcher.log').open('w') as handle:
    process = subprocess.Popen(args, cwd=root, env=env, stdout=handle,
                               stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                               start_new_session=True)
(exp / 'envscaler_resume.pid').write_text(str(process.pid) + '\n')
print(process.pid)
