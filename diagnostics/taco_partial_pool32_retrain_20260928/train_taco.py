"""Start and supervise a fresh full TACO hybrid-prefix run on H200."""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import psutil

ROOT = Path('/home/liyu.468/learning_kvcache')
PYTHON = '/usr/local/miniconda3/envs/kvcache/bin/python'
RUN = 'taco_3actor1teacher_batch24_pool32'
PREPARED = ROOT / 'outputs/prefix_composition_3domain_20260917/prepared/hybrid-G512-A64'
DATA = ROOT / 'outputs/prefix_composition_3domain_20260922_4gpu/data'


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temp, path)


def configuration(exp):
    overrides = dict(
        KVCACHE_PYTHON=PYTHON, PYTHON_BIN=PYTHON, USE_WANDB='0',
        MODEL=str(PREPARED), DATA_ROOT=str(DATA), EXPERIMENT_ROOT=str(exp),
        RUN_NAME=RUN, RUN_MODE='full', TRAINER_MODE='colocate_async',
        RESUME_MODE='disable', TOTAL_EPOCHS='1', TOTAL_TRAINING_STEPS='',
        ACTOR_GPUS='3', TEACHER_GPUS='1', CUDA_VISIBLE_DEVICES='0,1,2,3',
        TRAIN_BATCH_SIZE='24', PPO_MINI_BATCH_SIZE='24', PARTIAL_ROLLOUT_TARGET='32',
        PARTIAL_MAX_OFF_POLICY='8', PARTIAL_OFF_POLICY_STRATEGY='wait',
        MAX_RESPONSE_LENGTH='16384', MAX_MODEL_LEN='49152', PPO_MAX_TOKEN_LEN_PER_GPU='49152',
        NUM_WORKERS='24', GPU_MEMORY_UTILIZATION='0.65', ACTOR_LR='1e-5',
        ENABLE_THINKING='True', SAVE_FREQ='20', TEST_FREQ='20', VAL_BEFORE_TRAIN='True',
        RAY_TMPDIR='/tmp/kvcache-taco-pool32-retrain', RAY_ADDRESS='local',
        PYTHONUNBUFFERED='1',
    )
    args = ['bash', './experiment.sh', 'train', 'taco',
            '+ray_kwargs.ray_init.address=local', 'ray_kwargs.ray_init.num_cpus=64',
            '+ray_kwargs.ray_init.num_gpus=4', '+ray_kwargs.ray_init.include_dashboard=False',
            '+ray_kwargs.ray_init.object_store_memory=8589934592']
    return overrides, args


def supervise(exp):
    overrides, args = configuration(exp)
    env = {**os.environ, **overrides}
    seen = {}
    status = dict(state='running', phase='starting', started_at=stamp(),
                  supervisor_pid=os.getpid(), run_name=RUN, expected_steps=66,
                  latest_step=0, log=str(exp / 'launcher.log'))
    process = None
    offset = 0
    buffered = ''
    try:
        with (exp / 'launcher.log').open('w') as handle:
            process = subprocess.Popen(args, cwd=ROOT, env=env, stdout=handle,
                                       stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        status['trainer_pid'] = process.pid
        while True:
            try:
                parent = psutil.Process(process.pid)
                for child in [parent, *parent.children(recursive=True)]:
                    try:
                        seen[str(child.pid)] = child.create_time()
                    except psutil.NoSuchProcess:
                        pass
            except psutil.NoSuchProcess:
                pass
            with (exp / 'launcher.log').open() as handle:
                handle.seek(offset)
                buffered += re.sub(r'\x1b\[[0-9;]*m', '', handle.read()).replace('\r', '\n')
                offset = handle.tell()
            lines = buffered.split('\n')
            buffered = lines.pop()
            for line in lines:
                if 'Initial validation metrics:' in line or 'Validation metrics before training:' in line:
                    status['phase'] = 'training'
                if re.search(r'\bstep:\s*\d+\s*-', line):
                    row = {key: float(value) for key, value in re.findall(
                        r'([\w/.-]+):\s*(?:np\.\w+\()?([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|nan|inf)', line)}
                    status['latest_step'] = int(row['step'])
                    status['phase'] = 'training'
                    status['latest_metrics'] = {k:v for k,v in row.items() if k in [
                        'actor/loss', 'actor/grad_norm', 'timing_s/gen', 'timing_s/update_actor', 'timing_s/step',
                        'training/off_policy/trajectory_staleness/max'] or k.startswith('training/partial/')}
                    with (exp / 'metrics.jsonl').open('a') as handle:
                        handle.write(json.dumps(row) + '\n')
            status['updated_at'] = stamp()
            write_json(exp / 'status.json', status)
            write_json(exp / 'observed_processes.json', seen)
            code = process.poll()
            if code is not None:
                break
            time.sleep(5)
        status['trainer_exit_code'] = code
        if code:
            raise RuntimeError(f'Trainer exited with code {code}; see launcher.log')
        # Release only children observed in this exact training job, then export on CPU.
        cleanup(seen)
        status['phase'] = 'exporting'
        write_json(exp / 'status.json', status)
        sys.path.insert(0, str(ROOT))
        import torch
        from safetensors.torch import load_file
        from prefix_tuning.virtual_prefix.export_hybrid_delta_residual_attention_prefix_checkpoint import export
        checkpoints = exp / 'checkpoints' / RUN
        latest = max(checkpoints.glob('global_step_*'), key=lambda p:int(p.name.rsplit('_', 1)[1]))
        assert int(latest.name.rsplit('_', 1)[1]) == status['latest_step'] == 66
        row = json.loads((exp / 'metrics.jsonl').read_text().splitlines()[-1])
        assert row['training/partial/queued_prompts'] == 0
        assert math.isfinite(row['actor/loss']) and math.isfinite(row['actor/grad_norm'])
        output = exp / 'prefixes/taco'
        manifest = export(latest / 'actor', PREPARED, output, torch.float32, 'hardlink')
        assert manifest['world_size'] == 3 and manifest['trainable_numel'] == 78905344
        changes = {}
        for name in ['independent_delta_kv_prefix.safetensors', 'residual_attention_prefix.safetensors']:
            original, trained = load_file(str(PREPARED / name)), load_file(str(output / name))
            deltas = []
            for key, tensor in trained.items():
                assert torch.isfinite(tensor).all(), key
                delta = float((tensor.float() - original[key].float()).abs().max())
                assert delta > 0, key
                deltas.append(delta)
            changes[name] = {'changed_tensors':len(deltas), 'max_delta':max(deltas)}
        status.update(state='completed', phase='completed', completed_at=stamp(),
                      checkpoint=str(latest), prefix_model=str(output), prefix_changes=changes)
    except Exception as exc:
        status.update(state='failed', phase='failed', error=f'{type(exc).__name__}: {exc}', failed_at=stamp())
        if process is not None:
            cleanup(seen)
        raise
    finally:
        status['updated_at'] = stamp()
        write_json(exp / 'status.json', status)


def cleanup(seen):
    live = []
    for pid, created in seen.items():
        try:
            proc = psutil.Process(int(pid))
            if proc.create_time() == created and proc.status() != psutil.STATUS_ZOMBIE:
                proc.terminate()
                live.append(proc)
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(live, timeout=15)
    for proc in alive:
        try:
            if proc.create_time() == seen[str(proc.pid)]:
                proc.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(alive, timeout=10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-root', type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--worker', action='store_true')
    opts = parser.parse_args()
    exp = opts.experiment_root.resolve()
    exp.mkdir(parents=True, exist_ok=True)
    overrides, args = configuration(exp)
    if opts.worker:
        supervise(exp)
    elif opts.preflight:
        subprocess.run(args, cwd=ROOT, env={**os.environ, **overrides, 'PREFLIGHT_ONLY':'1'}, check=True)
    else:
        if (exp / 'launcher.log').exists() or (exp / 'supervisor.pid').exists():
            raise RuntimeError('Refusing to overwrite an existing run')
        tracked = ['prefix_tuning/virtual_prefix/internal/train_domain.bash',
                   'verl/verl/trainer/distillation/losses.py',
                   'verl/verl/trainer/ppo/v1/trainer_colocate_async.py',
                   'verl/verl/workers/rollout/llm_server.py',
                   'verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py']
        write_json(exp / 'launch_config.json', {'created_at':stamp(), 'env_overrides':overrides,
                   'command':args, 'expected_steps':66, 'source_sha256':{
                       name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in tracked}})
        with (exp / 'supervisor.log').open('w') as handle:
            proc = subprocess.Popen([PYTHON, str(Path(__file__).resolve()), '--worker', '--experiment-root', str(exp)],
                                    cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, start_new_session=True)
        (exp / 'supervisor.pid').write_text(str(proc.pid) + '\n')
        print(json.dumps({'supervisor_pid':proc.pid, 'experiment_root':str(exp), 'run_name':RUN}))


if __name__ == '__main__':
    main()
