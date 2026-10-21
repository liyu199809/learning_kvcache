"""Run the existing full benchmark protocol on the new SGLang model."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import urllib.request

ROOT = Path('/home/liyu.468/learning_kvcache')
DIAG = ROOT / 'diagnostics/b300_taco_compare_20260929'
OUT = ROOT / 'outputs/b300_taco_compare_20260929'
MODELS = {
    'new': ROOT / 'outputs/taco_partial_pool32_retrain_20260928_1830/prefixes/taco',
}


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    temp.replace(path)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def wait_service(name):
    config = json.loads((DIAG / f'{name}_service.json').read_text())
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        if not Path(f"/proc/{config['pid']}/cmdline").exists():
            raise RuntimeError(f'{name} SGLang service exited')
        try:
            with opener.open(f"http://127.0.0.1:{config['port']}/v1/models", timeout=3) as response:
                data = json.load(response)
            if any(m['id'] == f'prefix-taco-{name}' for m in data['data']):
                return config
        except OSError:
            pass
        time.sleep(3)
    raise TimeoutError(f'{name} SGLang service did not become ready')


def worker(name):
    config = wait_service(name)
    model = f'prefix-taco-{name}'
    model_out = OUT / name
    status_path = model_out / 'status.json'
    status = (json.loads(status_path.read_text()) if status_path.exists() else
              dict(started_at=now(), completed=[]))
    status.update(state='running', current_job=None, resumed_at=now())
    status.pop('error', None)
    status.pop('finished_at', None)
    write(status_path, status)
    base = ['bash', str(ROOT / 'benchmark/run.sh'), '--openai-base-url',
            f"http://127.0.0.1:{config['port']}/v1", '--model', model,
            '--temperature', '0.6', '--top-p', '1.0']
    jobs = []
    for suite, count in [('humaneval+', 164), ('mbpp+', 378), ('livecodebench', 1055)]:
        args = ['--benchmark', suite, '--code-execution-backend', 'nsjail',
                '--code-thinking', 'on', '--code-pass-k', '1', '--code-n-samples', '1',
                '--code-seed', '42', '--code-eval-workers', '16', '--code-memory', '64g',
                '--code-eval-timeout', '7200', '--code-api-timeout', '1800',
                '--llm-max-completion-tokens', '16384', '--concurrency', '64']
        if suite == 'livecodebench':
            args += ['--lcb-release', 'v6']
        jobs.append((suite, count, args))
    jobs.append(('bfcl_v3', 4441, ['--benchmark', 'bfcl_v3', '--bfcl-model-path',
                                 str(MODELS[name]), '--bfcl-categories', 'all', '--concurrency', '64']))
    for domain, count in [('airline', 50), ('retail', 114)]:
        jobs.append((f'tau2_{domain}', count, ['--benchmark', 'tau2', '--tau2-domain', domain,
                    '--tau2-split', 'base', '--tau2-num-trials', '1', '--tau2-max-steps', '200',
                    '--tau2-require-external-user', '--tau2-require-external-judge',
                    '--llm-max-completion-tokens', '16384', '--concurrency',
                    '16' if domain == 'retail' else '2']))
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               NUMEXPR_MAX_THREADS='1', TOKENIZERS_PARALLELISM='false')
    if (DIAG / 'tau2_entrypoint/bin/tau2').exists():
        env['TAU2_VENV'] = str(DIAG / 'tau2_entrypoint')
    try:
        for job, expected, args in jobs:
            dest = model_out / job
            if job in status['completed']:
                if not (dest / 'timing.json').exists():
                    raise RuntimeError(f'Missing timing for completed job {job}')
                continue
            status.update(current_job=job, updated_at=now())
            write(status_path, status)
            command = base + args + ['--output-dir', str(dest)]
            log = OUT / 'logs' / f'{name}_{job}.log'
            log.parent.mkdir(parents=True, exist_ok=True)
            started = time.monotonic()
            with log.open('w') as handle:
                handle.write(json.dumps(command) + '\n'); handle.flush()
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=handle,
                                        stderr=subprocess.STDOUT)
            elapsed = time.monotonic() - started
            if result.returncode:
                raise RuntimeError(f'{name}:{job} failed ({result.returncode}); {log}')
            timing = dict(wall_seconds=elapsed, expected_tasks=expected,
                          finished_at=now(), command=command)
            if job in ('humaneval+', 'mbpp+', 'livecodebench'):
                summary = json.loads((dest / 'summary.json').read_text())
                if summary['tasks'] != expected or summary['samples'] != expected:
                    raise RuntimeError(f'Incomplete coding suite: {summary}')
                rows = [json.loads(line) for line in (dest / 'raw_samples.jsonl').read_text().splitlines()]
                tokens = sum((row.get('usage') or {}).get('completion_tokens', 0) for row in rows)
                generation_seconds = (dest / 'raw_samples.jsonl').stat().st_mtime - (dest / 'manifest.json').stat().st_mtime
                timing.update(generation_seconds=generation_seconds, completion_tokens=tokens,
                              completion_tokens_per_second=tokens / generation_seconds,
                              tasks_per_hour=expected * 3600 / generation_seconds)
            write(dest / 'timing.json', timing)
            status['completed'].append(job)
            write(status_path, status)
        status.update(state='completed', current_job=None, finished_at=now())
    except BaseException:
        status.update(state='failed', error=traceback.format_exc(), finished_at=now())
        raise
    finally:
        write(status_path, status)


def run():
    parity = json.loads((DIAG / 'hf_parity.json').read_text())
    checks = parity + [row for item in parity for row in item['decode']]
    if len(parity) != 2 or any(row['hf_argmax'] != row['sglang_argmax']
                              or row['max_top10_logprob_delta'] > 0.30
                              or row['max_top10_probability_delta'] > 0.03 for row in checks):
        raise RuntimeError('SGLang/HF parity must pass before benchmark generation')
    resume = '--resume' in sys.argv
    OUT.mkdir(parents=True, exist_ok=resume)
    provenance = {}
    for name, model in MODELS.items():
        provenance[name] = dict(path=str(model), merge=json.loads((model / 'prefix_merge_manifest.json').read_text()),
            prefix_sha256={f: hashlib.sha256((model / f).read_bytes()).hexdigest() for f in
                           ('independent_delta_kv_prefix.safetensors', 'residual_attention_prefix.safetensors')})
    manifest_path = OUT / 'run_manifest.json'
    if not resume:
        write(manifest_path, dict(started_at=now(), engine='sglang==0.5.15.post1',
        models=provenance, gpus={'new':[0,1,2,3]}, tp=1, dp=4,
        sampling=dict(temperature=0.6, top_p=1, code_seed=42, code_thinking=True, code_max_tokens=16384),
        scopes={'humaneval+':164, 'mbpp+':378, 'livecodebench_v6':1055, 'bfcl_v3':4441,
                'tau2_airline':50, 'tau2_retail':114},
        interpretation='New model only. Reuse historical old vLLM benchmark scores. Training changes include partial rollout and corrected loss normalization.'))
    children = []
    try:
        for name in MODELS:
            with (OUT / f'{name}_worker.log').open('w') as handle:
                children.append(subprocess.Popen([sys.executable, __file__, name], cwd=ROOT,
                    stdout=handle, stderr=subprocess.STDOUT))
        codes = [p.wait() for p in children]
        write(OUT / 'status.json', dict(state='completed' if codes == [0] else 'failed',
                                      exit_codes=codes, finished_at=now()))
        return int(any(codes))
    finally:
        for name in MODELS:
            info = json.loads((DIAG / f'{name}_service.json').read_text())
            path = Path(f"/proc/{info['pid']}/cmdline")
            if path.exists() and b'sglang.launch_server' in path.read_bytes():
                os.killpg(info['pid'], signal.SIGTERM)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] != '--resume':
        worker(sys.argv[1])
    else:
        raise SystemExit(run())
