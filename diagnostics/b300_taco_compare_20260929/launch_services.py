"""Manage only the SGLang services owned by this paired evaluation."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path('/home/liyu.468/learning_kvcache')
OUT = ROOT / 'diagnostics/b300_taco_compare_20260929'
MODELS = {
    'new': ROOT / 'outputs/taco_partial_pool32_retrain_20260928_1830/prefixes/taco',
}


def stop(name):
    path = OUT / f'{name}_service.json'
    if not path.exists():
        return
    record = json.loads(path.read_text())
    proc = Path(f"/proc/{record['pid']}/cmdline")
    if proc.exists() and b'sglang.launch_server' in proc.read_bytes():
        os.killpg(record['pid'], signal.SIGTERM)
        deadline = time.monotonic() + 30
        while proc.exists() and proc.read_bytes() and time.monotonic() < deadline:
            time.sleep(1)


def start(name):
    stop(name)
    gpus, port = '0,1,2,3', 18340
    cmd = ['/usr/bin/python3', '-m', 'sglang.launch_server', '--model-path', str(MODELS[name]),
           '--served-model-name', f'prefix-taco-{name}', '--host', '127.0.0.1', '--port', str(port),
           '--trust-remote-code', '--json-model-override-args',
           json.dumps({'architectures':['Qwen3_5ForConditionalGeneration']}),
           '--dtype', 'bfloat16', '--context-length', '65536', '--dp-size', '4', '--tp-size', '1',
           '--mem-fraction-static', '0.65', '--max-running-requests', '128',
           '--max-mamba-cache-size', '80', '--chunked-prefill-size', '8192',
           '--linear-attn-backend', 'triton', '--mamba-ssm-dtype', 'float32',
           '--disable-prefill-cuda-graph', '--cuda-graph-max-bs-decode', '32',
           '--reasoning-parser', 'qwen3', '--tool-call-parser', 'qwen3_coder',
           '--enable-metrics', '--sampling-defaults', 'openai', '--random-seed', '42']
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus, SGLANG_EXTERNAL_MODEL_PACKAGE='sglang_prefix',
               PYTHONPATH=str(OUT)+':'+str(ROOT), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1')
    with (OUT/f'{name}_service_dp2.log').open('w') as handle:
        process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                   stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
    (OUT/f'{name}_service.json').write_text(json.dumps(dict(pid=process.pid, command=cmd,
                     gpus=gpus, port=port, started_at=time.time()), indent=2))
    print(name, process.pid, gpus, port, flush=True)


if __name__ == '__main__':
    for name in sys.argv[2:] or ['new']:
        (start if sys.argv[1] == 'start' else stop)(name)
