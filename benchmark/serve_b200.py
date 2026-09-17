"""Start/status/stop the bounded benchmark vLLM service; never touch training jobs."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/benchmark_bringup_20260917'
STATE = OUT / 'vllm_process.json'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['start', 'status', 'stop'])
    ap.add_argument('--gpu', type=int, default=3)
    ap.add_argument('--port', type=int, default=18081)
    ap.add_argument('--model', default='/home/liyu.468/models/Qwen3.5-4B')
    ap.add_argument('--alias', default='qwen3.5-4b-benchmark')
    ap.add_argument('--max-model-len', type=int, default=32768)
    ap.add_argument('--memory-fraction', type=float, default=.08)
    ap.add_argument('--python', default='/usr/local/miniconda3/envs/kvcache/bin/python')
    args = ap.parse_args()
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    pid = state.get('pid')
    live = bool(pid and Path(f'/proc/{pid}/cmdline').exists())
    if live:
        cmdline = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
        if 'vllm.entrypoints.openai.api_server' not in cmdline:
            raise SystemExit('Recorded PID no longer belongs to the benchmark server; refusing to act.')
    if args.action == 'stop':
        if live:
            if os.getpgid(pid) != pid:
                raise SystemExit('Benchmark process group does not match its recorded leader.')
            os.killpg(pid, signal.SIGTERM)
            print(f'Sent SIGTERM to benchmark process group {pid}; training processes untouched.')
        else:
            print('Benchmark service is already stopped.')
        return
    if args.action == 'status' or live:
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(f'http://127.0.0.1:{args.port}/v1/models', timeout=3) as response:
                print(json.dumps(json.load(response), indent=2))
        except Exception as error:
            print(f'pid={pid}, live={live}, endpoint not ready: {error}')
            raise SystemExit(2)
        return
    total, free = map(int, subprocess.check_output([
        'nvidia-smi', f'--id={args.gpu}', '--query-gpu=memory.total,memory.free',
        '--format=csv,noheader,nounits'], text=True).strip().split(','))
    if not 0 < args.memory_fraction < 1 or free < total * args.memory_fraction + 4096:
        raise SystemExit('Insufficient free GPU memory for the requested budget plus 4 GiB margin.')
    command = [args.python, '-m', 'vllm.entrypoints.openai.api_server', '--model', args.model,
        '--served-model-name', args.alias, '--host', '127.0.0.1', '--port', str(args.port),
        '--gpu-memory-utilization', str(args.memory_fraction), '--max-model-len', str(args.max_model_len),
        '--max-num-seqs', '2', '--max-num-batched-tokens', '2048', '--enforce-eager',
        '--enable-auto-tool-choice', '--tool-call-parser', 'qwen3_coder', '--reasoning-parser', 'qwen3',
        '--language-model-only', '--generation-config', 'vllm',
        '--default-chat-template-kwargs', '{"enable_thinking":false}']
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), TOKENIZERS_PARALLELISM='false',
               HF_HUB_OFFLINE='1', NO_PROXY='localhost,127.0.0.1,6.5.85.147')
    (OUT / 'logs').mkdir(parents=True, exist_ok=True)
    with (OUT / 'logs/vllm_benchmark.log').open('a') as log:
        process = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    STATE.write_text(json.dumps({'pid': process.pid, 'gpu': args.gpu, 'command': command,
                                'memory_utilization': args.memory_fraction}, indent=2))
    print(f'Started PID {process.pid}; check status before evaluation.')


if __name__ == '__main__':
    main()
