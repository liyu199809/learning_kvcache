"""Exercise the real local jail before accepting model-generated code."""
import argparse
import json
import os
from pathlib import Path
import time

from nsjail_backend import PROJECT, execute, provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    args.code_eval_workers = 1
    args.code_eval_timeout = 30
    args.code_memory = '512m'
    args.output_dir.mkdir(parents=True, exist_ok=True)
    host_pid = os.getpid()
    code = f'''
import ctypes, errno, json, os, pathlib, resource, socket
assert os.getuid() == {1000000 + host_pid}
status = pathlib.Path('/proc/self/status').read_text()
assert 'NoNewPrivs:\\t1' in status
assert 'CapEff:\\t0000000000000000' in status
assert 'Seccomp:\\t2' in status
assert not pathlib.Path({str(PROJECT)!r}).exists()
assert not pathlib.Path('/proc/{host_pid}').exists()
assert not pathlib.Path('/sys/fs/cgroup').exists()
assert not list(pathlib.Path('/dev').glob('nvidia*'))
assert 'SSH_AUTH_SOCK' not in os.environ and 'CODE_EVAL_API_KEY' not in os.environ
try:
    pathlib.Path('/app/score.py').open('a')
except OSError as e:
    assert e.errno in (errno.EROFS, errno.EACCES)
else:
    raise AssertionError('scorer mount must be read-only')
try:
    socket.socket(socket.AF_INET, socket.SOCK_STREAM)
except OSError as e:
    assert e.errno == errno.EPERM
else:
    raise AssertionError('INET sockets must be denied')
pair = socket.socketpair()
pair[0].send(b'ok'); assert pair[1].recv(2) == b'ok'
for s in pair: s.close()
libc = ctypes.CDLL(None, use_errno=True)
assert libc.ptrace(0, 0, 0, 0) == -1 and ctypes.get_errno() == errno.EPERM
assert resource.getrlimit(resource.RLIMIT_NPROC) == (64, 64)
assert resource.getrlimit(resource.RLIMIT_AS) == (512*1024**2, 512*1024**2)
pathlib.Path('/tmp/writable').write_text('ok')
print(json.dumps({{'isolation': 'passed', 'uid': os.getuid(), 'cpu_affinity': sorted(os.sched_getaffinity(0))}}))
'''
    reports = {}
    result = execute(['-c', code], args)
    (args.output_dir / 'probe.cfg').write_text(result['config'])
    (args.output_dir / 'probe.stderr.log').write_text(result['stderr'])
    if result['returncode']:
        raise RuntimeError(result['stderr'])
    reports['isolation'] = json.loads(result['stdout'])

    args.code_memory = '128m'
    memory = execute(['-c', "try:\n x=bytearray(256*1024**2)\nexcept MemoryError:\n print('memory_limit_passed')\nelse:\n raise AssertionError('limit failed')"], args)
    assert memory['returncode'] == 0 and 'memory_limit_passed' in memory['stdout'], memory
    reports['memory_limit'] = 'passed'

    args.code_eval_timeout = 2
    before = time.monotonic()
    timeout = execute(['-c', "import os,time; print(os.readlink('/proc/self/ns/pid'),flush=True); os.fork(); time.sleep(60)"], args)
    elapsed = time.monotonic() - before
    assert timeout['returncode'] != 0 and elapsed < 15, timeout
    pid_namespace = timeout['stdout'].strip()
    assert pid_namespace.startswith('pid:['), timeout
    remaining = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdecimal():
            continue
        try:
            if os.readlink(proc / 'ns/pid') == pid_namespace:
                remaining.append(proc.name)
        except OSError:
            pass
    assert not remaining, remaining
    reports['timeout_and_descendant_cleanup'] = {'passed': True, 'seconds': elapsed}
    reports['runtime'] = provenance()
    (args.output_dir / 'verification.json').write_text(json.dumps(reports, indent=2))
    print(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
