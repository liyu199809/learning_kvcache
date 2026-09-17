"""Local scoring sandbox: namespaces + seccomp + rlimits, with no cgroup writes.

The official evaluator runs unchanged. No fallback to unsandboxed execution.
Memory limits are per process (RLIMIT_AS), not an aggregate cgroup memory quota.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import signal
import subprocess
import sys
import sysconfig
import tempfile

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
RUNTIME = PROJECT / 'benchmark/runtime'
NSJAIL = RUNTIME / 'bin/nsjail'
POLICY = ROOT / 'nsjail.policy'


def memory_mib(value):
    match = re.fullmatch(r'(\d+)([gGmMkK]?)', str(value))
    if not match:
        raise ValueError('--code-memory must be an integer with g, m, k, or no suffix')
    amount, suffix = int(match[1]), match[2].lower()
    size = amount * {'g': 1024**3, 'm': 1024**2, 'k': 1024, '': 1}[suffix]
    if size < 128 * 1024**2:
        raise ValueError('NsJail requires at least 128 MiB per process')
    return (size + 1024**2 - 1) // 1024**2


def check_runtime():
    if sys.platform != 'linux' or not NSJAIL.is_file() or not os.access(NSJAIL, os.X_OK):
        raise RuntimeError(f'NsJail unavailable: {NSJAIL}. Run benchmark/setup_nsjail.sh first.')
    if not POLICY.is_file():
        raise RuntimeError(f'Missing required seccomp policy: {POLICY}')
    if os.geteuid() != 0:
        raise RuntimeError('This B200 configuration requires a root supervisor; jailed code drops to a unique non-root UID')


def provenance():
    check_runtime()
    return {'backend': 'nsjail', 'version': '3.6',
            'source_commit': 'f78475530b46d0186111a9096b30725f816b55fe',
            'binary_sha256': hashlib.sha256(NSJAIL.read_bytes()).hexdigest(),
            'policy_sha256': hashlib.sha256(POLICY.read_bytes()).hexdigest(),
            'cgroup_management': False, 'memory_limit_scope': 'per_process_RLIMIT_AS',
            'privilege_mode': 'root supervisor; unique non-root UID; no capabilities; no_new_privs',
            'network': 'isolated; AF_INET/AF_INET6 sockets denied',
            'host_project_and_gpu_exposed': False}


def configuration(output, args):
    # Use a different host UID for each supervising process when running as root.
    # This keeps RLIMIT_NPROC separate from training and from the TACO nobody UID.
    host_uid = 1000000 + os.getpid() if os.geteuid() == 0 else os.getuid()
    host_gid = host_uid if os.geteuid() == 0 else os.getgid()
    workers = args.code_eval_workers
    if not 1 <= workers <= 32:
        raise ValueError('Local NsJail scoring supports 1..32 workers; split larger runs into batches')
    if os.geteuid() == 0:
        os.chown(output, host_uid, host_gid)
    output.chmod(0o700)
    lines = [
        'name: "benchmark-coding"', 'mode: ONCE', 'hostname: "code-eval"',
        'cwd: "/tmp"', f'time_limit: {args.code_eval_timeout}',
        f'max_cpus: {workers}', 'nice_level: 19', 'log_level: WARNING',
        'clone_newuser: false', 'clone_newns: true', 'clone_newpid: true',
        'clone_newnet: true', 'clone_newipc: true', 'clone_newuts: true',
        'clone_newcgroup: false', 'keep_env: false', 'keep_caps: false',
        'disable_no_new_privs: false', 'mount_proc: false',
        f'uidmap {{ inside_id: "{host_uid}" outside_id: "{host_uid}" count: 1 }}',
        f'gidmap {{ inside_id: "{host_gid}" outside_id: "{host_gid}" count: 1 }}',
        f'rlimit_as: {memory_mib(args.code_memory)}', 'rlimit_core: 0',
        f'rlimit_stack: {memory_mib(args.code_memory)}', 'rlimit_stack_type: VALUE',
        f'rlimit_cpu: {args.code_eval_timeout}', 'rlimit_fsize: 64',
        'rlimit_nofile: 256', 'rlimit_nproc: 64', 'rlimit_nproc_type: VALUE',
        f'seccomp_policy_file: {json.dumps(str(POLICY))}',
    ]
    for item in ('HOME=/tmp', 'TMPDIR=/tmp', 'PATH=/opt/python/bin',
                 'PYTHONPATH=/opt/deps:/official', 'PYTHONDONTWRITEBYTECODE=1',
                 'XDG_CACHE_HOME=/tmp/cache', 'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1',
                 'MKL_NUM_THREADS=1', 'NUMEXPR_NUM_THREADS=1', 'TOKENIZERS_PARALLELISM=false',
                 'PYTHONHASHSEED=0', 'LANG=C.UTF-8'):
        lines.append(f'envar: {json.dumps(item)}')
    # EvalPlus sets AS/DATA/STACK itself. Keep its requested hard limits within
    # the jail's limits instead of letting it fail before evaluating an answer.
    lines.append(f'envar: "EVALPLUS_MAX_MEMORY_BYTES={min(4096, memory_mib(args.code_memory)) * 1024**2}"')

    def mount(**fields):
        def encode(value):
            return str(value).lower() if isinstance(value, bool) else json.dumps(str(value))
        lines.append('mount { ' + ' '.join(f'{key}: {encode(value)}' for key, value in fields.items()) + ' }')

    mount(dst='/', fstype='tmpfs', options='size=4m,mode=755', rw=False, nosuid=True, nodev=True)
    for source, target in ((Path(sys.base_prefix), '/opt/python'),
                           (Path(sysconfig.get_path('purelib')), '/opt/deps'),
                           (ROOT / 'official', '/official'), (ROOT / 'score.py', '/app/score.py'),
                           (Path('/lib'), '/lib'), (Path('/lib64'), '/lib64')):
        if not source.exists():
            raise RuntimeError(f'Required runtime path missing: {source}')
        mount(src=source.resolve(), dst=target, is_bind=True, rw=False, nosuid=True, nodev=True)
    mount(dst='/tmp', fstype='tmpfs', options='size=512m,mode=1777', rw=True, nosuid=True, nodev=True)
    mount(dst='/dev/shm', fstype='tmpfs', options='size=128m,mode=1777', rw=True, nosuid=True, nodev=True)
    for device in ('null', 'zero', 'urandom', 'random'):
        mount(src=f'/dev/{device}', dst=f'/dev/{device}', is_bind=True, rw=True, nosuid=True)
    mount(dst='/proc', fstype='proc', rw=False, nosuid=True, nodev=True, noexec=True)
    mount(src=output.resolve(), dst='/output', is_bind=True, rw=True, nosuid=True, nodev=True, noexec=True)
    return '\n'.join(lines) + '\n'


def execute(python_args, args, job=None):
    """Execute trusted scorer/probe entrypoints in the same sandbox; return files/logs.

    Callers may supply model code only through the serialized scoring job.
    """
    check_runtime()
    temp_root = RUNTIME / 'nsjail-tmp'
    temp_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='eval-', dir=temp_root) as temporary:
        work = Path(temporary)
        output = work / 'output'
        output.mkdir()
        config = configuration(output, args)
        config_path = work / 'jail.cfg'
        config_path.write_text(config)
        # glibc derives each new thread's default stack from the initial soft
        # limit. Keep a large hard limit for EvalPlus, but exec the scorer with
        # an 8 MiB soft limit so ordinary threads do not reserve 4 GiB each.
        bootstrap = (
            'import os,resource,sys; '
            'hard=resource.getrlimit(resource.RLIMIT_STACK)[1]; '
            'resource.setrlimit(resource.RLIMIT_STACK,(min(8*1024**2,hard),hard)); '
            'os.execv(sys.executable,[sys.executable,"-B",*sys.argv[1:]])'
        )
        command = [str(NSJAIL), '--config', str(config_path), '--',
                   '/opt/python/bin/python3.11', '-B', '-c', bootstrap, *python_args]
        job_path = work / 'job.pkl'
        with job_path.open('wb') as handle:
            if job is not None:
                pickle.dump(job, handle)
        with job_path.open('rb') as source, (work / 'stdout.log').open('wb') as stdout, \
                (work / 'stderr.log').open('wb') as stderr:
            proc = subprocess.Popen(command, stdin=source, stdout=stdout, stderr=stderr,
                                    env={'PATH': '/usr/bin:/bin', 'TMPDIR': str(work)},
                                    start_new_session=True)
            try:
                status = proc.wait(timeout=args.code_eval_timeout + 10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                raise RuntimeError('NsJail supervisor exceeded its wall-time limit')
        result = {'returncode': status, 'stdout': (work / 'stdout.log').read_text(errors='replace'),
                  'stderr': (work / 'stderr.log').read_text(errors='replace'), 'config': config}
        score_path = output / 'scores.json'
        if score_path.is_file() and not score_path.is_symlink():
            result['scores'] = json.loads(score_path.read_text())
        return result


def score_in_nsjail(job, args):
    result = execute(['/app/score.py'], args, job)
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for key, name in (('stdout', 'nsjail.stdout.log'), ('stderr', 'nsjail.stderr.log'),
                      ('config', 'nsjail.cfg')):
        (destination / name).write_text(result[key])
    worker_crash = re.search(r'^Process [^\n]+:\nTraceback \(most recent call last\):',
                             result['stderr'], re.MULTILINE)
    if result['returncode'] != 0 or 'scores' not in result or worker_crash:
        raise RuntimeError(f'NsJail scoring failed, exit={result["returncode"]}. '
                           f'See {destination / "nsjail.stderr.log"}\n{result["stderr"][-3000:]}')
    return result['scores']
