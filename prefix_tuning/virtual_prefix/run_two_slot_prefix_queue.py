#!/usr/bin/env python3
"""Run two isolated four-GPU prefix trainers, then occupy the freed slot."""
from __future__ import annotations

import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "outputs/prefix_composition_3domain_20260917"
SCRIPT = ROOT / "prefix_tuning/virtual_prefix"
SLOTS = {"a": "0,1,2,3", "b": "4,5,6,7"}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def atomic_json(path, value):
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temp, path)


def checkpoint_dir(domain):
    return EXP / "checkpoints" / f"{domain}_4gpu"


def capture(job):
    try:
        parent = psutil.Process(job["process"].pid)
        for proc in parent.children(recursive=True):
            try:
                job["children"][proc.pid] = proc.create_time()
            except psutil.NoSuchProcess:
                pass
    except psutil.NoSuchProcess:
        pass


def cleanup(job):
    # Only descendants observed while this exact trainer was alive. Never ray stop.
    live = []
    for pid, created in job["children"].items():
        try:
            proc = psutil.Process(pid)
            if proc.create_time() == created:
                proc.terminate()
                live.append(proc)
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(live, timeout=20)
    for proc in alive:
        try:
            if proc.create_time() == job["children"][proc.pid]:
                proc.kill()
        except psutil.NoSuchProcess:
            pass
    _, remaining = psutil.wait_procs(alive, timeout=10)
    assert not [p for p in remaining if p.status() != psutil.STATUS_ZOMBIE], "Job cleanup incomplete"


def start(domain, slot, resume):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=SLOTS[slot], ACTOR_GPUS="3", TEACHER_GPUS="1",
               RUN_MODE="full", RUN_NAME=f"{domain}_4gpu", RESUME_MODE="resume_path" if resume else "disable",
               CHECKPOINT_DIR=str(checkpoint_dir(domain)), VAL_BEFORE_TRAIN="False", TEST_FREQ="-1",
               TRAIN_BATCH_SIZE="24", PPO_MINI_BATCH_SIZE="24", PYTHONUNBUFFERED="1",
               RAY_TMPDIR=f"/tmp/kvcache-prefix-{domain}", RAY_ADDRESS="local",
               EXPERIMENT_ROOT=str(EXP))
    args = ["bash", str(SCRIPT / "run_three_domain_prefix.sh"), domain,
            "+ray_kwargs.ray_init.address=local", "ray_kwargs.ray_init.num_cpus=48",
            "+ray_kwargs.ray_init.num_gpus=4", "+ray_kwargs.ray_init.include_dashboard=False",
            "+ray_kwargs.ray_init.object_store_memory=8589934592"]
    if resume:
        args.append(f"trainer.resume_from_path={resume}")
    logfile = EXP / "logs" / f"parallel_{domain}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    with logfile.open("a") as handle:
        process = subprocess.Popen(args, cwd=ROOT, env=env, stdout=handle,
                                   stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                   start_new_session=True)
    print(f"{now()} started {domain} pid={process.pid} slot={slot} GPUs={SLOTS[slot]}", flush=True)
    return dict(domain=domain, slot=slot, gpus=SLOTS[slot], process=process,
                children={}, log=str(logfile), started_at=now())


def export(domain):
    checkpoints = [p for p in checkpoint_dir(domain).glob("global_step_*")
                   if (p / "actor/trainable_only_meta.json").is_file()]
    expected = 66 if domain == "taco" else 67
    latest = max(checkpoints, key=lambda p: int(p.name.rsplit("_", 1)[1]))
    assert latest.name == f"global_step_{expected}", (domain, latest)
    subprocess.run([sys.executable, str(SCRIPT / "export_hybrid_delta_residual_attention_prefix_checkpoint.py"),
                    "--checkpoint", str(latest / "actor"), "--prepared-model",
                    str(EXP / "prepared/hybrid-G512-A64"), "--output", str(EXP / "prefixes" / domain)],
                   cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, check=True)


def occupy(slot):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": SLOTS[slot]}
    log = EXP / "logs" / f"gpu_occupier_slot_{slot}.log"
    with log.open("a") as handle:
        process = subprocess.Popen([sys.executable, str(ROOT / "gpu_load.py"), "--devices", "all",
                                    "--memory-percent", "70", "--utilization", "100",
                                    "--duration", "86400"], cwd=ROOT, env=env, stdout=handle,
                                   stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                   start_new_session=True)
    record = dict(pid=process.pid, slot=slot, gpus=SLOTS[slot], started_at=now(),
                  duration_seconds=86400, log=str(log))
    atomic_json(EXP / "gpu_occupier_status.json", record)
    print(f"{now()} occupier {record}", flush=True)
    return record


def run(resume):
    (EXP / "logs").mkdir(parents=True, exist_ok=True)
    with (EXP / "training_queue.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pending = ["taco"]
        running = {"a": start("awm", "a", resume), "b": start("envscaler", "b", None)}
        completed, failed, holder = [], {}, None

        def status(state="running"):
            atomic_json(EXP / "queue_status.json", dict(
                phase="parallel_training", state=state, pid=os.getpid(), updated_at=now(),
                evaluation_status="pending_dataset", pending=pending, completed=completed, failed=failed,
                running={s: {k: v for k, v in j.items() if k not in ("process", "children")}
                         | {"pid": j["process"].pid} for s, j in running.items()}, occupier=holder))

        status()
        while running:
            for slot, job in list(running.items()):
                capture(job)
                code = job["process"].poll()
                if code is None:
                    continue
                cleanup(job)
                del running[slot]
                if code != 0:
                    failed[job["domain"]] = dict(exit_code=code, log=job["log"])
                    print(f"{now()} failed {job['domain']} code={code}", flush=True)
                    # Preserve the other domain's progress; do not silently reuse a failed slot.
                else:
                    try:
                        export(job["domain"])
                        completed.append(job["domain"])
                    except Exception as exc:
                        failed[job["domain"]] = dict(export_error=repr(exc))
                    if pending and not failed:
                        running[slot] = start(pending.pop(0), slot, None)
                    elif not pending and len(running) == 1 and holder is None and not failed:
                        holder = occupy(slot)
                status("failed" if failed else "running")
            status("failed" if failed else "running")
            time.sleep(5)
        if failed or pending:
            status("failed")
            raise RuntimeError(f"Training queue failed: {failed}; pending={pending}")
        for domains in (("awm", "envscaler"), ("awm", "taco"), ("envscaler", "taco"),
                        ("awm", "envscaler", "taco")):
            command = [sys.executable, "-m", "prefix_tuning.virtual_prefix.compose_hybrid_prefixes"]
            for domain in domains:
                command += ["--input", str(EXP / "prefixes" / domain)]
            command += ["--output", str(EXP / "models" / "_".join(domains))]
            subprocess.run(command, cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, check=True)
        status("awaiting_evaluation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--awm-resume", type=Path)
    options = parser.parse_args()
    if options.awm_resume:
        assert (options.awm_resume / "actor/trainable_only_meta.json").is_file()
    for domain in ("awm", "envscaler", "taco"):
        if list(checkpoint_dir(domain).glob("global_step_*")):
            raise FileExistsError(f"Refusing a fresh start over existing {checkpoint_dir(domain)}")
        if (EXP / "prefixes" / domain).exists():
            raise FileExistsError(EXP / "prefixes" / domain)
    run(options.awm_resume.resolve() if options.awm_resume else None)
