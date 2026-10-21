#!/usr/bin/env python3
"""Run the three domain trainers serially, each owning all eight GPUs.

Every job uses six actor (student) GPUs and two teacher GPUs; the domains run
one after another so only one trainer touches the node at a time.
"""
from __future__ import annotations

import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[3]
EXP = Path(os.environ.get(
    "EXPERIMENT_ROOT", ROOT / "outputs/prefix_composition_3domain_20260917"
)).resolve()
LAUNCHER = ROOT / "experiment.sh"
SCRIPT = ROOT / "prefix_tuning/virtual_prefix"
DEVICES = os.environ.get("TRAIN_DEVICES", "0,1,2,3,4,5,6,7")
ACTOR_GPUS = os.environ.get("TRAIN_ACTOR_GPUS", "6")
TEACHER_GPUS = os.environ.get("TRAIN_TEACHER_GPUS", "2")
DOMAINS = ("awm", "envscaler", "taco")
GPUS_PER_JOB = len([device for device in DEVICES.split(",") if device.strip()])


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def atomic_json(path, value):
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temp, path)


def run_name(domain):
    target = os.environ.get("PARTIAL_ROLLOUT_TARGET", "32")
    suffix = f"_partial_pool{target}" if os.environ.get("TRAINER_MODE", "colocate_async") == "colocate_async" else ""
    return f"{domain}_{GPUS_PER_JOB}gpu{suffix}"


def checkpoint_dir(domain):
    return EXP / "checkpoints" / run_name(domain)


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


def start(domain, resume):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=DEVICES, ASCEND_RT_VISIBLE_DEVICES=DEVICES,
               ACTOR_GPUS=ACTOR_GPUS, TEACHER_GPUS=TEACHER_GPUS,
               RUN_MODE="full", RUN_NAME=run_name(domain),
               RESUME_MODE="resume_path" if resume else "disable",
               CHECKPOINT_DIR=str(checkpoint_dir(domain)),
               TRAIN_BATCH_SIZE="24", PPO_MINI_BATCH_SIZE="24", PYTHONUNBUFFERED="1",
               RAY_TMPDIR=f"/tmp/kvcache-prefix-{domain}", RAY_ADDRESS="local",
               EXPERIMENT_ROOT=str(EXP))
    args = ["bash", str(LAUNCHER), "_train-domain", domain,
            "+ray_kwargs.ray_init.address=local", "ray_kwargs.ray_init.num_cpus=28",
            f"+ray_kwargs.ray_init.num_gpus={GPUS_PER_JOB}",
            "+ray_kwargs.ray_init.include_dashboard=False",
            "+ray_kwargs.ray_init.object_store_memory=8589934592"]
    if resume:
        args.append(f"trainer.resume_from_path={resume}")
    logfile = EXP / "logs" / f"serial_{domain}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    with logfile.open("a") as handle:
        process = subprocess.Popen(args, cwd=ROOT, env=env, stdout=handle,
                                   stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                   start_new_session=True)
    print(f"{now()} started {domain} pid={process.pid} GPUs={DEVICES} "
          f"actor={ACTOR_GPUS} teacher={TEACHER_GPUS}", flush=True)
    return dict(domain=domain, gpus=DEVICES, actor=ACTOR_GPUS, teacher=TEACHER_GPUS,
                process=process, children={}, log=str(logfile), started_at=now())


def export(domain):
    checkpoints = [p for p in checkpoint_dir(domain).glob("global_step_*")
                   if (p / "actor/trainable_only_meta.json").is_file()]
    expected = 66 if domain == "taco" else 67
    latest = max(checkpoints, key=lambda p: int(p.name.rsplit("_", 1)[1]))
    assert latest.name == f"global_step_{expected}", (domain, latest)
    subprocess.run([sys.executable, str(SCRIPT / "export_hybrid_delta_residual_attention_prefix_checkpoint.py"),
                    "--checkpoint", str(latest / "actor"), "--prepared-model",
                    os.environ.get("MODEL", str(EXP / "prepared/hybrid-G512-A64")), "--output", str(EXP / "prefixes" / domain)],
                   cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, check=True)


def run(resume):
    (EXP / "logs").mkdir(parents=True, exist_ok=True)
    with (EXP / "training_queue.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        completed, failed = [], {}

        def status(state="running", current=None):
            atomic_json(EXP / "queue_status.json", dict(
                phase="serial_training", state=state, pid=os.getpid(), updated_at=now(),
                evaluation_status="pending_dataset", pending=[], completed=completed, failed=failed,
                running={} if current is None else {
                    current["domain"]: {k: v for k, v in current.items()
                                        if k not in ("process", "children")}
                    | {"pid": current["process"].pid}}))

        first_resume = resume
        for domain in DOMAINS:
            job = start(domain, first_resume if domain == "awm" else None)
            first_resume = None
            status(current=job)
            while True:
                capture(job)
                code = job["process"].poll()
                if code is not None:
                    break
                status(current=job)
                time.sleep(5)
            cleanup(job)
            if code != 0:
                failed[domain] = dict(exit_code=code, log=job["log"])
                print(f"{now()} failed {domain} code={code}", flush=True)
                # Preserve the remaining domains' progress.
            else:
                try:
                    export(domain)
                    completed.append(domain)
                except Exception as exc:
                    failed[domain] = dict(export_error=repr(exc))
            status("failed" if failed else "running")
        if failed:
            status("failed")
            raise RuntimeError(f"Training queue failed: {failed}")
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
