#!/usr/bin/env python3
"""Supervise replacement domain trainers after a partial queue recovery."""
from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil


ROOT = Path(__file__).resolve().parents[3]
EXP = Path(os.environ["EXPERIMENT_ROOT"]).resolve()
SCRIPT = ROOT / "prefix_tuning/virtual_prefix"
EXPECTED_STEPS = {"awm": 67, "envscaler": 67, "taco": 66}


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def atomic_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temp, path)


def checkpoint(domain: str) -> Path | None:
    root = EXP / "checkpoints" / f"{domain}_2gpu"
    candidates = [
        path
        for path in root.glob("global_step_*")
        if (path / "actor/trainable_only_meta.json").is_file()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: int(path.name.rsplit("_", 1)[1]))


def export(domain: str, ckpt: Path) -> None:
    output = EXP / "prefixes" / domain
    if output.exists():
        return
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT / "export_hybrid_delta_residual_attention_prefix_checkpoint.py"),
            "--checkpoint",
            str(ckpt / "actor"),
            "--prepared-model",
            str(EXP / "prepared/hybrid-G512-A64"),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        check=True,
    )


def compose() -> None:
    for domains in (
        ("awm", "envscaler"),
        ("awm", "taco"),
        ("envscaler", "taco"),
        ("awm", "envscaler", "taco"),
    ):
        output = EXP / "models" / "_".join(domains)
        if output.exists():
            continue
        command = [sys.executable, "-m", "prefix_tuning.virtual_prefix.compose_hybrid_prefixes"]
        for domain in domains:
            command += ["--input", str(EXP / "prefixes" / domain)]
        command += ["--output", str(output)]
        subprocess.run(command, cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, check=True)


def main() -> None:
    status_path = EXP / "recovery_status.json"
    status = json.loads(status_path.read_text())
    jobs = status["jobs"]
    status["supervisor_pid"] = os.getpid()
    status["state"] = "running"
    for job in jobs.values():
        job.setdefault("state", "running")
    atomic_json(status_path, status)

    while True:
        failures: dict[str, str] = {}
        complete = []
        for domain in EXPECTED_STEPS:
            prefix = EXP / "prefixes" / domain
            if prefix.exists():
                complete.append(domain)
                if domain in jobs:
                    jobs[domain]["state"] = "exported"
                continue

            ckpt = checkpoint(domain)
            expected = EXPECTED_STEPS[domain]
            if ckpt is not None and ckpt.name == f"global_step_{expected}":
                try:
                    export(domain, ckpt)
                    complete.append(domain)
                    if domain in jobs:
                        jobs[domain]["state"] = "exported"
                except Exception as exc:
                    failures[domain] = f"export failed: {exc!r}"
                continue

            if domain in jobs:
                pid = int(jobs[domain]["pid"])
                if not psutil.pid_exists(pid):
                    jobs[domain]["state"] = "failed"
                    failures[domain] = f"trainer exited before global_step_{expected}"

        status["updated_at"] = now()
        status["completed"] = sorted(complete)
        status["failures"] = failures
        if failures:
            status["state"] = "failed"
            atomic_json(status_path, status)
            raise RuntimeError(f"Recovery failed: {failures}")
        if len(complete) == len(EXPECTED_STEPS):
            compose()
            status["state"] = "awaiting_evaluation"
            status["updated_at"] = now()
            atomic_json(status_path, status)
            return
        atomic_json(status_path, status)
        time.sleep(30)


if __name__ == "__main__":
    main()
