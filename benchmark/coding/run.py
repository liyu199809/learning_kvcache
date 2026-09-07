"""Generate through an OpenAI-compatible endpoint; score with official evaluators."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import pickle
import re
import subprocess
import sys
import sysconfig
import tempfile
import uuid
from collections import Counter
from pathlib import Path

from data import lcb_sample, load_tasks, select_tasks

ROOT = Path(__file__).resolve().parent


def extract_code(text):
    # Discard raw Qwen/DeepSeek reasoning before extracting the final answer.
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    blocks = re.findall(r"```(?:python|py)?[ \t]*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return (blocks[-1] if blocks else text).strip()


def normalize_solution(text, task, benchmark):
    code = extract_code(text)
    if benchmark != "livecodebench":
        from evalplus.sanitize import sanitize
        code = sanitize(code, entrypoint=task["problem"]["entry_point"])
    return code


async def generate_jobs(jobs, args, path):
    from openai import AsyncOpenAI

    semaphore = asyncio.Semaphore(args.concurrency)
    async with AsyncOpenAI(base_url=args.openai_base_url,
                           api_key=os.environ.get("CODE_EVAL_API_KEY", "EMPTY"),
                           timeout=args.step_timeout, max_retries=2) as client:
        async def one(task, sample_index):
            async with semaphore:
                error = None
                text = ""
                finish_reason = None
                try:
                    response = await client.chat.completions.create(
                        model=args.model, messages=[{"role": "user", "content": task["prompt"]}],
                        temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.llm_max_completion_tokens or 16384,
                    )
                    text = response.choices[0].message.content or ""
                    finish_reason = response.choices[0].finish_reason
                    if not text.strip():
                        error = "empty_final_answer"
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                row = {"task_id": task["task_id"], "sample_index": sample_index,
                       "solution": normalize_solution(text, task, args.benchmark),
                       "raw_response": text, "finish_reason": finish_reason, "error": error}
                with path.open("a", encoding="utf-8") as out:
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                return row
        return await asyncio.gather(*(one(task, sample_index) for task, sample_index in jobs))


async def generate(tasks, args, path):
    jobs = [(task, sample_index) for task in tasks
            for sample_index in range(args.code_n_samples)]
    return await generate_jobs(jobs, args, path)


def read_samples(path, tasks, args):
    by_id = {t["task_id"]: t for t in tasks}
    rows = []
    with Path(path).open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("task_id") not in by_id:
                continue
            if not isinstance(row.get("solution"), str):
                raise ValueError("--code-samples requires task_id and full solution strings")
            rows.append({**row, "solution": normalize_solution(
                row["solution"], by_id[row["task_id"]], args.benchmark,
            )})
    counts = Counter(r["task_id"] for r in rows)
    if any(counts[t["task_id"]] != args.code_n_samples for t in tasks):
        raise ValueError("Every selected task must have exactly --code-n-samples solutions; "
                         "use --task-ids/--tasks for a subset")
    return rows


def prepare_retry_samples(path, tasks, args, output_path):
    """Keep successful rows and return jobs for missing or failed samples."""
    by_id = {t["task_id"]: t for t in tasks}
    prior = {}
    with Path(path).open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            task_id = row.get("task_id")
            sample_index = row.get("sample_index")
            if task_id not in by_id or not isinstance(sample_index, int):
                continue
            if 0 <= sample_index < args.code_n_samples:
                prior[(task_id, sample_index)] = row
    kept = []
    jobs = []
    for task in tasks:
        for sample_index in range(args.code_n_samples):
            row = prior.get((task["task_id"], sample_index))
            if row is None or row.get("error"):
                jobs.append((task, sample_index))
                continue
            if not isinstance(row.get("solution"), str):
                raise ValueError("Retry source rows require task_id, sample_index, and solution")
            kept.append({**row, "solution": normalize_solution(
                row["solution"], task, args.benchmark,
            )})
    output_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in kept),
        encoding="utf-8",
    )
    return kept, jobs


def score_in_container(job, args):
    # Only evaluator runtime and this run's results are mounted. The job is streamed
    # over stdin: LCB hidden tests can be several GiB, and mounting a directory that
    # contains the serialized job can stall Docker's container-create path.
    # No project, credentials, model weights, Docker socket or host network enter.
    name = "self-evolver-code-" + uuid.uuid4().hex[:12]
    # Keep the large unmounted job and the mounted output in separate parents.
    # Some Docker/storage configurations inspect the bind source's parent.
    output_parent = "/dev/shm" if Path("/dev/shm").is_dir() else None
    with (tempfile.TemporaryDirectory(prefix="code-eval-job-") as job_temp,
          tempfile.TemporaryDirectory(prefix="code-eval-output-", dir=output_parent) as output_temp):
        source = Path(job_temp)
        output = Path(output_temp)
        source.chmod(0o755)
        output.chmod(0o777)
        with (source / "job.pkl").open("wb") as out:
            pickle.dump(job, out)
        job_path = source / "job.pkl"
        job_path.chmod(0o644)
        mounts = [
            (Path(sys.base_prefix), "/opt/python"),
            (Path(sysconfig.get_path("purelib")), "/opt/deps"),
            (ROOT / "official", "/official"),
            (ROOT / "score.py", "/app/score.py"),
        ]
        command = ["docker", "create", "--interactive", "--name", name, "--network", "none",
                   "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges",
                   "--user", "65534:65534", "--pids-limit", "512",
                   "--memory", args.code_memory, "--cpus", str(args.code_eval_workers),
                   "--tmpfs", "/tmp:rw,exec,size=2g,mode=1777", "--shm-size", "256m",
                   "--env", "HOME=/tmp", "--env", "XDG_CACHE_HOME=/tmp/cache",
                   "--env", "PYTHONPATH=/opt/deps:/official", "--env", "OMP_NUM_THREADS=1",
                   "--env", "OPENBLAS_NUM_THREADS=1", "--env", "TOKENIZERS_PARALLELISM=false"]
        for host, target in mounts:
            command.extend(["--mount", f"type=bind,src={host.resolve()},dst={target},readonly"])
        command.extend(["--mount", f"type=bind,src={output},dst=/output",
                        "--workdir", "/tmp", "--entrypoint", "/opt/python/bin/python3.11",
                        args.code_docker_image, "-B", "/app/score.py"])
        try:
            # Create first, then attach the multi-GiB stdin stream.  `docker run -i`
            # can block before container creation while it prepares a large stdin.
            subprocess.run(command, check=True, timeout=args.code_eval_timeout,
                           stdout=subprocess.DEVNULL)
            with job_path.open("rb") as job_input:
                subprocess.run(["docker", "start", "--attach", "--interactive", name],
                               stdin=job_input, check=True, timeout=args.code_eval_timeout)
            return json.loads((output / "scores.json").read_text())
        finally:
            subprocess.run(["docker", "rm", "-f", name], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, check=False)


def main(args):
    for name in ("concurrency", "code_n_samples", "code_eval_workers", "code_timeout", "code_eval_timeout"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    k = sorted({int(v) for v in args.code_pass_k.split(",")})
    if not k or min(k) < 1:
        raise ValueError("--code-pass-k requires positive integers")
    k = [v for v in k if v <= args.code_n_samples]
    if not k:
        raise ValueError("No requested pass@k has k <= --code-n-samples")
    if args.code_samples and args.code_retry_errors_from:
        raise ValueError("--code-samples and --code-retry-errors-from are mutually exclusive")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name in ("samples.jsonl", "manifest.json", "results.json", "summary.json"):
        if (output / name).exists():
            raise ValueError(f"Output already contains {name}; choose a fresh --output-dir")
    if not args.code_generate_only:
        subprocess.run(["docker", "image", "inspect", args.code_docker_image], check=True,
                       stdout=subprocess.DEVNULL)
    tasks, provenance = load_tasks(args)
    total = len(tasks)
    tasks = select_tasks(tasks, args)
    manifest = {"benchmark": args.benchmark, **provenance, "available_tasks": total,
                "selected_tasks": [t["task_id"] for t in tasks], "num_samples": args.code_n_samples,
                "model": args.model, "endpoint": args.openai_base_url,
                "temperature": args.temperature, "top_p": args.top_p,
                "max_tokens": args.llm_max_completion_tokens or 16384,
                "timeout": args.code_timeout, "workers": args.code_eval_workers,
                "base_only": False, "pass_k": k,
                "samples_source": args.code_samples,
                "retry_errors_from": args.code_retry_errors_from,
                "prompt_mode": "chat_full_solution"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[coding] {args.benchmark}: {len(tasks)}/{total} tasks, n={args.code_n_samples}", flush=True)
    if args.code_samples:
        rows = read_samples(args.code_samples, tasks, args)
        (output / "samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    elif args.code_retry_errors_from:
        sample_path = output / "samples.jsonl"
        rows, jobs = prepare_retry_samples(args.code_retry_errors_from, tasks, args, sample_path)
        print(f"[coding] retrying {len(jobs)} missing/error samples; reusing {len(rows)}", flush=True)
        rows.extend(asyncio.run(generate_jobs(jobs, args, sample_path)))
    else:
        rows = asyncio.run(generate(tasks, args, output / "samples.jsonl"))
    errors = sum(bool(r.get("error")) for r in rows)
    if args.code_generate_only:
        print(f"Saved {len(rows)} samples ({errors} generation errors) to {output}")
        return 1 if errors else 0
    grouped = {t["task_id"]: [] for t in tasks}
    for row in rows:
        grouped[row["task_id"]].append(row["solution"])
    problems = []
    for task in tasks:
        if args.benchmark == "livecodebench":
            with open(task["path"], "rb") as source:
                source.seek(task["offset"])
                problems.append(lcb_sample(json.loads(source.readline())))
        else:
            problems.append(task["problem"])
    job = {"dataset": "livecodebench" if args.benchmark == "livecodebench" else provenance["dataset"],
           "task_ids": list(grouped), "problems": problems, "solutions": list(grouped.values()),
           "k": k, "workers": args.code_eval_workers, "timeout": args.code_timeout,
           "base_only": False}
    results = score_in_container(job, args)
    (output / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    summary = {"benchmark": args.benchmark, "model": args.model, "tasks": len(tasks),
               "samples": len(rows), "generation_errors": errors, "metrics": results["metrics"],
               "subset": len(tasks) != total, "provenance": provenance}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Output: {output}")
    # Generation failures are already represented as failed samples in the
    # official score and recorded in summary.json.  Once scoring completed,
    # keep the suite runner moving so one failed request does not skip every
    # subsequent benchmark.
    return 0


if __name__ == "__main__":
    raise SystemExit(main(argparse.Namespace(**json.load(sys.stdin))))
