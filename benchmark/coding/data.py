"""Official datasets and prompt construction; no training data conversion."""
from __future__ import annotations

import base64
import io
import hashlib
import json
import pickle
import re
import zlib
from datetime import date
from pathlib import Path

LCB_REVISION = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"
LCB_COMMIT = "28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24"
LCB_DATASET = "livecodebench/code_generation_lite"
LCB_SHA256 = {
    "test.jsonl": "2bd02b38beb48e8c46b5b9987095d999ff38cd8efc255ea5d58974317c48f63f",
    "test2.jsonl": "095df7c5daf15f882c51a9deb84085cff1e073495a5dbcf95015a564d485f3a3",
    "test3.jsonl": "28ed26cc83363ce3f1fe2d5fad9f8393077beb1907b167a31bd3b32f80801b79",
    "test4.jsonl": "d711138ddaebfcf5f8ec6a4283ee677298c0f5c5d374a235af92aaf0584510da",
    "test5.jsonl": "7f77571c2a6df0c2a72a3277650309f67e01e0008e18117e624633df53f81214",
    "test6.jsonl": "bb4c364f71921c4495a6ad15abe1a927350b720009f4933e2e71f8af0f6fd1f5",
}


def verify_lcb_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != LCB_SHA256[path.name]:
        raise ValueError(f"LCB checksum mismatch: {path}; expected revision {LCB_REVISION}")


def release_files(release):
    if release not in {"v5", "v6"}:
        raise ValueError(f"Unsupported LCB release: {release!r}; expected 'v5' or 'v6'")
    numbers = range(1, int(release[-1]) + 1)
    return ["test.jsonl" if n == 1 else f"test{n}.jsonl" for n in numbers]


class _StringUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError("LCB private tests may not contain Python globals")


def decode_tests(raw):
    if isinstance(raw, list):
        return raw
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        packed = zlib.decompress(base64.b64decode(raw, validate=True))
        value = _StringUnpickler(io.BytesIO(packed)).load()
        if not isinstance(value, (str, bytes)):
            raise ValueError("Encoded LCB tests must contain a JSON string")
        value = json.loads(value)
    if not isinstance(value, list):
        raise ValueError("LCB tests must be a list")
    return value


def lcb_sample(row):
    tests = decode_tests(row["public_test_cases"]) + decode_tests(row["private_test_cases"])
    if not tests:
        raise ValueError("LCB task has no tests")
    metadata = row["metadata"]
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    # Preserve official newline-separated JSON arguments for call-based tasks.
    return {"input_output": json.dumps({
        "inputs": [t["input"] for t in tests],
        "outputs": [t["output"] for t in tests],
        "fn_name": metadata.get("func_name"),
    })}


def lcb_prompt(row):
    text = "### Question:\n" + row["question_content"]
    starter = row.get("starter_code", "")
    if starter:
        text += "\n\nImplement the following interface:\n```python\n" + starter + "\n```"
    else:
        text += "\n\nRead from standard input and write the answer to standard output."
    return text + "\n\nReturn the complete solution in a single fenced Python code block."


def load_tasks(args):
    if args.benchmark == "livecodebench":
        from huggingface_hub import hf_hub_download

        start = date.fromisoformat(args.lcb_start_date) if args.lcb_start_date else None
        end = date.fromisoformat(args.lcb_end_date) if args.lcb_end_date else None
        if start and end and start > end:
            raise ValueError("LCB start date is after end date")
        tasks = []
        for name in release_files(args.lcb_release):
            local_dir = Path(__file__).parent / "data" / LCB_REVISION
            path = (Path(args.lcb_data_dir) if args.lcb_data_dir else local_dir) / name
            if not args.lcb_data_dir and not path.exists():
                path = Path(hf_hub_download(
                    LCB_DATASET, name, repo_type="dataset", revision=LCB_REVISION, local_dir=local_dir,
                ))
            verify_lcb_file(path)
            # Retain offsets, not multi-GB hidden-test strings, in the task catalog.
            with path.open("rb") as source:
                while True:
                    offset = source.tell()
                    line = source.readline()
                    if not line:
                        break
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    contest_date = date.fromisoformat(row["contest_date"][:10])
                    if (start and contest_date < start) or (end and contest_date > end):
                        continue
                    tasks.append({
                        "task_id": str(row["question_id"]), "prompt": lcb_prompt(row),
                        "path": str(path.resolve()), "offset": offset,
                        "platform": row["platform"], "difficulty": row["difficulty"],
                        "contest_date": row["contest_date"],
                    })
        return tasks, {"dataset": LCB_DATASET, "revision": LCB_REVISION,
                       "release": f"release_{args.lcb_release}", "evaluator_commit": LCB_COMMIT,
                       "start_date": args.lcb_start_date, "end_date": args.lcb_end_date}
    from evalplus.data import (get_human_eval_plus, get_mbpp_plus,
                              get_human_eval_plus_hash, get_mbpp_plus_hash)

    dataset = "humaneval" if args.benchmark.startswith("humaneval") else "mbpp"
    version = "v0.1.10" if dataset == "humaneval" else "v0.2.0"
    problems = (get_human_eval_plus if dataset == "humaneval" else get_mbpp_plus)(version=version)
    tasks = [{"task_id": key, "problem": problem,
              "prompt": "Complete the following Python task. Return the full solution, including "
                        "the function signature and imports, in a single fenced Python code block.\n\n"
                        + problem["prompt"]}
             for key, problem in problems.items()]
    dataset_hash = (get_human_eval_plus_hash if dataset == "humaneval" else get_mbpp_plus_hash)(version=version)
    return tasks, {"dataset": dataset, "version": version, "hash": dataset_hash,
                   "evaluator": "evalplus==0.3.1"}


def select_tasks(tasks, args):
    ids = [t["task_id"] for t in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate task IDs in dataset")
    selected = list(tasks)
    if args.tasks:
        indices = set()
        for part in args.tasks.split(","):
            if not re.fullmatch(r"\d+(?:-\d+)?", part.strip()):
                raise ValueError(f"Invalid task index: {part!r}")
            bounds = [int(x) for x in part.strip().split("-")]
            lo, hi = bounds[0], bounds[-1]
            if lo > hi or hi >= len(tasks):
                raise ValueError(f"Task index out of range: {part!r}")
            indices.update(range(lo, hi + 1))
        selected = [t for i, t in enumerate(tasks) if i in indices]
    if args.task_ids:
        wanted = {v.strip() for v in args.task_ids.split(",")}
        missing = wanted - set(ids)
        if missing:
            raise ValueError(f"Unknown task IDs: {sorted(missing)}")
        selected = [t for t in selected if t["task_id"] in wanted]
    if args.max_tasks is not None:
        if args.max_tasks < 1:
            raise ValueError("--max-tasks must be positive")
        selected = selected[:args.max_tasks]
    if not selected:
        raise ValueError("No tasks selected")
    return selected
