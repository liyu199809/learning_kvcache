"""Compare optimized extraction with previously saved official outputs in parallel."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from data import load_tasks
from run import normalize_solution


def check(job):
    source, task_id, raw, expected, entry_point, benchmark = job
    actual = normalize_solution(raw, {"problem": {"entry_point": entry_point}}, benchmark)
    if actual != expected:
        raise AssertionError(f"Changed sanitizer output: {source}, {task_id}")
    return source, task_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    jobs = []
    hashes = {}
    for benchmark, name in (("humaneval+", "humaneval_plus"), ("mbpp+", "mbpp_plus")):
        tasks, _ = load_tasks(SimpleNamespace(benchmark=benchmark))
        entries = {t["task_id"]: t["problem"]["entry_point"] for t in tasks}
        for path in sorted(args.suite.glob(f"*/thinking_off/{name}_generation/samples.jsonl")):
            contents = path.read_bytes()
            hashes[str(path)] = hashlib.sha256(contents).hexdigest()
            for line in contents.splitlines():
                row = json.loads(line)
                jobs.append((str(path), row["task_id"], row["raw_response"], row["solution"],
                             entries[row["task_id"]], benchmark))
    print(f"Checking {len(jobs)} already saved official outputs", flush=True)
    done = 0
    with ProcessPoolExecutor(max_workers=16) as pool:
        futures = [pool.submit(check, job) for job in jobs]
        for future in as_completed(futures):
            future.result()
            done += 1
            if done % 100 == 0 or done == len(jobs):
                print(f"Matched {done}/{len(jobs)}", flush=True)
    with args.output.open("x") as out:
        json.dump({"status": "passed", "exact_matches": done, "input_sha256": hashes}, out, indent=2)


if __name__ == "__main__":
    main()
