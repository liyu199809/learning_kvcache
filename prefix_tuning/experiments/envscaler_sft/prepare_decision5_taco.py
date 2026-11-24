"""Check each pinned TACO reference in an isolated grader and retain the shortest pass."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
from pathlib import Path
import re
import subprocess
import sys

from audit_sft_schemas import REPO
from prepare import OFFENSIVE, OUT, lines

ROOT = OUT / "sft-decision5"
SERVER = REPO / "OpenEnv/envs/code_judge_env/server"
sys.path.insert(0, str(SERVER.parent.parent))
from code_judge_env.server.data_loader import TacoDataLoader

BWRAP = REPO / "benchmark/runtime/bwrap/usr/bin/bwrap"
PYTHON = "/usr/local/bin/python3.12"
WORKER = SERVER / "judge_worker.py"
SYSTEM = "Solve the current independent benchmark task. Each task starts with fresh state. Use only the tools offered for this task. For programming questions return Python in a fenced code block."


def judge(code, tests):
    command = [str(BWRAP), "--unshare-all", "--die-with-parent", "--uid", "65534", "--gid", "65534",
               "--clearenv", "--tmpfs", "/tmp", "--dev", "/dev", "--chdir", "/tmp",
               "--setenv", "HOME", "/tmp", "--setenv", "TMPDIR", "/tmp",
               "--ro-bind", PYTHON, PYTHON, "--ro-bind", "/usr/local/lib/python3.12", "/usr/local/lib/python3.12",
               "--ro-bind", "/lib", "/lib", "--ro-bind", "/lib64", "/lib64",
               "--ro-bind", str(WORKER), "/judge_worker.py", "--",
               PYTHON, "-I", "-c",
               "import os,resource,runpy; assert os.getuid()==65534 and not os.path.exists('/host-ssd'); "
               "resource.setrlimit(resource.RLIMIT_AS,(2147483648,2147483648)); "
               "resource.setrlimit(resource.RLIMIT_CPU,(100,100)); "
               "resource.setrlimit(resource.RLIMIT_FSIZE,(16777216,16777216)); "
               "resource.setrlimit(resource.RLIMIT_NPROC,(128,128)); "
               "runpy.run_path('/judge_worker.py',run_name='__main__')"]
    try:
        proc = subprocess.run(command, input=json.dumps({"code": code, "tests": tests}),
                              text=True, capture_output=True, timeout=100)
    except subprocess.TimeoutExpired:
        return "timeout"
    from code_judge_env.server.judge_worker import RESULT_MARKER
    report = next((l[len(RESULT_MARKER):] for l in reversed(proc.stdout.splitlines())
                   if l.startswith(RESULT_MARKER)), None)
    if report is None:
        raise RuntimeError(f"Isolated worker did not start or report (exit {proc.returncode}): {proc.stderr[:250]}")
    outcome = json.loads(report)
    if not outcome["ok"]:
        raise RuntimeError("Isolated TACO grader reported an internal error")
    return "pass" if outcome["result"]["all_passed"] else "fail"


def median_candidates(candidates):
    normalized = [(position, re.sub(r"\n{3,}", "\n\n", "\n".join(
        line.rstrip() for line in code.replace("\t", "    ").splitlines())).strip())
        for position, code in candidates]
    filtered = [(position, code) for position, code in normalized
                if sum(bool(line) for line in code.splitlines()) >= 3
                and max(map(len, code.splitlines())) <= 160]
    ordered = sorted(filtered, key=lambda candidate: len(candidate[1]))
    middle = (len(ordered) - 1) / 2
    return [ordered[index] for index in
            sorted(range(len(ordered)), key=lambda index: abs(index - middle))[:20]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--decision9", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    data = TacoDataLoader(OUT / "verifier-inputs/taco")
    cohort = OUT / ("sft-decision6/t3-data/official-arm" if args.decision9 else "sft-800")
    cases = [(s, r) for s in ("train", "val") for r in lines(cohort / f"{s}.jsonl") if r["source"] == "taco"]
    if args.probe:
        cases = cases[:1]

    def evaluate(case):
        split, old = case
        idx = int(old["task_id"].rsplit(":", 1)[1])
        task = data.get_task(idx, include_solutions=True)
        question = task["problem"].strip()
        assert question == old["messages"][1]["content"].strip()
        outcomes = ["screened"] * len(task["solutions"])
        candidates = []
        for position, code in enumerate(task["solutions"]):
            if not isinstance(code, str) or not code.strip() or OFFENSIVE.search(question + "\n" + code):
                continue
            outcomes[position] = "not_needed"
            candidates.append((position, code))
        selected = None
        # Stable ordering retains the original tie-break between equal-length solutions.
        ordered = (median_candidates(candidates) if args.decision9 else
                   sorted(candidates, key=lambda candidate: len(candidate[1].strip())))
        for position, code in ordered:
            result = outcomes[position] = judge(code, task["tests"])
            if result == "pass":
                selected = code.strip()
                break
        audit = {"id": old["task_id"], "split": split, "outcomes": outcomes}
        if args.decision9:
            audit["fallback_shortest"] = selected is None
            row = old
            if selected is not None:
                assistant = next(message for message in row["messages"] if message["loss"])
                assistant["content"] = "```python\n" + selected + "\n```"
            return {**audit, "selected": row}
        if selected is None:
            return {**audit, "reason": "official_solution_fails_tests"}
        row = {"id": old["task_id"], "source": "taco", "type": "SFT", "enable_thinking": False, "tools": [],
               "messages": [{"role": "system", "content": SYSTEM, "loss": False},
                            {"role": "user", "content": question, "loss": False},
                            {"role": "assistant", "content": "```python\n" + selected + "\n```", "loss": True}]}
        return {**audit, "selected": row}

    ROOT.mkdir(exist_ok=True)
    if args.decision9:
        output = OUT / "sft-decision9" / ("taco-probe.jsonl" if args.probe else "taco-audit.jsonl")
    else:
        output = ROOT / ("taco-probe-fast.jsonl" if args.probe else "taco-audit.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    expected = {row["task_id"]: split for split, row in cases}
    if len(expected) != len(cases):
        raise ValueError("Duplicate task in TACO cohort")
    with output.open("a+", buffering=1) as stream, ThreadPoolExecutor(max_workers=args.workers) as pool:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stream.seek(0)
        completed = set()
        for line in stream:
            result = json.loads(line)
            ident = result["id"]
            if ident in completed or expected.get(ident) != result["split"]:
                raise ValueError(f"Unexpected existing TACO audit row: {ident}")
            completed.add(ident)
        jobs = {pool.submit(evaluate, case): case[1]["task_id"]
                for case in cases if case[1]["task_id"] not in completed}
        print(f"Resuming TACO audit with {len(completed)}/{len(expected)} already recorded", flush=True)
        for job in as_completed(jobs):
            try:
                result = job.result()
            except Exception as error:
                print(f"TACO audit error for {jobs[job]}: {error}", file=sys.stderr, flush=True)
                continue
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            completed.add(result["id"])
            i = len(completed)
            if i % 10 == 0 or i == len(expected) or args.probe:
                print(f"{i}/{len(expected)} {result['id']} "
                      f"{sum(outcome not in {'not_needed', 'screened'} for outcome in result['outcomes'])} tested",
                      flush=True)
    if len(completed) != len(expected):
        raise SystemExit(f"Incomplete TACO audit: {len(completed)}/{len(expected)} recorded. Rerun to resume.")


if __name__ == "__main__":
    main()
