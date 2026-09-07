"""Validate the unified CLI against real official tasks, without model/GPU calls.

Run with benchmark/coding/.venv/bin/python benchmark/coding/smoke.py.
Uses cached EvalPlus data and the pinned cumulative LCB v5/v6 files, downloading if needed.
"""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from data import load_tasks

ROOT = Path(__file__).resolve().parents[2]


def main():
    output = Path(tempfile.mkdtemp(prefix="coding-real-smoke-", dir=ROOT / "benchmark/coding/data"))
    cases = (
        ("humaneval+", None, None),
        ("mbpp+", None, None),
        ("livecodebench", "v5", "1873_A"),
        ("livecodebench", "v6", "abc387_b"),
    )
    for benchmark, release, expected_id in cases:
        args = argparse.Namespace(benchmark=benchmark, lcb_release=release or "v6", lcb_data_dir=None,
                                  lcb_start_date=None, lcb_end_date=None)
        tasks, _ = load_tasks(args)
        task = next((item for item in tasks if item["task_id"] == expected_id), tasks[0])
        if benchmark == "livecodebench":
            assert task["task_id"] == expected_id
            if release == "v5":
                solution = ("t = int(input())\nfor _ in range(t):\n s = input().strip()\n "
                            "print('YES' if sum(a != b for a, b in zip(s, 'abc')) in (0, 2) else 'NO')")
            else:
                solution = "x = int(input())\nprint(sum(i*j for i in range(1,10) for j in range(1,10) if i*j != x))"
        else:
            solution = task["problem"]["prompt"] + task["problem"]["canonical_solution"]
        label = f"{benchmark}_{release}" if release else benchmark
        samples = output / f"{label}.jsonl"
        samples.write_text("".join(json.dumps({"task_id": task["task_id"], "solution": code}) + "\n"
                                   for code in (solution, "raise RuntimeError('intentionally wrong')")))
        destination = output / label
        subprocess.run([
            str(ROOT / ".venv/bin/python"), "-m", "benchmark.eval.run_eval", "--benchmark", benchmark,
            "--lcb-release", release or "v6", "--task-ids", task["task_id"], "--code-n-samples", "2",
            "--code-pass-k", "1,2", "--code-samples", str(samples), "--output-dir", str(destination),
            "--code-timeout", "2", "--code-eval-workers", "2",
        ], cwd=ROOT, check=True)
        summary = json.loads((destination / "summary.json").read_text())
        metrics = summary["metrics"]
        suites = [metrics] if benchmark == "livecodebench" else [metrics["base"], metrics["plus"]]
        for suite in suites:
            assert suite["pass@1"] == .5, summary
            assert suite["pass@2"] == 1., summary
    print(f"All four real-dataset smoke runs passed. Artifacts: {output}")


if __name__ == "__main__":
    main()
