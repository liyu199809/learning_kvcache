"""Runs only inside the network-disabled scoring container."""
from __future__ import annotations

import json
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def score(job):
    if job["dataset"] == "livecodebench":
        from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics

        metrics, raw, metadata = codegen_metrics(
            job["problems"], job["solutions"], k_list=job["k"],
            num_process_evaluate=job["workers"], timeout=job["timeout"],
        )
        rows = []
        for i, task_id in enumerate(job["task_ids"]):
            rows.append({"task_id": task_id, "tests": raw[i], "metadata": metadata[i],
                         "passed": [bool(r) and all(v is True or v == 1 for v in r) for r in raw[i]]})
        return {"metrics": metrics, "results": rows}

    from evalplus.evaluate import check_correctness, get_groundtruth
    from evalplus.eval import PASS, estimate_pass_at_k
    from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS

    problems = dict(zip(job["task_ids"], job["problems"]))
    expected = get_groundtruth(
        problems, "selected_tasks", MBPP_OUTPUT_NOT_NONE_TASKS if job["dataset"] == "mbpp" else [],
    )
    rows = []
    with ProcessPoolExecutor(max_workers=job["workers"]) as pool:
        futures = []
        for task_id, solutions in zip(job["task_ids"], job["solutions"]):
            for index, solution in enumerate(solutions):
                futures.append(pool.submit(
                    check_correctness, job["dataset"], index, problems[task_id], solution,
                    expected[task_id], base_only=job["base_only"], fast_check=True,
                    min_time_limit=job["timeout"],
                ))
        for future in futures:
            result = future.result()
            base = result["base"][0] == PASS
            plus = base and result.get("plus", (None,))[0] == PASS
            rows.append({"task_id": result["task_id"], "sample_index": result["completion_id"],
                         "base_passed": base, "plus_passed": plus if not job["base_only"] else None,
                         "base_status": result["base"][0],
                         "plus_status": result.get("plus", (None,))[0]})
    metrics = {}
    for suite in (["base"] if job["base_only"] else ["base", "plus"]):
        correct = [sum(r[f"{suite}_passed"] for r in rows if r["task_id"] == task_id)
                   for task_id in job["task_ids"]]
        totals = [len(s) for s in job["solutions"]]
        metrics[suite] = {f"pass@{k}": float(estimate_pass_at_k(totals, correct, k).mean())
                          for k in job["k"] if min(totals) >= k}
    return {"metrics": metrics, "results": rows}


if __name__ == "__main__":
    job = pickle.load(sys.stdin.buffer)  # Produced locally by run.py, never downloaded.
    result = score(job)
    Path("/output/scores.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
