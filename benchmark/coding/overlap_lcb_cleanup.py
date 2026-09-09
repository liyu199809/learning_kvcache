"""One-shot continuation: overlap completed MBPP inference's CPU cleanup with LCB.

Pause only the suite coordinator, not its CPU preprocessing child or DP server.
After LCB generation and v5 scoring, the coordinator reuses those artifacts and
performs the remaining MBPP/v6 scoring and final aggregation normally.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
from types import SimpleNamespace
import urllib.request

from run_cleaned_taco_suite import Suite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--coordinator-pid", type=int, required=True)
    args = parser.parse_args()
    root = Path("/disk3/self_evolver")
    output = Path(args.suite).resolve()
    cmdline = Path(f"/proc/{args.coordinator_pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    assert "run_cleaned_taco_suite.py" in cmdline and str(output) in cmdline
    assert "--models off_on" in cmdline
    log = output / "server_off_on_resume_1.log"
    completed = sum("POST /v1/chat/completions" in line and "200 OK" in line
                    for line in log.read_text().splitlines())
    assert completed == 32, f"Expected all 32 retry requests finished, got {completed}"
    with urllib.request.urlopen("http://127.0.0.1:9400/v1/models", timeout=10) as response:
        assert json.load(response)["data"][0]["id"] == "off_on"
    parent = output / "off_on/thinking_off"
    assert not (parent / "lcb_v5_generation").exists()
    suite = Suite(SimpleNamespace(root=str(root), output_root=str(output), resume=True,
                                 port=9400, thinking="off", models=["off_on"]))
    os.kill(args.coordinator_pid, signal.SIGSTOP)
    suite.event("cpu_cleanup_overlap_start", coordinator_pid=args.coordinator_pid)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            v5_args = ["--benchmark", "livecodebench", "--lcb-release", "v5"]
            v5 = suite.generate("off_on", "off", "lcb_v5", v5_args, parent, 880)
            scoring = pool.submit(suite.score, "off_on", "off", "lcb_v5", v5_args,
                                  v5 / "samples.jsonl", parent)
            suite.generate("off_on", "off", "lcb_v6_new",
                           ["--benchmark", "livecodebench", "--lcb-release", "v6", "--tasks", "880-1054"],
                           parent, 175)
            scoring.result()
        suite.event("cpu_cleanup_overlap_complete")
    finally:
        os.kill(args.coordinator_pid, signal.SIGCONT)
        suite.event("coordinator_resumed", coordinator_pid=args.coordinator_pid)


if __name__ == "__main__":
    main()
