"""Dispatch coding evaluation to an isolated dependency environment."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


def add_coding_args(parser):
    group = parser.add_argument_group("LiveCodeBench / EvalPlus")
    group.add_argument("--code-thinking", choices=("on", "off"), default=None,
                       help="Explicit chat-template thinking mode; omitted preserves server default.")
    group.add_argument("--code-top-k", type=int, default=None)
    group.add_argument("--code-min-p", type=float, default=None)
    group.add_argument("--code-presence-penalty", type=float, default=None)
    group.add_argument("--code-repetition-penalty", type=float, default=None)
    group.add_argument("--code-seed", type=int, default=None,
                       help="Request seed; sample index is added for multi-sample evaluation.")
    group.add_argument("--code-n-samples", type=int, default=1)
    group.add_argument("--code-pass-k", default="1,5,10")
    group.add_argument("--code-eval-workers", type=int, default=4)
    group.add_argument("--code-timeout", type=int, default=6,
                       help="LCB per-test timeout; EvalPlus minimum time limit (seconds).")
    group.add_argument("--code-eval-timeout", type=int, default=7200)
    group.add_argument("--code-samples", help="Score an existing task_id/solution JSONL; no model calls.")
    group.add_argument(
        "--code-retry-errors-from",
        help="Reuse successful rows from an existing samples JSONL and regenerate only missing/error rows.",
    )
    group.add_argument("--code-generate-only", action="store_true")
    group.add_argument("--code-docker-image", default="ubuntu:latest")
    group.add_argument("--code-memory", default="8g")
    group.add_argument("--lcb-release", choices=("v5", "v6"), default="v6",
                       help="Official cumulative release: v5 (880 tasks) or v6 (1055 tasks).")
    group.add_argument("--lcb-start-date", help="Inclusive contest date, YYYY-MM-DD.")
    group.add_argument("--lcb-end-date", help="Inclusive contest date, YYYY-MM-DD.")
    group.add_argument("--lcb-data-dir", help="Directory containing official test*.jsonl; skips download.")


def run_coding(args):
    root = Path(__file__).resolve().parents[1] / "coding"
    python = root / ".venv/bin/python"
    if not python.exists() or not (root / "official/lcb_runner/evaluation/testing_util.py").exists():
        raise SystemExit(f"Coding dependencies missing. Run: bash {root / 'setup.sh'}")
    config = vars(args).copy()
    # Keep credentials out of argv and persisted manifests.
    key = config.pop("api_key")
    env = dict(os.environ, CODE_EVAL_API_KEY=key)
    return subprocess.run(
        [str(python), str(root / "run.py")], input=json.dumps(config), text=True, env=env,
    ).returncode
