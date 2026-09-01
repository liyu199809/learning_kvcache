"""Runtime configuration for the CodeJudge service."""

from __future__ import annotations

import os
from pathlib import Path


DATASET_REVISION = "177913a7bd43791646ef6a43645caa3c871ab3db"


def _default_data_dir() -> Path:
    configured = os.environ.get("DEEPCODER_TACO_DATA_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    repo_root = Path(__file__).resolve().parents[4]
    return (repo_root / "data" / "deepcoder_taco" / DATASET_REVISION).resolve()


DATA_DIR = _default_data_dir()
MAX_CONCURRENT_ENVS = int(os.environ.get("OPENENV_CODE_JUDGE_MAX_CONCURRENT_ENVS", "128"))
MAX_CONCURRENT_JUDGES = int(os.environ.get("OPENENV_CODE_JUDGE_MAX_CONCURRENT_JUDGES", "32"))
PER_TEST_TIMEOUT = float(os.environ.get("OPENENV_CODE_JUDGE_PER_TEST_TIMEOUT", "3"))
TOTAL_TIMEOUT = float(os.environ.get("OPENENV_CODE_JUDGE_TOTAL_TIMEOUT", "90"))
MEMORY_LIMIT_MB = int(os.environ.get("OPENENV_CODE_JUDGE_MEMORY_MB", "1024"))
MAX_PROCESSES = int(os.environ.get("OPENENV_CODE_JUDGE_MAX_PROCESSES", "256"))
MAX_OUTPUT_CHARS = int(os.environ.get("OPENENV_CODE_JUDGE_MAX_OUTPUT_CHARS", "1000000"))
