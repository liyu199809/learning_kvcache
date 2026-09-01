"""Sandbox launcher and answer extraction for TACO verification."""

from __future__ import annotations

import json
import os
import re
import resource
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

from .config import (
    DATA_DIR,
    MAX_CONCURRENT_JUDGES,
    MAX_OUTPUT_CHARS,
    MAX_PROCESSES,
    MEMORY_LIMIT_MB,
    PER_TEST_TIMEOUT,
    TOTAL_TIMEOUT,
)
from .judge_worker import RESULT_MARKER


_PYTHON_FENCE = re.compile(r"```(?:python|py)\s*\n(.*?)```", re.IGNORECASE | re.DOTALL)
_JUDGE_SEMAPHORE = threading.BoundedSemaphore(MAX_CONCURRENT_JUDGES)


def extract_python_code(model_response: str) -> str | None:
    """Return the final Python fenced block, matching the DeepCoder contract."""
    if not isinstance(model_response, str):
        return None
    matches = _PYTHON_FENCE.findall(model_response)
    if not matches:
        return None
    code = matches[-1].strip()
    return code or None


def _limit_worker() -> None:
    os.setsid()
    memory_bytes = MEMORY_LIMIT_MB * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    resource.setrlimit(resource.RLIMIT_DATA, (memory_bytes, memory_bytes))
    resource.setrlimit(resource.RLIMIT_CPU, (int(TOTAL_TIMEOUT) + 5, int(TOTAL_TIMEOUT) + 5))
    resource.setrlimit(resource.RLIMIT_NPROC, (MAX_PROCESSES, MAX_PROCESSES))
    resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024 * 1024, 16 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))


def _sandbox_command(worker_path: Path) -> list[str]:
    unshare = shutil.which("unshare")
    python_command = [sys.executable, "-I", "-B", str(worker_path)]
    if unshare is None:
        return python_command
    namespace_args = ["--net", "--mount", "--fork", "--pid", "--mount-proc"]
    if os.geteuid() != 0:
        namespace_args = ["--user", "--map-root-user", *namespace_args]
    return [unshare, *namespace_args, *python_command]


def _parse_worker_output(stdout: str) -> dict[str, Any] | None:
    marker_position = stdout.rfind(RESULT_MARKER)
    if marker_position < 0:
        return None
    encoded = stdout[marker_position + len(RESULT_MARKER) :].splitlines()[0]
    try:
        value = json.loads(encoded)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def judge_answer(final_answer: str, tests: dict[str, Any]) -> dict[str, Any]:
    """Extract and score candidate code, returning only aggregate test details."""
    code = extract_python_code(final_answer)
    if code is None:
        return {
            "ok": True,
            "reward": 0.0,
            "reward_type": "format_error",
            "verify_result": {
                "passed_tests": 0,
                "total_tests": len(tests.get("inputs", [])),
                "all_passed": False,
                "format_error": True,
            },
        }

    payload = {
        "code": code,
        "tests": tests,
        "per_test_timeout": PER_TEST_TIMEOUT,
        "total_timeout": TOTAL_TIMEOUT,
        "max_output_chars": MAX_OUTPUT_CHARS,
        "hide_paths": [str(DATA_DIR)],
    }
    worker_path = Path(__file__).with_name("judge_worker.py")
    with _JUDGE_SEMAPHORE, tempfile.TemporaryDirectory(prefix="openenv_taco_judge_") as work_dir:
        work_path = Path(work_dir)
        work_path.chmod(0o700)
        if os.geteuid() == 0:
            try:
                os.chown(work_path, 65534, 65534)
            except OSError:
                work_path.chmod(0o777)
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": work_dir,
            "TMPDIR": work_dir,
            "PYTHONHASHSEED": "0",
            "PYTHONIOENCODING": "utf-8",
            "OMP_NUM_THREADS": "1",
        }
        try:
            completed = subprocess.run(
                _sandbox_command(worker_path),
                input=json.dumps(payload, ensure_ascii=False),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=TOTAL_TIMEOUT + 10.0,
                cwd=work_dir,
                env=env,
                preexec_fn=_limit_worker,
            )
        except subprocess.TimeoutExpired:
            return {
                "ok": True,
                "reward": 0.0,
                "reward_type": "timeout",
                "verify_result": {
                    "passed_tests": 0,
                    "total_tests": len(tests.get("inputs", [])),
                    "all_passed": False,
                    "judge_timeout": True,
                },
            }
        except BaseException as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    worker_output = _parse_worker_output(completed.stdout[-MAX_OUTPUT_CHARS:])
    if not worker_output:
        return {
            "ok": False,
            "error": f"Judge worker exited {completed.returncode} without a valid result",
        }
    if not worker_output.get("ok"):
        return {
            "ok": False,
            "error": str(worker_output.get("error") or worker_output.get("kind") or "judge failed"),
        }
    result = worker_output.get("result")
    if not isinstance(result, dict):
        return {"ok": False, "error": "Judge worker returned invalid result data"}
    passed = int(result.get("passed_tests", 0))
    total = int(result.get("total_tests", 0))
    if total <= 0 or passed < 0 or passed > total:
        return {"ok": False, "error": f"Invalid judge counts: passed={passed}, total={total}"}
    reward = passed / total
    if bool(result.get("all_passed")) and passed == total:
        reward_type = "complete"
    elif bool(result.get("compile_error")):
        reward_type = "compile_error"
    elif int(result.get("timeouts", 0)) > 0:
        reward_type = "timeout"
    else:
        reward_type = "incomplete"
    return {
        "ok": True,
        "reward": reward,
        "reward_type": reward_type,
        "verify_result": result,
    }
