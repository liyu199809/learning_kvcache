"""Isolated TACO test worker.

The test-shape handling follows the public TACO/APPS and rLLM evaluators, but
candidate programs are always executed in a child process. Hidden expected
outputs remain in this parent worker and are never written beside candidate
code.
"""

from __future__ import annotations

import argparse
import ast
import builtins
import contextlib
import ctypes
import json
import math
import os
import resource
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


RESULT_MARKER = "__OPENENV_TACO_RESULT__"
CANDIDATE_PYTHON = "/usr/bin/python3" if Path("/usr/bin/python3").is_file() else sys.executable
FUNCTIONAL_WORKER_PATH = Path(__file__).resolve()
COMMON_PRELUDE = """\
from typing import *
from collections import Counter, OrderedDict, defaultdict, deque
from functools import cache, lru_cache, reduce
from itertools import accumulate, chain, combinations, permutations, product
from bisect import bisect, bisect_left, bisect_right, insort
from heapq import *
from math import *
import collections, functools, itertools, math, random, re
"""


class CandidateTimeout(Exception):
    """Raised when candidate import or execution exceeds its alarm."""


def _alarm_handler(_signum: int, _frame: Any) -> None:
    raise CandidateTimeout("candidate timed out")


def _set_not_dumpable() -> None:
    """Prevent same-UID candidate children from reading this process memory."""
    try:
        libc = ctypes.CDLL(None)
        libc.prctl(4, 0, 0, 0, 0)  # PR_SET_DUMPABLE
    except Exception:
        pass


def _hide_paths(paths: list[str]) -> None:
    """Cover dataset directories with empty tmpfs mounts in this namespace."""
    if os.geteuid() != 0:
        return
    mount = shutil.which("mount")
    if mount is None:
        return
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_dir():
            continue
        subprocess.run(
            [mount, "-t", "tmpfs", "-o", "size=1m,mode=000", "tmpfs", str(path)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def _drop_privileges() -> None:
    """Use the conventional nobody identity when the namespace maps it."""
    if os.geteuid() != 0:
        return
    try:
        os.setgroups([])
        os.setgid(65534)
        os.setuid(65534)
    except OSError:
        # A root-mapped unprivileged user namespace normally maps only uid 0.
        # It is still isolated from host-root permissions in that case.
        pass


def _reliability_guard() -> None:
    """Disable common destructive APIs in functional-call candidate children."""
    builtins.exit = None
    builtins.quit = None
    os.environ["OMP_NUM_THREADS"] = "1"
    for name in (
        "kill",
        "killpg",
        "system",
        "putenv",
        "remove",
        "removedirs",
        "rmdir",
        "fork",
        "forkpty",
        "rename",
        "renames",
        "truncate",
        "replace",
        "unlink",
        "chmod",
        "chown",
        "chroot",
    ):
        if hasattr(os, name):
            setattr(os, name, None)
    for name in ("rmtree", "move", "chown"):
        if hasattr(shutil, name):
            setattr(shutil, name, None)


def _safe_json_value(value: Any, *, depth: int = 0) -> Any:
    if depth > 30:
        return {"__repr__": "maximum nesting exceeded"}
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, tuple | list):
        return [_safe_json_value(item, depth=depth + 1) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _safe_json_value(item, depth=depth + 1)
            for key, item in value.items()
        }
    if isinstance(value, set | frozenset):
        items = [_safe_json_value(item, depth=depth + 1) for item in value]
        return {"__set__": sorted(items, key=repr)}
    return {"__repr__": repr(value)[:1000]}


def _emit_result(result: dict[str, Any]) -> None:
    data = RESULT_MARKER + json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n"
    os.write(1, data.encode("utf-8", errors="replace"))


def _functional_child(timeout: float) -> int:
    """Execute one function-call test without receiving its expected output."""
    try:
        payload = json.load(sys.stdin)
        code = payload["code"]
        fn_name = payload["fn_name"]
        arguments = payload["arguments"]
        if not isinstance(code, str) or not isinstance(fn_name, str):
            raise TypeError("invalid functional child payload")

        _set_not_dumpable()
        _reliability_guard()
        signal.signal(signal.SIGALRM, _alarm_handler)
        signal.setitimer(signal.ITIMER_REAL, timeout)
        namespace: dict[str, Any] = {"__name__": "candidate_solution"}
        with open(os.devnull, "w", encoding="utf-8") as discarded:
            with contextlib.redirect_stdout(discarded), contextlib.redirect_stderr(discarded):
                exec(compile(COMMON_PRELUDE + "\n" + code, "<candidate>", "exec"), namespace)
                target: Any = namespace
                solution_class = namespace.get("Solution")
                if isinstance(solution_class, type):
                    target = solution_class()
                function = getattr(target, fn_name) if not isinstance(target, dict) else target[fn_name]
                if isinstance(arguments, list):
                    value = function(*arguments)
                else:
                    value = function(arguments)
        signal.setitimer(signal.ITIMER_REAL, 0)
        _emit_result({"ok": True, "value": _safe_json_value(value)})
        return 0
    except CandidateTimeout:
        _emit_result({"ok": False, "kind": "timeout"})
        return 124
    except SyntaxError as exc:
        _emit_result({"ok": False, "kind": "compile_error", "error": str(exc)[:500]})
        return 2
    except BaseException as exc:
        _emit_result(
            {
                "ok": False,
                "kind": "runtime_error",
                "error": f"{type(exc).__name__}: {exc}"[:500],
            }
        )
        return 1


def _parse_marked_result(stdout: str) -> dict[str, Any] | None:
    marker_position = stdout.rfind(RESULT_MARKER)
    if marker_position < 0:
        return None
    encoded = stdout[marker_position + len(RESULT_MARKER) :].splitlines()[0]
    try:
        value = json.loads(encoded)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _numeric_equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if not isinstance(left, int | float) or not isinstance(right, int | float):
        return False
    return math.isclose(float(left), float(right), rel_tol=1e-5, abs_tol=1e-6)


def _values_equal(actual: Any, expected: Any) -> bool:
    actual = _safe_json_value(actual)
    expected = _safe_json_value(expected)
    if actual == expected or _numeric_equal(actual, expected):
        return True
    if isinstance(expected, list) and len(expected) == 1:
        if _values_equal(actual, expected[0]):
            return True
    if isinstance(actual, list) and isinstance(expected, list) and len(actual) == len(expected):
        return all(_values_equal(left, right) for left, right in zip(actual, expected, strict=True))
    if isinstance(actual, dict) and isinstance(expected, dict) and actual.keys() == expected.keys():
        return all(_values_equal(actual[key], expected[key]) for key in actual)
    return False


def _text_equal(actual: str, expected: Any) -> bool:
    if isinstance(expected, list):
        expected = "\n".join(str(item) for item in expected)
    else:
        expected = str(expected)
    actual_lines = [line.strip() for line in actual.strip().splitlines() if line.strip()]
    expected_lines = [line.strip() for line in expected.strip().splitlines() if line.strip()]
    if actual_lines == expected_lines:
        return True
    actual_tokens = " ".join(actual_lines).split()
    expected_tokens = " ".join(expected_lines).split()
    if len(actual_tokens) != len(expected_tokens):
        return False
    for actual_token, expected_token in zip(actual_tokens, expected_tokens, strict=True):
        if actual_token == expected_token:
            continue
        try:
            if math.isclose(
                float(actual_token),
                float(expected_token),
                rel_tol=1e-5,
                abs_tol=1e-6,
            ):
                continue
        except (TypeError, ValueError, OverflowError):
            pass
        return False
    return True


def _base_result(total_tests: int) -> dict[str, Any]:
    return {
        "passed_tests": 0,
        "total_tests": total_tests,
        "wrong_answers": 0,
        "timeouts": 0,
        "runtime_errors": 0,
        "not_run": 0,
        "compile_error": False,
    }


def _remaining_timeout(deadline: float, per_test_timeout: float) -> float:
    return max(0.0, min(per_test_timeout, deadline - time.monotonic()))


def _judge_standard_input(
    code: str,
    tests: dict[str, Any],
    *,
    per_test_timeout: float,
    total_timeout: float,
    max_output_chars: int,
) -> dict[str, Any]:
    inputs = tests["inputs"]
    outputs = tests["outputs"]
    result = _base_result(len(inputs))
    try:
        compile(code, "<candidate>", "exec")
    except SyntaxError:
        result["compile_error"] = True
        result["not_run"] = len(inputs)
        return result

    solution_path = Path("solution.py")
    solution_path.write_text(code, encoding="utf-8")
    deadline = time.monotonic() + total_timeout
    for index, (test_input, expected) in enumerate(zip(inputs, outputs, strict=True)):
        timeout = _remaining_timeout(deadline, per_test_timeout)
        if timeout <= 0:
            result["not_run"] += len(inputs) - index
            break
        if isinstance(test_input, list):
            test_input = "\n".join(str(item) for item in test_input)
        if not isinstance(test_input, str):
            test_input = str(test_input)
        output_path = Path("candidate.stdout")
        try:
            with output_path.open("w+", encoding="utf-8", errors="replace") as candidate_stdout:
                completed = subprocess.run(
                [CANDIDATE_PYTHON, "-I", "-B", str(solution_path)],
                    input=test_input,
                    text=True,
                    stdout=candidate_stdout,
                    stderr=subprocess.DEVNULL,
                    timeout=timeout,
                    env={
                        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                        "HOME": os.getcwd(),
                        "TMPDIR": os.getcwd(),
                        "PYTHONHASHSEED": "0",
                        "PYTHONIOENCODING": "utf-8",
                        "OMP_NUM_THREADS": "1",
                    },
                    start_new_session=True,
                )
                candidate_stdout.seek(0)
                candidate_output = candidate_stdout.read(max_output_chars + 1)
        except subprocess.TimeoutExpired:
            result["timeouts"] += 1
            continue
        except BaseException:
            result["runtime_errors"] += 1
            continue
        if completed.returncode != 0 or len(candidate_output) > max_output_chars:
            result["runtime_errors"] += 1
        elif _text_equal(candidate_output, expected):
            result["passed_tests"] += 1
        else:
            result["wrong_answers"] += 1
    return result


def _judge_functional(
    code: str,
    tests: dict[str, Any],
    *,
    per_test_timeout: float,
    total_timeout: float,
    max_output_chars: int,
) -> dict[str, Any]:
    inputs = tests["inputs"]
    outputs = tests["outputs"]
    fn_name = tests["fn_name"]
    result = _base_result(len(inputs))
    try:
        compile(COMMON_PRELUDE + "\n" + code, "<candidate>", "exec")
    except SyntaxError:
        result["compile_error"] = True
        result["not_run"] = len(inputs)
        return result

    deadline = time.monotonic() + total_timeout
    worker_path = FUNCTIONAL_WORKER_PATH
    for index, (arguments, expected) in enumerate(zip(inputs, outputs, strict=True)):
        timeout = _remaining_timeout(deadline, per_test_timeout)
        if timeout <= 0:
            result["not_run"] += len(inputs) - index
            break
        payload = json.dumps(
            {"code": code, "fn_name": fn_name, "arguments": arguments},
            ensure_ascii=False,
        )
        try:
            completed = subprocess.run(
                [CANDIDATE_PYTHON, "-I", "-B", str(worker_path), "--functional-child", "--timeout", str(timeout)],
                input=payload,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout + 1.0,
                env={
                    "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                    "HOME": os.getcwd(),
                    "TMPDIR": os.getcwd(),
                    "PYTHONHASHSEED": "0",
                    "PYTHONIOENCODING": "utf-8",
                    "OMP_NUM_THREADS": "1",
                },
                start_new_session=True,
            )
        except subprocess.TimeoutExpired:
            result["timeouts"] += 1
            continue
        except BaseException:
            result["runtime_errors"] += 1
            continue
        child_result = _parse_marked_result(completed.stdout[-max_output_chars:])
        if not child_result or not child_result.get("ok"):
            kind = child_result.get("kind") if child_result else "runtime_error"
            if kind == "timeout":
                result["timeouts"] += 1
            elif kind == "compile_error":
                result["compile_error"] = True
                result["not_run"] += len(inputs) - index
                break
            else:
                result["runtime_errors"] += 1
        elif _values_equal(child_result.get("value"), expected):
            result["passed_tests"] += 1
        else:
            result["wrong_answers"] += 1
    return result


def judge(payload: dict[str, Any]) -> dict[str, Any]:
    code = payload.get("code")
    tests = payload.get("tests")
    if not isinstance(code, str) or not isinstance(tests, dict):
        raise TypeError("judge payload must contain string code and object tests")
    inputs = tests.get("inputs")
    outputs = tests.get("outputs")
    if not isinstance(inputs, list) or not isinstance(outputs, list) or len(inputs) != len(outputs):
        raise ValueError("invalid TACO tests")
    per_test_timeout = max(0.1, float(payload.get("per_test_timeout", 3.0)))
    total_timeout = max(per_test_timeout, float(payload.get("total_timeout", 90.0)))
    max_output_chars = max(1000, int(payload.get("max_output_chars", 1_000_000)))
    started = time.monotonic()
    if tests.get("fn_name"):
        result = _judge_functional(
            code,
            tests,
            per_test_timeout=per_test_timeout,
            total_timeout=total_timeout,
            max_output_chars=max_output_chars,
        )
        result["test_type"] = "functional"
    else:
        result = _judge_standard_input(
            code,
            tests,
            per_test_timeout=per_test_timeout,
            total_timeout=total_timeout,
            max_output_chars=max_output_chars,
        )
        result["test_type"] = "stdin"
    result["all_passed"] = result["passed_tests"] == result["total_tests"]
    result["duration_seconds"] = round(time.monotonic() - started, 4)
    return result


def main() -> int:
    global FUNCTIONAL_WORKER_PATH
    parser = argparse.ArgumentParser()
    parser.add_argument("--functional-child", action="store_true")
    parser.add_argument("--timeout", type=float, default=3.0)
    args = parser.parse_args()
    if args.functional_child:
        return _functional_child(args.timeout)

    try:
        payload = json.load(sys.stdin)
        hide_paths = payload.pop("hide_paths", [])
        if not isinstance(hide_paths, list):
            hide_paths = []
        _set_not_dumpable()
        _hide_paths([str(path) for path in hide_paths])
        copied_worker = Path(os.environ.get("HOME", os.getcwd())) / "judge_worker.py"
        shutil.copyfile(Path(__file__).resolve(), copied_worker)
        copied_worker.chmod(0o755)
        if os.geteuid() == 0:
            try:
                os.chown(copied_worker, 65534, 65534)
            except OSError:
                pass
        FUNCTIONAL_WORKER_PATH = copied_worker
        _drop_privileges()
        os.chdir(os.environ.get("HOME", os.getcwd()))
        result = judge(payload)
        _emit_result({"ok": True, "result": result})
        return 0
    except BaseException as exc:
        _emit_result(
            {
                "ok": False,
                "kind": "worker_error",
                "error": f"{type(exc).__name__}: {exc}"[:1000],
            }
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
