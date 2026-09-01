"""Isolated per-episode execution for synthesized EnvScaler Python code."""

from __future__ import annotations

import dataclasses
import datetime as datetime_module
import json
import multiprocessing
import os
import signal
import threading
import traceback
import types
from copy import deepcopy
from pathlib import Path
from typing import Any

import jsonschema

from .config import (
    MAX_TOOL_RESULT_CHARS,
    MAX_VERIFY_DETAILS,
    TOOL_CALL_TIMEOUT,
    VERIFY_TIMEOUT,
    WORKER_MEMORY_LIMIT_MB,
    WORKER_START_TIMEOUT,
)


def _state_snapshot(instance: Any) -> dict[str, Any]:
    return deepcopy(
        {
            key: value
            for key, value in vars(instance).items()
            if not (key.startswith("__") and key.endswith("__"))
        }
    )


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (datetime_module.date, datetime_module.time)):
        return value.isoformat()
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    return str(value)


def _truncate_result(value: Any) -> Any:
    normalized = _jsonable(value)
    serialized = json.dumps(normalized, ensure_ascii=False)
    if len(serialized) <= MAX_TOOL_RESULT_CHARS:
        return normalized
    return {
        "truncated": True,
        "original_characters": len(serialized),
        "content": serialized[:MAX_TOOL_RESULT_CHARS],
    }


def _apply_resource_limits() -> None:
    try:
        import resource

        if WORKER_MEMORY_LIMIT_MB > 0:
            memory_bytes = WORKER_MEMORY_LIMIT_MB * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (ImportError, OSError, ValueError):
        pass


def _instantiate_environment(code: str, class_name: str, init_config: dict[str, Any]) -> Any:
    module = types.ModuleType("envscaler_dynamic_env")
    exec(code, module.__dict__)
    if not hasattr(module, class_name):
        raise ValueError(f"Class {class_name!r} was not defined by env_class_code")
    environment_class = getattr(module, class_name)
    config = deepcopy(init_config)
    try:
        instance = environment_class(config if config else {})
    except TypeError:
        instance = environment_class()
    for key, value in config.items():
        setattr(instance, key, deepcopy(value))
    return instance


def _verify(
    checklist: list[dict[str, Any]],
    initial_state: dict[str, Any],
    final_state: dict[str, Any],
) -> dict[str, Any]:
    passed = 0
    checker_errors = 0
    failed_indices: list[int] = []
    details: list[dict[str, Any]] = []
    for index, item in enumerate(checklist):
        code = item.get("check_func", "") if isinstance(item, dict) else ""
        globals_dict = {"__builtins__": __builtins__, "initial_state": deepcopy(initial_state)}
        error = None
        result = False
        try:
            exec(code, globals_dict)
            function = globals_dict.get("check_func")
            if not callable(function):
                raise ValueError("check_func was not defined")
            raw_result = function(deepcopy(final_state))
            if not isinstance(raw_result, bool):
                raise TypeError(f"check_func returned {type(raw_result).__name__}, expected bool")
            result = raw_result
        except Exception as exc:
            checker_errors += 1
            error = f"{type(exc).__name__}: {exc}"
        if result:
            passed += 1
        else:
            failed_indices.append(index)
        if error and len(details) < MAX_VERIFY_DETAILS:
            details.append({"index": index, "check_item": item.get("check_item"), "error": error})
    total = len(checklist)
    score = passed / total if total else 0.0
    return {
        "score": score,
        "passed_checks": passed,
        "total_checks": total,
        "checker_errors": checker_errors,
        "failed_check_indices": failed_indices,
        "error_details": details,
    }


def _worker_main(connection, payload: dict[str, Any], work_dir: str) -> None:
    _apply_resource_limits()
    os.chdir(work_dir)
    try:
        environment = _instantiate_environment(
            payload["env_class_code"],
            payload["env_class_name"],
            payload["init_config"],
        )
        initial_state = _state_snapshot(environment)
        schemas = {
            tool["function"]["name"]: tool["function"].get("parameters", {"type": "object"})
            for tool in payload["tools"]
        }
        checklist = payload["checklist_with_func"]
        connection.send({"ok": True, "type": "ready"})
    except BaseException as exc:
        connection.send(
            {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
        connection.close()
        return

    while True:
        try:
            request = connection.recv()
        except EOFError:
            break
        command = request.get("command")
        if command == "close":
            connection.send({"ok": True})
            break
        if command == "call_tool":
            tool_name = request.get("tool_name")
            arguments = request.get("arguments", {})
            try:
                if tool_name not in schemas:
                    raise ValueError(f"Unknown tool: {tool_name}")
                jsonschema.validate(instance=arguments, schema=schemas[tool_name])
                method = getattr(environment, tool_name, None)
                if not callable(method):
                    raise AttributeError(f"Environment has no callable tool {tool_name!r}")
                result = method(**arguments)
                connection.send({"ok": True, "result": _truncate_result(result)})
            except BaseException as exc:
                connection.send(
                    {
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    }
                )
            continue
        if command == "verify":
            try:
                result = _verify(checklist, initial_state, _state_snapshot(environment))
                connection.send({"ok": True, "result": result})
            except BaseException as exc:
                connection.send(
                    {
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    }
                )
            continue
        connection.send({"ok": False, "error": f"Unknown worker command: {command}"})
    connection.close()


class EpisodeWorker:
    """Own one subprocess and serialize all requests through one pipe."""

    def __init__(self) -> None:
        self._context = multiprocessing.get_context("spawn")
        self._process: multiprocessing.Process | None = None
        self._connection = None
        self._lock = threading.Lock()
        self._work_dir: Path | None = None

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.is_alive()

    def start(self, payload: dict[str, Any], work_dir: str | os.PathLike[str]) -> None:
        self.stop()
        parent_connection, child_connection = self._context.Pipe()
        self._work_dir = Path(work_dir)
        self._process = self._context.Process(
            target=_worker_main,
            args=(child_connection, payload, str(self._work_dir)),
            daemon=True,
        )
        self._connection = parent_connection
        self._process.start()
        child_connection.close()
        response = self._receive(WORKER_START_TIMEOUT, stage="worker startup")
        if not response.get("ok"):
            error = response.get("error", "unknown worker startup error")
            self.stop()
            raise RuntimeError(error)

    def _receive(self, timeout: float, *, stage: str) -> dict[str, Any]:
        if self._connection is None or self._process is None:
            raise RuntimeError("Episode worker is not running")
        if not self._connection.poll(timeout):
            self.stop()
            raise TimeoutError(f"EnvScaler {stage} timed out after {timeout:.1f}s")
        try:
            return self._connection.recv()
        except EOFError as exc:
            exit_code = self._process.exitcode
            self.stop()
            raise RuntimeError(f"EnvScaler worker exited unexpectedly (exit code {exit_code})") from exc

    def _request(self, request: dict[str, Any], timeout: float, *, stage: str) -> dict[str, Any]:
        with self._lock:
            if not self.is_running or self._connection is None:
                raise RuntimeError("Episode worker is not running")
            self._connection.send(request)
            return self._receive(timeout, stage=stage)

    def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        timeout: float | None = None,
    ) -> dict[str, Any]:
        return self._request(
            {"command": "call_tool", "tool_name": tool_name, "arguments": arguments},
            timeout or TOOL_CALL_TIMEOUT,
            stage=f"tool call {tool_name}",
        )

    def verify(self, timeout: float | None = None) -> dict[str, Any]:
        return self._request(
            {"command": "verify"},
            timeout or VERIFY_TIMEOUT,
            stage="verification",
        )

    def stop(self) -> None:
        process, self._process = self._process, None
        connection, self._connection = self._connection, None
        if process is None:
            if connection is not None:
                connection.close()
            return
        if process.is_alive() and connection is not None:
            try:
                connection.send({"command": "close"})
                if connection.poll(1.0):
                    connection.recv()
            except (BrokenPipeError, EOFError, OSError):
                pass
        process.join(timeout=1.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=2.0)
        if process.is_alive() and process.pid:
            try:
                os.kill(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.join(timeout=1.0)
        if connection is not None:
            connection.close()
