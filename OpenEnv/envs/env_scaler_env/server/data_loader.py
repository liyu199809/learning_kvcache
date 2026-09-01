"""Load and validate the official EnvScaler environment and RL task data."""

from __future__ import annotations

import json
import os
import re
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download


ENV_REPO_ID = "XXHStudyHard/EnvScaler-191-Env"
TASK_REPO_ID = "XXHStudyHard/EnvScaler-RL-Scenario"
ENV_REMOTE_FILENAME = "191_env_metadata_processed.json"
TASK_REMOTE_FILENAME = "envscaler_rl_scenario_metadata.json"
ENV_LOCAL_FILENAMES = (ENV_REMOTE_FILENAME, "191_env_metadata.json")
TASK_LOCAL_FILENAMES = (TASK_REMOTE_FILENAME, "rl_scenario_metadata.json")


def _default_cache_dir() -> Path:
    configured = os.environ.get("ENVSCALER_DATA_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path("~/.cache/openenv/envscaler").expanduser().resolve()


def _decode_json_field(value: Any, *, field: str, expected_type: type) -> Any:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {field}: {exc}") from exc
    if not isinstance(value, expected_type):
        raise TypeError(f"{field} must be {expected_type.__name__}, got {type(value).__name__}")
    return value


def _task_sort_key(task: dict[str, Any]) -> tuple[int, str]:
    task_id = str(task.get("task_id", ""))
    match = re.search(r"-task_(\d+)$", task_id)
    return (int(match.group(1)) if match else 2**31 - 1, task_id)


class EnvScalerDataLoader:
    """Thread-safe lazy accessor for the two official EnvScaler datasets."""

    def __init__(
        self,
        cache_dir: str | os.PathLike[str] | None = None,
        *,
        env_revision: str | None = None,
        task_revision: str | None = None,
        download_if_missing: bool = True,
    ) -> None:
        self.cache_dir = Path(cache_dir).expanduser().resolve() if cache_dir else _default_cache_dir()
        self.env_revision = env_revision or os.environ.get("ENVSCALER_ENV_REVISION", "main")
        self.task_revision = task_revision or os.environ.get("ENVSCALER_TASK_REVISION", "main")
        self.download_if_missing = download_if_missing
        self._lock = threading.Lock()
        self._loaded = False
        self._envs: dict[str, dict[str, Any]] = {}
        self._tasks_by_env: dict[str, list[dict[str, Any]]] = {}
        self._env_path: Path | None = None
        self._task_path: Path | None = None

    def _find_local(self, names: tuple[str, ...]) -> Path | None:
        for name in names:
            candidate = self.cache_dir / name
            if candidate.is_file():
                return candidate
        return None

    def _resolve_files(self) -> tuple[Path, Path]:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        env_path = self._find_local(ENV_LOCAL_FILENAMES)
        task_path = self._find_local(TASK_LOCAL_FILENAMES)
        if (env_path is None or task_path is None) and not self.download_if_missing:
            raise FileNotFoundError(
                f"EnvScaler data is incomplete in {self.cache_dir}; expected one of "
                f"{ENV_LOCAL_FILENAMES} and one of {TASK_LOCAL_FILENAMES}"
            )
        if env_path is None:
            env_path = Path(
                hf_hub_download(
                    repo_id=ENV_REPO_ID,
                    repo_type="dataset",
                    filename=ENV_REMOTE_FILENAME,
                    revision=self.env_revision,
                    local_dir=self.cache_dir,
                )
            )
        if task_path is None:
            task_path = Path(
                hf_hub_download(
                    repo_id=TASK_REPO_ID,
                    repo_type="dataset",
                    filename=TASK_REMOTE_FILENAME,
                    revision=self.task_revision,
                    local_dir=self.cache_dir,
                )
            )
        return env_path, task_path

    def _load(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            env_path, task_path = self._resolve_files()
            with env_path.open(encoding="utf-8") as source:
                raw_envs = json.load(source)
            with task_path.open(encoding="utf-8") as source:
                raw_tasks = json.load(source)

            env_values = list(raw_envs.values()) if isinstance(raw_envs, dict) else raw_envs
            if not isinstance(env_values, list):
                raise TypeError("Environment dataset must be a list or env_id mapping")
            envs: dict[str, dict[str, Any]] = {}
            for raw_env in env_values:
                if not isinstance(raw_env, dict):
                    raise TypeError("Every environment row must be a mapping")
                item = dict(raw_env)
                env_id = str(item.get("env_id", ""))
                if not env_id:
                    raise ValueError("Environment row is missing env_id")
                item["tools"] = _decode_json_field(item.get("tools"), field=f"{env_id}.tools", expected_type=list)
                if env_id in envs:
                    raise ValueError(f"Duplicate env_id: {env_id}")
                envs[env_id] = item

            if not isinstance(raw_tasks, list):
                raise TypeError("RL scenario dataset must be a list")
            tasks_by_env: dict[str, list[dict[str, Any]]] = defaultdict(list)
            seen_task_ids: set[str] = set()
            for raw_task in raw_tasks:
                if not isinstance(raw_task, dict):
                    raise TypeError("Every task row must be a mapping")
                task = dict(raw_task)
                env_id = str(task.get("env_id", ""))
                task_id = str(task.get("task_id", ""))
                if env_id not in envs:
                    raise ValueError(f"Task {task_id!r} references missing environment {env_id!r}")
                if not task_id or task_id in seen_task_ids:
                    raise ValueError(f"Missing or duplicate task_id: {task_id!r}")
                seen_task_ids.add(task_id)
                task["init_config"] = _decode_json_field(
                    task.get("init_config"), field=f"{task_id}.init_config", expected_type=dict
                )
                task["checklist_with_func"] = _decode_json_field(
                    task.get("checklist_with_func"),
                    field=f"{task_id}.checklist_with_func",
                    expected_type=list,
                )
                if task.get("env_class_name") != envs[env_id].get("env_class_name"):
                    raise ValueError(f"Class mismatch for task {task_id}")
                tasks_by_env[env_id].append(task)

            for env_id, tasks in tasks_by_env.items():
                tasks.sort(key=_task_sort_key)
                tool_names = []
                for tool in envs[env_id]["tools"]:
                    function = tool.get("function", {}) if isinstance(tool, dict) else {}
                    name = function.get("name")
                    parameters = function.get("parameters", {})
                    if tool.get("type") != "function" or not name or parameters.get("type") != "object":
                        raise ValueError(f"Invalid OpenAI tool schema in {env_id}: {tool!r}")
                    tool_names.append(name)
                if len(tool_names) != len(set(tool_names)):
                    raise ValueError(f"Duplicate tool names in {env_id}")

            self._envs = envs
            self._tasks_by_env = dict(tasks_by_env)
            self._env_path = env_path
            self._task_path = task_path
            self._loaded = True

    def get_environment(self, env_id: str) -> dict[str, Any]:
        self._load()
        try:
            return self._envs[env_id]
        except KeyError as exc:
            raise ValueError(f"Unknown EnvScaler environment: {env_id}") from exc

    def get_task(self, env_id: str, task_idx: int) -> dict[str, Any]:
        self._load()
        tasks = self._tasks_by_env.get(env_id, [])
        if task_idx < 0 or task_idx >= len(tasks):
            raise ValueError(f"task_idx {task_idx} out of range for {env_id} (0..{len(tasks) - 1})")
        return tasks[task_idx]

    def list_scenarios(self) -> list[dict[str, Any]]:
        self._load()
        return [
            {
                "name": env_id,
                "description": self._envs[env_id].get("environment_summary", ""),
                "num_tasks": len(tasks),
            }
            for env_id, tasks in sorted(self._tasks_by_env.items())
        ]

    def stats(self) -> dict[str, Any]:
        self._load()
        return {
            "env_count": len(self._tasks_by_env),
            "task_count": sum(len(tasks) for tasks in self._tasks_by_env.values()),
            "metadata_env_count": len(self._envs),
            "env_data_path": str(self._env_path),
            "task_data_path": str(self._task_path),
            "env_revision": self.env_revision,
            "task_revision": self.task_revision,
        }
