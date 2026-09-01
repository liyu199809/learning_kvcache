"""Indexed, lazy access to the pinned DeepCoder TACO Parquet shards."""

from __future__ import annotations

import bisect
import json
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

import pyarrow.parquet as pq

from .config import DATASET_REVISION


EXPECTED_ROWS = 7436
SHARD_NAMES = tuple(f"train-{index:05d}-of-00004.parquet" for index in range(4))


def parse_tests(raw_tests: Any, *, task_idx: int | None = None) -> dict[str, Any]:
    """Parse and validate one TACO inputs/outputs payload."""
    if isinstance(raw_tests, str):
        try:
            raw_tests = json.loads(raw_tests)
        except json.JSONDecodeError as exc:
            prefix = f"task_idx {task_idx}: " if task_idx is not None else ""
            raise ValueError(f"{prefix}invalid tests JSON: {exc}") from exc
    if not isinstance(raw_tests, dict):
        raise TypeError(f"tests must decode to an object, got {type(raw_tests).__name__}")
    inputs = raw_tests.get("inputs")
    outputs = raw_tests.get("outputs")
    if not isinstance(inputs, list) or not isinstance(outputs, list):
        raise TypeError("tests.inputs and tests.outputs must both be lists")
    if not inputs:
        raise ValueError("tests.inputs must not be empty")
    if len(inputs) != len(outputs):
        raise ValueError(f"tests inputs/outputs length mismatch: {len(inputs)} != {len(outputs)}")
    fn_name = raw_tests.get("fn_name")
    if fn_name is not None and (not isinstance(fn_name, str) or not fn_name.strip()):
        raise TypeError("tests.fn_name must be a non-empty string when present")
    result = {"inputs": inputs, "outputs": outputs}
    if fn_name is not None:
        result["fn_name"] = fn_name
    return result


class TacoDataLoader:
    """Thread-safe random row access without loading the 1.6 GB table in memory."""

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir).expanduser().resolve()
        self._lock = threading.Lock()
        self._loaded = False
        self._shards: list[Path] = []
        self._parquet_files: list[pq.ParquetFile] = []
        self._row_group_ends: list[int] = []
        self._row_group_locations: list[tuple[int, int, int]] = []
        self._total_rows = 0

    def _load(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            shards = [self.data_dir / name for name in SHARD_NAMES]
            missing = [str(path) for path in shards if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"Missing DeepCoder TACO shards: {missing}")

            parquet_files: list[pq.ParquetFile] = []
            row_group_ends: list[int] = []
            row_group_locations: list[tuple[int, int, int]] = []
            total_rows = 0
            for shard_idx, path in enumerate(shards):
                parquet_file = pq.ParquetFile(path)
                names = set(parquet_file.schema_arrow.names)
                if not {"problem", "tests", "solutions"}.issubset(names):
                    raise ValueError(f"Unexpected TACO schema in {path}: {sorted(names)}")
                parquet_files.append(parquet_file)
                shard_row = 0
                for row_group_idx in range(parquet_file.num_row_groups):
                    num_rows = parquet_file.metadata.row_group(row_group_idx).num_rows
                    total_rows += num_rows
                    row_group_ends.append(total_rows)
                    row_group_locations.append((shard_idx, row_group_idx, shard_row))
                    shard_row += num_rows
            if total_rows != EXPECTED_ROWS:
                raise ValueError(
                    f"Pinned TACO revision should contain {EXPECTED_ROWS} rows, found {total_rows}"
                )

            self._shards = shards
            self._parquet_files = parquet_files
            self._row_group_ends = row_group_ends
            self._row_group_locations = row_group_locations
            self._total_rows = total_rows
            self._loaded = True

    @property
    def total_rows(self) -> int:
        self._load()
        return self._total_rows

    @lru_cache(maxsize=512)
    def get_task(self, task_idx: int, *, include_solutions: bool = False) -> dict[str, Any]:
        self._load()
        if task_idx < 0 or task_idx >= self._total_rows:
            raise ValueError(f"task_idx {task_idx} out of range (0..{self._total_rows - 1})")
        group_position = bisect.bisect_right(self._row_group_ends, task_idx)
        previous_end = self._row_group_ends[group_position - 1] if group_position else 0
        offset = task_idx - previous_end
        shard_idx, row_group_idx, _ = self._row_group_locations[group_position]
        columns = ["problem", "tests"]
        if include_solutions:
            columns.append("solutions")
        table = self._parquet_files[shard_idx].read_row_group(row_group_idx, columns=columns)
        row = table.slice(offset, 1).to_pylist()[0]
        problem = row.get("problem")
        if not isinstance(problem, str) or not problem.strip():
            raise ValueError(f"task_idx {task_idx} has an empty problem")
        result = {
            "task_id": f"taco_{task_idx}",
            "task_idx": task_idx,
            "problem": problem,
            "tests": parse_tests(row.get("tests"), task_idx=task_idx),
        }
        if include_solutions:
            solutions = row.get("solutions")
            if not isinstance(solutions, list):
                raise TypeError(f"task_idx {task_idx} solutions must be a list")
            result["solutions"] = solutions
        return result

    def iter_rows(self, *, batch_size: int = 32) -> Iterator[tuple[int, dict[str, Any]]]:
        """Yield raw rows in the exact order used by ``task_idx``."""
        self._load()
        task_idx = 0
        for parquet_file in self._parquet_files:
            for batch in parquet_file.iter_batches(
                batch_size=batch_size,
                columns=["problem", "tests"],
            ):
                for row in batch.to_pylist():
                    yield task_idx, row
                    task_idx += 1

    def stats(self) -> dict[str, Any]:
        self._load()
        return {
            "scenario_count": 1,
            "task_count": self._total_rows,
            "dataset_revision": DATASET_REVISION,
            "data_dir": str(self.data_dir),
            "shards": [str(path) for path in self._shards],
        }
