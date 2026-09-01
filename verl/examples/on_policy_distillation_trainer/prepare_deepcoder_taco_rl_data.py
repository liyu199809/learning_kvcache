#!/usr/bin/env python3
"""Convert the pinned DeepCoder TACO subset to the shared verl row schema."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[3]
OPENENV_ENVS = REPO_ROOT / "OpenEnv" / "envs"
if str(OPENENV_ENVS) not in sys.path:
    sys.path.insert(0, str(OPENENV_ENVS))

from code_judge_env.server.config import DATASET_REVISION  # noqa: E402
from code_judge_env.server.data_loader import TacoDataLoader, parse_tests  # noqa: E402


SYSTEM_PROMPT = """You are a competitive programmer. Solve the problem in Python.
Reason carefully, then put the final complete program in the LAST fenced Python block:
```python
# complete solution
```
Only that final Python block is executed against hidden tests. Do not include tests,
explanations, or additional code blocks after it."""


def _parse_indices(value: str | None) -> set[int] | None:
    if not value:
        return None
    result: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        index = int(part)
        if index < 0:
            raise ValueError("task indices must be non-negative")
        result.add(index)
    if not result:
        raise ValueError("--task-indices did not contain any indices")
    return result


def build_row(
    *,
    task_idx: int,
    problem: str,
    tests: dict[str, Any],
    base_url: str,
) -> dict[str, Any]:
    test_count = len(tests["inputs"])
    return {
        "data_source": "deepcoder_taco",
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem},
        ],
        "agent_name": "awm_agent",
        "tools": "[]",
        "env_config": {
            "scenario": "taco",
            "task_idx": task_idx,
            "awm_base_url": base_url.rstrip("/"),
        },
        "scenario": "taco",
        "task_idx": task_idx,
        "reward_model": {"style": "rule", "ground_truth": ""},
        "extra_info": {
            "problem": problem,
            "task_id": f"taco_{task_idx}",
            "scenario": "taco",
            "task_idx": task_idx,
            "source_row": task_idx,
            "test_count": test_count,
            "has_func_name": bool(tests.get("fn_name")),
            "dataset_revision": DATASET_REVISION,
        },
    }


def _prompt_tokens(row: dict[str, Any], tokenizer: Any) -> int:
    encoded = tokenizer.apply_chat_template(
        row["prompt"],
        tools=[],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=False,
    )
    return len(encoded)


def main() -> None:
    default_data_dir = REPO_ROOT / "data" / "deepcoder_taco" / DATASET_REVISION
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=default_data_dir)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "traj_data" / "deepcoder_taco")
    parser.add_argument("--base-url", default="http://127.0.0.1:8901")
    parser.add_argument("--task-indices", help="Comma-separated source row indices for a smoke subset")
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--val-size", type=int, default=128)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--max-prompt-tokens", type=int, default=8192)
    args = parser.parse_args()
    if args.max_rows is not None and args.max_rows < 1:
        parser.error("--max-rows must be positive")
    if args.val_size < 1:
        parser.error("--val-size must be positive")
    if args.repeat < 1:
        parser.error("--repeat must be positive")

    selected_indices = _parse_indices(args.task_indices)
    loader = TacoDataLoader(args.data_dir)
    rows: list[dict[str, Any]] = []
    test_counts: list[int] = []
    functional_count = 0
    stdin_count = 0
    for task_idx, raw_row in loader.iter_rows():
        if selected_indices is not None and task_idx not in selected_indices:
            continue
        problem = raw_row.get("problem")
        if not isinstance(problem, str) or not problem.strip():
            raise ValueError(f"task_idx {task_idx} has an empty problem")
        tests = parse_tests(raw_row.get("tests"), task_idx=task_idx)
        row = build_row(
            task_idx=task_idx,
            problem=problem,
            tests=tests,
            base_url=args.base_url,
        )
        rows.append(row)
        test_counts.append(len(tests["inputs"]))
        if tests.get("fn_name"):
            functional_count += 1
        else:
            stdin_count += 1
        if args.max_rows is not None and len(rows) >= args.max_rows:
            break
    if selected_indices is not None:
        found_indices = {int(row["task_idx"]) for row in rows}
        missing = sorted(selected_indices - found_indices)
        if missing:
            raise ValueError(f"Requested task indices were not found: {missing}")
    if not rows:
        raise ValueError("No TACO rows selected")

    token_counts: list[int] = []
    if args.model:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        for row in rows:
            token_count = _prompt_tokens(row, tokenizer)
            token_counts.append(token_count)
            if token_count > args.max_prompt_tokens:
                raise ValueError(
                    f"task_idx {row['task_idx']} prompt has {token_count} tokens, "
                    f"exceeding {args.max_prompt_tokens}"
                )

    if len(rows) == 1:
        train_rows = rows * args.repeat
        val_rows = [rows[0]]
    else:
        indices = list(range(len(rows)))
        random.Random(args.seed).shuffle(indices)
        val_size = min(args.val_size, len(rows) - 1)
        val_indices = set(indices[:val_size])
        val_rows = [row for index, row in enumerate(rows) if index in val_indices]
        base_train_rows = [row for index, row in enumerate(rows) if index not in val_indices]
        train_rows = base_train_rows * args.repeat

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(train_rows), args.output_dir / "train.parquet")
    pq.write_table(pa.Table.from_pylist(val_rows), args.output_dir / "val.parquet")
    manifest = {
        "dataset": "agentica-org/DeepCoder-Preview-Dataset",
        "subset": "taco",
        "revision": DATASET_REVISION,
        "data_dir": str(args.data_dir.resolve()),
        "base_url": args.base_url.rstrip("/"),
        "selected_rows": len(rows),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "repeat": args.repeat,
        "stdin_tasks": stdin_count,
        "functional_tasks": functional_count,
        "test_count": {
            "min": min(test_counts),
            "max": max(test_counts),
            "mean": round(statistics.fmean(test_counts), 3),
        },
        "prompt_tokens": (
            {
                "min": min(token_counts),
                "max": max(token_counts),
                "mean": round(statistics.fmean(token_counts), 3),
            }
            if token_counts
            else None
        ),
    }
    if len(rows) <= 256:
        manifest["task_indices"] = [int(row["task_idx"]) for row in rows]
    else:
        manifest["task_index_range"] = [
            min(int(row["task_idx"]) for row in rows),
            max(int(row["task_idx"]) for row in rows),
        ]
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
