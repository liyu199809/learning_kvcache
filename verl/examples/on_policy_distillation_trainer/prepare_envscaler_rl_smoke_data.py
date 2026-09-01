#!/usr/bin/env python3
"""Build a native-verl smoke dataset from official EnvScaler RL scenarios."""

from __future__ import annotations

import argparse
import copy
import json
import re
from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer


TASK_SUFFIX = re.compile(r"-task_(\d+)$")
SYSTEM_PROMPT = """You are an agent operating in a stateful tool environment.
Use the provided tools to complete the user's task. Inspect the environment before
making changes, obey all stated constraints, and do not invent tool results.
The harness manages verification; never call verify or done yourself. When the task
is complete, stop calling tools and give the user a concise final answer.
Current date: {current_date}."""

# An empty JSON Schema is the standard representation for an unconstrained
# value. verl's lightweight OpenAI schema model requires an explicit ``type``,
# so preserve the same semantics with a union of every JSON value type.
ANY_JSON_TYPES = ["string", "number", "boolean", "object", "array", "null"]


def task_number(task: dict) -> int:
    match = TASK_SUFFIX.search(str(task.get("task_id", "")))
    if not match:
        raise ValueError(f"Task id has no numeric suffix: {task.get('task_id')!r}")
    return int(match.group(1))


def load_official_data(data_dir: Path) -> tuple[dict, list[dict]]:
    envs = json.loads((data_dir / "191_env_metadata.json").read_text(encoding="utf-8"))
    tasks = json.loads(
        (data_dir / "envscaler_rl_scenario_metadata.json").read_text(encoding="utf-8")
    )
    return envs, tasks


def build_task_indices(tasks: list[dict]) -> dict[str, int]:
    grouped: dict[str, list[dict]] = {}
    for task in tasks:
        grouped.setdefault(str(task["env_id"]), []).append(task)
    result = {}
    for env_tasks in grouped.values():
        for task_idx, task in enumerate(sorted(env_tasks, key=task_number)):
            result[str(task["task_id"])] = task_idx
    return result


def normalize_tool_schemas(tools: list[dict]) -> tuple[list[dict], int]:
    """Make official JSON Schemas consumable by verl without narrowing Any."""
    normalized = copy.deepcopy(tools)
    repaired = 0
    for tool in normalized:
        properties = tool.get("function", {}).get("parameters", {}).get("properties", {})
        for property_schema in properties.values():
            if isinstance(property_schema, dict) and "type" not in property_schema:
                property_schema["type"] = list(ANY_JSON_TYPES)
                repaired += 1
    return normalized, repaired


def build_row(
    task: dict,
    env: dict,
    task_idx: int,
    base_url: str,
    current_date: str,
    source_row: int,
) -> dict:
    tools = env.get("tools")
    if not isinstance(tools, list) or not tools:
        raise ValueError(f"Environment {task['env_id']} has no tools")
    tools, repaired_schema_count = normalize_tool_schemas(tools)
    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT.format(current_date=current_date)},
        {"role": "user", "content": str(task["task"])},
    ]
    return {
        "data_source": "envscaler_rl",
        "prompt": prompt,
        "agent_name": "awm_agent",
        "tools": json.dumps(tools, ensure_ascii=False),
        "env_config": {
            "scenario": str(task["env_id"]),
            "task_idx": int(task_idx),
            "awm_base_url": base_url.rstrip("/"),
        },
        "scenario": str(task["env_id"]),
        "task_idx": int(task_idx),
        "reward_model": {"style": "rule", "ground_truth": ""},
        "extra_info": {
            "problem": str(task["task"]),
            "task_id": str(task["task_id"]),
            "scenario": str(task["env_id"]),
            "task_idx": int(task_idx),
            "source_row": int(source_row),
            "checklist_count": len(task.get("checklist_with_func", [])),
            "repaired_schema_count": repaired_schema_count,
        },
    }


def prompt_tokens(row: dict, tokenizer) -> int:
    tools = json.loads(row["tools"])
    encoded = tokenizer.apply_chat_template(
        row["prompt"],
        tools=tools,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=False,
    )
    return len(encoded)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("envscaler_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("traj_data/envscaler_rl_smoke"))
    parser.add_argument("--task-id", default="env_144_rl-task_37")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--base-url", default="http://127.0.0.1:8900")
    parser.add_argument("--model", default="/mnt/storage/disk1/verl_data/base_model/Qwen3.5-4B")
    parser.add_argument("--max-prompt-tokens", type=int, default=8192)
    parser.add_argument("--current-date", default=date.today().isoformat())
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")

    envs, tasks = load_official_data(args.data_dir)
    task_indices = build_task_indices(tasks)
    matching = [(index, task) for index, task in enumerate(tasks) if task["task_id"] == args.task_id]
    if len(matching) != 1:
        raise ValueError(f"Expected one task named {args.task_id!r}, found {len(matching)}")
    source_row, task = matching[0]
    env = envs[str(task["env_id"])]
    row = build_row(
        task=task,
        env=env,
        task_idx=task_indices[args.task_id],
        base_url=args.base_url,
        current_date=args.current_date,
        source_row=source_row,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    token_count = prompt_tokens(row, tokenizer)
    if token_count > args.max_prompt_tokens:
        raise ValueError(
            f"Prompt and tools use {token_count} tokens, exceeding {args.max_prompt_tokens}"
        )

    rows = [row for _ in range(args.repeat)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), args.output_dir / "train.parquet")
    pq.write_table(pa.Table.from_pylist([row]), args.output_dir / "val.parquet")
    manifest = {
        "task_id": args.task_id,
        "scenario": task["env_id"],
        "task_idx": task_indices[args.task_id],
        "source_row": source_row,
        "train_rows": len(rows),
        "val_rows": 1,
        "tool_count": len(env["tools"]),
        "repaired_schema_count": row["extra_info"]["repaired_schema_count"],
        "checklist_count": len(task["checklist_with_func"]),
        "prompt_tokens": token_count,
        "base_url": args.base_url.rstrip("/"),
        "current_date": args.current_date,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
