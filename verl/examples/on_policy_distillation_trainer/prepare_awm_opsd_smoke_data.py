#!/usr/bin/env python3
"""Convert real AWM OPSD JSONL rows into a native-verl dataset.

The source file contains both first-attempt tasks and retry tasks. Retry rows use
Swift's ``tool_call`` / ``tool_response`` roles, so tool calls are rendered into
Qwen3.5's native markup and tool responses are normalized to the ``tool`` role.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer


def load_rows(path: Path, row_index: int, num_rows: int) -> list[tuple[int, dict]]:
    selected = []
    with path.open(encoding="utf-8") as source:
        for index, line in enumerate(source):
            if row_index <= index < row_index + num_rows:
                selected.append((index, json.loads(line)))
            if len(selected) == num_rows:
                return selected
    raise IndexError(
        f"Requested rows [{row_index}, {row_index + num_rows}) from {path}, "
        f"but only found {len(selected)}"
    )


def load_all_rows(path: Path) -> list[tuple[int, dict]]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for index, line in enumerate(source):
            if line.strip():
                rows.append((index, json.loads(line)))
    if not rows:
        raise ValueError(f"No JSON rows found in {path}")
    return rows


def parse_tools(raw_tools) -> list[dict]:
    tools = json.loads(raw_tools) if isinstance(raw_tools, str) else raw_tools
    if not isinstance(tools, list) or not tools:
        raise ValueError("The selected row must contain a non-empty OpenAI tool list")
    names = []
    for tool in tools:
        function = tool.get("function", {}) if isinstance(tool, dict) else {}
        name = function.get("name")
        parameters = function.get("parameters", {})
        if tool.get("type") != "function" or not name or parameters.get("type") != "object":
            raise ValueError(f"Invalid OpenAI tool schema: {tool!r}")
        names.append(name)
    if len(names) != len(set(names)):
        raise ValueError("The selected row contains duplicate tool names")
    return tools


def render_qwen_tool_call(name: str, arguments: dict) -> str:
    lines = ["<tool_call>", f"<function={name}>"]
    for argument_name, argument_value in arguments.items():
        lines.append(f"<parameter={argument_name}>")
        if isinstance(argument_value, (dict, list)):
            lines.append(json.dumps(argument_value, ensure_ascii=False))
        else:
            lines.append(str(argument_value))
        lines.append("</parameter>")
    lines.extend(["</function>", "</tool_call>"])
    return "\n".join(lines)


def normalize_messages(raw_messages: list[dict]) -> list[dict]:
    """Map Swift tool roles to Qwen messages without losing retry history.

    Historical calls are stored as native Qwen markup in assistant content.
    Keeping heterogeneous tool arguments out of nested Parquet structs avoids
    Arrow coercing a parameter used as an integer in one call and a string in
    another into one invalid fixed type.
    """
    normalized: list[dict] = []
    for message_index, message in enumerate(raw_messages):
        if not isinstance(message, dict):
            raise ValueError(f"messages[{message_index}] must be a mapping")
        role = message.get("role")
        if role in {"system", "user", "assistant", "tool"}:
            normalized.append({"role": role, **{key: value for key, value in message.items() if key != "role"}})
            continue
        if role == "tool_call":
            if not normalized or normalized[-1].get("role") != "assistant":
                raise ValueError(f"messages[{message_index}] is a tool_call without a preceding assistant message")
            try:
                tool_call = json.loads(message.get("content", "{}"))
            except json.JSONDecodeError as exc:
                raise ValueError(f"messages[{message_index}] contains invalid tool-call JSON") from exc
            name = tool_call.get("name")
            arguments = tool_call.get("arguments", {})
            if not name or not isinstance(arguments, dict):
                raise ValueError(f"messages[{message_index}] contains an invalid tool call: {tool_call!r}")
            content = str(normalized[-1].get("content", ""))
            separator = "\n\n" if "<tool_call>" not in content else "\n"
            normalized[-1]["content"] = content + separator + render_qwen_tool_call(name, arguments)
            continue
        if role == "tool_response":
            normalized.append({"role": "tool", "content": str(message.get("content", ""))})
            continue
        raise ValueError(f"messages[{message_index}] has unsupported role {role!r}")
    return normalized


def build_verl_row(source: dict, awm_base_url: str) -> dict:
    required = ("messages", "teacher_prompt", "scenario", "task_idx", "env_config", "tools")
    missing = [key for key in required if key not in source]
    if missing:
        raise ValueError(f"Source row is missing fields: {missing}")
    raw_messages = source["messages"]
    if not isinstance(raw_messages, list) or not raw_messages:
        raise ValueError("messages must be a non-empty list")
    messages = normalize_messages(raw_messages)
    problem = next(
        (str(message.get("content", "")) for message in messages if message.get("role") == "user"),
        "",
    )
    env_config = dict(source["env_config"])
    env_config.update(
        scenario=str(source["scenario"]),
        task_idx=int(source["task_idx"]),
        awm_base_url=awm_base_url.rstrip("/"),
    )
    return {
        "data_source": "awm",
        "prompt": messages,
        "agent_name": "awm_agent",
        # Preserve the source representation: AWMAgentLoop deliberately accepts
        # both this JSON string form and an already-decoded list.
        "tools": source["tools"],
        "env_config": env_config,
        "scenario": str(source["scenario"]),
        "task_idx": int(source["task_idx"]),
        "teacher_prompt": str(source["teacher_prompt"]),
        "reward_model": {"style": "rule", "ground_truth": str(source["teacher_prompt"])},
        "extra_info": {
            "problem": problem,
            "scenario": str(source["scenario"]),
            "task_idx": int(source["task_idx"]),
            "source_row": int(source.get("source_row", 0)),
        },
    }


def validate_token_budget(row: dict, tokenizer, max_prompt_tokens: int) -> int:
    tools = parse_tools(row["tools"])
    encoded = tokenizer.apply_chat_template(
        row["prompt"],
        tools=tools,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=False,
    )
    token_count = len(encoded)
    if token_count > max_prompt_tokens:
        raise ValueError(
            f"Prompt plus {len(tools)} tools is {token_count} tokens, exceeding max_prompt_tokens={max_prompt_tokens}"
        )
    return token_count


def write_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)
    print(f"wrote {len(rows)} row(s) to {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, default=Path("traj_data/swift_stage2_opsd.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("traj_data/awm_opsd_smoke"))
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--num-rows", type=int, default=1)
    parser.add_argument("--all-rows", action="store_true", help="Convert every non-empty source row")
    parser.add_argument(
        "--val-size",
        type=int,
        default=1,
        help="Random validation subset size. With --all-rows, train still contains every source row.",
    )
    parser.add_argument("--val-seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--model", default="/mnt/storage/disk1/verl_data/base_model/Qwen3.5-4B")
    parser.add_argument("--awm-base-url", default="http://localhost:8899")
    parser.add_argument("--max-prompt-tokens", type=int, default=16384)
    args = parser.parse_args()
    if args.row_index < 0 or args.num_rows < 1 or args.val_size < 1 or args.progress_every < 1:
        parser.error(
            "--row-index must be non-negative; --num-rows, --val-size, and --progress-every must be positive"
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    rows = []
    token_counts = []
    selected_sources = (
        load_all_rows(args.input_jsonl)
        if args.all_rows
        else load_rows(args.input_jsonl, args.row_index, args.num_rows)
    )
    if args.val_size > len(selected_sources):
        parser.error(f"--val-size={args.val_size} exceeds the selected row count {len(selected_sources)}")
    for selected_index, (source_row, source) in enumerate(selected_sources):
        source["source_row"] = source_row
        row = build_verl_row(source, args.awm_base_url)
        token_count = validate_token_budget(row, tokenizer, args.max_prompt_tokens)
        if (
            selected_index == 0
            or selected_index + 1 == len(selected_sources)
            or (selected_index + 1) % args.progress_every == 0
        ):
            print(
                f"validated {selected_index + 1}/{len(selected_sources)} (source row {source_row}): "
                f"{row['scenario']} task {row['task_idx']}, "
                f"{len(parse_tools(row['tools']))} tools, {token_count} prompt tokens"
            )
        rows.append(row)
        token_counts.append(token_count)

    write_parquet(args.output_dir / "train.parquet", rows)
    val_positions = sorted(random.Random(args.val_seed).sample(range(len(rows)), args.val_size))
    write_parquet(args.output_dir / "val.parquet", [rows[position] for position in val_positions])
    manifest = {
        "input_jsonl": str(args.input_jsonl.resolve()),
        "input_sha256": hashlib.sha256(args.input_jsonl.read_bytes()).hexdigest(),
        "train_rows": len(rows),
        "train_contains_all_selected_rows": True,
        "val_rows": len(val_positions),
        "val_seed": args.val_seed,
        "val_source_rows": [int(rows[position]["extra_info"]["source_row"]) for position in val_positions],
        "prompt_tokens": {
            "min": min(token_counts),
            "max": max(token_counts),
            "mean": sum(token_counts) / len(token_counts),
        },
    }
    manifest_path = args.output_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote split manifest to {manifest_path}")


if __name__ == "__main__":
    main()
