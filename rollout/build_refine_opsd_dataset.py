#!/usr/bin/env python3
"""Build a small, quality-filtered mixed OPSD dataset from refine traces.

The output keeps the native verl row schema used by the first AWM OPSD run,
while preserving per-row environment routing for AWM, EnvScaler and TACO.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from rollout.refine2swift import (
    ENV_RESET_NOTE,
    _ADVICE_USER_BLOCK,
    _CODE_REFERENCE_BLOCK,
    _TRAJECTORY_HINT_BLOCK,
    _fetch_tools_by_scenario,
    _lenient_json,
    _round_messages,
    _select_candidate,
    _task_prefix,
)


SAMPLE_PRIORITY = {
    "improved_partial": 1,
    "complete_after_refine": 2,
    "complete_first_try": 3,
}
DATA_SOURCE = {
    "awm": "awm",
    "envscaler": "envscaler_rl",
    "taco": "deepcoder_taco",
}
DEFAULT_SOURCE_FILES = {
    "awm": ["traj_data/awm_refine_full.jsonl"],
    "envscaler": [
        "traj_data/envscaler_refine_full.jsonl",
        "traj_data/envscaler_refine_16iter_full.jsonl",
    ],
    "taco": ["traj_data/deepcoder_taco_refine_full.jsonl"],
}
SOURCE_LABELS = {
    "awm_refine_full.jsonl": "awm",
    "envscaler_refine_full.jsonl": "env8",
    "envscaler_refine_16iter_full.jsonl": "env16",
    "deepcoder_taco_refine_full.jsonl": "taco",
}

FAILURE_ANSWER_RE = re.compile(
    r"\b(?:unable to complete|could not complete|couldn't complete|"
    r"cannot complete|failed to complete|task (?:is|was) not complete|"
    r"was not completed)\b",
    re.IGNORECASE,
)
TEXT_ERROR_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"^\s*error\b",
        r"error executing tool",
        r"unknown tool",
        r"status code:\s*[45]\d\d",
        r"validation error",
        r"traceback \(most recent call last\)",
    )
)
TACO_CONTAMINATION_RE = re.compile(
    r"(?:^\s*we need (?:answer|solve|respond)|developer instruction|"
    r"higher priority|must call audit_analysis|user says do not call)",
    re.IGNORECASE,
)
FENCED_PYTHON_RE = re.compile(r"```(?:python)?\s*[\s\S]+?```", re.IGNORECASE)


@dataclass
class CandidateRow:
    dataset: str
    source_path: Path
    source_label: str
    source_line: int
    record: dict[str, Any]
    candidate: dict[str, Any]
    quality_tier: str

    @property
    def key(self) -> tuple[str, str, str]:
        task_id = self.record.get("task_id")
        identity = task_id if task_id is not None else self.record.get("task_idx")
        return (
            str(self.record.get("data_source") or DATA_SOURCE[self.dataset]),
            str(self.record.get("scenario")),
            str(identity),
        )

    @property
    def rank(self) -> tuple[int, float, int]:
        score = self.candidate.get("target_score")
        source_preference = 1 if self.source_label == "env16" else 0
        return (
            SAMPLE_PRIORITY[self.candidate["sample_type"]],
            float(score) if score is not None else -math.inf,
            source_preference,
        )


def _nonempty_json_value(value: Any) -> bool:
    if value is None or value is False or value == "":
        return False
    if isinstance(value, dict):
        return any(_nonempty_json_value(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_nonempty_json_value(item) for item in value)
    return True


def _leading_json(text: str) -> Any | None:
    try:
        value, _ = json.JSONDecoder().raw_decode(text.lstrip())
        return value
    except (json.JSONDecodeError, TypeError):
        return None


def is_explicit_tool_error(text: str | None) -> bool:
    """Recognize real failures without flagging harmless ``error: null`` text."""
    if text is None or not text.strip():
        return True
    if any(pattern.search(text) for pattern in TEXT_ERROR_PATTERNS):
        return True
    payload = _leading_json(text)
    if not isinstance(payload, dict):
        return False
    if payload.get("success") is False:
        return True
    status = str(payload.get("status") or "").strip().lower()
    if status in {"error", "failed", "failure"}:
        return True
    return "error" in payload and _nonempty_json_value(payload.get("error"))


def round_tool_results(record: dict[str, Any], detail: dict[str, Any]) -> list[tuple[dict, str | None]]:
    messages = _round_messages(record, detail)
    paired: list[tuple[dict, str | None]] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        calls = message.get("tool_calls") or [] if message.get("role") == "assistant" else []
        if not calls:
            index += 1
            continue
        results: list[str] = []
        next_index = index + 1
        while (
            next_index < len(messages)
            and messages[next_index].get("role") == "tool"
            and len(results) < len(calls)
        ):
            results.append(str(messages[next_index].get("content") or ""))
            next_index += 1
        paired.extend(
            (call, results[call_index] if call_index < len(results) else None)
            for call_index, call in enumerate(calls)
        )
        index = next_index
    return paired


def render_success_plan(record: dict[str, Any], detail: dict[str, Any]) -> str:
    messages = _round_messages(record, detail)
    groups: list[list[tuple[str, dict[str, Any]]]] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        calls = message.get("tool_calls") or [] if message.get("role") == "assistant" else []
        if not calls:
            index += 1
            continue
        results: list[str] = []
        next_index = index + 1
        while (
            next_index < len(messages)
            and messages[next_index].get("role") == "tool"
            and len(results) < len(calls)
        ):
            results.append(str(messages[next_index].get("content") or ""))
            next_index += 1
        kept: list[tuple[str, dict[str, Any]]] = []
        for call_index, call in enumerate(calls):
            result = results[call_index] if call_index < len(results) else None
            if is_explicit_tool_error(result):
                continue
            function = call.get("function") or {}
            arguments = _lenient_json(function.get("arguments"))
            kept.append(
                (
                    str(function.get("name") or ""),
                    arguments if isinstance(arguments, dict) else {},
                )
            )
        if kept:
            groups.append(kept)
        index = next_index

    lines: list[str] = []
    for step, group in enumerate(groups, start=1):
        if len(group) == 1:
            name, arguments = group[0]
            lines.append(f"Step {step}: call {name}({json.dumps(arguments, ensure_ascii=False)})")
        else:
            lines.append(f"Step {step} (parallel):")
            for name, arguments in group:
                lines.append(f"    - call {name}({json.dumps(arguments, ensure_ascii=False)})")
    return "\n".join(lines)


def _judge_complete_confidence(detail: dict[str, Any]) -> float | None:
    raw = detail.get("judge_confidence_score")
    if isinstance(raw, list) and raw:
        raw = raw[0]
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def quality_reason(dataset: str, record: dict[str, Any], candidate: dict[str, Any]) -> tuple[str | None, str]:
    target = candidate["target"]
    target_answer = str(target.get("student_final_answer") or "").strip()
    # Tool environments can complete state-changing work even when the final
    # prose turn is empty. The successful tool plan/advice remains valid OPSD
    # evidence. TACO, in contrast, has no useful target without runnable code.
    if dataset == "taco" and not target_answer:
        return "empty_target_answer", "drop"

    if dataset in {"awm", "envscaler"}:
        paired = round_tool_results(record, target)
        if not paired:
            return "target_has_no_tool_calls", "drop"
        if all(is_explicit_tool_error(result) for _, result in paired):
            return "target_all_tool_errors", "drop"
        if candidate["sample_type"].startswith("complete") and FAILURE_ANSWER_RE.search(target_answer):
            return "complete_answer_admits_failure", "drop"

    if dataset == "awm" and candidate["sample_type"].startswith("complete"):
        if target.get("verify_reward_type") != "complete":
            confidence = _judge_complete_confidence(target)
            if confidence is None or confidence < 80:
                return "low_confidence_judge_only_complete", "drop"
            return None, "B"

    if dataset == "taco":
        if not FENCED_PYTHON_RE.search(target_answer):
            return "taco_target_missing_python", "drop"
        if candidate["sample_type"] == "improved_partial":
            source = candidate["source"]
            if source.get("student_failure_type") == "reasoning_without_answer":
                return "taco_reasoning_only_partial", "drop"
            advice = str(source.get("teacher_advice") or "").strip()
            if len(advice) > 12_000:
                return "taco_partial_advice_too_long", "drop"
            if TACO_CONTAMINATION_RE.search(advice):
                return "taco_partial_advice_contaminated", "drop"

    return None, "A"


def load_pool(dataset: str, paths: Iterable[Path]) -> tuple[list[CandidateRow], Counter]:
    rows_by_task: dict[tuple[str, str, str], CandidateRow] = {}
    stats: Counter = Counter()
    for path in paths:
        source_label = SOURCE_LABELS.get(path.name, path.stem)
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                stats["input_rows"] += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    stats["drop:invalid_json"] += 1
                    continue
                candidate, reason = _select_candidate(record)
                if candidate is None:
                    stats[f"drop:{reason}"] += 1
                    continue
                reason, tier = quality_reason(dataset, record, candidate)
                if reason:
                    stats[f"drop:{reason}"] += 1
                    continue
                row = CandidateRow(
                    dataset=dataset,
                    source_path=path,
                    source_label=source_label,
                    source_line=line_number,
                    record=record,
                    candidate=candidate,
                    quality_tier=tier,
                )
                previous = rows_by_task.get(row.key)
                if previous is None or row.rank > previous.rank:
                    if previous is not None:
                        stats["drop:duplicate_replaced"] += 1
                    rows_by_task[row.key] = row
                else:
                    stats["drop:duplicate_lower_rank"] += 1
    rows = list(rows_by_task.values())
    stats["quality_pool"] = len(rows)
    for row in rows:
        stats[f"pool:{row.candidate['sample_type']}"] += 1
        stats[f"pool_tier:{row.quality_tier}"] += 1
    return rows, stats


def stratified_take(rows: list[CandidateRow], count: int, seed: int) -> tuple[list[CandidateRow], list[CandidateRow]]:
    if count < 0 or count > len(rows):
        raise ValueError(f"Cannot take {count} rows from pool of {len(rows)}")
    if count == 0:
        return [], list(rows)
    groups: dict[str, list[CandidateRow]] = defaultdict(list)
    for row in rows:
        groups[row.candidate["sample_type"]].append(row)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)

    exact = {name: count * len(group) / len(rows) for name, group in groups.items()}
    quota = {name: min(len(groups[name]), math.floor(value)) for name, value in exact.items()}
    remaining = count - sum(quota.values())
    order = sorted(
        groups,
        key=lambda name: (exact[name] - quota[name], len(groups[name]), name),
        reverse=True,
    )
    while remaining:
        progressed = False
        for name in order:
            if quota[name] < len(groups[name]):
                quota[name] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            raise RuntimeError("Unable to allocate stratified sample quota")

    selected: list[CandidateRow] = []
    leftovers: list[CandidateRow] = []
    for name in sorted(groups):
        selected.extend(groups[name][: quota[name]])
        leftovers.extend(groups[name][quota[name] :])
    rng.shuffle(selected)
    rng.shuffle(leftovers)
    return selected, leftovers


def render_qwen_tool_call(name: str, arguments: dict[str, Any]) -> str:
    lines = ["<tool_call>", f"<function={name}>"]
    for argument_name, argument_value in arguments.items():
        lines.append(f"<parameter={argument_name}>")
        lines.append(
            json.dumps(argument_value, ensure_ascii=False)
            if isinstance(argument_value, (dict, list))
            else str(argument_value)
        )
        lines.append("</parameter>")
    lines.extend(["</function>", "</tool_call>"])
    return "\n".join(lines)


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for message in messages:
        role = message.get("role")
        if role == "assistant":
            reasoning = str(message.get("reasoning") or "").strip()
            content = str(message.get("content") or "").strip()
            if reasoning:
                content = f"<think>{reasoning}</think>{content}"
            rendered_calls = []
            for tool_call in message.get("tool_calls") or []:
                function = tool_call.get("function") or {}
                arguments = _lenient_json(function.get("arguments"))
                rendered_calls.append(
                    render_qwen_tool_call(
                        str(function.get("name") or ""),
                        arguments if isinstance(arguments, dict) else {},
                    )
                )
            if rendered_calls:
                separator = "\n\n" if content else ""
                content = content + separator + "\n".join(rendered_calls)
            normalized.append({"role": "assistant", "content": content})
        elif role == "tool":
            normalized.append({"role": "tool", "content": str(message.get("content") or "")})
        elif role in {"system", "user"}:
            normalized.append({"role": str(role), "content": str(message.get("content") or "")})
        else:
            raise ValueError(f"Unsupported message role: {role!r}")
    return normalized


def build_teacher_and_prompt(row: CandidateRow) -> tuple[list[dict[str, str]], str]:
    record = row.record
    candidate = row.candidate
    target = candidate["target"]
    conversation = record.get("student_conversation") or []
    task_prefix = _task_prefix(conversation)
    if not task_prefix:
        raise ValueError(f"Task {row.key} has no task prefix")

    if candidate["sample_type"] == "complete_first_try":
        messages = task_prefix
        task_text = str(messages[-1].get("content") or "")
        if row.dataset == "taco":
            solution = str(target.get("student_final_answer") or "").strip()
            teacher_prompt = _CODE_REFERENCE_BLOCK.format(solution=solution)
        else:
            plan = render_success_plan(record, target)
            if not plan:
                raise ValueError(f"Task {row.key} has no successful tool plan")
            teacher_prompt = _TRAJECTORY_HINT_BLOCK.format(trajectory=plan)
        teacher_prompt += "\n\n[User Question]" + task_text
    else:
        source = candidate["source"]
        source_round = _round_messages(record, source)
        if not source_round:
            raise ValueError(f"Task {row.key} has an invalid source message range")
        messages = task_prefix + source_round + [{"role": "user", "content": ENV_RESET_NOTE}]
        if row.dataset == "taco" and candidate["sample_type"] == "complete_after_refine":
            solution = str(target.get("student_final_answer") or "").strip()
            teacher_prompt = _CODE_REFERENCE_BLOCK.format(solution=solution) + "\n\n" + ENV_RESET_NOTE
        else:
            advice = str(source.get("teacher_advice") or "").strip()
            teacher_prompt = _ADVICE_USER_BLOCK.format(advice=advice) + "\n\n" + ENV_RESET_NOTE
    return normalize_messages(messages), teacher_prompt


def load_awm_tool_cache(path: Path) -> dict[str, str]:
    cache: dict[str, str] = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            scenario = row.get("scenario")
            tools = row.get("tools")
            if scenario and tools:
                cache.setdefault(str(scenario), tools if isinstance(tools, str) else json.dumps(tools, ensure_ascii=False))
    return cache


ANY_JSON_TYPES = ["string", "number", "boolean", "object", "array", "null"]


def normalize_envscaler_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Round-trip copy is sufficient for JSON-only tool metadata.
    normalized = json.loads(json.dumps(tools))
    for tool in normalized:
        properties = tool.get("function", {}).get("parameters", {}).get("properties", {})
        for schema in properties.values():
            if isinstance(schema, dict) and "type" not in schema:
                schema["type"] = list(ANY_JSON_TYPES)
    return normalized


def attach_tools(
    selected: dict[str, list[CandidateRow]],
    awm_cache_path: Path,
    env_metadata_path: Path,
    awm_base_url: str,
) -> dict[str, str]:
    tools_by_scenario = load_awm_tool_cache(awm_cache_path)
    awm_scenarios = {str(row.record.get("scenario")) for row in selected["awm"]}
    missing = sorted(awm_scenarios - tools_by_scenario.keys())
    if missing:
        fetched = asyncio.run(_fetch_tools_by_scenario(missing, awm_base_url))
        tools_by_scenario.update({key: value for key, value in fetched.items() if value})
    still_missing = sorted(awm_scenarios - tools_by_scenario.keys())
    if still_missing:
        raise RuntimeError(f"Missing AWM tools for {len(still_missing)} scenarios: {still_missing[:10]}")

    env_metadata = json.loads(env_metadata_path.read_text(encoding="utf-8"))
    for row in selected["envscaler"]:
        scenario = str(row.record.get("scenario"))
        metadata = env_metadata.get(scenario)
        if not isinstance(metadata, dict) or not metadata.get("tools"):
            raise RuntimeError(f"Missing EnvScaler tools for {scenario}")
        tools_by_scenario[f"envscaler:{scenario}"] = json.dumps(
            normalize_envscaler_tools(metadata["tools"]), ensure_ascii=False
        )
    return tools_by_scenario


def build_verl_row(
    selected: CandidateRow,
    tools_by_scenario: dict[str, str],
    base_urls: dict[str, str],
) -> dict[str, Any]:
    messages, teacher_prompt = build_teacher_and_prompt(selected)
    record = selected.record
    candidate = selected.candidate
    scenario = str(record.get("scenario"))
    if selected.dataset == "awm":
        tools = tools_by_scenario[scenario]
    elif selected.dataset == "envscaler":
        tools = tools_by_scenario[f"envscaler:{scenario}"]
    else:
        tools = "[]"
    task_id = record.get("task_id")
    if task_id is None:
        task_id = f"{scenario}:{record.get('task_idx')}"
    baseline = candidate.get("baseline_score")
    target_score = candidate.get("target_score")
    score_delta = (
        float(target_score) - float(baseline)
        if target_score is not None and baseline is not None
        else None
    )
    extra_info = {
        "problem": str(record.get("task") or ""),
        "task_id": str(task_id),
        "scenario": scenario,
        "task_idx": int(record.get("task_idx")),
        "sample_type": str(candidate["sample_type"]),
        "source_round": int(candidate["source"].get("round")) if candidate.get("source") else None,
        "target_round": int(candidate["target"].get("round")),
        "baseline_score": float(baseline) if baseline is not None else None,
        "source_score": (
            float(candidate["source_score"]) if candidate.get("source_score") is not None else None
        ),
        "target_score": float(target_score) if target_score is not None else None,
        "score_delta": score_delta,
        "quality_tier": selected.quality_tier,
        "source_dataset": selected.source_label,
        "source_row": int(selected.source_line),
    }
    return {
        "data_source": DATA_SOURCE[selected.dataset],
        "prompt": messages,
        "agent_name": "awm_agent",
        "tools": tools,
        "env_config": {
            "scenario": scenario,
            "task_idx": int(record.get("task_idx")),
            "awm_base_url": base_urls[selected.dataset].rstrip("/"),
        },
        "scenario": scenario,
        "task_idx": int(record.get("task_idx")),
        "teacher_prompt": teacher_prompt,
        "reward_model": {"style": "rule", "ground_truth": teacher_prompt},
        "extra_info": extra_info,
    }


def token_stats(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    max_prompt_tokens: int,
    max_response_tokens: int,
    max_model_tokens: int,
    privileged_overhead_tokens: int,
) -> dict[str, Any]:
    prompt_counts: list[int] = []
    teacher_counts: list[int] = []
    violations: list[dict[str, Any]] = []
    for row in rows:
        tools = json.loads(row["tools"])
        prompt_count = len(
            tokenizer.apply_chat_template(
                row["prompt"],
                tools=tools,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_dict=False,
            )
        )
        teacher_count = len(tokenizer.encode(row["teacher_prompt"], add_special_tokens=False))
        teacher_total = prompt_count + teacher_count + max_response_tokens + privileged_overhead_tokens
        if prompt_count > max_prompt_tokens or teacher_total > max_model_tokens:
            violations.append(
                {
                    "data_source": row["data_source"],
                    "task_id": row["extra_info"]["task_id"],
                    "prompt_tokens": prompt_count,
                    "teacher_total_tokens": teacher_total,
                }
            )
        prompt_counts.append(prompt_count)
        teacher_counts.append(teacher_count)
    if violations:
        raise ValueError(f"Token limits exceeded by {len(violations)} rows: {violations[:10]}")

    def summarize(values: list[int]) -> dict[str, float | int]:
        ordered = sorted(values)
        return {
            "min": ordered[0],
            "max": ordered[-1],
            "mean": round(statistics.fmean(ordered), 3),
            "p95": ordered[int(0.95 * (len(ordered) - 1))],
        }

    return {"prompt": summarize(prompt_counts), "teacher_prompt": summarize(teacher_counts)}


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def distribution(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row["extra_info"][key]) for row in rows).items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=Path("traj_data/opsd_mixed_1625_v1"))
    parser.add_argument("--train-per-source", type=int, default=1625)
    parser.add_argument("--env-val-size", type=int, default=50)
    parser.add_argument("--taco-val-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="/mnt/storage/disk1/verl_data/base_model/Qwen3.5-4B")
    parser.add_argument("--max-prompt-tokens", type=int, default=28672)
    parser.add_argument("--max-response-tokens", type=int, default=8192)
    parser.add_argument("--max-model-tokens", type=int, default=40960)
    parser.add_argument("--privileged-overhead-tokens", type=int, default=32)
    parser.add_argument("--awm-base-url", default="http://localhost:8899")
    parser.add_argument("--envscaler-base-url", default="http://127.0.0.1:8900")
    parser.add_argument("--taco-base-url", default="http://127.0.0.1:8901")
    args = parser.parse_args()
    if args.train_per_source < 1 or args.env_val_size < 0 or args.taco_val_size < 0:
        parser.error("Training count must be positive and validation counts non-negative")

    root = args.repo_root.resolve()
    output_dir = (root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    source_paths = {
        dataset: [root / path for path in paths]
        for dataset, paths in DEFAULT_SOURCE_FILES.items()
    }
    pools: dict[str, list[CandidateRow]] = {}
    filter_stats: dict[str, dict[str, int]] = {}
    for dataset, paths in source_paths.items():
        pool, stats = load_pool(dataset, paths)
        pools[dataset] = pool
        filter_stats[dataset] = dict(sorted(stats.items()))
        print(f"{dataset}: quality pool {len(pool)}")

    selected_train: dict[str, list[CandidateRow]] = {}
    selected_val: dict[str, list[CandidateRow]] = {"awm": []}
    for dataset in ("awm", "envscaler", "taco"):
        train_rows, leftovers = stratified_take(
            pools[dataset], args.train_per_source, args.seed + {"awm": 0, "envscaler": 100, "taco": 200}[dataset]
        )
        val_size = {"awm": 0, "envscaler": args.env_val_size, "taco": args.taco_val_size}[dataset]
        val_rows, _ = stratified_take(leftovers, val_size, args.seed + {"awm": 1, "envscaler": 101, "taco": 201}[dataset])
        selected_train[dataset] = train_rows
        selected_val[dataset] = val_rows

    all_selected = {
        dataset: selected_train[dataset] + selected_val[dataset]
        for dataset in ("awm", "envscaler", "taco")
    }
    tools_by_scenario = attach_tools(
        all_selected,
        root / "traj_data/task1/swift_opsd.jsonl",
        root / "envscaler_data/191_env_metadata.json",
        args.awm_base_url,
    )
    base_urls = {
        "awm": args.awm_base_url,
        "envscaler": args.envscaler_base_url,
        "taco": args.taco_base_url,
    }

    train_by_source = {
        dataset: [build_verl_row(row, tools_by_scenario, base_urls) for row in selected_train[dataset]]
        for dataset in ("awm", "envscaler", "taco")
    }
    val_by_source = {
        dataset: [build_verl_row(row, tools_by_scenario, base_urls) for row in selected_val[dataset]]
        for dataset in ("awm", "envscaler", "taco")
    }
    mixed_train = [row for dataset in ("awm", "envscaler", "taco") for row in train_by_source[dataset]]
    mixed_val = [row for dataset in ("envscaler", "taco") for row in val_by_source[dataset]]
    random.Random(args.seed + 1000).shuffle(mixed_train)
    random.Random(args.seed + 1001).shuffle(mixed_val)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    train_token_stats = token_stats(
        mixed_train,
        tokenizer,
        args.max_prompt_tokens,
        args.max_response_tokens,
        args.max_model_tokens,
        args.privileged_overhead_tokens,
    )
    val_token_stats = token_stats(
        mixed_val,
        tokenizer,
        args.max_prompt_tokens,
        args.max_response_tokens,
        args.max_model_tokens,
        args.privileged_overhead_tokens,
    )

    for dataset in ("awm", "envscaler", "taco"):
        write_rows(output_dir / "sources" / dataset / "train.parquet", train_by_source[dataset])
        if val_by_source[dataset]:
            write_rows(output_dir / "sources" / dataset / "val.parquet", val_by_source[dataset])
    write_rows(output_dir / "train.parquet", mixed_train)
    write_rows(output_dir / "val.parquet", mixed_val)

    train_keys = {(row["data_source"], row["extra_info"]["task_id"]) for row in mixed_train}
    val_keys = {(row["data_source"], row["extra_info"]["task_id"]) for row in mixed_val}
    overlap = train_keys & val_keys
    if overlap:
        raise RuntimeError(f"Train/val overlap detected: {sorted(overlap)[:10]}")

    manifest = {
        "output_dir": str(output_dir),
        "seed": args.seed,
        "train_rows": len(mixed_train),
        "val_rows": len(mixed_val),
        "train_rows_by_source": dict(Counter(row["data_source"] for row in mixed_train)),
        "val_rows_by_source": dict(Counter(row["data_source"] for row in mixed_val)),
        "train_sample_types_by_source": {
            dataset: distribution(rows, "sample_type") for dataset, rows in train_by_source.items()
        },
        "val_sample_types_by_source": {
            dataset: distribution(rows, "sample_type") for dataset, rows in val_by_source.items() if rows
        },
        "train_quality_tiers_by_source": {
            dataset: distribution(rows, "quality_tier") for dataset, rows in train_by_source.items()
        },
        "filter_stats": filter_stats,
        "token_limits": {
            "max_prompt_tokens": args.max_prompt_tokens,
            "max_response_tokens": args.max_response_tokens,
            "max_model_tokens": args.max_model_tokens,
            "privileged_overhead_tokens": args.privileged_overhead_tokens,
        },
        "train_token_stats": train_token_stats,
        "val_token_stats": val_token_stats,
        # Compatibility with the existing AWM OPSD launcher, which reads
        # split_manifest.json["prompt_tokens"]["max"].
        "prompt_tokens": train_token_stats["prompt"],
        "train_contains_all_selected_rows": True,
        "train_val_task_overlap": 0,
        "source_files": {
            dataset: [
                {"path": str(path), "sha256": sha256(path)} for path in paths
            ]
            for dataset, paths in source_paths.items()
        },
        "base_urls": base_urls,
    }
    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    (output_dir / "manifest.json").write_text(manifest_text, encoding="utf-8")
    (output_dir / "split_manifest.json").write_text(manifest_text, encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
