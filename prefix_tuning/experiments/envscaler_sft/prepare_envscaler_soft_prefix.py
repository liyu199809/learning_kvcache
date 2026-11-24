"""Prepare a single-source EnvScaler dataset for the existing soft-prefix trainer."""
import argparse
import ast
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from hashlib import file_digest
from itertools import islice
import json
from pathlib import Path
import re

import ijson
from transformers import AutoTokenizer

from audit_sft_schemas import canonical_tools, parameters_match
from prepare import OFFENSIVE, OUT, lines
from prepare_sft_decision6 import clean
from prepare_sft_800 import normalize
from sft_trajectory_tokens import marked_template, tokenize_trajectory
from train_prefix import MODEL

RUN = OUT / "envscaler-soft-prefix"
META = OUT / "envscaler-reference"
SOURCE = OUT / "sft-decision5/envscaler-official/trajectories.json"
ALIAS = re.compile(r"\bmcp__episode_?\b")
CONTEXT = 16384
PREFIX_LENGTH = 32


def setup():
    global tokenizer, template
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    template = marked_template(tokenizer.chat_template)


def prepare_trajectory(pair):
    index, source = pair
    original = json.loads(source["messages"])
    tools = canonical_tools(json.loads(source["tools"]))
    if OFFENSIVE.search(json.dumps([original, tools], ensure_ascii=False)):
        return "content_screen", []
    try:
        messages = normalize(original, False, merge_assistant=False)
        for message in messages:
            for call in message.get("tool_calls", []):
                function = call["function"]
                if isinstance(function["arguments"], str):
                    function["arguments"] = json.loads(function["arguments"])
                if not isinstance(function["arguments"], dict):
                    return "schema_mismatch", []
    except (KeyError, TypeError, ValueError):
        return "schema_mismatch", []
    last_assistant = max((i for i, message in enumerate(messages) if message["role"] == "assistant"), default=-1)
    messages = messages[:last_assistant + 1]
    if not parameters_match(messages, tools)[0]:
        return "schema_mismatch", []
    for message in messages:
        for field in ("content", "reasoning_content"):
            if field in message:
                message[field] = ALIAS.sub("the tool", message[field].replace("mcp__episode__", ""))
    if "mcp__" in json.dumps([messages, tools], ensure_ascii=False):
        return "tool_alias", []
    users = [index for index, message in enumerate(messages) if message["role"] == "user"]
    if not users:
        return "no_user_turn", []
    ident = f"envscaler-official:{index+1}"
    output = []
    for part, (start, stop) in enumerate(zip(users, users[1:] + [len(messages)]), 1):
        end = next((i+1 for i in range(stop-1, start, -1) if messages[i]["role"] == "assistant"), 0)
        if not end:
            return "no_assistant_turn", []
        section = [{**message, "loss": i > start and message["role"] == "assistant"}
                   for i, message in enumerate(messages[:end])]
        row = {"id": f"{ident}:user-{part}", "trajectory_id": ident, "source": "envscaler", "type": "SFT",
               "task_id": str(source["task_info"]["task_id"]), "env_id": source["task_info"]["env_id"],
               "messages": section, "tools": tools, "enable_thinking": True, "sample_weight": 1 / len(users)}
        if any(a["role"] == b["role"] == "assistant" for a, b in zip(section, section[1:])):
            return "adjacent_assistant_turns", []
        kwargs = {"tools": tools, "tokenize": False, "enable_thinking": True}
        try:
            prompt = tokenizer.apply_chat_template(section[:start+1], add_generation_prompt=True, **kwargs)
            full = tokenizer.apply_chat_template(section, add_generation_prompt=False, **kwargs)
        except (TypeError, ValueError):
            return "prompt_boundary", []
        if not full.startswith(prompt):
            return "prompt_boundary", []
        try:
            ids, targets, _ = tokenize_trajectory(tokenizer, row, template)
        except ValueError:
            return "loss_boundary", []
        if len(ids) + PREFIX_LENGTH > CONTEXT:
            return "over_context", []
        if tokenizer.decode(targets, clean_up_tokenization_spaces=False).count("<|im_end|>") != sum(
                message["loss"] for message in section):
            return "loss_boundary", []
        output.append(row)
    return "included", output


def resume_progress(log, partial):
    match = next((m for line in reversed(log.read_text().splitlines())
                  if (m := re.fullmatch(r"(\d+) (\d+) (\{.*\})", line))), None)
    if match is None:
        raise ValueError("Previous preprocessing has no complete batch")
    count, segments = map(int, match.groups()[:2])
    counts = Counter(ast.literal_eval(match[3]))
    if count != sum(counts.values()) or not 0 < count < 9022 or not segments:
        raise ValueError("Previous preprocessing counts are inconsistent")
    with partial.open("r+b") as stream:
        for _ in range(segments):
            line = stream.readline()
            if not line.endswith(b"\n"):
                raise ValueError("Previous preprocessing is missing completed rows")
        last = json.loads(line)
        if int(last["trajectory_id"].split(":")[-1]) > count:
            raise ValueError("Previous preprocessing does not match the saved batch")
        stream.truncate()
    return count, counts, segments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--resume-log", type=Path)
    args = parser.parse_args()
    if args.probe and args.resume_log:
        parser.error("--probe cannot resume a full preprocessing run")
    tree = json.loads((META / "dataset-tree.json").read_text())
    entry = next(row for row in tree if row["path"] == "envscaler_sft_traj_9k_metadata.json")
    with SOURCE.open("rb") as stream:
        sha256 = file_digest(stream, "sha256").hexdigest()
    assert sha256 == entry["lfs"]["oid"] and SOURCE.stat().st_size == entry["size"]
    val = [row for row in lines(OUT / "sft-800/val.jsonl") if row["source"] == "envscaler"]
    val_environments = {row["split_key"] for row in val}
    assert len(val) == 15
    dest = RUN / ("probe-data" if args.probe else "data")
    dest.mkdir(parents=True, exist_ok=True)
    partial = dest / "train.tmp"
    offset, counts, segments = resume_progress(args.resume_log, partial) if args.resume_log else (0, Counter(), 0)
    mode = "a" if args.resume_log else "w"
    with ProcessPoolExecutor(args.workers, initializer=setup) as pool, SOURCE.open("rb") as stream, partial.open(mode) as output:
        records = islice(enumerate(ijson.items(stream, "item")), offset, None)
        if args.probe:
            records = islice(records, 96)
        while batch := list(islice(records, 96)):
            for status, rows in pool.map(prepare_trajectory, batch, chunksize=2):
                counts[status] += 1
                for row in rows:
                    assert row["env_id"] not in val_environments
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
                segments += len(rows)
            output.flush()
            print(sum(counts.values()), segments, dict(counts), flush=True)
    assert sum(counts.values()) == (96 if args.probe else 9022) and segments
    partial.replace(dest / "train.jsonl")
    setup()
    validation_tasks = 0
    with (dest / "val.jsonl").open("w") as stream:
        for row in val:
            row = clean({**row, "enable_thinking": True})
            ids, _, _ = tokenize_trajectory(tokenizer, row, template)
            if len(ids) + PREFIX_LENGTH <= CONTEXT:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                validation_tasks += 1
    assert validation_tasks > 0
    (dest / "manifest.json").write_text(json.dumps({"source_sha256": sha256, "trajectory_counts": counts,
                                                   "train_segments": segments, "validation_tasks": validation_tasks,
                                                   "over_context_validation_tasks": len(val) - validation_tasks},
                                                  indent=2) + "\n")


if __name__ == "__main__":
    main()
