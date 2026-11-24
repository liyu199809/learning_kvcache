"""Build the grouped 720/80 split specified by SFT-DECISION.md."""
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

from transformers import AutoTokenizer
from audit_sft_schemas import ARCHIVE, CAPTURE, EPISODE, canonical_tools, digest, schema_index
from prepare import APPROVED, OFFENSIVE, OUT, SOURCE, dump, lines

RESULT = OUT / "sft-800"
MODEL = "/tmp/prefix-vllm-4b/base"
CODING = Path("/host-ssd/learning-kvcache-agent-world-pr/benchmark/coding")
FEEDBACK = re.compile(
    r"\b(?:feedback|teacher|coach|hint|previous attempt|my (?:previous|earlier) (?:answer|solution)|"
    r"user (?:said|says|suggests|pointed|points|corrected|indicated))\b", re.I,
)
EXAMPLES = re.compile(r"(?im)^\s*(?:sample (?:input|output)|examples?|样例)\b")


def phrases(question):
    start = EXAMPLES.search(question)
    words = re.findall(r"\w+", question[:start.start() if start else None].casefold())
    size = min(4, len(words))
    return frozenset(tuple(words[i:i + size]) for i in range(len(words) - size + 1)) if size else frozenset()


class Similarity:
    def __init__(self):
        self.entries = {}
        self.index = defaultdict(list)

    def add(self, key, values):
        self.entries[key] = values
        for item in values:
            self.index[item].append(key)

    def matches(self, values, threshold):
        overlaps = Counter(key for item in values for key in self.index.get(item, ()))
        return sorted(key for key, shared in overlaps.items()
                      if shared >= threshold * (len(values) + len(self.entries[key]) - shared))


def benchmark_index():
    index = Similarity()
    proc = subprocess.run([str(CODING / ".venv/bin/python"), "-c", """
import json
from argparse import Namespace
from data import load_tasks
for name in ('humaneval+', 'mbpp+', 'livecodebench'):
    tasks, _ = load_tasks(Namespace(benchmark=name, lcb_release='v6', lcb_start_date=None,
                                    lcb_end_date=None, lcb_data_dir=None))
    for task in tasks:
        print(json.dumps([name, task['task_id'], task['prompt']]))
"""], cwd=CODING, capture_output=True, text=True,
        env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"})
    if proc.returncode:
        raise RuntimeError(f"Could not load benchmark questions: {proc.stderr}")
    for line in proc.stdout.splitlines():
        if not line.startswith("["):
            continue
        name, task_id, prompt = json.loads(line)
        question = re.sub(r"^### Question:\n|^Complete the following Python task[^\n]+\n\n", "", prompt)
        question = question.split("\n\nReturn the complete solution", 1)[0]
        index.add(f"{name}:{task_id}", phrases(question))
    return index


def normalize(messages, loss, empty_thought=False, *, merge_assistant=True):
    clean = []
    for message in messages:
        role = message["role"]
        row = {"role": role, "content": message.get("content") or "",
               "loss": loss and role == "assistant"}
        if role == "assistant":
            row["reasoning_content"] = "" if empty_thought else message.get("reasoning_content") or ""
            if calls := message.get("tool_calls"):
                row["tool_calls"] = []
                for call in calls:
                    fn = call["function"]
                    arguments = fn["arguments"]
                    row["tool_calls"].append({"id": call.get("id"), "type": "function", "function": {
                        "name": fn["name"].removeprefix("mcp__episode__"),
                        "arguments": json.loads(arguments) if isinstance(arguments, str) else arguments,
                    }})
        if role == "tool":
            row.update(name=(message.get("name") or "").removeprefix("mcp__episode__"),
                       tool_call_id=message.get("tool_call_id"))
        if merge_assistant and role == "assistant" and clean and clean[-1]["role"] == role:
            old = clean[-1]
            for key in ("content", "reasoning_content"):
                old[key] = "\n".join(filter(None, (old[key], row[key])))
            if row.get("tool_calls"):
                old.setdefault("tool_calls", []).extend(row["tool_calls"])
        else:
            clean.append(row)
    return clean


def verify_size(tokenizer, row):
    rendered = tokenizer.apply_chat_template(row["messages"], tools=row["tools"],
        tokenize=False, add_generation_prompt=False, enable_thinking=True)
    return len(tokenizer.encode(rendered, add_special_tokens=False)) + 32


def add_candidate(pools, rejected, tokenizer, row):
    row["tokens"] = verify_size(tokenizer, row)
    if row["tokens"] > 32768:
        rejected.append({"id": row["id"], "source": row["source"], "reason": "over_context"})
    else:
        pools[row["source"]].append(row)


def row_for(source, ident, environment, row_type, messages, tools, source_line, reason, schema=None):
    return {"source": source, "id": ident, "split_key": environment, "type": row_type,
            "messages": messages, "tools": tools, "source_line": source_line,
            "reason": reason, **(schema or {})}


def build(tokenizer, rejected):
    pool = defaultdict(list)
    audit = {(r["source"], r["source_line"]): r for r in lines(OUT / "sft-schema-audit.jsonl")}
    excluded_environments = {(r["source"], r["environment"]) for r in audit.values()
                             if r["status"] == "tau2_domain"}
    archived_schemas = schema_index()
    contamination = benchmark_index()
    for number, row in enumerate(lines(APPROVED), 1):
        checked = audit["agent_world", number]
        if checked["status"] != "eligible_before_length_and_dedup":
            continue
        history = row["prompt"][:2] + [msg for msg in row["prompt"][2:] if msg["role"] != "user"]
        messages = normalize(history, False) + normalize(row["continuation"], True, empty_thought=True)
        task = f"aw:{row['scenario']}:{row['task_idx']}"
        tools = canonical_tools(row["tools"])
        entry = row_for("agent_world", f"{task}:r{row['round']}:{number}", row["scenario"], "C",
                        messages, tools, number, row["correction_type"],
                        {k: checked[k] for k in ("schema_source", "schema_sha256")})
        entry.update(task_id=task, score_gain=row["after_score"] - row["before_score"])
        add_candidate(pool, rejected, tokenizer, entry)

    for number, row in enumerate(lines(SOURCE), 1):
        meta = row["task_source"]
        source = meta["dataset"]
        if meta["status"] != "scored" or not meta["is_canonical"]:
            continue
        env, idx = EPISODE.fullmatch(meta["trajectory"].rsplit("/", 1)[-1]).groups()
        checked = audit.get((source, number), {})
        turns = [i for i, msg in enumerate(row["messages"]) if msg["role"] == "user"]
        bounds = turns + [len(row["messages"])]
        rewards = row["rounds_rewards"]
        if len(turns) != len(rewards) or not turns or 1 not in rewards:
            continue
        if source == "taco":
            tools, schema = [], {}
            attempt = rewards.index(1)
            response = row["messages"][bounds[attempt] + 1:bounds[attempt + 1]]
            if len(response) != 1 or response[0]["role"] != "assistant":
                continue
            reasoning = response[0].get("reasoning_content") or ""
            if not reasoning or (attempt and FEEDBACK.search(reasoning)):
                continue
            if OFFENSIVE.search(json.dumps(row["messages"][:2] + response, ensure_ascii=False)):
                rejected.append({"source": source, "source_line": number, "reason": "safety_screen"})
                continue
            question = row["messages"][1]["content"]
            if matches := contamination.matches(phrases(question), .7):
                rejected.append({"source": source, "source_line": number, "reason": "benchmark_overlap",
                                 "benchmark_ids": matches})
                continue
            env = hashlib.sha256(" ".join(question.casefold().split()).encode()).hexdigest()
            rank = "rescued_without_feedback" if attempt else "first_pass"
        else:
            if checked.get("status") != "eligible_before_length_and_dedup":
                continue
            if (source, env) in excluded_environments:
                rejected.append({"source": source, "source_line": number, "reason": "tau2_domain"})
                continue
            capture = CAPTURE / meta["trajectory"] / "tools.json"
            tools = canonical_tools(json.loads(capture.read_text())) if capture.is_file() else (
                archived_schemas[source, env, int(idx)]
            )
            assert digest(tools) == checked["schema_sha256"]
            schema = {k: checked[k] for k in ("schema_source", "schema_sha256")}
            response = row["messages"][2:bounds[1]]
            rank = "first_pass"
        ident = f"{source}:{env}:{idx}"
        entry = row_for(source, ident, env, "D", normalize(row["messages"][:2], False)
                        + normalize(response, True), tools, number, rank, schema)
        entry.update(task_id=ident, trajectory=meta["trajectory"])
        if source == "taco":
            entry["reasoning_characters"] = len(reasoning)
        add_candidate(pool, rejected, tokenizer, entry)
    return pool


def choose(pools, rejected):
    chosen = []
    seen = Similarity()
    seen_by_source = defaultdict(Similarity)
    sizes = Counter()
    task_rows = defaultdict(list)
    aw_readback = 0
    for source in ("agent_world", "taco", "envscaler", "awm"):
        pool = pools[source]
        if source == "agent_world":
            non_readback = {row["task_id"] for row in pool if row["reason"] != "verify_readback"}
            pool.sort(key=lambda row: (row["reason"] == "verify_readback", row["task_id"] in non_readback,
                                       -row["score_gain"], row["tokens"], row["id"]))
        else:
            pool.sort(key=lambda row: (row["reason"] != "rescued_without_feedback",
                                       row.get("reasoning_characters", 0) > 8000, row["tokens"], row["id"]))
        count = Counter()
        rescued = 0
        # Give TACO its share of the Agent-World gap. EnvScaler stays capped;
        # section 6 sends its unfilled allocation to AWM.
        taco_target = 253 + round(max(0, 167 - sizes["agent_world"]) * .4)
        target = {"taco": taco_target, "awm": 800 - sum(sizes.values())}.get(source)
        for row in pool:
            task = row["task_id"]
            previously = task_rows[task]
            reason = None
            if target is not None and sizes[source] >= target:
                reason = "quota"
            elif source == "taco" and row["reason"] == "rescued_without_feedback" and rescued >= 151:
                reason = "quota"
            elif source == "awm" and count[row["split_key"]] >= 1:
                reason = "quota"
            elif source == "envscaler" and count[row["split_key"]] >= 4:
                reason = "quota"
            elif source == "agent_world" and (len(previously) >= 2 or any(
                prev["reason"] == row["reason"] for prev in previously
            )):
                reason = "quota"
            elif source == "agent_world" and row["reason"] == "verify_readback" and (
                aw_readback + 1 > .35 * (sum(count.values()) + 1)
            ):
                reason = "quota"
            values = phrases(row["messages"][1]["content"])
            if reason is None and not previously:
                if any(not key.startswith(source + "/") for key in seen.matches(values, .8)):
                    reason = "cross_source_duplicate"
                elif source != "agent_world" and seen_by_source[source].matches(values, .999):
                    reason = "duplicate"
            if reason:
                rejected.append({"id": row["id"], "source": source, "reason": reason})
                continue
            if not previously:
                sizes[source] += 1
                seen.add(source + "/" + task, values)
                seen_by_source[source].add(task, values)
            task_rows[task].append(row)
            count[row["split_key"]] += 1
            rescued += int(row["reason"] == "rescued_without_feedback")
            aw_readback += int(source == "agent_world" and row["reason"] == "verify_readback")
            chosen.append(row)
    return chosen, sizes


def split(rows, sizes):
    groups = defaultdict(lambda: defaultdict(set))
    taco_index = Similarity()
    parents = {}
    def root(key):
        while key != parents[key]:
            key = parents[key]
        return key
    for row in rows:
        if row["source"] == "taco":
            key = row["split_key"]
            parents[key] = key
            values = phrases(row["messages"][1]["content"])
            for match in taco_index.matches(values, .7):
                parents[root(match)] = root(key)
            taco_index.add(key, values)
    for row in rows:
        if row["source"] == "taco":
            row["split_key"] = root(row["split_key"])
        groups[row["source"]][row["split_key"]].add(row["task_id"])
    targets = {source: count // 10 for source, count in sizes.items()}
    for source in sorted(sizes, key=lambda source: (-(sizes[source] % 10), source))[:80 - sum(targets.values())]:
        targets[source] += 1
    validation = {}
    for source, count in targets.items():
        options = {0: []}
        for key in sorted(groups[source], key=lambda k: hashlib.sha256(k.encode()).digest()):
            size = len(groups[source][key])
            for n, chosen in list(options.items())[::-1]:
                if n + size <= count and n + size not in options:
                    options[n + size] = [*chosen, key]
        if count not in options:
            raise ValueError(f"No grouped {source} validation split of {count} questions")
        validation[source] = set(options[count])
    for row in rows:
        row["split"] = "val" if row["split_key"] in validation[row["source"]] else "train"
    assert len({r["task_id"] for r in rows if r["split"] == "val"}) == 80
    return targets


def main():
    RESULT.mkdir(exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    rejected = []
    pools = build(tokenizer, rejected)
    chosen, sizes = choose(pools, rejected)
    dump(RESULT / "rejected.jsonl", [*rejected, *(r for r in lines(OUT / "sft-schema-audit.jsonl")
                                               if r["status"] != "eligible_before_length_and_dedup")])
    print("Available samples:", {k: len(v) for k, v in pools.items()}, "Selected questions:", sizes, flush=True)
    if sizes["taco"] != 253 + round(max(0, 167 - sizes["agent_world"]) * .4) or sum(sizes.values()) != 800:
        raise ValueError("The authorized sources do not fill the required quotas")
    val_sizes = split(chosen, sizes)
    by_id = defaultdict(list)
    for row in chosen:
        by_id[row["task_id"]].append(row)
    dump(RESULT / "manifest-800.jsonl", ({k: group[0][k] for k in
         ("source", "task_id", "split_key", "split", "type")} | {"samples": [
         {k: v for k, v in r.items() if k not in ("messages", "tools")} for r in group]}
         for group in by_id.values()))
    for name in ("train", "val"):
        dump(RESULT / f"{name}.jsonl", (r for r in chosen if r["split"] == name))
    summary = {"questions": dict(sizes), "validation_questions": val_sizes,
               "sample_counts": dict(Counter(f"{r['split']}/{r['source']}" for r in chosen))}
    (RESULT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
