"""Prepare the Decision 6 datasets without changing their task cohorts."""

import hashlib
import json
import re

from safetensors.torch import load_file
from transformers import AutoTokenizer

from audit_sft_schemas import parameters_match
from prepare import OUT, lines
from sft_trajectory_tokens import marked_template, non_thinking_template, tokenize_trajectory
from train_prefix import MODEL
from train_prefix_distributed import PREFIX_LENGTH

ROOT = OUT / "sft-decision6"
MATCHED = OUT / "sft-decision5/matched-c"
E1 = OUT / "sft-800/eval-decision-3"
INITIAL = E1 / "initial.safetensors"
ALIAS = re.compile(r"\bmcp__episode_?\b")


def clean(row):
    for message in row["messages"]:
        for field in ("content", "reasoning_content"):
            value = message.get(field)
            if isinstance(value, str):
                message[field] = ALIAS.sub("the tool", value.replace("mcp__episode__", ""))
    if "mcp__" in json.dumps(row, ensure_ascii=False):
        raise ValueError(f"Unexpected MCP prefix in {row['id']}")
    return row


def check_render(row, tokenizer, marked):
    messages, tools = row["messages"], row["tools"]
    assert parameters_match(messages, tools) == (True, False), row["id"]
    assert all(not message["loss"] or message["role"] == "assistant" for message in messages), row["id"]
    thinking = row["enable_thinking"]
    first = next(i for i, message in enumerate(messages) if message["loss"])
    template = tokenizer.chat_template if thinking else non_thinking_template(tokenizer.chat_template)
    kwargs = dict(tools=tools, chat_template=template, tokenize=False, enable_thinking=thinking)
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=False, **kwargs)
    prompt = tokenizer.apply_chat_template(messages[:first], add_generation_prompt=True, **kwargs)
    assert text.startswith(prompt), row["id"]
    ids, targets, positions = tokenize_trajectory(tokenizer, row, marked)
    assert len(ids) + PREFIX_LENGTH <= 16384, row["id"]
    assert tokenizer.decode(targets, clean_up_tokenization_spaces=False).count("<|im_end|>") == sum(
        message["loss"] for message in messages
    ), row["id"]
    assert all(position < len(ids) for position in positions), row["id"]


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    with temp.open("w") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp.replace(path)


def prepare():
    ROOT.mkdir(exist_ok=True)
    prefix = load_file(INITIAL)["soft_prefix"]
    distances = {}
    for rate in ("1e-4", "3e-4"):
        folder = E1 / f"decision4-e1-{rate}"
        checkpoint = load_file(folder / "quarter-4-soft-prefix.safetensors")["soft_prefix"]
        measured = (checkpoint - prefix).norm(dim=1).mean().item()
        recorded = json.loads((folder / "quarter-4-geometry.json").read_text())["mean_l2_from_initial"]
        assert abs(measured - recorded) < 1e-7
        distances[rate] = measured
    (ROOT / "initializer.json").write_text(json.dumps({
        "path": str(INITIAL), "sha256": hashlib.sha256(INITIAL.read_bytes()).hexdigest(),
        "text_source": "train_prefix.COACH", "verified_e1_mean_l2": distances,
    }, indent=2) + "\n")
    old = OUT / "sft-800"
    for split in ("train", "val"):
        rows = [clean(row) for row in lines(old / f"{split}.jsonl") if row["type"] == "D" or split == "val"]
        assert len(rows) == (616 if split == "train" else 80)
        write(ROOT / "t2-data" / f"{split}.jsonl", rows)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    marked = marked_template(tokenizer.chat_template)
    report = {}
    for split, expected in (("train", 667), ("val", 31)):
        cohorts = []
        for arm in ("c-arm", "official-arm"):
            rows = list(lines(MATCHED / arm / f"{split}.jsonl"))
            for row in rows:
                clean(row)
                if arm == "c-arm" and row["source"] in {"agent_world", "taco"}:
                    row["enable_thinking"] = True
                assert isinstance(row["enable_thinking"], bool), row["id"]
                if row["source"] == "agent_world" or (arm == "c-arm" and row["source"] == "taco"):
                    check_render(row, tokenizer, marked)
            write(ROOT / "t3-data" / arm / f"{split}.jsonl", rows)
            cohorts.append({row["task_id"] for row in rows})
        assert cohorts[0] == cohorts[1] and len(cohorts[0]) == expected
        report[split] = expected
    (ROOT / "preflight.json").write_text(json.dumps({"paired_task_counts": report, "mcp_prefixes": 0}, indent=2) + "\n")


if __name__ == "__main__":
    prepare()
