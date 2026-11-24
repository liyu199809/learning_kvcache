"""Tokenize full Qwen trajectories with loss restricted to selected assistant spans."""
from collections import defaultdict
import re

from prepare import lines

START = "\ue000"
END = "\ue001"
ASSISTANT = "{%- elif message.role == \"assistant\" %}"
REASONING_END = "+ '\\n</think>\\n\\n' + content }}"
HISTORY_REASONING = "loop.index0 > ns.last_query_index"


def marked_template(template):
    before, assistant = template.split(ASSISTANT, 1)
    assistant, after = assistant.split("{%- elif message.role == \"tool\" %}", 1)
    assistant = assistant.replace(
        "+ '\\n<think>\\n' + reasoning_content",
        "+ '\\n' + ('" + START + "' if message.loss and message.reasoning_content and "
        "not message.loss_reasoning_offset else '') + '<think>\\n' + "
        "reasoning_content[:message.loss_reasoning_offset|default(0)] + "
        "('" + START + "' if message.loss and message.loss_reasoning_offset else '') + "
        "reasoning_content[message.loss_reasoning_offset|default(0):]",
    ).replace(REASONING_END,
        "+ '\\n</think>\\n\\n' + ('" + START + "' if message.loss and "
        "not message.reasoning_content else '') + content }}")
    # Include the assistant's end-of-turn token and exclude the following separator.
    assistant = assistant.replace("'\u003c|im_end|>\\n'", "'<|im_end|>' + ('" + END
                                  + "' if message.loss else '') + '\\n'")
    if assistant.count(START) != 3 or assistant.count(END) != 1:
        raise ValueError("Unexpected Qwen chat template")
    return before + ASSISTANT + assistant + "{%- elif message.role == \"tool\" %}" + after


def trajectory_render_options(row, template):
    if not row.get("preserve_history_reasoning"):
        return template
    if template.count(HISTORY_REASONING) != 1:
        raise ValueError("Unexpected Qwen history reasoning condition")
    return template.replace(HISTORY_REASONING, HISTORY_REASONING + " or preserve_history_reasoning")


def tokenize_trajectory(tokenizer, row, template):
    if row.get("enable_thinking") is False:
        return tokenize_non_thinking(tokenizer, row)
    for message in row["messages"]:
        if "loss_reasoning_offset" in message:
            thought = message.get("reasoning_content", "").strip()
            offset = message["loss_reasoning_offset"]
            if not (message["role"] == "assistant" and message["loss"] and
                    type(offset) is int and 0 < offset < len(thought)):
                raise ValueError(f"Invalid reflection loss boundary in {row['id']}")
    kwargs = {"tools": row["tools"], "tokenize": False,
              "add_generation_prompt": False, "enable_thinking": True,
              "preserve_history_reasoning": row.get("preserve_history_reasoning", False)}
    marked = tokenizer.apply_chat_template(row["messages"],
        chat_template=trajectory_render_options(row, template), **kwargs)
    text = marked.replace(START, "").replace(END, "")
    if text != tokenizer.apply_chat_template(row["messages"],
            chat_template=trajectory_render_options(row, tokenizer.chat_template), **kwargs):
        raise ValueError(f"Loss markers altered the chat rendering for {row['id']}")
    spans = []
    offset = 0
    for match in re.finditer(f"{START}(.*?){END}", marked, re.DOTALL):
        spans.append((match.start() - offset, match.end() - offset - 2))
        offset += 2
    expected = sum(bool(msg["loss"]) for msg in row["messages"])
    if len(spans) != expected or not expected:
        raise ValueError(f"Missing supervised span in {row['id']}")
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    positions = []
    span_iter = iter(spans)
    current = next(span_iter, None)
    for pos, (start, end) in enumerate(encoded.offset_mapping):
        while current and start >= current[1]:
            current = next(span_iter, None)
        if current and start >= current[0] and end <= current[1] and end > start:
            positions.append(pos)
    if not positions:
        raise ValueError(f"No supervised tokens in {row['id']}")
    return encoded.input_ids, [encoded.input_ids[i] for i in positions], positions


def non_thinking_template(template, *, mark=False):
    before, assistant = template.split(ASSISTANT, 1)
    assistant, after = assistant.split('{%- elif message.role == "tool" %}', 1)
    assistant = assistant.replace(
        "{%- if loop.index0 > ns.last_query_index %}",
        "{%- if not enable_thinking and loop.index0 > ns.last_query_index %}\n"
        "{{- '<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' + "
        + ("('" + START + "' if message.loss else '') + " if mark else "") + "content }}\n"
        "{%- elif loop.index0 > ns.last_query_index %}", 1)
    if mark:
        assistant = assistant.replace("'<|im_end|>\\n'", "'<|im_end|>' + ('" + END
                                      + "' if message.loss else '') + '\\n'", 1)
    return before + ASSISTANT + assistant + '{%- elif message.role == "tool" %}' + after


def tokenize_non_thinking(tokenizer, row):
    template = non_thinking_template(tokenizer.chat_template, mark=True)
    messages = row["messages"]
    kwargs = {"tools": row["tools"], "tokenize": False, "enable_thinking": False}
    marked = tokenizer.apply_chat_template(messages, chat_template=template,
                                           add_generation_prompt=False, **kwargs)
    text = marked.replace(START, "").replace(END, "")
    spans = [(m.start() - 2*i, m.end() - 2*i - 2)
             for i, m in enumerate(re.finditer(f"{START}(.*?){END}", marked, re.DOTALL))]
    if len(spans) != sum(bool(msg["loss"]) for msg in messages) or not spans:
        raise ValueError(f"Missing non-thinking target in {row['id']}")
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    positions = []
    spans_iter = iter(spans)
    block = next(spans_iter, None)
    for i, (start, end) in enumerate(encoded.offset_mapping):
        while block and start >= block[1]:
            block = next(spans_iter, None)
        if block and block[0] <= start < end <= block[1]:
            positions.append(i)
    if not positions:
        raise ValueError(f"Empty non-thinking target in {row['id']}")
    return encoded.input_ids, [encoded.input_ids[i] for i in positions], positions


def tokenize_split(tokenizer, data_dir, split, context, prefix_length, include_c):
    template = marked_template(tokenizer.chat_template)
    stats = defaultdict(lambda: {"included": 0, "input_tokens": 0, "target_tokens": 0})
    examples = []
    for row in lines(data_dir / f"{split}.jsonl"):
        if split == "train" and row["type"] == "C" and not include_c:
            continue
        ids, targets, positions = tokenize_trajectory(tokenizer, row, template)
        if len(ids) + prefix_length > context:
            raise ValueError(f"Previously selected sample exceeds context: {row['id']}")
        example = (row["source"], ids, targets, positions)
        if "sample_weight" in row:
            weight = row["sample_weight"]
            if not 0 < weight <= 1:
                raise ValueError(f"Invalid sample weight: {row['id']}")
            example += (weight,)
        examples.append(example)
        stats[row["source"]]["included"] += 1
        stats[row["source"]]["input_tokens"] += len(ids)
        stats[row["source"]]["target_tokens"] += len(targets)
    return examples, dict(stats)
