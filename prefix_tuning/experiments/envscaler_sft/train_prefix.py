"""Fit a shared input-embedding soft prefix to the next expert experience."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers import AutoModelForImageTextToText, AutoTokenizer

from prepare import OUT, lines

MODEL = "/tmp/prefix-vllm-4b/base"
COACH = (
    "You are the expert coach. The conversation below is a task followed by the "
    "student's attempts and any previous coaching. Write the next concise correction "
    "using the evidence already visible. Give a concrete next step and leave the "
    "answer for the student to obtain and verify."
)


def encode_history(tokenizer, messages):
    history = [dict(message) for message in messages]
    history[0]["content"] = COACH + "\n\nOriginal student instructions:\n" + history[0]["content"]
    # Keep reasoning from historical attempts across prior coaching boundaries.
    template = tokenizer.chat_template.replace(
        "{%- if loop.index0 > ns.last_query_index %}",
        "{%- if true %}",
    )
    context = tokenizer.apply_chat_template(
        history, chat_template=template, tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )
    return tokenizer.encode(context, add_special_tokens=False)


def tokenize(tokenizer, split, max_length, prefix_length, data_dir=OUT, objective="next_expert_turn"):
    encoded = []
    stats = Counter()
    for row in lines(data_dir / f"{split}.jsonl"):
        source = row["source"]
        if objective == "direct_code":
            prompt_text = tokenizer.apply_chat_template(
                row["history"], tokenize=False, add_generation_prompt=True, enable_thinking=True,
            )
            if not prompt_text.rstrip().endswith("<think>"):
                raise ValueError("Expected the coding chat template to open the thinking block")
            prompt = tokenizer.encode(prompt_text, add_special_tokens=False)
            target_text = "</think>\n\n" + row["target"]
        else:
            prompt = encode_history(tokenizer, row["history"])
            target_text = row["target"]
        target = tokenizer.encode(target_text, add_special_tokens=False) + [tokenizer.eos_token_id]
        if len(prompt) + len(target) + prefix_length > max_length:
            stats[source, "over_context"] += 1
            continue
        encoded.append((source, prompt, target))
        stats[source, "included"] += 1
        stats[source, "tokens"] += len(prompt) + len(target)
    by_source = defaultdict(dict)
    for (source, metric), count in stats.items():
        by_source[source][metric] = count
    return encoded, by_source


def loss_sum(model, row, prefix):
    _, prompt, target, *positions_in_trajectory = row
    device = model.device
    ids = torch.tensor([prompt if positions_in_trajectory else prompt + target], device=device)
    embedding = model.get_input_embeddings()(ids)
    offset = 0
    if prefix is not None:
        offset = prefix.shape[0]
        embedding = torch.cat((prefix.to(embedding.dtype).unsqueeze(0), embedding), dim=1)
    # Position p-1 predicts token p. No logits or loss for prompt tokens.
    if positions_in_trajectory:
        target_positions = torch.tensor(positions_in_trajectory[0], device=device)
        positions = offset + target_positions - 1
        labels = ids[0, target_positions]
    else:
        first = offset + len(prompt) - 1
        positions = torch.arange(first, first + len(target), device=device)
        labels = ids[0, -len(target):]
    logits = model(inputs_embeds=embedding, logits_to_keep=positions, use_cache=False).logits[0]
    return F.cross_entropy(logits.float(), labels, reduction="sum")


@torch.no_grad()
def evaluate(model, rows, prefix):
    sums = Counter()
    tokens = Counter()
    counts = Counter()
    for row in rows:
        source, _, target = row
        sums[source] += loss_sum(model, row, prefix).item()
        tokens[source] += len(target)
        counts[source] += 1
    result = {source: {"rows": counts[source], "tokens": tokens[source],
                       "nll": sums[source] / tokens[source]} for source in counts}
    result["combined"] = {"rows": sum(counts.values()), "tokens": sum(tokens.values()),
                          "nll": sum(sums.values()) / sum(tokens.values())}
    return result


def write_json(path, value):
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    pending.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--prefix-length", type=int, default=32)
    parser.add_argument("--limit", type=int, help="Train on a stratified smoke sample.")
    parser.add_argument("--eval-limit", type=int, help="Evaluate a stratified smoke sample.")
    parser.add_argument("--tokenize-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20261123)
    torch.manual_seed(20261123)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    train, train_stats = tokenize(tokenizer, "train", args.context, args.prefix_length)
    validation, validation_stats = tokenize(tokenizer, "validation", args.context, args.prefix_length)
    coverage = {"train": train_stats, "validation": validation_stats, "context": args.context,
                "prefix_length": args.prefix_length}
    write_json(args.output / "coverage.json", coverage)
    print("coverage", json.dumps(coverage), flush=True)
    if args.tokenize_only:
        return

    def sample(rows, limit):
        rng.shuffle(rows)
        if limit is None:
            return rows
        counts = Counter()
        selected = []
        for row in rows:
            source = row[0]
            if counts[source] < limit:
                selected.append(row)
                counts[source] += 1
        return selected

    train = sample(train, args.limit)
    validation = sample(validation, args.eval_limit)
    if not train or not validation:
        raise ValueError("Train and validation both need at least one row")
    started = time.monotonic()
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to("cuda")
    model.requires_grad_(False).eval()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    vocab = model.get_input_embeddings().weight
    initial_tokens = torch.tensor(tokenizer.encode(COACH, add_special_tokens=False)[:args.prefix_length], device="cuda")
    if len(initial_tokens) != args.prefix_length:
        raise ValueError("Prefix length exceeds the initialization text")
    prefix = torch.nn.Parameter(vocab[initial_tokens].detach().float().clone())
    initial = prefix.detach().clone()
    optimizer = torch.optim.AdamW([prefix], lr=0.003, weight_decay=0.0)
    print("trainable", prefix.numel(), "train rows", len(train), "validation rows", len(validation), flush=True)
    baseline = evaluate(model, validation, None)
    before = evaluate(model, validation, initial)
    print("baseline", baseline, flush=True)
    print("initial", before, flush=True)
    write_json(args.output / "baseline.json", {"no_prefix": baseline, "initial_prefix": before})
    batch_size = 8
    model.train()
    for step, start in enumerate(range(0, len(train), batch_size), 1):
        batch = train[start:start + batch_size]
        num_tokens = sum(len(row[2]) for row in batch)
        observed = 0.0
        for row in batch:
            loss = loss_sum(model, row, prefix)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at step {step}")
            (loss / num_tokens).backward()
            observed += loss.detach().item()
        norm = torch.nn.utils.clip_grad_norm_([prefix], 1.0, error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        print(f"step {step} rows {min(start + batch_size, len(train))}/{len(train)} "
              f"nll {observed / num_tokens:.4f} grad_norm {norm.item():.5f} "
              f"seconds {time.monotonic() - started:.1f}", flush=True)
        if step == 1 or step % 10 == 0:
            save_file({"soft_prefix": prefix.detach().cpu()}, args.output / "prefix-in-progress.safetensors")
    model.eval()
    after = evaluate(model, validation, prefix)
    delta = (prefix.detach() - initial).norm().item()
    assert delta > 0 and all(parameter.grad is None for parameter in model.parameters())
    save_file({"soft_prefix": prefix.detach().cpu()}, args.output / "soft-prefix.safetensors",
              metadata={"base_model": MODEL, "format": "input_embedding_prefix", "objective": "next_expert_turn"})
    write_json(args.output / "result.json", {
        "train_rows": len(train), "optimizer_steps": step, "prefix_tokens": args.prefix_length,
        "trainable_parameters": prefix.numel(), "prefix_change_l2": delta, "context": args.context,
        "no_prefix": baseline, "initial_prefix": before, "trained_prefix": after,
        "wall_seconds": round(time.monotonic() - started, 1),
    })
    print("validation", after, flush=True)


if __name__ == "__main__":
    main()
