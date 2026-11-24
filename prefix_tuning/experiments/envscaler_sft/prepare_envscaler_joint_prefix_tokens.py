"""Encode the same masked EnvScaler rows for the joint-softmax experiment."""
from concurrent.futures import ProcessPoolExecutor
import itertools
from pathlib import Path

import torch
from transformers import AutoTokenizer

from prepare import lines
from sft_trajectory_tokens import marked_template, tokenize_trajectory
from train_prefix import MODEL, write_json


def init_tokenizer():
    global tokenizer, template
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    template = marked_template(tokenizer.chat_template)


def encode(row):
    ids, targets, positions = tokenize_trajectory(tokenizer, row, template)
    if len(ids) + 32 > 16384:
        raise ValueError(f"Previously selected sample exceeds context: {row['id']}")
    result = (row["source"], ids, targets, positions)
    if "sample_weight" in row:
        weight = row["sample_weight"]
        if not 0 < weight <= 1:
            raise ValueError(f"Invalid sample weight: {row['id']}")
        result += (weight,)
    return result


def batches(rows, size):
    while batch := list(itertools.islice(rows, size)):
        yield batch


def main():
    root = Path(__file__).resolve().parent
    run = root / "envscaler-joint-prefix"
    data = root / "envscaler-soft-prefix" / "data"
    output = run / "encoded.pt"
    if output.exists():
        return
    splits, coverage = {}, {}
    with ProcessPoolExecutor(max_workers=24, initializer=init_tokenizer) as pool:
        for split, filename in (("validation", "val"), ("train", "train")):
            results = []
            for batch in batches(lines(data / f"{filename}.jsonl"), 1536):
                results.extend(pool.map(encode, batch, chunksize=16))
                print(split, len(results), flush=True)
            splits[split] = results
            coverage[split] = {"envscaler": {
                "included": len(results), "input_tokens": sum(len(r[1]) for r in results),
                "target_tokens": sum(len(r[2]) for r in results),
            }}
    old = root / "envscaler-soft-prefix/train/config.json"
    import json
    if coverage != json.loads(old.read_text())["coverage"]:
        raise ValueError(f"Coverage differs from the original run: {coverage}")
    tmp = output.with_suffix(".parallel.tmp")
    torch.save(splits, tmp)
    write_json(run / "coverage.json", coverage)
    tmp.replace(output)


if __name__ == "__main__":
    main()
