#!/usr/bin/env python3
"""Export a verl trainable-only FSDP checkpoint as a deployable HF model."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch.distributed.tensor import DTensor, Replicate, Shard


PREFIX_WEIGHTS_FILE = "delta_virtual_prefix.safetensors"


def _copy_prepared_model(source: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    for source_path in source.iterdir():
        target_path = output / source_path.name
        if source_path.is_dir():
            shutil.copytree(source_path, target_path, symlinks=True)
        elif source_path.suffix == ".safetensors" and source_path.name != PREFIX_WEIGHTS_FILE:
            try:
                os.link(source_path, target_path)
            except OSError:
                shutil.copy2(source_path, target_path)
        elif source_path.name != PREFIX_WEIGHTS_FILE:
            shutil.copy2(source_path, target_path)


def _gather_tensor(shards: list[torch.Tensor]) -> torch.Tensor:
    first = shards[0]
    if not isinstance(first, DTensor):
        return first.detach().cpu()
    placements = first.placements
    if len(placements) != 1:
        raise ValueError(f"Only one-dimensional FSDP meshes are supported, got {placements}")
    placement = placements[0]
    if isinstance(placement, Replicate):
        return first.to_local().detach().cpu()
    if not isinstance(placement, Shard) or placement.dim != 0:
        raise ValueError(f"Only dim-0 FSDP shards are supported, got {placement}")
    full = torch.cat([tensor.to_local().detach().cpu() for tensor in shards], dim=0)
    return full.narrow(0, 0, first.shape[0]).reshape(first.shape)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prepared-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    prepared_model = args.prepared_model.resolve()
    output = args.output.resolve()
    metadata = json.loads((checkpoint / "trainable_only_meta.json").read_text())
    if metadata.get("checkpoint_type") != "trainable_only":
        raise ValueError("Expected a trainable-only verl checkpoint")
    world_size = int(metadata["world_size"])
    rank_states = [
        torch.load(
            checkpoint / f"model_world_size_{world_size}_rank_{rank}.pt",
            map_location="cpu",
            weights_only=False,
        )
        for rank in range(world_size)
    ]
    expected_names = set(metadata["parameters"])
    if any(set(state) != expected_names for state in rank_states):
        raise ValueError("Checkpoint rank shards do not contain identical trainable parameter names")
    if not all(name.endswith(".linear_attn.prefix_tokens") for name in expected_names):
        raise ValueError("Checkpoint contains non-prefix trainable parameters")

    prefix_weights = {
        name: _gather_tensor([state[name] for state in rank_states]).to(torch.bfloat16).contiguous()
        for name in sorted(expected_names)
    }
    for name, tensor in prefix_weights.items():
        expected_shape = metadata["parameters"][name]["shape"]
        if list(tensor.shape) != expected_shape:
            raise ValueError(f"Exported shape mismatch for {name}: {list(tensor.shape)} != {expected_shape}")

    prepared_config = json.loads((prepared_model / "config.json").read_text())
    configured_m = prepared_config["text_config"].get("delta_prefix_num_virtual_tokens")
    if configured_m != metadata.get("num_virtual_tokens"):
        raise ValueError(
            f"Prepared model M={configured_m} does not match checkpoint M={metadata.get('num_virtual_tokens')}"
        )

    _copy_prepared_model(prepared_model, output)
    save_file(prefix_weights, output / PREFIX_WEIGHTS_FILE)
    (output / "trainable_only_meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "prefix_tensors": len(prefix_weights),
                "prefix_numel": sum(tensor.numel() for tensor in prefix_weights.values()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
