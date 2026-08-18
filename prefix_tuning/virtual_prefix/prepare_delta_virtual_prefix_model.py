#!/usr/bin/env python3
"""Create a Qwen3.5 checkpoint with per-GDN-layer virtual-token tables."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file


ARCHITECTURE = "Qwen3_5DeltaVirtualPrefixForConditionalGeneration"
MODELING_FILE = "modeling_qwen3_5_delta_virtual_prefix.py"
PREFIX_WEIGHTS_FILE = "delta_virtual_prefix.safetensors"


def copy_checkpoint(source: Path, output: Path, copy_mode: str) -> None:
    output.mkdir(parents=True, exist_ok=False)
    for source_path in source.iterdir():
        target_path = output / source_path.name
        if source_path.is_dir():
            shutil.copytree(source_path, target_path, symlinks=True)
        elif copy_mode == "hardlink" and source_path.suffix == ".safetensors":
            try:
                os.link(source_path, target_path)
            except OSError:
                shutil.copy2(source_path, target_path)
        else:
            shutil.copy2(source_path, target_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--modeling-file", type=Path, required=True)
    parser.add_argument("--num-virtual-tokens", type=int, default=256)
    parser.add_argument("--copy-mode", choices=("hardlink", "copy"), default="hardlink")
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    modeling_file = args.modeling_file.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if not modeling_file.is_file():
        raise FileNotFoundError(modeling_file)
    if args.num_virtual_tokens <= 0:
        raise ValueError("--num-virtual-tokens must be positive")

    config = json.loads((source / "config.json").read_text())
    if config.get("model_type") != "qwen3_5":
        raise ValueError(f"Expected qwen3_5, got {config.get('model_type')!r}")
    text_config = config["text_config"]
    layer_types = text_config["layer_types"]
    hidden_size = int(text_config["hidden_size"])

    copy_checkpoint(source, output, args.copy_mode)
    shutil.copy2(modeling_file, output / MODELING_FILE)

    prefix_weights: dict[str, torch.Tensor] = {}
    linear_layer_indices: list[int] = []
    for layer_idx, layer_type in enumerate(layer_types):
        if layer_type != "linear_attention":
            continue
        linear_layer_indices.append(layer_idx)
        name = f"model.language_model.layers.{layer_idx}.linear_attn.prefix_tokens"
        prefix_weights[name] = torch.zeros(
            args.num_virtual_tokens,
            hidden_size,
            dtype=torch.bfloat16,
        )
    save_file(prefix_weights, output / PREFIX_WEIGHTS_FILE)

    index_path = output / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    for name in prefix_weights:
        if name in index["weight_map"]:
            raise RuntimeError(f"Duplicate tensor in source checkpoint: {name}")
        index["weight_map"][name] = PREFIX_WEIGHTS_FILE
    prefix_bytes = sum(tensor.numel() * tensor.element_size() for tensor in prefix_weights.values())
    index.setdefault("metadata", {})["total_size"] = int(
        index.get("metadata", {}).get("total_size", 0)
    ) + prefix_bytes
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n")

    text_config["delta_prefix_num_virtual_tokens"] = args.num_virtual_tokens
    config["architectures"] = [ARCHITECTURE]
    config["auto_map"] = {
        "AutoModelForImageTextToText": f"{MODELING_FILE[:-3]}.{ARCHITECTURE}",
    }
    config["delta_virtual_prefix"] = {
        "method": "per_layer_continuous_virtual_tokens",
        "linear_layer_indices": linear_layer_indices,
        "num_virtual_tokens": args.num_virtual_tokens,
        "token_width": hidden_size,
        "insertion": "gdn_mixer_input_before_user_tokens",
        "initialization": "zeros",
    }
    (output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    (output / "delta_virtual_prefix_config.json").write_text(
        json.dumps(config["delta_virtual_prefix"], ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "model_type": config["model_type"],
                "architecture": ARCHITECTURE,
                "linear_layers": len(linear_layer_indices),
                "num_virtual_tokens": args.num_virtual_tokens,
                "token_width": hidden_size,
                "prefix_parameters": sum(t.numel() for t in prefix_weights.values()),
                "prefix_weight_bytes": prefix_bytes,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
