#!/usr/bin/env python3
"""Build a Qwen3.5 checkpoint with independent Delta K/V/beta/a prefixes."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file


ARCHITECTURE = "Qwen3_5IndependentDeltaKVPrefixForConditionalGeneration"
MODELING_FILE = "modeling_qwen3_5_independent_delta_kv_prefix.py"
PREFIX_WEIGHTS_FILE = "independent_delta_kv_prefix.safetensors"


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
    parser.add_argument("--num-virtual-tokens", type=int, default=2048)
    parser.add_argument("--prefix-v-init-std", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20260819)
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
    if args.prefix_v_init_std <= 0:
        raise ValueError("--prefix-v-init-std must be positive")

    config = json.loads((source / "config.json").read_text())
    if config.get("model_type") != "qwen3_5":
        raise ValueError(f"Expected qwen3_5, got {config.get('model_type')!r}")
    text_config = config["text_config"]
    layer_types = text_config["layer_types"]
    key_dim = int(text_config["linear_key_head_dim"]) * int(
        text_config["linear_num_key_heads"]
    )
    value_dim = int(text_config["linear_value_head_dim"]) * int(
        text_config["linear_num_value_heads"]
    )
    value_heads = int(text_config["linear_num_value_heads"])
    conv_kernel = int(text_config["linear_conv_kernel_dim"])

    copy_checkpoint(source, output, args.copy_mode)
    shutil.copy2(modeling_file, output / MODELING_FILE)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    prefix_weights: dict[str, torch.Tensor] = {}
    linear_layer_indices: list[int] = []
    for layer_idx, layer_type in enumerate(layer_types):
        if layer_type != "linear_attention":
            continue
        linear_layer_indices.append(layer_idx)
        base = f"model.language_model.layers.{layer_idx}.linear_attn"
        prefix_weights[f"{base}.prefix_k"] = torch.zeros(
            args.num_virtual_tokens, key_dim, dtype=torch.bfloat16
        )
        prefix_v = torch.randn(
            args.num_virtual_tokens,
            value_dim,
            generator=generator,
            dtype=torch.float32,
        ).mul_(args.prefix_v_init_std).to(torch.bfloat16)
        prefix_v[-(conv_kernel - 1) :].zero_()
        prefix_weights[f"{base}.prefix_v"] = prefix_v
        prefix_weights[f"{base}.prefix_beta_logits"] = torch.zeros(
            args.num_virtual_tokens, value_heads, dtype=torch.bfloat16
        )
        prefix_weights[f"{base}.prefix_a"] = torch.zeros(
            args.num_virtual_tokens, value_heads, dtype=torch.bfloat16
        )
    save_file(prefix_weights, output / PREFIX_WEIGHTS_FILE)

    index_path = output / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    for name in prefix_weights:
        if name in index["weight_map"]:
            raise RuntimeError(f"Duplicate tensor in source checkpoint: {name}")
        index["weight_map"][name] = PREFIX_WEIGHTS_FILE
    prefix_numel = sum(tensor.numel() for tensor in prefix_weights.values())
    prefix_bytes = sum(tensor.numel() * tensor.element_size() for tensor in prefix_weights.values())
    index.setdefault("metadata", {})["total_size"] = int(
        index.get("metadata", {}).get("total_size", 0)
    ) + prefix_bytes
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n")

    text_config["independent_delta_prefix_num_virtual_tokens"] = args.num_virtual_tokens
    text_config["independent_delta_prefix_v_init_std"] = args.prefix_v_init_std
    config["architectures"] = [ARCHITECTURE]
    config["auto_map"] = {
        "AutoModelForImageTextToText": f"{MODELING_FILE[:-3]}.{ARCHITECTURE}",
    }
    metadata = {
        "method": "independent_delta_k_v_beta_a",
        "linear_layer_indices": linear_layer_indices,
        "num_virtual_tokens": args.num_virtual_tokens,
        "key_dim": key_dim,
        "value_dim": value_dim,
        "value_heads": value_heads,
        "prefix_v_init_std": args.prefix_v_init_std,
        "initialization": {
            "prefix_k": "zeros",
            "prefix_v": f"normal_std_{args.prefix_v_init_std}",
            "prefix_v_final_conv_history_rows": "zeros",
            "prefix_beta_logits": "zeros_beta_0.5",
            "prefix_a": "zeros",
        },
        "conv_kernel_size": conv_kernel,
        "conv_history_length": conv_kernel - 1,
        "trainable_tensor_count": len(prefix_weights),
        "trainable_numel": prefix_numel,
        "base_model": str(source),
    }
    config["independent_delta_kv_prefix"] = metadata
    (output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    (output / "independent_delta_kv_prefix_config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "model_type": config["model_type"],
                "architecture": ARCHITECTURE,
                "linear_layers": len(linear_layer_indices),
                "num_virtual_tokens": args.num_virtual_tokens,
                "trainable_tensors": len(prefix_weights),
                "prefix_parameters": prefix_numel,
                "prefix_weight_bytes": prefix_bytes,
                "prefix_v_init_std": args.prefix_v_init_std,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
