#!/usr/bin/env python3
"""Build Qwen3.5 with Independent Delta and residual attention prefixes."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file


ARCHITECTURE = "Qwen3_5HybridDeltaResidualAttentionPrefixForConditionalGeneration"
HYBRID_MODELING_FILE = "modeling_qwen3_5_hybrid_delta_residual_attention_prefix.py"
DELTA_MODELING_FILE = "modeling_qwen3_5_independent_delta_kv_prefix.py"
ATTENTION_WEIGHTS_FILE = "residual_attention_prefix.safetensors"


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
    parser.add_argument("--hybrid-modeling-file", type=Path, required=True)
    parser.add_argument("--num-attention-prefix-tokens", type=int, default=256)
    parser.add_argument("--key-init-std", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--copy-mode", choices=("hardlink", "copy"), default="hardlink")
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    hybrid_modeling_file = args.hybrid_modeling_file.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if not hybrid_modeling_file.is_file():
        raise FileNotFoundError(hybrid_modeling_file)
    if not (source / DELTA_MODELING_FILE).is_file():
        raise FileNotFoundError(source / DELTA_MODELING_FILE)
    if args.num_attention_prefix_tokens <= 0:
        raise ValueError("--num-attention-prefix-tokens must be positive")
    if args.key_init_std <= 0:
        raise ValueError("--key-init-std must be positive")

    config = json.loads((source / "config.json").read_text())
    if config.get("model_type") != "qwen3_5":
        raise ValueError(f"Expected qwen3_5, got {config.get('model_type')!r}")
    text_config = config["text_config"]
    if int(text_config.get("independent_delta_prefix_num_virtual_tokens", 0)) <= 0:
        raise ValueError("Source must already contain the Independent Delta prefix")
    layer_types = text_config["layer_types"]
    hidden_size = int(text_config["hidden_size"])

    copy_checkpoint(source, output, args.copy_mode)
    shutil.copy2(hybrid_modeling_file, output / HYBRID_MODELING_FILE)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    attention_weights: dict[str, torch.Tensor] = {}
    full_attention_layer_indices: list[int] = []
    for layer_idx, layer_type in enumerate(layer_types):
        if layer_type != "full_attention":
            continue
        full_attention_layer_indices.append(layer_idx)
        base = f"model.language_model.layers.{layer_idx}.self_attn"
        key_tokens = torch.randn(
            args.num_attention_prefix_tokens,
            hidden_size,
            generator=generator,
            dtype=torch.float32,
        ).mul_(args.key_init_std).to(torch.bfloat16)
        attention_weights[f"{base}.prefix_key_tokens"] = key_tokens
        attention_weights[f"{base}.prefix_value_tokens"] = torch.zeros(
            args.num_attention_prefix_tokens,
            hidden_size,
            dtype=torch.bfloat16,
        )
    save_file(attention_weights, output / ATTENTION_WEIGHTS_FILE)

    index_path = output / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    for name in attention_weights:
        if name in index["weight_map"]:
            raise RuntimeError(f"Duplicate tensor in source checkpoint: {name}")
        index["weight_map"][name] = ATTENTION_WEIGHTS_FILE
    attention_numel = sum(tensor.numel() for tensor in attention_weights.values())
    attention_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in attention_weights.values()
    )
    index.setdefault("metadata", {})["total_size"] = int(
        index.get("metadata", {}).get("total_size", 0)
    ) + attention_bytes
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n")

    delta_metadata = config["independent_delta_kv_prefix"]
    delta_numel = int(delta_metadata["trainable_numel"])
    text_config["residual_attention_prefix_num_virtual_tokens"] = (
        args.num_attention_prefix_tokens
    )
    text_config["residual_attention_prefix_key_init_std"] = args.key_init_std
    metadata = {
        "method": "residual_soft_prefix_attention_independent_hidden_kv",
        "full_attention_layer_indices": full_attention_layer_indices,
        "num_virtual_tokens": args.num_attention_prefix_tokens,
        "hidden_size": hidden_size,
        "key_initialization": f"normal_std_{args.key_init_std}",
        "value_initialization": "zeros",
        "prefix_positions": [
            -args.num_attention_prefix_tokens,
            -1,
        ],
        "insertion": "before_native_query_gate_and_output_projection",
        "joint_softmax_with_user_attention": False,
        "uses_user_kv_cache": False,
        "trainable_tensor_count": len(attention_weights),
        "trainable_numel": attention_numel,
    }
    config["residual_attention_prefix"] = metadata
    config["hybrid_prefix"] = {
        "delta_prefix_num_virtual_tokens": text_config[
            "independent_delta_prefix_num_virtual_tokens"
        ],
        "attention_prefix_num_virtual_tokens": args.num_attention_prefix_tokens,
        "trainable_tensor_count": int(delta_metadata["trainable_tensor_count"])
        + len(attention_weights),
        "trainable_numel": delta_numel + attention_numel,
    }
    config["architectures"] = [ARCHITECTURE]
    config["auto_map"] = {
        "AutoModelForImageTextToText": f"{HYBRID_MODELING_FILE[:-3]}.{ARCHITECTURE}",
    }
    (output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    (output / "residual_attention_prefix_config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "architecture": ARCHITECTURE,
                "delta_prefix_tokens": text_config[
                    "independent_delta_prefix_num_virtual_tokens"
                ],
                "attention_prefix_tokens": args.num_attention_prefix_tokens,
                "attention_layers": len(full_attention_layer_indices),
                "attention_prefix_tensors": len(attention_weights),
                "attention_prefix_numel": attention_numel,
                "hybrid_trainable_tensors": config["hybrid_prefix"][
                    "trainable_tensor_count"
                ],
                "hybrid_trainable_numel": config["hybrid_prefix"]["trainable_numel"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
