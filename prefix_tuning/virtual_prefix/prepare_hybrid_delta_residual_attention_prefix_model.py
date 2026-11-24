#!/usr/bin/env python3
"""Build Qwen3.5 with Independent Delta and residual attention prefixes."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


ARCHITECTURE = "Qwen3_5HybridDeltaResidualAttentionPrefixForConditionalGeneration"
HYBRID_MODELING_FILE = "modeling_qwen3_5_hybrid_delta_residual_attention_prefix.py"
DELTA_MODELING_FILE = "modeling_qwen3_5_independent_delta_kv_prefix.py"
DELTA_WEIGHTS_FILE = "independent_delta_kv_prefix.safetensors"
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
    parser.add_argument("--joint-softmax", action="store_true")
    parser.add_argument("--attention-prefix-mode", choices=("joint", "anchored_additive", "token_null_additive"), default="joint")
    parser.add_argument("--gdn-prefix-mode", choices=("state_rows_joint_softmax", "anchored_additive", "token_null_additive"),
                        default="state_rows_joint_softmax")
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--copy-mode", choices=("hardlink", "copy"), default="hardlink")
    args = parser.parse_args()
    if args.attention_prefix_mode != "joint" and not args.joint_softmax:
        parser.error("Additive attention requires --joint-softmax")
    anchored_gdn = args.gdn_prefix_mode != "state_rows_joint_softmax"
    if anchored_gdn and not args.joint_softmax:
        parser.error("--gdn-prefix-mode anchored_additive requires --joint-softmax")

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
    for dependency in (DELTA_MODELING_FILE, "joint_prefix_attention.py"):
        shutil.copy2(hybrid_modeling_file.with_name(dependency), output / dependency)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    attention_weights: dict[str, torch.Tensor] = {}
    delta_weights: dict[str, torch.Tensor] = {}
    full_attention_layer_indices: list[int] = []
    delta_tokens = int(text_config["independent_delta_prefix_num_virtual_tokens"])
    attention_heads = int(text_config["num_attention_heads"])
    if args.joint_softmax:
        gdn_heads = int(text_config["linear_num_value_heads"])
        gdn_key_dim = int(text_config["linear_key_head_dim"])
        gdn_value_dim = int(text_config["linear_value_head_dim"])
    for layer_idx, layer_type in enumerate(layer_types):
        if layer_type != "full_attention":
            if args.joint_softmax and layer_type == "linear_attention":
                base = f"model.language_model.layers.{layer_idx}.linear_attn"
                delta_weights[f"{base}.prefix_state_key"] = torch.randn(
                    1, gdn_heads, delta_tokens, gdn_key_dim, generator=generator,
                ).mul_(args.key_init_std).to(torch.bfloat16)
                delta_weights[f"{base}.prefix_state_value"] = torch.zeros(
                    1, gdn_heads, delta_tokens, gdn_value_dim, dtype=torch.bfloat16,
                )
                delta_weights[f"{base}.prefix_state_temperature"] = torch.ones(gdn_heads)
                bias = -math.log(delta_tokens) - 8.0 if anchored_gdn else 0.0
                delta_weights[f"{base}.prefix_state_bias"] = torch.full((gdn_heads,), bias)
                if args.gdn_prefix_mode == "token_null_additive":
                    delta_weights[f"{base}.prefix_null_proj"] = torch.zeros(gdn_heads, hidden_size)
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
        if args.joint_softmax:
            initial_bias = -math.log(args.num_attention_prefix_tokens) - 8.0
            attention_weights[f"{base}.prefix_logit_bias"] = torch.full(
                (attention_heads,), initial_bias,
            )
            if args.attention_prefix_mode == "token_null_additive":
                attention_weights[f"{base}.prefix_null_proj"] = torch.zeros(attention_heads, hidden_size)
    save_file(attention_weights, output / ATTENTION_WEIGHTS_FILE)

    index_path = output / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    delta_metadata = config["independent_delta_kv_prefix"]
    delta_numel = int(delta_metadata["trainable_numel"])
    delta_bytes_change = 0
    if args.joint_softmax:
        legacy_weights = load_file(source / DELTA_WEIGHTS_FILE)
        mapped_names = {name for name, shard in index["weight_map"].items() if shard == DELTA_WEIGHTS_FILE}
        if mapped_names != set(legacy_weights):
            raise ValueError("Source Delta shard and weight map disagree")
        for name in mapped_names:
            del index["weight_map"][name]
        for name in delta_weights:
            if name in index["weight_map"]:
                raise ValueError(f"Duplicate joint GDN tensor: {name}")
            index["weight_map"][name] = DELTA_WEIGHTS_FILE
        delta_path = output / DELTA_WEIGHTS_FILE
        delta_path.unlink()
        save_file(delta_weights, delta_path)
        trained_delta = {name: tensor for name, tensor in delta_weights.items()
                         if not (anchored_gdn and name.endswith(".prefix_state_temperature"))}
        delta_numel = sum(tensor.numel() for tensor in trained_delta.values())
        delta_bytes_change = sum(t.numel() * t.element_size() for t in delta_weights.values()) - sum(
            t.numel() * t.element_size() for t in legacy_weights.values()
        )
        delta_metadata.update(
            method=args.gdn_prefix_mode,
            trainable_tensor_count=len(trained_delta),
            trainable_numel=delta_numel,
        )
        (output / "independent_delta_kv_prefix_config.json").write_text(
            json.dumps(delta_metadata, ensure_ascii=False, indent=2) + "\n"
        )
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
    ) + attention_bytes + delta_bytes_change
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n")

    text_config["residual_attention_prefix_num_virtual_tokens"] = (
        args.num_attention_prefix_tokens
    )
    text_config["residual_attention_prefix_key_init_std"] = args.key_init_std
    text_config["residual_attention_prefix_joint_softmax"] = args.joint_softmax
    text_config["residual_attention_prefix_mode"] = args.attention_prefix_mode
    if args.joint_softmax:
        text_config["independent_delta_prefix_method"] = args.gdn_prefix_mode
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
        "joint_softmax_with_user_attention": args.joint_softmax and args.attention_prefix_mode == "joint",
        "attention_prefix_mode": args.attention_prefix_mode,
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
