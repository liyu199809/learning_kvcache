#!/usr/bin/env python3
"""Export a verl hybrid-prefix checkpoint as a deployable HF model tree."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

try:
    from prefix_tuning.virtual_prefix.merge_prefix_checkpoint import (
        _gather_tensor,
        _read_json,
        _validate_rank_state,
    )
except ModuleNotFoundError:
    from merge_prefix_checkpoint import _gather_tensor, _read_json, _validate_rank_state


DELTA_FILE = "independent_delta_kv_prefix.safetensors"
ATTENTION_FILE = "residual_attention_prefix.safetensors"
DELTA_SUFFIXES = (".prefix_k", ".prefix_v", ".prefix_beta_logits", ".prefix_a")
ATTENTION_SUFFIXES = (".prefix_key_tokens", ".prefix_value_tokens")


def _safetensor_shapes(path: Path) -> dict[str, list[int]]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        return {name: list(handle.get_slice(name).get_shape()) for name in handle.keys()}


def _copy_prepared(source: Path, output: Path, copy_mode: str) -> None:
    excluded = {
        DELTA_FILE,
        ATTENTION_FILE,
        "trainable_only_meta.json",
        "prefix_merge_manifest.json",
    }
    for source_path in source.iterdir():
        if source_path.name in excluded:
            continue
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


def export(checkpoint: Path, prepared: Path, output: Path, dtype, copy_mode: str):
    checkpoint, prepared, output = checkpoint.resolve(), prepared.resolve(), output.resolve()
    metadata = _read_json(checkpoint / "trainable_only_meta.json")
    if metadata.get("checkpoint_type") != "trainable_only":
        raise ValueError("Expected a trainable-only checkpoint")
    if metadata.get("trainable_type") != "hybrid_delta_residual_attention_prefix":
        raise ValueError(f"Unexpected trainable_type: {metadata.get('trainable_type')!r}")
    parameters = metadata.get("parameters")
    names = metadata.get("parameter_names")
    if not isinstance(parameters, dict) or not isinstance(names, list) or set(names) != set(parameters):
        raise ValueError("Invalid checkpoint parameter metadata")
    delta_names = [name for name in names if name.endswith(DELTA_SUFFIXES)]
    attention_names = [name for name in names if name.endswith(ATTENTION_SUFFIXES)]
    if len(delta_names) != 96 or len(attention_names) != 16:
        raise ValueError(
            f"Expected 96 Delta and 16 attention tensors, got {len(delta_names)} and {len(attention_names)}"
        )
    if len(delta_names) + len(attention_names) != len(names):
        raise ValueError("Checkpoint contains unsupported trainable parameters")
    if metadata.get("num_virtual_tokens") != 2048:
        raise ValueError("Checkpoint Delta prefix length is not 2048")
    if metadata.get("attention_num_virtual_tokens") != 256:
        raise ValueError("Checkpoint attention prefix length is not 256")

    config = _read_json(prepared / "config.json")
    if config.get("architectures") != [
        "Qwen3_5HybridDeltaResidualAttentionPrefixForConditionalGeneration"
    ]:
        raise ValueError("Prepared model is not the hybrid prefix architecture")
    delta_shapes = _safetensor_shapes(prepared / DELTA_FILE)
    attention_shapes = _safetensor_shapes(prepared / ATTENTION_FILE)
    expected_shapes = {**delta_shapes, **attention_shapes}
    checkpoint_shapes = {name: parameters[name]["shape"] for name in names}
    if checkpoint_shapes != expected_shapes:
        raise ValueError("Checkpoint tensor names/shapes do not match the prepared model")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")

    world_size = metadata.get("world_size")
    if not isinstance(world_size, int) or world_size <= 0:
        raise ValueError(f"Invalid world_size: {world_size!r}")
    rank_states = []
    for rank in range(world_size):
        path = checkpoint / f"model_world_size_{world_size}_rank_{rank}.pt"
        state = torch.load(path, map_location="cpu", weights_only=False)
        _validate_rank_state(state, names, parameters, rank)
        rank_states.append(state)

    merged = {
        name: _gather_tensor([state[name] for state in rank_states], name).to(dtype).contiguous()
        for name in names
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        _copy_prepared(prepared, staging, copy_mode)
        save_file({name: merged[name] for name in delta_names}, staging / DELTA_FILE)
        save_file({name: merged[name] for name in attention_names}, staging / ATTENTION_FILE)
        index_path = staging / "model.safetensors.index.json"
        index = _read_json(index_path)
        for name in delta_names:
            if index["weight_map"].get(name) != DELTA_FILE:
                raise ValueError(f"Bad prepared weight-map entry for {name}")
        for name in attention_names:
            if index["weight_map"].get(name) != ATTENTION_FILE:
                raise ValueError(f"Bad prepared weight-map entry for {name}")
        index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
        (staging / "trainable_only_meta.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        manifest = {
            "format_version": 1,
            "trainable_type": "hybrid_delta_residual_attention_prefix",
            "source_checkpoint": str(checkpoint),
            "prepared_model": str(prepared),
            "world_size": world_size,
            "output_dtype": str(dtype),
            "delta_prefix_tensors": len(delta_names),
            "attention_prefix_tensors": len(attention_names),
            "trainable_numel": sum(t.numel() for t in merged.values()),
        }
        (staging / "prefix_merge_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {"output": str(output), **manifest}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prepared-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--copy-mode", choices=("hardlink", "copy"), default="hardlink")
    args = parser.parse_args()
    result = export(
        args.checkpoint,
        args.prepared_model,
        args.output,
        {"bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype],
        args.copy_mode,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
