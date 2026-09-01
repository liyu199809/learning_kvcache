#!/usr/bin/env python3
"""Merge ordered hybrid-prefix HF partitions or verl checkpoints into one model."""

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
PART_MANIFEST = "prefix_split_manifest.json"
DELTA_SUFFIXES = (".prefix_k", ".prefix_v", ".prefix_beta_logits", ".prefix_a")
ATTENTION_SUFFIXES = (".prefix_key_tokens", ".prefix_value_tokens")


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _safetensors(path: Path) -> dict[str, torch.Tensor]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        return {name: handle.get_tensor(name) for name in handle.keys()}


def _load_hf_part(path: Path) -> tuple[dict, dict[str, torch.Tensor], str]:
    manifest = _read_json(path / PART_MANIFEST)
    tensors = {
        **_safetensors(path / DELTA_FILE),
        **_safetensors(path / ATTENTION_FILE),
    }
    return manifest, tensors, "hf_model"


def _load_checkpoint(path: Path) -> tuple[dict, dict[str, torch.Tensor], str]:
    metadata = _read_json(path / "trainable_only_meta.json")
    if metadata.get("trainable_type") != "hybrid_delta_residual_attention_prefix":
        raise ValueError(f"Unexpected checkpoint type in {path}: {metadata.get('trainable_type')!r}")
    names = metadata.get("parameter_names")
    parameters = metadata.get("parameters")
    if not isinstance(names, list) or not isinstance(parameters, dict) or set(names) != set(parameters):
        raise ValueError(f"Invalid parameter metadata in {path}")
    world_size = metadata.get("world_size")
    if not isinstance(world_size, int) or world_size <= 0:
        raise ValueError(f"Invalid world_size in {path}: {world_size!r}")
    states = []
    for rank in range(world_size):
        rank_path = path / f"model_world_size_{world_size}_rank_{rank}.pt"
        state = torch.load(rank_path, map_location="cpu", weights_only=False)
        _validate_rank_state(state, names, parameters, rank)
        states.append(state)
    tensors = {
        name: _gather_tensor([state[name] for state in states], name).contiguous()
        for name in names
    }
    prepared = Path(metadata.get("base_model_path", ""))
    if not prepared.is_dir():
        raise NotADirectoryError(f"Checkpoint base_model_path is unavailable: {prepared}")
    manifest = _read_json(prepared / PART_MANIFEST)
    return manifest, tensors, "verl_checkpoint"


def _load_part(path: Path) -> tuple[dict, dict[str, torch.Tensor], str]:
    path = path.resolve()
    if (path / "model_world_size_1_rank_0.pt").is_file() or list(path.glob("model_world_size_*_rank_0.pt")):
        return _load_checkpoint(path)
    if (path / DELTA_FILE).is_file() and (path / ATTENTION_FILE).is_file():
        return _load_hf_part(path)
    raise ValueError(f"Not a supported HF partition or verl actor checkpoint: {path}")


def _copy_source(source: Path, output: Path, copy_mode: str) -> None:
    excluded = {
        DELTA_FILE,
        ATTENTION_FILE,
        "trainable_only_meta.json",
        "prefix_merge_manifest.json",
        "prefix_partition_merge_manifest.json",
        PART_MANIFEST,
        "split_manifest.json",
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


def _validate_and_order(parts: list[tuple[dict, dict[str, torch.Tensor], str]]):
    if len(parts) < 2:
        raise ValueError("At least two partitions are required")
    ordered = sorted(parts, key=lambda item: item[0].get("split_index", -1))
    first = ordered[0][0]
    expected_indices = list(range(first.get("num_splits", -1)))
    indices = [item[0].get("split_index") for item in ordered]
    if indices != expected_indices or len(ordered) != len(expected_indices):
        raise ValueError(f"Expected split indices {expected_indices}, got {indices}")
    common_keys = (
        "format_version",
        "partition_method",
        "source_model",
        "source_prefix_sha256",
        "num_splits",
        "delta_source_tokens",
        "attention_source_tokens",
    )
    for manifest, _, _ in ordered[1:]:
        mismatches = {key: (first.get(key), manifest.get(key)) for key in common_keys if first.get(key) != manifest.get(key)}
        if mismatches:
            raise ValueError(f"Partitions do not share one source: {mismatches}")
    for prefix in ("delta", "attention"):
        ranges = [item[0][f"{prefix}_range"] for item in ordered]
        cursor = 0
        for start, end in ranges:
            if start != cursor or end <= start:
                raise ValueError(f"Non-contiguous {prefix} ranges: {ranges}")
            cursor = end
        if cursor != first[f"{prefix}_source_tokens"]:
            raise ValueError(f"Incomplete {prefix} ranges: {ranges}")
    names = list(ordered[0][1])
    if not names or any(set(item[1]) != set(names) for item in ordered):
        raise ValueError("Partition tensor names do not match")
    if not all(name.endswith(DELTA_SUFFIXES + ATTENTION_SUFFIXES) for name in names):
        raise ValueError("Partitions contain unsupported tensor names")
    return ordered, names


def merge_partitions(
    inputs: list[Path],
    output: Path,
    output_dtype: torch.dtype,
    copy_mode: str,
    expect_source_exact: bool,
) -> dict:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    loaded = [_load_part(path) for path in inputs]
    input_by_index = {
        item[0].get("split_index"): str(path.resolve()) for path, item in zip(inputs, loaded)
    }
    ordered, names = _validate_and_order(loaded)
    manifest = ordered[0][0]
    source = Path(manifest["source_model"]).resolve()
    if not source.is_dir():
        raise NotADirectoryError(source)
    merged = {
        name: torch.cat([item[1][name] for item in ordered], dim=0).to(output_dtype).contiguous()
        for name in names
    }
    source_tensors = {
        **_safetensors(source / DELTA_FILE),
        **_safetensors(source / ATTENTION_FILE),
    }
    if set(merged) != set(source_tensors):
        raise ValueError("Merged tensor names do not match the source model")
    bad_shapes = {
        name: (tuple(merged[name].shape), tuple(source_tensors[name].shape))
        for name in merged
        if merged[name].shape != source_tensors[name].shape
    }
    if bad_shapes:
        raise ValueError(f"Merged shapes do not match the source model: {bad_shapes}")
    if expect_source_exact:
        unequal = [
            name
            for name in merged
            if not torch.equal(merged[name], source_tensors[name].to(output_dtype))
        ]
        if unequal:
            raise ValueError(f"Round-trip differs from source for {len(unequal)} tensors: {unequal[:5]}")

    old_bytes = sum(t.numel() * t.element_size() for t in source_tensors.values())
    new_bytes = sum(t.numel() * t.element_size() for t in merged.values())
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        _copy_source(source, staging, copy_mode)
        delta = {name: tensor for name, tensor in merged.items() if name.endswith(DELTA_SUFFIXES)}
        attention = {
            name: tensor for name, tensor in merged.items() if name.endswith(ATTENTION_SUFFIXES)
        }
        save_file(delta, staging / DELTA_FILE)
        save_file(attention, staging / ATTENTION_FILE)
        index_path = staging / "model.safetensors.index.json"
        index = _read_json(index_path)
        metadata = index.setdefault("metadata", {})
        if isinstance(metadata.get("total_size"), int):
            metadata["total_size"] = metadata["total_size"] - old_bytes + new_bytes
        _write_json(index_path, index)
        result_manifest = {
            "format_version": 1,
            "merge_method": "concatenate_virtual_token_rows",
            "source_model": str(source),
            "source_prefix_sha256": manifest["source_prefix_sha256"],
            "input_types": [item[2] for item in ordered],
            "ordered_inputs": [input_by_index[item[0]["split_index"]] for item in ordered],
            "num_splits": len(ordered),
            "delta_tokens": manifest["delta_source_tokens"],
            "attention_tokens": manifest["attention_source_tokens"],
            "output_dtype": str(output_dtype),
            "trainable_tensor_count": len(merged),
            "trainable_numel": sum(t.numel() for t in merged.values()),
            "source_exact_verified": expect_source_exact,
        }
        _write_json(staging / "prefix_partition_merge_manifest.json", result_manifest)
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {"output": str(output), **result_manifest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--copy-mode", choices=("hardlink", "copy"), default="hardlink")
    parser.add_argument("--expect-source-exact", action="store_true")
    args = parser.parse_args()
    result = merge_partitions(
        args.input,
        args.output,
        {"bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype],
        args.copy_mode,
        args.expect_source_exact,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
