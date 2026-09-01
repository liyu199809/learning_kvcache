#!/usr/bin/env python3
"""Split a deployed hybrid-prefix HF model into contiguous token partitions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from copy import deepcopy
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


DELTA_FILE = "independent_delta_kv_prefix.safetensors"
ATTENTION_FILE = "residual_attention_prefix.safetensors"
DELTA_CONFIG_FILE = "independent_delta_kv_prefix_config.json"
ATTENTION_CONFIG_FILE = "residual_attention_prefix_config.json"
PART_MANIFEST = "prefix_split_manifest.json"
ROOT_MANIFEST = "split_manifest.json"
DELTA_SUFFIXES = (".prefix_k", ".prefix_v", ".prefix_beta_logits", ".prefix_a")
ATTENTION_SUFFIXES = (".prefix_key_tokens", ".prefix_value_tokens")


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_tensors(path: Path, suffixes: tuple[str, ...]) -> dict[str, torch.Tensor]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        tensors = {name: handle.get_tensor(name) for name in handle.keys()}
    if not tensors or not all(name.endswith(suffixes) for name in tensors):
        raise ValueError(f"Unexpected tensors in {path}")
    return tensors


def _copy_shared_files(source: Path, output: Path, copy_mode: str) -> None:
    excluded = {
        DELTA_FILE,
        ATTENTION_FILE,
        DELTA_CONFIG_FILE,
        ATTENTION_CONFIG_FILE,
        "config.json",
        "model.safetensors.index.json",
        "trainable_only_meta.json",
        "prefix_merge_manifest.json",
        "prefix_partition_merge_manifest.json",
        PART_MANIFEST,
        ROOT_MANIFEST,
    }
    output.mkdir()
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


def _validate_source(
    source: Path,
    config: dict,
    delta: dict[str, torch.Tensor],
    attention: dict[str, torch.Tensor],
) -> tuple[int, int]:
    expected_architecture = ["Qwen3_5HybridDeltaResidualAttentionPrefixForConditionalGeneration"]
    if config.get("architectures") != expected_architecture:
        raise ValueError(f"Unexpected hybrid architecture: {config.get('architectures')!r}")
    text_config = config.get("text_config")
    if not isinstance(text_config, dict):
        raise ValueError("Source config has no text_config object")
    delta_tokens = text_config.get("independent_delta_prefix_num_virtual_tokens")
    attention_tokens = text_config.get("residual_attention_prefix_num_virtual_tokens")
    if not isinstance(delta_tokens, int) or delta_tokens <= 0:
        raise ValueError(f"Invalid Delta prefix length: {delta_tokens!r}")
    if not isinstance(attention_tokens, int) or attention_tokens <= 0:
        raise ValueError(f"Invalid attention prefix length: {attention_tokens!r}")
    if len(delta) != 96 or any(tensor.shape[0] != delta_tokens for tensor in delta.values()):
        raise ValueError(f"{source / DELTA_FILE} does not match G={delta_tokens}")
    if len(attention) != 16 or any(tensor.shape[0] != attention_tokens for tensor in attention.values()):
        raise ValueError(f"{source / ATTENTION_FILE} does not match A={attention_tokens}")
    index = _read_json(source / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("Source model index has no weight_map")
    for name in delta:
        if weight_map.get(name) != DELTA_FILE:
            raise ValueError(f"Bad source index mapping for {name}")
    for name in attention:
        if weight_map.get(name) != ATTENTION_FILE:
            raise ValueError(f"Bad source index mapping for {name}")
    return delta_tokens, attention_tokens


def _partition_config(
    source_config: dict,
    delta_tokens: int,
    attention_tokens: int,
    delta_numel: int,
    attention_numel: int,
) -> dict:
    config = deepcopy(source_config)
    config["text_config"]["independent_delta_prefix_num_virtual_tokens"] = delta_tokens
    config["text_config"]["residual_attention_prefix_num_virtual_tokens"] = attention_tokens
    delta_config = config["independent_delta_kv_prefix"]
    delta_config["num_virtual_tokens"] = delta_tokens
    delta_config["trainable_numel"] = delta_numel
    attention_config = config["residual_attention_prefix"]
    attention_config["num_virtual_tokens"] = attention_tokens
    attention_config["prefix_positions"] = [-attention_tokens, -1]
    attention_config["trainable_numel"] = attention_numel
    hybrid_config = config["hybrid_prefix"]
    hybrid_config["delta_prefix_num_virtual_tokens"] = delta_tokens
    hybrid_config["attention_prefix_num_virtual_tokens"] = attention_tokens
    hybrid_config["trainable_numel"] = delta_numel + attention_numel
    return config


def split_model(source: Path, output_root: Path, num_splits: int, copy_mode: str) -> dict:
    source = source.resolve()
    output_root = output_root.resolve()
    if not source.is_dir():
        raise NotADirectoryError(source)
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite {output_root}")
    if num_splits <= 1:
        raise ValueError("num_splits must be greater than one")

    config = _read_json(source / "config.json")
    delta_config_file = _read_json(source / DELTA_CONFIG_FILE)
    attention_config_file = _read_json(source / ATTENTION_CONFIG_FILE)
    index = _read_json(source / "model.safetensors.index.json")
    delta = _load_tensors(source / DELTA_FILE, DELTA_SUFFIXES)
    attention = _load_tensors(source / ATTENTION_FILE, ATTENTION_SUFFIXES)
    delta_tokens, attention_tokens = _validate_source(source, config, delta, attention)
    if delta_tokens % num_splits or attention_tokens % num_splits:
        raise ValueError(
            f"Prefix lengths G={delta_tokens}, A={attention_tokens} are not divisible by {num_splits}"
        )
    delta_chunk = delta_tokens // num_splits
    attention_chunk = attention_tokens // num_splits
    source_hashes = {
        DELTA_FILE: _sha256(source / DELTA_FILE),
        ATTENTION_FILE: _sha256(source / ATTENTION_FILE),
    }
    old_prefix_bytes = sum(t.numel() * t.element_size() for t in (*delta.values(), *attention.values()))

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir=output_root.parent))
    parts = []
    try:
        for split_index in range(num_splits):
            delta_start = split_index * delta_chunk
            attention_start = split_index * attention_chunk
            part_dir = staging / f"cluster{split_index}"
            _copy_shared_files(source, part_dir, copy_mode)
            delta_part = {
                name: tensor[delta_start : delta_start + delta_chunk].contiguous()
                for name, tensor in delta.items()
            }
            attention_part = {
                name: tensor[attention_start : attention_start + attention_chunk].contiguous()
                for name, tensor in attention.items()
            }
            delta_numel = sum(t.numel() for t in delta_part.values())
            attention_numel = sum(t.numel() for t in attention_part.values())
            save_file(delta_part, part_dir / DELTA_FILE)
            save_file(attention_part, part_dir / ATTENTION_FILE)

            part_config = _partition_config(
                config, delta_chunk, attention_chunk, delta_numel, attention_numel
            )
            _write_json(part_dir / "config.json", part_config)
            part_delta_config = deepcopy(delta_config_file)
            part_delta_config["num_virtual_tokens"] = delta_chunk
            part_delta_config["trainable_numel"] = delta_numel
            _write_json(part_dir / DELTA_CONFIG_FILE, part_delta_config)
            part_attention_config = deepcopy(attention_config_file)
            part_attention_config["num_virtual_tokens"] = attention_chunk
            part_attention_config["prefix_positions"] = [-attention_chunk, -1]
            part_attention_config["trainable_numel"] = attention_numel
            _write_json(part_dir / ATTENTION_CONFIG_FILE, part_attention_config)

            part_index = deepcopy(index)
            new_prefix_bytes = sum(
                t.numel() * t.element_size() for t in (*delta_part.values(), *attention_part.values())
            )
            metadata = part_index.setdefault("metadata", {})
            if isinstance(metadata.get("total_size"), int):
                metadata["total_size"] = metadata["total_size"] - old_prefix_bytes + new_prefix_bytes
            _write_json(part_dir / "model.safetensors.index.json", part_index)
            part_manifest = {
                "format_version": 1,
                "partition_method": "contiguous_virtual_token_rows",
                "source_model": str(source),
                "source_prefix_sha256": source_hashes,
                "num_splits": num_splits,
                "split_index": split_index,
                "delta_source_tokens": delta_tokens,
                "delta_range": [delta_start, delta_start + delta_chunk],
                "attention_source_tokens": attention_tokens,
                "attention_range": [attention_start, attention_start + attention_chunk],
                "delta_partition_tokens": delta_chunk,
                "attention_partition_tokens": attention_chunk,
                "trainable_tensor_count": len(delta_part) + len(attention_part),
                "trainable_numel": delta_numel + attention_numel,
            }
            _write_json(part_dir / PART_MANIFEST, part_manifest)
            parts.append({"path": str(output_root / part_dir.name), **part_manifest})

        root_manifest = {
            "format_version": 1,
            "partition_method": "contiguous_virtual_token_rows",
            "source_model": str(source),
            "source_prefix_sha256": source_hashes,
            "num_splits": num_splits,
            "delta_source_tokens": delta_tokens,
            "attention_source_tokens": attention_tokens,
            "parts": parts,
        }
        _write_json(staging / ROOT_MANIFEST, root_manifest)
        os.replace(staging, output_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {"output_root": str(output_root), **root_manifest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--num-splits", type=int, default=4)
    parser.add_argument("--copy-mode", choices=("hardlink", "copy"), default="hardlink")
    args = parser.parse_args()
    result = split_model(args.source_model, args.output_root, args.num_splits, args.copy_mode)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
