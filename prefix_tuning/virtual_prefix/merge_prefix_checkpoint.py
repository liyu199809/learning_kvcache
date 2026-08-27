#!/usr/bin/env python3
"""Merge a verl trainable-only prefix checkpoint into a deployable HF tree.

The base model is not rewritten.  Its safetensor shards are copied or
hardlinked from the prepared prefix model, while the FSDP prefix shards are
gathered into the prefix safetensor referenced by the model index.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch.distributed.tensor import DTensor, Replicate, Shard


@dataclass(frozen=True)
class PrefixSpec:
    trainable_type: str
    weights_file: str
    config_key: str
    virtual_tokens_key: str
    suffixes: tuple[str, ...]


SPECS = {
    "delta_virtual_prefix": PrefixSpec(
        trainable_type="delta_virtual_prefix",
        weights_file="delta_virtual_prefix.safetensors",
        config_key="delta_virtual_prefix",
        virtual_tokens_key="delta_prefix_num_virtual_tokens",
        suffixes=(".linear_attn.prefix_tokens",),
    ),
    "independent_delta_kv_prefix": PrefixSpec(
        trainable_type="independent_delta_kv_prefix",
        weights_file="independent_delta_kv_prefix.safetensors",
        config_key="independent_delta_kv_prefix",
        virtual_tokens_key="independent_delta_prefix_num_virtual_tokens",
        suffixes=(".prefix_k", ".prefix_v", ".prefix_beta_logits", ".prefix_a"),
    ),
}

_SAFETENSOR_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _infer_spec(parameter_names: set[str]) -> PrefixSpec:
    matches = [
        spec
        for spec in SPECS.values()
        if parameter_names and all(name.endswith(spec.suffixes) for name in parameter_names)
    ]
    if len(matches) != 1:
        raise ValueError(
            "Unable to infer one supported prefix type from checkpoint parameters: "
            + ", ".join(sorted(parameter_names)[:5])
        )
    return matches[0]


def _validate_metadata(metadata: dict, expected_type: str | None) -> tuple[PrefixSpec, list[str]]:
    if metadata.get("checkpoint_type") != "trainable_only":
        raise ValueError("Expected a trainable-only verl checkpoint")
    parameters = metadata.get("parameters")
    if not isinstance(parameters, dict) or not parameters:
        raise ValueError("Checkpoint metadata has no parameter descriptions")
    parameter_names = metadata.get("parameter_names", list(parameters))
    if not isinstance(parameter_names, list) or set(parameter_names) != set(parameters):
        raise ValueError("Checkpoint parameter_names do not match parameters")
    if len(parameter_names) != len(set(parameter_names)):
        raise ValueError("Checkpoint parameter_names contain duplicates")

    spec = _infer_spec(set(parameter_names))
    recorded_type = metadata.get("trainable_type")
    # Old delta-virtual-prefix checkpoints predate trainable_type metadata.
    if recorded_type not in (None, spec.trainable_type):
        raise ValueError(
            f"Checkpoint trainable_type={recorded_type!r} conflicts with parameters for {spec.trainable_type!r}"
        )
    if expected_type is not None and spec.trainable_type != expected_type:
        raise ValueError(
            f"Expected {expected_type!r}, but checkpoint contains {spec.trainable_type!r}"
        )
    return spec, parameter_names


def _prepared_prefix_layout(prepared_model: Path, spec: PrefixSpec) -> tuple[dict, dict[str, list[int]], int]:
    config = _read_json(prepared_model / "config.json")
    if config.get("model_type") != "qwen3_5":
        raise ValueError(f"Expected a qwen3_5 prepared model, got {config.get('model_type')!r}")
    text_config = config.get("text_config")
    if not isinstance(text_config, dict):
        raise ValueError("Prepared model config has no text_config object")
    configured_m = text_config.get(spec.virtual_tokens_key)
    if not isinstance(configured_m, int) or configured_m <= 0:
        raise ValueError(
            f"Prepared model text_config.{spec.virtual_tokens_key} must be a positive integer"
        )
    prefix_config = config.get(spec.config_key)
    if not isinstance(prefix_config, dict):
        raise ValueError(f"Prepared model config has no {spec.config_key} object")

    weights_path = prepared_model / spec.weights_file
    if not weights_path.is_file():
        raise FileNotFoundError(weights_path)
    shapes: dict[str, list[int]] = {}
    old_tensor_bytes = 0
    with safe_open(weights_path, framework="pt", device="cpu") as handle:
        for name in handle.keys():
            tensor_slice = handle.get_slice(name)
            shape = list(tensor_slice.get_shape())
            dtype = tensor_slice.get_dtype()
            try:
                dtype_bytes = _SAFETENSOR_DTYPE_BYTES[dtype]
            except KeyError as exc:
                raise ValueError(f"Unsupported prepared prefix dtype {dtype!r} for {name}") from exc
            shapes[name] = shape
            old_tensor_bytes += int(torch.Size(shape).numel()) * dtype_bytes
    if not shapes or not all(name.endswith(spec.suffixes) for name in shapes):
        raise ValueError(f"Prepared model {weights_path} does not contain {spec.trainable_type} tensors")

    index = _read_json(prepared_model / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("Prepared model index has no weight_map")
    bad_mappings = {
        name: weight_map.get(name)
        for name in shapes
        if weight_map.get(name) != spec.weights_file
    }
    if bad_mappings:
        raise ValueError(f"Prepared model index has invalid prefix mappings: {bad_mappings}")
    return config, shapes, old_tensor_bytes


def _validate_rank_state(
    state: dict,
    parameter_names: list[str],
    parameter_meta: dict,
    rank: int,
) -> None:
    if list(state) != parameter_names:
        missing = sorted(set(parameter_names) - set(state))
        unexpected = sorted(set(state) - set(parameter_names))
        raise ValueError(
            f"Rank {rank} checkpoint parameters differ from metadata: "
            f"missing={missing}, unexpected={unexpected}"
        )
    for name, tensor in state.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Rank {rank} value for {name} is not a tensor: {type(tensor)!r}")
        expected_shape = parameter_meta[name].get("shape")
        if list(tensor.shape) != expected_shape:
            raise ValueError(
                f"Rank {rank} global shape mismatch for {name}: {list(tensor.shape)} != {expected_shape}"
            )
        expected_dtype = parameter_meta[name].get("dtype")
        if expected_dtype is not None and str(tensor.dtype) != expected_dtype:
            raise ValueError(
                f"Rank {rank} dtype mismatch for {name}: {tensor.dtype} != {expected_dtype}"
            )


def _gather_tensor(shards: list[torch.Tensor], name: str) -> torch.Tensor:
    first = shards[0]
    if not isinstance(first, DTensor):
        if any(isinstance(tensor, DTensor) for tensor in shards[1:]):
            raise ValueError(f"Mixed DTensor and Tensor shards for {name}")
        if any(tensor.shape != first.shape or tensor.dtype != first.dtype for tensor in shards[1:]):
            raise ValueError(f"Replicated tensor shards differ in shape or dtype for {name}")
        return first.detach().cpu()

    if any(not isinstance(tensor, DTensor) for tensor in shards[1:]):
        raise ValueError(f"Mixed DTensor and Tensor shards for {name}")
    placements = tuple(first.placements)
    if len(placements) != 1:
        raise ValueError(f"Only one-dimensional FSDP meshes are supported for {name}, got {placements}")
    for rank, tensor in enumerate(shards[1:], start=1):
        if tuple(tensor.placements) != placements or tensor.shape != first.shape or tensor.dtype != first.dtype:
            raise ValueError(f"Inconsistent DTensor metadata for {name} at rank {rank}")

    placement = placements[0]
    if isinstance(placement, Replicate):
        return first.to_local().detach().cpu()
    if not isinstance(placement, Shard):
        raise ValueError(f"Unsupported FSDP placement for {name}: {placement}")
    shard_dim = placement.dim
    full = torch.cat([tensor.to_local().detach().cpu() for tensor in shards], dim=shard_dim)
    expected_shape = tuple(first.shape)
    if full.shape[shard_dim] < expected_shape[shard_dim]:
        raise ValueError(
            f"FSDP shards for {name} are too short on dim {shard_dim}: "
            f"{full.shape[shard_dim]} < {expected_shape[shard_dim]}"
        )
    slices = [slice(None)] * full.ndim
    slices[shard_dim] = slice(0, expected_shape[shard_dim])
    full = full[tuple(slices)]
    if tuple(full.shape) != expected_shape:
        raise ValueError(f"Gathered shape mismatch for {name}: {tuple(full.shape)} != {expected_shape}")
    return full


def _copy_prepared_model(source: Path, output: Path, spec: PrefixSpec, copy_mode: str) -> None:
    excluded = {spec.weights_file, "trainable_only_meta.json", "prefix_merge_manifest.json"}
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


def _update_index(output: Path, spec: PrefixSpec, names: list[str], old_bytes: int, new_bytes: int) -> None:
    index_path = output / "model.safetensors.index.json"
    index = _read_json(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("Output model index has no weight_map")
    for name in names:
        if weight_map.get(name) != spec.weights_file:
            raise ValueError(f"Output model index does not map {name} to {spec.weights_file}")
    metadata = index.setdefault("metadata", {})
    if isinstance(metadata.get("total_size"), int):
        metadata["total_size"] = metadata["total_size"] - old_bytes + new_bytes
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def merge_prefix_checkpoint(
    checkpoint: Path,
    output: Path,
    prepared_model: Path | None = None,
    *,
    expected_trainable_type: str | None = None,
    output_dtype: torch.dtype = torch.bfloat16,
    copy_mode: str = "hardlink",
) -> dict:
    checkpoint = checkpoint.resolve()
    output = output.resolve()
    metadata = _read_json(checkpoint / "trainable_only_meta.json")
    spec, parameter_names = _validate_metadata(metadata, expected_trainable_type)

    if prepared_model is None:
        recorded_path = metadata.get("base_model_path")
        if not recorded_path:
            raise ValueError("--prepared-model is required because metadata has no base_model_path")
        prepared_model = Path(recorded_path)
    prepared_model = prepared_model.resolve()
    if not prepared_model.is_dir():
        raise NotADirectoryError(prepared_model)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    _, prepared_shapes, old_prefix_bytes = _prepared_prefix_layout(prepared_model, spec)
    checkpoint_shapes = {name: metadata["parameters"][name].get("shape") for name in parameter_names}
    if checkpoint_shapes != prepared_shapes:
        missing = sorted(set(prepared_shapes) - set(checkpoint_shapes))
        unexpected = sorted(set(checkpoint_shapes) - set(prepared_shapes))
        mismatched = sorted(
            name
            for name in set(checkpoint_shapes).intersection(prepared_shapes)
            if checkpoint_shapes[name] != prepared_shapes[name]
        )
        raise ValueError(
            "Checkpoint prefix layout does not match prepared model: "
            f"missing={missing}, unexpected={unexpected}, shape_mismatches={mismatched}"
        )
    configured_m = _read_json(prepared_model / "config.json")["text_config"][spec.virtual_tokens_key]
    if configured_m != metadata.get("num_virtual_tokens"):
        raise ValueError(
            f"Prepared model M={configured_m} does not match checkpoint M={metadata.get('num_virtual_tokens')}"
        )

    world_size = metadata.get("world_size")
    if not isinstance(world_size, int) or world_size <= 0:
        raise ValueError(f"Invalid checkpoint world_size: {world_size!r}")
    rank_states = []
    for rank in range(world_size):
        rank_path = checkpoint / f"model_world_size_{world_size}_rank_{rank}.pt"
        if not rank_path.is_file():
            raise FileNotFoundError(rank_path)
        state = torch.load(rank_path, map_location="cpu", weights_only=False)
        if not isinstance(state, dict):
            raise TypeError(f"Expected a state dict in {rank_path}, got {type(state)!r}")
        _validate_rank_state(state, parameter_names, metadata["parameters"], rank)
        rank_states.append(state)

    merged = {}
    for name in parameter_names:
        merged[name] = _gather_tensor([state[name] for state in rank_states], name).to(
            output_dtype
        ).contiguous()
        if list(merged[name].shape) != prepared_shapes[name]:
            raise ValueError(
                f"Merged shape mismatch for {name}: {list(merged[name].shape)} != {prepared_shapes[name]}"
            )
    new_prefix_bytes = sum(tensor.numel() * tensor.element_size() for tensor in merged.values())

    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        _copy_prepared_model(prepared_model, staging, spec, copy_mode)
        save_file(merged, staging / spec.weights_file)
        _update_index(staging, spec, parameter_names, old_prefix_bytes, new_prefix_bytes)

        normalized_metadata = dict(metadata)
        normalized_metadata["trainable_type"] = spec.trainable_type
        (staging / "trainable_only_meta.json").write_text(
            json.dumps(normalized_metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "format_version": 1,
            "trainable_type": spec.trainable_type,
            "source_checkpoint": str(checkpoint),
            "prepared_model": str(prepared_model),
            "world_size": world_size,
            "output_dtype": str(output_dtype),
            "prefix_weights_file": spec.weights_file,
            "prefix_tensors": len(merged),
            "prefix_numel": sum(tensor.numel() for tensor in merged.values()),
            "prefix_weight_bytes": new_prefix_bytes,
        }
        (staging / "prefix_merge_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        with safe_open(staging / spec.weights_file, framework="pt", device="cpu") as handle:
            written = {name: list(handle.get_slice(name).get_shape()) for name in handle.keys()}
        if written != prepared_shapes:
            raise RuntimeError("Written prefix safetensor failed key/shape verification")
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return {"output": str(output), **manifest}


def main(expected_trainable_type: str | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Gather a trainable-only prefix checkpoint and package a deployable HF model"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--prepared-model",
        type=Path,
        default=None,
        help="Prepared prefix model; defaults to trainable_only_meta.json base_model_path",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--copy-mode", choices=("hardlink", "copy"), default="hardlink")
    args = parser.parse_args()
    result = merge_prefix_checkpoint(
        args.checkpoint,
        args.output,
        args.prepared_model,
        expected_trainable_type=expected_trainable_type,
        output_dtype={"bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype],
        copy_mode=args.copy_mode,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
