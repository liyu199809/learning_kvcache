#!/usr/bin/env python3
"""CPU-only conversion of trusted local FSDP2 prefix training checkpoints.

Preserves full model/Adam tensors, scheduler and dataloader state. Rank RNGs
are mapped to source ranks; changing world size is not bitwise replay.
Requires the installed torch DeviceMesh API with explicit _rank support.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import tempfile
from pathlib import Path

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta

from prefix_tuning.virtual_prefix.merge_prefix_checkpoint import _gather_tensor


def convert_tree(values, meshes, path, counts):
    first = values[0]
    if isinstance(first, DTensor):
        full = _gather_tensor(values, path).contiguous()
        if not torch.isfinite(full).all():
            raise ValueError(f"Nonfinite tensor: {path}")
        placement, = first.placements
        if isinstance(placement, Shard):
            if full.size(placement.dim) < len(meshes):
                raise ValueError(f"Empty target shard unsupported: {path}")
            pieces = torch.chunk(full, len(meshes), dim=placement.dim)
            assert len(pieces) == len(meshes)
        elif isinstance(placement, Replicate):
            pieces = [full] * len(meshes)
        else:
            raise ValueError(f"Unsupported placement {placement}")
        result = [DTensor(piece.clone().contiguous(), DTensorSpec(
            mesh, first.placements, TensorMeta(full.shape, full.stride(), full.dtype)
        ), requires_grad=False) for mesh, piece in zip(meshes, pieces)]
        assert torch.equal(full, _gather_tensor(result, path)), path
        counts["sharded_tensors"] += 1
        counts["verified_elements"] += full.numel()
        return result
    if isinstance(first, torch.Tensor):
        assert all(torch.equal(first, value) for value in values), path
        counts["replicated_tensors"] += 1
        return [first.clone() for _ in meshes]
    if isinstance(first, dict):
        assert all(list(v) == list(first) for v in values), path
        out = [type(first)() for _ in meshes]
        for key in first:
            converted = convert_tree([v[key] for v in values], meshes, f"{path}/{key}", counts)
            for rank, item in enumerate(converted):
                out[rank][key] = item
        return out
    if isinstance(first, (list, tuple)):
        assert all(len(v) == len(first) for v in values), path
        columns = [convert_tree([v[i] for v in values], meshes, f"{path}/{i}", counts)
                   for i in range(len(first))]
        return [type(first)(column[r] for column in columns) for r in range(len(meshes))]
    assert all(v == first for v in values), path
    return [copy.deepcopy(first) for _ in meshes]


def convert(source: Path, output: Path, target: int):
    assert source.name.startswith("global_step_") and target > 0
    if output.exists():
        raise FileExistsError(output)
    actor = source / "actor"
    meta = json.loads((actor / "trainable_only_meta.json").read_text())
    assert meta["checkpoint_type"] == "trainable_only"
    old = meta["world_size"]
    meshes = [DeviceMesh("cuda", torch.arange(target), mesh_dim_names=("fsdp",),
                         _init_backend=False, _rank=r) for r in range(target)]
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".reshard-", dir=output.parent))
    dest = staging / "actor"
    dest.mkdir()
    counts = dict(sharded_tensors=0, replicated_tensors=0, verified_elements=0)
    for kind in ("model", "optim"):
        shards = [torch.load(actor / f"{kind}_world_size_{old}_rank_{r}.pt",
                             map_location="cpu", weights_only=False) for r in range(old)]
        converted = convert_tree(shards, meshes, kind, counts)
        for r, value in enumerate(converted):
            torch.save(value, dest / f"{kind}_world_size_{target}_rank_{r}.pt")
        # Check serialized tensor payloads, not just the in-memory conversion.
        loaded = [torch.load(dest / f"{kind}_world_size_{target}_rank_{r}.pt",
                             map_location="cpu", weights_only=False) for r in range(target)]
        # A second reconstruction verifies all saved model/optimizer shards.
        checked = convert_tree(loaded, meshes, f"reload/{kind}",
                               dict(sharded_tensors=0, replicated_tensors=0, verified_elements=0))
        del shards, converted, loaded, checked
    rng_mapping = {r: r * old // target for r in range(target)}
    for rank, source_rank in rng_mapping.items():
        shutil.copy2(actor / f"extra_state_world_size_{old}_rank_{source_rank}.pt",
                     dest / f"extra_state_world_size_{target}_rank_{rank}.pt")
    for item in actor.iterdir():
        if item.name.endswith(".pt") or item.name == "trainable_only_meta.json":
            continue
        if item.is_dir():
            shutil.copytree(item, dest / item.name, symlinks=True)
        else:
            shutil.copy2(item, dest / item.name)
    meta["world_size"] = target
    (dest / "trainable_only_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    shutil.copy2(source / "data.pt", staging / "data.pt")
    manifest = dict(source=str(source), output=str(output), source_world_size=old,
                    target_world_size=target, rng_rank_mapping=rng_mapping,
                    bitwise_trajectory_replay=False, **counts)
    (staging / "reshard_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(staging, output)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=3)
    args = parser.parse_args()
    torch.set_num_threads(4)
    convert(args.source.resolve(), args.output.resolve(), args.world_size)
