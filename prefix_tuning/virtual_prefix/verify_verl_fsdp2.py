#!/usr/bin/env python3
"""Two-rank FSDP2 smoke for trainable-only virtual-prefix training."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import torch

from verl.trainer.config import CheckpointConfig
from verl.utils.distributed import initialize_global_process_group
from verl.workers.config import FSDPEngineConfig, FSDPOptimizerConfig, HFModelConfig
from verl.workers.engine.fsdp.transformer_impl import FSDPEngine


MODEL = os.getenv(
    "MODEL",
    "/mnt/storage/disk1/verl_data/base_model/Qwen3.5-4B-DeltaVirtualPrefix-2048",
)
PREFIX_PATTERN = r".*\.linear_attn\.prefix_tokens$"
EXPECTED_COUNT = 24
EXPECTED_NUMEL = 125829120


def _local_tensor(parameter):
    tensor = parameter.to_local() if hasattr(parameter, "to_local") else parameter
    return tensor.detach().cpu().clone()


def main() -> None:
    _, rank, world_size = initialize_global_process_group()
    if world_size != 2:
        raise ValueError(f"This smoke test requires exactly two ranks, got {world_size}")

    model_config = HFModelConfig(
        path=MODEL,
        load_tokenizer=False,
        trust_remote_code=True,
        use_remove_padding=True,
        enable_gradient_checkpointing=os.getenv("GRADIENT_CHECKPOINTING", "1") != "0",
        trainable_param_patterns=[PREFIX_PATTERN],
        expected_trainable_param_count=EXPECTED_COUNT,
        expected_trainable_numel=EXPECTED_NUMEL,
        rollout_sync_trainable_only=True,
    )
    engine_config = FSDPEngineConfig(
        strategy="fsdp2",
        model_dtype="bf16",
        use_torch_compile=False,
        ulysses_sequence_parallel_size=1,
        param_offload=False,
        optimizer_offload=False,
        mixed_precision={"param_dtype": "bf16", "reduce_dtype": "fp32", "buffer_dtype": "fp32"},
    )
    optimizer_config = FSDPOptimizerConfig(
        lr=5e-6,
        weight_decay=0.0,
        total_training_steps=2,
        lr_warmup_steps=0,
    )
    checkpoint_config = CheckpointConfig(save_trainable_only=True)
    engine = FSDPEngine(model_config, engine_config, optimizer_config, checkpoint_config)
    engine.initialize()

    named = dict(engine.module.named_parameters())
    trainable = {name: parameter for name, parameter in named.items() if parameter.requires_grad}
    if set(trainable) != engine._trainable_param_names:
        raise AssertionError("FSDP2 changed the selected trainable parameter names")
    if len(trainable) != EXPECTED_COUNT or sum(parameter.numel() for parameter in trainable.values()) != EXPECTED_NUMEL:
        raise AssertionError("Unexpected FSDP2 trainable parameter inventory")

    prefix_name = sorted(trainable)[0]
    base_name = next(name for name, parameter in named.items() if not parameter.requires_grad)
    prefix_before = _local_tensor(named[prefix_name])
    base_before = _local_tensor(named[base_name])

    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9]], device="cuda")
    position_ids = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3, 4]], device="cuda")
    cu_seqlens = torch.tensor([0, 4, 9], device="cuda", dtype=torch.int32)
    seq_idx = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, 1]], device="cuda", dtype=torch.int32)
    output = engine.module(
        input_ids=ids,
        attention_mask=None,
        position_ids=position_ids,
        use_cache=False,
        cu_seq_lens_q=cu_seqlens,
        cu_seq_lens_k=cu_seqlens,
        max_length_q=5,
        max_length_k=5,
        seq_idx=seq_idx,
    )
    loss = output.logits[:, -1, 123].float().mean()
    loss.backward()
    finite_nonzero_grads = 0
    for parameter in trainable.values():
        grad = parameter.grad
        if grad is None:
            finite = torch.tensor(0, device="cuda", dtype=torch.int32)
            nonzero = torch.tensor(0, device="cuda", dtype=torch.int32)
        else:
            local_grad = grad.to_local() if hasattr(grad, "to_local") else grad
            finite = torch.tensor(int(torch.isfinite(local_grad).all()), device="cuda", dtype=torch.int32)
            nonzero = torch.tensor(int(bool(torch.count_nonzero(local_grad))), device="cuda", dtype=torch.int32)
        torch.distributed.all_reduce(finite, op=torch.distributed.ReduceOp.MIN)
        torch.distributed.all_reduce(nonzero, op=torch.distributed.ReduceOp.MAX)
        if finite.item() and nonzero.item():
            finite_nonzero_grads += 1
    if finite_nonzero_grads != EXPECTED_COUNT:
        raise AssertionError(f"Only {finite_nonzero_grads}/{EXPECTED_COUNT} prefix gradients were finite and nonzero")
    engine.optimizer.step()
    engine.optimizer.zero_grad(set_to_none=True)

    prefix_after = _local_tensor(named[prefix_name])
    base_after = _local_tensor(named[base_name])
    prefix_changed = torch.tensor(
        int(not torch.equal(prefix_before, prefix_after)), device="cuda", dtype=torch.int32
    )
    base_unchanged = torch.tensor(int(torch.equal(base_before, base_after)), device="cuda", dtype=torch.int32)
    torch.distributed.all_reduce(prefix_changed, op=torch.distributed.ReduceOp.MAX)
    torch.distributed.all_reduce(base_unchanged, op=torch.distributed.ReduceOp.MIN)
    if not prefix_changed.item():
        raise AssertionError("Prefix parameter did not change after optimizer step")
    if not base_unchanged.item():
        raise AssertionError(f"Frozen base parameter changed: {base_name}")

    synced, peft_config = engine.get_per_tensor_param(trainable_only=True)
    synced = list(synced)
    if peft_config is not None or len(synced) != EXPECTED_COUNT:
        raise AssertionError("Trainable-only rollout sync did not return exactly 24 standard tensors")
    if sum(tensor.numel() for _, tensor in synced) != EXPECTED_NUMEL:
        raise AssertionError("Trainable-only rollout sync returned the wrong number of elements")

    checkpoint_dir = os.getenv(
        "CHECKPOINT",
        os.path.join(tempfile.gettempdir(), "verl_delta_prefix_fsdp2_smoke"),
    )
    engine.save_checkpoint(checkpoint_dir, global_step=1, max_ckpt_to_keep=1)
    checkpoint_prefix = _local_tensor(named[prefix_name])
    with torch.no_grad():
        named[prefix_name].add_(1)
    engine.load_checkpoint(checkpoint_dir)
    if not torch.equal(_local_tensor(named[prefix_name]), checkpoint_prefix):
        raise AssertionError("Trainable-only checkpoint did not restore the prefix parameter")

    if rank == 0:
        checkpoint_path = Path(checkpoint_dir)
        metadata = json.loads((checkpoint_path / "trainable_only_meta.json").read_text())
        if metadata["parameter_names"] != list(engine._trainable_param_order):
            raise AssertionError("Checkpoint parameter order does not match optimizer/model order")
        hf_files = {path.name for path in (checkpoint_path / "huggingface").iterdir()}
        if "modeling_qwen3_5_delta_virtual_prefix.py" not in hf_files:
            raise AssertionError("Checkpoint did not preserve the remote modeling file")
        if any(name.startswith("_fsdp_") for name in hf_files):
            raise AssertionError("Checkpoint incorrectly saved FSDP internals as remote model code")
        print(
            {
                "trainable_tensors": len(trainable),
                "trainable_numel": sum(parameter.numel() for parameter in trainable.values()),
                "sync_tensors": len(synced),
                "loss": float(loss.detach()),
                "checkpoint": checkpoint_dir,
            }
        )
        print("verl_delta_virtual_prefix_fsdp2_ok", True)
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
