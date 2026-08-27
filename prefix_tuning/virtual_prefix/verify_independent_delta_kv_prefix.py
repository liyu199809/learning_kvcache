#!/usr/bin/env python3
"""Focused functional checks for the independent Delta K/V prefix."""

from __future__ import annotations

import json

import torch
from transformers import DynamicCache
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

from prefix_tuning.virtual_prefix.modeling_qwen3_5_independent_delta_kv_prefix import (
    Qwen3_5IndependentDeltaKVPrefixGatedDeltaNet,
)


def make_config() -> Qwen3_5TextConfig:
    config = Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        layer_types=["linear_attention"],
        linear_conv_kernel_dim=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        dtype=torch.bfloat16,
    )
    config.independent_delta_prefix_num_virtual_tokens = 64
    config.independent_delta_prefix_v_init_std = 3e-4
    return config


def max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.float() - right.float()).abs().max())


def grad_stats(parameter: torch.nn.Parameter) -> dict[str, float | int | bool]:
    grad = parameter.grad
    if grad is None:
        return {"finite": True, "nonzero": 0, "nonzero_rows": 0, "norm": 0.0}
    rows = grad.float().reshape(grad.shape[0], -1).norm(dim=1)
    return {
        "finite": bool(torch.isfinite(grad).all()),
        "nonzero": int(torch.count_nonzero(grad)),
        "nonzero_rows": int(torch.count_nonzero(rows)),
        "norm": float(grad.float().norm()),
    }


def main() -> None:
    torch.manual_seed(7)
    config = make_config()
    module = Qwen3_5IndependentDeltaKVPrefixGatedDeltaNet(config, 0).cuda().bfloat16()
    base = Qwen3_5GatedDeltaNet(config, 0).cuda().bfloat16()
    base_state = {
        name: tensor for name, tensor in module.state_dict().items() if not name.startswith("prefix_")
    }
    base.load_state_dict(base_state, strict=True)
    # A directly constructed tiny mixer has uninitialized random decay values.
    # Use a stable long-memory setting representative of the slow heads in the
    # trained Qwen3.5 checkpoint so all prefix rows are observable in BF16.
    with torch.no_grad():
        module.A_log.fill_(torch.log(torch.tensor(0.1)))
        module.dt_bias.fill_(-6.0)
        base.A_log.copy_(module.A_log)
        base.dt_bias.copy_(module.dt_bias)
    for name, parameter in module.named_parameters():
        parameter.requires_grad_(name.startswith("prefix_"))

    x = torch.randn(2, 23, config.hidden_size, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        initial_output = module(x)
        base_output = base(x)
        prefix_tail, hf_conv, prefix_state = module._compute_prefix_states(x.dtype)
    initial_metrics = {
        "base_output_max_abs": max_abs(initial_output, base_output),
        "prefix_state_max_abs": float(prefix_state.float().abs().max()),
        "packed_conv_tail_max_abs": float(prefix_tail.float().abs().max()),
        "hf_conv_cache_max_abs": float(hf_conv.float().abs().max()),
        "prefix_k_exact_zero": bool(torch.count_nonzero(module.prefix_k) == 0),
    }

    weight = torch.randn_like(initial_output, dtype=torch.float32)
    output = module(x)
    (output.float() * weight).sum().backward()
    step1 = {
        name: grad_stats(getattr(module, name))
        for name in ("prefix_k", "prefix_v", "prefix_beta_logits", "prefix_a")
    }
    if step1["prefix_k"]["nonzero_rows"] != module.num_virtual_tokens:
        raise AssertionError(f"Step 1 K row coverage is incomplete: {step1['prefix_k']}")
    if not all(stats["finite"] for stats in step1.values()):
        raise AssertionError(f"Non-finite step 1 gradient: {step1}")

    with torch.no_grad():
        k_grad = module.prefix_k.grad.float()
        module.prefix_k.copy_((-1e-3 * k_grad / k_grad.norm().clamp_min(1e-12)).to(module.prefix_k))
    module.zero_grad(set_to_none=True)
    output = module(x)
    (output.float() * weight).sum().backward()
    step2 = {
        name: grad_stats(getattr(module, name))
        for name in ("prefix_k", "prefix_v", "prefix_beta_logits", "prefix_a")
    }
    for name in ("prefix_v", "prefix_beta_logits", "prefix_a"):
        if step2[name]["nonzero_rows"] == 0:
            raise AssertionError(f"Step 2 {name} did not enter the recurrent gradient path")
        if not step2[name]["finite"]:
            raise AssertionError(f"Step 2 {name} gradient is non-finite")
    module.zero_grad(set_to_none=True)

    lengths = [19, 11, 27]
    segments = [
        torch.randn(1, length, config.hidden_size, device="cuda", dtype=torch.bfloat16)
        for length in lengths
    ]
    with torch.no_grad():
        separate = torch.cat([module(segment) for segment in segments], dim=1)
        packed_input = torch.cat(segments, dim=1)
        cu_seqlens = torch.tensor(
            [0, *torch.tensor(lengths).cumsum(0).tolist()], device="cuda", dtype=torch.int32
        )
        packed = module(packed_input, cu_seq_lens_q=cu_seqlens)

        padded_input = torch.zeros(
            len(lengths), max(lengths), config.hidden_size, device="cuda", dtype=torch.bfloat16
        )
        mask = torch.zeros(len(lengths), max(lengths), device="cuda", dtype=torch.long)
        for index, segment in enumerate(segments):
            padded_input[index, -segment.shape[1] :] = segment[0]
            mask[index, -segment.shape[1] :] = 1
        padded = module(padded_input, attention_mask=mask)
        padded_compact = torch.cat(
            [padded[index : index + 1, -length:] for index, length in enumerate(lengths)], dim=1
        )

        whole = module(segments[0])
        cache = DynamicCache(config=config)
        first = module(segments[0][:, :13], cache_params=cache)
        second = module(segments[0][:, 13:], cache_params=cache)
        chunked = torch.cat((first, second), dim=1)
    behavior = {
        "packed_vs_separate_max_abs": max_abs(packed, separate),
        "padded_vs_separate_max_abs": max_abs(padded_compact, separate),
        "prefill_decode_vs_whole_max_abs": max_abs(chunked, whole),
    }
    print(json.dumps({"initial": initial_metrics, "step1": step1, "step2": step2, "behavior": behavior}, indent=2))


if __name__ == "__main__":
    main()
