#!/usr/bin/env python3
"""Functional and two-step gradient checks for residual soft-prefix attention."""

from __future__ import annotations

import json

import torch
from transformers import DynamicCache
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Attention,
    Qwen3_5TextRotaryEmbedding,
)

from prefix_tuning.virtual_prefix.modeling_qwen3_5_hybrid_delta_residual_attention_prefix import (
    Qwen3_5ResidualSoftPrefixAttention,
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
        layer_types=["full_attention"],
        dtype=torch.bfloat16,
    )
    config.residual_attention_prefix_num_virtual_tokens = 64
    config.residual_attention_prefix_key_init_std = 0.02
    config._attn_implementation = "eager"
    return config


def causal_mask(batch: int, query_positions: torch.Tensor, key_length: int) -> torch.Tensor:
    key_positions = torch.arange(key_length, device=query_positions.device)
    allowed = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
    mask = torch.zeros(
        query_positions.numel(), key_length, device=query_positions.device, dtype=torch.bfloat16
    )
    mask.masked_fill_(~allowed, torch.finfo(torch.bfloat16).min)
    return mask.unsqueeze(0).unsqueeze(0).expand(batch, 1, -1, -1)


def rotary(rotary_emb, hidden_states, positions):
    return rotary_emb(hidden_states, positions.unsqueeze(0).expand(hidden_states.shape[0], -1))


def stats(parameter: torch.nn.Parameter) -> dict:
    grad = parameter.grad
    if grad is None:
        return {"finite": True, "norm": 0.0, "nonzero_rows": 0}
    row_norm = grad.float().reshape(grad.shape[0], -1).norm(dim=1)
    return {
        "finite": bool(torch.isfinite(grad).all()),
        "norm": float(grad.float().norm()),
        "nonzero_rows": int(torch.count_nonzero(row_norm)),
    }


def main() -> None:
    torch.manual_seed(7)
    config = make_config()
    module = Qwen3_5ResidualSoftPrefixAttention(config, 0).cuda().bfloat16()
    base = Qwen3_5Attention(config, 0).cuda().bfloat16()
    base.load_state_dict(
        {
            name: tensor
            for name, tensor in module.state_dict().items()
            if not name.startswith("prefix_") and not name.startswith("_cached_")
        },
        strict=True,
    )
    rotary_emb = Qwen3_5TextRotaryEmbedding(config).cuda().bfloat16()
    for name, parameter in module.named_parameters():
        parameter.requires_grad_(name.startswith("prefix_"))

    batch, length = 2, 23
    x = torch.randn(batch, length, config.hidden_size, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(length, device="cuda")
    position_embeddings = rotary(rotary_emb, x, positions)
    mask = causal_mask(batch, positions, length)

    with torch.no_grad():
        initial = module(x, position_embeddings, mask)[0]
        reference = base(x, position_embeddings, mask)[0]
    initial_error = float((initial.float() - reference.float()).abs().max())
    if initial_error != 0:
        raise AssertionError(f"Initial residual prefix changed the base function: {initial_error}")

    weight = torch.randn_like(initial, dtype=torch.float32)
    output = module(x, position_embeddings, mask)[0]
    (output.float() * weight).sum().backward()
    step1 = {
        "key": stats(module.prefix_key_tokens),
        "value": stats(module.prefix_value_tokens),
    }
    if step1["value"]["nonzero_rows"] != module.num_prefix_tokens:
        raise AssertionError(f"Incomplete step-1 value row coverage: {step1}")
    if step1["key"]["norm"] != 0:
        raise AssertionError(f"Key prefix should be dormant while V=0: {step1}")

    with torch.no_grad():
        value_grad = module.prefix_value_tokens.grad.float()
        module.prefix_value_tokens.copy_(
            (-1e-3 * value_grad / value_grad.norm().clamp_min(1e-12)).to(
                module.prefix_value_tokens
            )
        )
    module.zero_grad(set_to_none=True)
    output = module(x, position_embeddings, mask)[0]
    (output.float() * weight).sum().backward()
    step2 = {
        "key": stats(module.prefix_key_tokens),
        "value": stats(module.prefix_value_tokens),
    }
    if any(item["nonzero_rows"] != module.num_prefix_tokens for item in step2.values()):
        raise AssertionError(f"Incomplete step-2 gradient coverage: {step2}")

    module.eval()
    module.zero_grad(set_to_none=True)
    split = 13
    with torch.no_grad():
        whole = module(x, position_embeddings, mask)[0]
        cache = DynamicCache(config=config)
        first_positions = positions[:split]
        first = module(
            x[:, :split],
            rotary(rotary_emb, x[:, :split], first_positions),
            causal_mask(batch, first_positions, split),
            past_key_values=cache,
        )[0]
        second_positions = positions[split:]
        second = module(
            x[:, split:],
            rotary(rotary_emb, x[:, split:], second_positions),
            causal_mask(batch, second_positions, length),
            past_key_values=cache,
        )[0]
        chunked = torch.cat((first, second), dim=1)
    cache_error = float((whole.float() - chunked.float()).abs().max())
    print(
        json.dumps(
            {
                "initial_base_max_abs": initial_error,
                "step1": step1,
                "step2": step2,
                "prefill_decode_max_abs": cache_error,
                "prefix_projection_cache_misses": module._prefix_projection_cache_misses,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
