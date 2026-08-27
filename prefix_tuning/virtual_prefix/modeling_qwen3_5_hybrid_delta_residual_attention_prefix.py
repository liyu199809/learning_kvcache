"""Qwen3.5 with Delta-space GDN prefixes and residual attention prefixes.

The 24 Gated DeltaNet layers retain their independent K/V/beta/a prefix.
Each of the eight full-attention layers additionally owns independent
hidden-space key/value soft tokens.  They form a shared, non-causal residual
attention branch for user queries and never enter input ids or the user KV
cache.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen3_5
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention

from .modeling_qwen3_5_independent_delta_kv_prefix import (
    Qwen3_5IndependentDeltaKVPrefixForConditionalGeneration,
)


class Qwen3_5ResidualSoftPrefixAttention(Qwen3_5Attention):
    """Native causal self-attention plus a shared residual soft-prefix branch."""

    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.num_prefix_tokens = int(
            getattr(config, "residual_attention_prefix_num_virtual_tokens", 0)
        )
        if self.num_prefix_tokens <= 0:
            raise ValueError(
                "text_config.residual_attention_prefix_num_virtual_tokens must be positive"
            )
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.prefix_key_tokens = torch.nn.Parameter(
            torch.empty(self.num_prefix_tokens, config.hidden_size)
        )
        self.prefix_value_tokens = torch.nn.Parameter(
            torch.zeros(self.num_prefix_tokens, config.hidden_size)
        )
        init_std = float(
            getattr(config, "residual_attention_prefix_key_init_std", 0.02)
        )
        if init_std <= 0:
            raise ValueError("residual_attention_prefix_key_init_std must be positive")
        torch.nn.init.normal_(self.prefix_key_tokens, mean=0.0, std=init_std)

        # Prefix positions are logically immediately before user position 0.
        # Keeping user positions unchanged preserves the base model's cache and
        # context semantics while RoPE sees the correct relative distance.
        self.prefix_rotary_emb = qwen3_5.Qwen3_5TextRotaryEmbedding(config)
        self.register_buffer("_cached_prefix_key", None, persistent=False)
        self.register_buffer("_cached_prefix_value", None, persistent=False)
        self._cached_prefix_projection_key = None
        self._prefix_projection_cache_misses = 0

    def invalidate_residual_attention_prefix_cache(self) -> None:
        self._cached_prefix_key = None
        self._cached_prefix_value = None
        self._cached_prefix_projection_key = None

    def _compute_prefix_key_value(
        self,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key_tokens = self.prefix_key_tokens.to(dtype)
        value_tokens = self.prefix_value_tokens.to(dtype)
        prefix_key = self.k_norm(
            self.k_proj(key_tokens).view(
                1, self.num_prefix_tokens, self.num_key_value_heads, self.head_dim
            )
        ).transpose(1, 2)
        prefix_value = self.v_proj(value_tokens).view(
            1, self.num_prefix_tokens, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        prefix_positions = torch.arange(
            -self.num_prefix_tokens,
            0,
            device=key_tokens.device,
            dtype=torch.long,
        ).unsqueeze(0)
        prefix_cos, prefix_sin = self.prefix_rotary_emb(prefix_key, prefix_positions)
        _, prefix_key = qwen3_5.apply_rotary_pos_emb(
            prefix_key,
            prefix_key,
            prefix_cos,
            prefix_sin,
        )
        return prefix_key.contiguous(), prefix_value.contiguous()

    def _get_prefix_key_value(
        self,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Training and gradient-checkpoint recomputation must retain the graph.
        if self.training or torch.is_grad_enabled():
            return self._compute_prefix_key_value(dtype)

        cache_key = (
            self.prefix_key_tokens._version,
            self.prefix_value_tokens._version,
            self.prefix_key_tokens.device,
            dtype,
        )
        if cache_key != self._cached_prefix_projection_key:
            prefix_key, prefix_value = self._compute_prefix_key_value(dtype)
            self._cached_prefix_key = prefix_key.detach()
            self._cached_prefix_value = prefix_value.detach()
            self._cached_prefix_projection_key = cache_key
            self._prefix_projection_cache_misses += 1
        return self._cached_prefix_key, self._cached_prefix_value

    def _residual_prefix_attention(
        self,
        query_states: torch.Tensor,
    ) -> torch.Tensor:
        prefix_key, prefix_value = self._get_prefix_key_value(query_states.dtype)
        batch_size = query_states.shape[0]
        prefix_key = prefix_key.expand(batch_size, -1, -1, -1)
        prefix_value = prefix_value.expand(batch_size, -1, -1, -1)
        if self.num_key_value_groups > 1:
            prefix_key = qwen3_5.repeat_kv(prefix_key, self.num_key_value_groups)
            prefix_value = qwen3_5.repeat_kv(prefix_value, self.num_key_value_groups)
        dropout_p = self.attention_dropout if self.training else 0.0
        output = F.scaled_dot_product_attention(
            query_states,
            prefix_key,
            prefix_value,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=False,
            scale=self.scaling,
        )
        return output.transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values=None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2),
            2,
            dim=-1,
        )
        gate = gate.reshape(*input_shape, -1)
        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = qwen3_5.apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx
            )

        attention_interface = qwen3_5.ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation,
            qwen3_5.eager_attention_forward,
        )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )
        attn_output = attn_output + self._residual_prefix_attention(query_states)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        return self.o_proj(attn_output), attn_weights


class Qwen3_5HybridDeltaResidualAttentionPrefixForConditionalGeneration(
    Qwen3_5IndependentDeltaKVPrefixForConditionalGeneration
):
    """Qwen3.5 with both recurrent Delta and residual attention prefixes."""

    def __init__(self, config):
        original_cls = qwen3_5.Qwen3_5Attention
        qwen3_5.Qwen3_5Attention = Qwen3_5ResidualSoftPrefixAttention
        try:
            super().__init__(config)
        finally:
            qwen3_5.Qwen3_5Attention = original_cls

    def invalidate_residual_attention_prefix_caches(self) -> None:
        for module in self.modules():
            if isinstance(module, Qwen3_5ResidualSoftPrefixAttention):
                module.invalidate_residual_attention_prefix_cache()


__all__ = [
    "Qwen3_5HybridDeltaResidualAttentionPrefixForConditionalGeneration",
    "Qwen3_5ResidualSoftPrefixAttention",
]
