"""Qwen3.5 with Delta-space GDN prefixes and residual attention prefixes.

The 24 Gated DeltaNet layers retain their independent K/V/beta/a prefix.
Each of the eight full-attention layers additionally owns independent
hidden-space key/value soft tokens.  They form a shared, non-causal residual
attention branch for user queries and never enter input ids or the user KV
cache.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen3_5
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention

from .joint_prefix_attention import (
    anchored_prefix_attention,
    compiled_gdn_anchored_attention,
    compiled_gdn_token_null_attention,
    gdn_anchored_attention,
    gdn_token_null_attention,
    token_null_attention,
    flex_full_attention_with_prefix,
    full_attention_with_prefix,
    compiled_gdn_state_rows_attention,
    gdn_state_rows_attention,
)
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
        self.joint_prefix = bool(getattr(config, "residual_attention_prefix_joint_softmax", False))
        self.prefix_mode = getattr(config, "residual_attention_prefix_mode", "joint")
        if self.prefix_mode not in ("joint", "anchored_additive", "token_null_additive"):
            raise ValueError(f"Unknown full-attention prefix mode: {self.prefix_mode}")
        if self.prefix_mode != "joint" and not self.joint_prefix:
            raise ValueError("Anchored additive attention requires a joint-softmax hybrid checkpoint")
        if self.joint_prefix:
            initial_bias = -math.log(self.num_prefix_tokens) - 8.0
            self.prefix_logit_bias = torch.nn.Parameter(
                torch.full((self.num_attention_heads,), initial_bias)
            )
        if self.prefix_mode == "token_null_additive":
            self.prefix_null_proj = torch.nn.Parameter(torch.zeros(self.num_attention_heads, config.hidden_size))
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

        enabled = not getattr(self, "prefix_disabled", False)
        if enabled and self.joint_prefix and self.prefix_mode == "joint":
            prefix_key, prefix_value = self._get_prefix_key_value(query_states.dtype)
            options = dict(dropout=self.attention_dropout if self.training else 0.0, scale=self.scaling)
            causal_attention = full_attention_with_prefix
            if (
                attention_mask is None
                and self.config._attn_implementation == "flash_attention_2"
                and query_states.is_cuda
                and query_states.dtype in (torch.float16, torch.bfloat16)
                and options["dropout"] == 0.0
            ):
                causal_attention = flex_full_attention_with_prefix
                options.pop("dropout")
            cu_seqlens = kwargs.get("cu_seq_lens_q")
            if cu_seqlens is None:
                if attention_mask is not None:
                    options["context_mask"] = attention_mask
                attn_output = causal_attention(
                    query_states, key_states, value_states, prefix_key, prefix_value,
                    self.prefix_logit_bias, **options,
                )
            else:
                if past_key_values is not None or attention_mask is not None:
                    raise ValueError("Packed joint prefix attention expects no cache or explicit mask")
                boundaries = cu_seqlens.tolist()
                attn_output = torch.cat([
                    causal_attention(
                        query_states[..., start:end, :], key_states[..., start:end, :],
                        value_states[..., start:end, :], prefix_key, prefix_value,
                        self.prefix_logit_bias, **options,
                    )
                    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True)
                ], dim=-2)
            attn_output = attn_output.transpose(1, 2)
            attn_weights = None
        else:
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
            if enabled and self.prefix_mode in ("anchored_additive", "token_null_additive"):
                key, value = self._get_prefix_key_value(query_states.dtype)
                if self.prefix_mode == "token_null_additive":
                    null = F.linear(hidden_states.float(), self.prefix_null_proj.float()).transpose(1, 2).unsqueeze(-1)
                    prefix_output, mass, raw_lse = token_null_attention(
                        query_states, key, value, self.prefix_logit_bias, null, scale=self.scaling,
                    )
                    calibration = getattr(self, "prefix_calibration", None)
                    if calibration is not None:
                        calibration(raw_lse.detach())
                else:
                    prefix_output, mass = anchored_prefix_attention(
                        query_states, key, value, self.prefix_logit_bias, scale=self.scaling,
                    )
                monitor = getattr(self, "prefix_monitor", None)
                if monitor is not None:
                    monitor(mass.detach(), prefix_output.detach(), attn_output.detach())
                attn_output = attn_output + prefix_output.transpose(1, 2).to(attn_output.dtype)
            elif enabled:
                attn_output = attn_output + self._residual_prefix_attention(query_states)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        return self.o_proj(attn_output), attn_weights


class Qwen3_5StateRowsPrefixGatedDeltaNet(qwen3_5.Qwen3_5GatedDeltaNet):
    """Compete with the native post-convolution readout without seeding its state."""

    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        count = int(config.independent_delta_prefix_num_virtual_tokens)
        if count <= 0:
            raise ValueError("GDN prefix length must be positive")
        self.prefix_state_key = torch.nn.Parameter(
            torch.empty(1, self.num_v_heads, count, self.head_k_dim)
        )
        self.prefix_state_value = torch.nn.Parameter(
            torch.zeros(1, self.num_v_heads, count, self.head_v_dim)
        )
        self.prefix_state_temperature = torch.nn.Parameter(torch.ones(self.num_v_heads))
        self.prefix_mode = getattr(config, "independent_delta_prefix_method", "state_rows_joint_softmax")
        self.anchored_prefix = self.prefix_mode in ("anchored_additive", "token_null_additive")
        if self.anchored_prefix:
            self.prefix_state_temperature.requires_grad_(False)
        initial_bias = -math.log(count) - 8.0 if self.anchored_prefix else 0.0
        self.prefix_state_bias = torch.nn.Parameter(torch.full((self.num_v_heads,), initial_bias))
        if self.prefix_mode == "token_null_additive":
            self.prefix_null_proj = torch.nn.Parameter(torch.zeros(self.num_v_heads, config.hidden_size))
        torch.nn.init.normal_(self.prefix_state_key, std=0.02)
        self._use_compiled_prefix = config._attn_implementation == "flash_attention_2"
        self.register_buffer("_cached_state_prefix_key", None, persistent=False)
        self.register_buffer("_cached_state_prefix_value", None, persistent=False)
        self._state_prefix_cache_key = None
        self._native_chunk_rule = self.chunk_gated_delta_rule
        self._native_recurrent_rule = self.recurrent_gated_delta_rule
        self.chunk_gated_delta_rule = self._chunk_rule_with_prefix
        self.recurrent_gated_delta_rule = self._recurrent_rule_with_prefix

    def _project_state_prefix(self):
        if self.training or torch.is_grad_enabled():
            return F.normalize(self.prefix_state_key.float(), dim=-1), self.prefix_state_value.float()
        cache_key = (
            self.prefix_state_key._version, self.prefix_state_value._version,
            self.prefix_state_key.device, self.prefix_state_key.dtype,
        )
        if self._state_prefix_cache_key != cache_key:
            self._cached_state_prefix_key = F.normalize(self.prefix_state_key.float(), dim=-1)
            self._cached_state_prefix_value = self.prefix_state_value.float()
            self._state_prefix_cache_key = cache_key
        return self._cached_state_prefix_key, self._cached_state_prefix_value

    def forward(self, hidden_states, *args, **kwargs):
        if self.prefix_mode != "token_null_additive" or getattr(self, "prefix_disabled", False):
            return super().forward(hidden_states, *args, **kwargs)
        self._null_projection = F.linear(hidden_states.float(), self.prefix_null_proj.float()).transpose(1, 2).unsqueeze(-1)
        try:
            return super().forward(hidden_states, *args, **kwargs)
        finally:
            self._null_projection = None

    def _add_prefix(self, query, readout):
        if getattr(self, "prefix_disabled", False):
            return readout
        query = query.transpose(1, 2)
        readout = readout.transpose(1, 2)
        if self.anchored_prefix:
            key, value = self._project_state_prefix()
            attention = (compiled_gdn_anchored_attention() if self._use_compiled_prefix and query.is_cuda
                         else gdn_anchored_attention)
            if self.prefix_mode == "token_null_additive":
                attention = (compiled_gdn_token_null_attention() if self._use_compiled_prefix and query.is_cuda
                             else gdn_token_null_attention)
                contribution, mass, raw_lse = attention(query, key, value, self.prefix_state_bias, self._null_projection)
                calibration = getattr(self, "prefix_calibration", None)
                if calibration is not None:
                    calibration(raw_lse.detach())
            else:
                contribution, mass = attention(query, key, value, self.prefix_state_bias)
            monitor = getattr(self, "prefix_monitor", None)
            if monitor is not None:
                monitor(mass.detach(), contribution.detach(), readout.detach().transpose(1, 2))
            return (readout + contribution.to(readout.dtype)).transpose(1, 2)
        clamp = (-80.0, 80.0)
        if self._use_compiled_prefix and query.is_cuda:
            prefix_key, prefix_value = self._project_state_prefix()
            output = compiled_gdn_state_rows_attention()(
                readout, query, prefix_key, prefix_value,
                self.prefix_state_temperature, self.prefix_state_bias, clamp,
            )
        else:
            output = gdn_state_rows_attention(
                readout, query, self.prefix_state_key, self.prefix_state_value,
                self.prefix_state_temperature, self.prefix_state_bias, clamp=clamp,
            )
        return output.transpose(1, 2)

    def _chunk_rule_with_prefix(self, query, key, value, **kwargs):
        readout, state = self._native_chunk_rule(query, key, value, **kwargs)
        return self._add_prefix(query, readout), state

    def _recurrent_rule_with_prefix(self, query, key, value, **kwargs):
        readout, state = self._native_recurrent_rule(query, key, value, **kwargs)
        return self._add_prefix(query, readout), state


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

    def _delta_net_class(self, config):
        if getattr(config.text_config, "residual_attention_prefix_joint_softmax", False):
            return Qwen3_5StateRowsPrefixGatedDeltaNet
        return super()._delta_net_class(config)

    def invalidate_residual_attention_prefix_caches(self) -> None:
        for module in self.modules():
            if isinstance(module, Qwen3_5ResidualSoftPrefixAttention):
                module.invalidate_residual_attention_prefix_cache()


__all__ = [
    "Qwen3_5HybridDeltaResidualAttentionPrefixForConditionalGeneration",
    "Qwen3_5ResidualSoftPrefixAttention",
]
