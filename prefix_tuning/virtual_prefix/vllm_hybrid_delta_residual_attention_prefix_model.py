"""Native vLLM Qwen3.5 model with Delta and residual attention prefixes."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models import qwen3_5
from vllm.model_executor.models.qwen3_next import Qwen3NextAttention
from vllm.model_executor.utils import set_weight_attrs

from .vllm_independent_delta_kv_prefix_model import (
    Qwen3_5IndependentDeltaKVPrefixForConditionalGeneration,
)


class Qwen3_5ResidualSoftPrefixAttention(Qwen3NextAttention):
    """Native self-attention plus a shared non-causal prefix-only branch."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_prefix_tokens = int(
            getattr(
                self.config,
                "residual_attention_prefix_num_virtual_tokens",
                0,
            )
        )
        if self.num_prefix_tokens <= 0:
            raise ValueError(
                "text_config.residual_attention_prefix_num_virtual_tokens "
                "must be positive"
            )

        self.prefix_key_tokens = torch.nn.Parameter(
            torch.empty(self.num_prefix_tokens, self.hidden_size)
        )
        self.prefix_value_tokens = torch.nn.Parameter(
            torch.zeros(self.num_prefix_tokens, self.hidden_size)
        )
        init_std = float(
            getattr(self.config, "residual_attention_prefix_key_init_std", 0.02)
        )
        if init_std <= 0:
            raise ValueError("residual_attention_prefix_key_init_std must be positive")
        torch.nn.init.normal_(self.prefix_key_tokens, mean=0.0, std=init_std)
        for parameter in (self.prefix_key_tokens, self.prefix_value_tokens):
            # Hidden-space soft tokens are replicated. TP=1 is used by the OPSD
            # launcher; the default loader also gives correct replicated TP loads.
            set_weight_attrs(parameter, {"weight_loader": default_weight_loader})

        self.register_buffer("_cached_prefix_key", None, persistent=False)
        self.register_buffer("_cached_prefix_value", None, persistent=False)
        self._cached_prefix_projection_key = None
        self._prefix_projection_cache_misses = 0

    def invalidate_residual_attention_prefix_cache(self) -> None:
        self._cached_prefix_key = None
        self._cached_prefix_value = None
        self._cached_prefix_projection_key = None

    def _split_qkv(self, qkv: torch.Tensor):
        if self.attn_output_gate:
            return qkv.split([self.q_size * 2, self.kv_size, self.kv_size], dim=-1)
        return qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

    def _compute_prefix_key_value(
        self, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key_qkv, _ = self.qkv_proj(self.prefix_key_tokens.to(dtype))
        value_qkv, _ = self.qkv_proj(self.prefix_value_tokens.to(dtype))
        _, prefix_key, _ = self._split_qkv(key_qkv)
        _, _, prefix_value = self._split_qkv(value_qkv)

        prefix_key = self.k_norm(
            prefix_key.view(-1, self.num_kv_heads, self.head_dim)
        ).view(-1, self.kv_size)
        prefix_positions = torch.arange(
            -self.num_prefix_tokens,
            0,
            device=prefix_key.device,
            dtype=torch.float32,
        )
        # vLLM's normal RoPE path indexes a cache and therefore cannot accept
        # negative positions.  Compute the same default partial RoPE directly
        # for logical positions [-M, -1].  Text tokens use the same coordinate
        # for all three MRoPE axes, so interleaving does not change these
        # frequencies.
        rotary_dim = self.rotary_emb.rotary_dim
        inv_freq = 1.0 / (
            self.rotary_emb.base
            ** (
                torch.arange(
                    0,
                    rotary_dim,
                    2,
                    device=prefix_key.device,
                    dtype=torch.float32,
                )
                / rotary_dim
            )
        )
        freqs = torch.outer(prefix_positions, inv_freq)
        cos = freqs.cos().to(prefix_key.dtype).unsqueeze(1)
        sin = freqs.sin().to(prefix_key.dtype).unsqueeze(1)
        prefix_key = prefix_key.view(-1, self.num_kv_heads, self.head_dim)
        rotated, passed = prefix_key[..., :rotary_dim], prefix_key[..., rotary_dim:]
        if self.rotary_emb.is_neox_style:
            first, second = rotated.chunk(2, dim=-1)
            rotated = torch.cat(
                (first * cos - second * sin, second * cos + first * sin), dim=-1
            )
        else:
            even, odd = rotated[..., 0::2], rotated[..., 1::2]
            rotated = torch.stack(
                (even * cos - odd * sin, odd * cos + even * sin), dim=-1
            ).flatten(-2)
        prefix_key = torch.cat((rotated, passed), dim=-1)
        return (
            prefix_key.contiguous(),
            prefix_value.view(-1, self.num_kv_heads, self.head_dim).contiguous(),
        )

    def _get_prefix_key_value(
        self, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

    def _residual_prefix_attention(self, query: torch.Tensor) -> torch.Tensor:
        prefix_key, prefix_value = self._get_prefix_key_value(query.dtype)
        kv_groups = self.num_heads // self.num_kv_heads
        if kv_groups > 1:
            prefix_key = prefix_key.repeat_interleave(kv_groups, dim=1)
            prefix_value = prefix_value.repeat_interleave(kv_groups, dim=1)
        # vLLM flattens all live request tokens.  Every query attends only to
        # the same immutable prefix table, so this branch cannot mix requests.
        output = F.scaled_dot_product_attention(
            query.transpose(0, 1).unsqueeze(0),
            prefix_key.transpose(0, 1).unsqueeze(0),
            prefix_value.transpose(0, 1).unsqueeze(0),
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=self.scaling,
        )
        return output.squeeze(0).transpose(0, 1).reshape(-1, self.q_size)

    def forward(
        self,
        positions: torch.Tensor,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
    ):
        qkv, _ = self.qkv_proj(hidden_states)
        if self.attn_output_gate:
            q_gate, k, v = self._split_qkv(qkv)
            q_gate = q_gate.view(-1, self.num_heads, self.head_dim * 2)
            q, gate = torch.chunk(q_gate, 2, dim=-1)
            q = q.reshape(-1, self.q_size)
            gate = gate.reshape(-1, self.q_size)
        else:
            q, k, v = self._split_qkv(qkv)

        q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view(
            -1, self.q_size
        )
        k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).view(
            -1, self.kv_size
        )
        q, k = self.rotary_emb(positions, q, k)

        attn_output = self.attn(q, k, v)
        query = q.view(-1, self.num_heads, self.head_dim)
        attn_output = attn_output + self._residual_prefix_attention(query)
        if self.attn_output_gate:
            attn_output = attn_output * torch.sigmoid(gate)
        output[:], _ = self.o_proj(attn_output)


class Qwen3_5HybridDeltaResidualAttentionPrefixForConditionalGeneration(
    Qwen3_5IndependentDeltaKVPrefixForConditionalGeneration
):
    """Native vLLM Qwen3.5 with both approved prefix mechanisms."""

    def __init__(self, *, vllm_config, prefix: str = "model"):
        original_cls = qwen3_5.Qwen3NextAttention
        qwen3_5.Qwen3NextAttention = Qwen3_5ResidualSoftPrefixAttention
        try:
            super().__init__(vllm_config=vllm_config, prefix=prefix)
        finally:
            qwen3_5.Qwen3NextAttention = original_cls

    def invalidate_residual_attention_prefix_caches(self) -> None:
        for module in self.modules():
            if isinstance(module, Qwen3_5ResidualSoftPrefixAttention):
                module.invalidate_residual_attention_prefix_cache()

    def load_weights(self, weights):
        loaded = super().load_weights(weights)
        expected = {
            name
            for name, _ in self.named_parameters()
            if name.endswith((".prefix_key_tokens", ".prefix_value_tokens"))
        }
        missing = expected.difference(loaded)
        if missing:
            raise RuntimeError(
                "vLLM did not load all residual attention prefix tensors: "
                + ", ".join(sorted(missing))
            )
        self.invalidate_residual_attention_prefix_caches()
        return loaded


__all__ = ["Qwen3_5HybridDeltaResidualAttentionPrefixForConditionalGeneration"]
