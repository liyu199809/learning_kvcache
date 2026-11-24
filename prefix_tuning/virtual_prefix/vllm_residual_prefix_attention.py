"""Shared residual attention branch used by native Qwen3 and Qwen3.5 vLLM models."""
from __future__ import annotations
import torch
import torch.nn.functional as F


class ResidualPrefixAttentionMixin:
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
        # Weight loading refreshes these buffers outside the captured forward.
        if (getattr(self, "_prefix_cache_refreshed", False)
                and self._cached_prefix_key is not None
                and self._cached_prefix_key.dtype == dtype):
            return self._cached_prefix_key, self._cached_prefix_value
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

