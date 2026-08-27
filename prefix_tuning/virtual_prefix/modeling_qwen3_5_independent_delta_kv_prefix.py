"""Qwen3.5 with independent Delta-space K/V/beta/a prefixes.

Each Gated DeltaNet layer owns four prefix tables in the projected Delta
space.  The prefix is executed only to build the convolution and recurrent
states from which user tokens start; it never enters input_ids, positions, or
the model context budget.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen3_5
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    Qwen3_5GatedDeltaNet,
)


class _FunctionalDeltaCache:
    """Small differentiable cache used between prefix and user chunks."""

    def __init__(self, layer_idx: int, conv_states: torch.Tensor, recurrent_states: torch.Tensor):
        self.layer_idx = layer_idx
        self.layers = [None] * (layer_idx + 1)
        self.layers[layer_idx] = SimpleNamespace(
            conv_states=conv_states,
            recurrent_states=recurrent_states,
        )

    def has_previous_state(self, layer_idx: int | None = None) -> bool:
        return layer_idx is None or layer_idx == self.layer_idx

    def update_conv_state(self, conv_states: torch.Tensor, layer_idx: int, **kwargs):
        if layer_idx != self.layer_idx:
            raise IndexError(layer_idx)
        self.layers[layer_idx].conv_states = conv_states
        return conv_states

    def update_recurrent_state(self, recurrent_states: torch.Tensor, layer_idx: int, **kwargs):
        if layer_idx != self.layer_idx:
            raise IndexError(layer_idx)
        self.layers[layer_idx].recurrent_states = recurrent_states
        return recurrent_states


class Qwen3_5IndependentDeltaKVPrefixGatedDeltaNet(Qwen3_5GatedDeltaNet):
    """Stock Qwen3.5 GDN seeded by independently learned Delta parameters."""

    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.num_virtual_tokens = int(
            getattr(config, "independent_delta_prefix_num_virtual_tokens", 0)
        )
        if self.num_virtual_tokens <= 0:
            raise ValueError(
                "text_config.independent_delta_prefix_num_virtual_tokens must be positive"
            )

        self.prefix_k = torch.nn.Parameter(
            torch.zeros(self.num_virtual_tokens, self.key_dim)
        )
        self.prefix_v = torch.nn.Parameter(
            torch.empty(self.num_virtual_tokens, self.value_dim)
        )
        self.prefix_beta_logits = torch.nn.Parameter(
            torch.zeros(self.num_virtual_tokens, self.num_v_heads)
        )
        self.prefix_a = torch.nn.Parameter(
            torch.zeros(self.num_virtual_tokens, self.num_v_heads)
        )

        init_std = float(getattr(config, "independent_delta_prefix_v_init_std", 3e-4))
        if init_std <= 0:
            raise ValueError("independent_delta_prefix_v_init_std must be positive")
        torch.nn.init.normal_(self.prefix_v, mean=0.0, std=init_std)
        # The user convolution sees the final K-1 raw projected values.  Keep
        # that direct path exactly zero at initialization so the prepared model
        # is initially function-identical to the base model.
        history_length = self.conv_kernel_size - 1
        with torch.no_grad():
            self.prefix_v[-history_length:].zero_()

    def _raw_prefix_qkv(self, dtype: torch.dtype) -> torch.Tensor:
        """Return raw pre-convolution [Q=0, K, V] in [1, C, M] layout."""

        prefix_k = self.prefix_k.to(dtype)
        prefix_v = self.prefix_v.to(dtype)
        prefix_q = prefix_k.new_zeros(prefix_k.shape)
        return torch.cat((prefix_q, prefix_k, prefix_v), dim=-1).transpose(0, 1).unsqueeze(0)

    def _compute_prefix_states(
        self,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return packed conv tail, HF conv cache, and final recurrent state."""

        raw_qkv = self._raw_prefix_qkv(dtype)
        history_length = self.conv_kernel_size - 1
        packed_conv_tail = F.pad(
            raw_qkv,
            (max(history_length - raw_qkv.shape[-1], 0), 0),
        )[:, :, -history_length:]
        # The Transformers cache stores K raw values.  Its chunk-continuation
        # path concatenates this cache before the new chunk and then retains
        # only the final user outputs.
        hf_conv_state = F.pad(
            raw_qkv,
            (self.conv_kernel_size - raw_qkv.shape[-1], 0),
        )[:, :, -self.conv_kernel_size:]

        convolved = self.causal_conv1d_fn(
            x=raw_qkv,
            weight=self.conv1d.weight.squeeze(1),
            bias=self.conv1d.bias,
            activation=self.activation,
        ).transpose(1, 2)
        query, key, value = torch.split(
            convolved,
            [self.key_dim, self.key_dim, self.value_dim],
            dim=-1,
        )
        query = query.reshape(1, self.num_virtual_tokens, -1, self.head_k_dim)
        key = key.reshape(1, self.num_virtual_tokens, -1, self.head_k_dim)
        value = value.reshape(1, self.num_virtual_tokens, -1, self.head_v_dim)
        beta = self.prefix_beta_logits.to(dtype).sigmoid().unsqueeze(0)
        a = self.prefix_a.to(dtype).unsqueeze(0)
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            repeats = self.num_v_heads // self.num_k_heads
            query = query.repeat_interleave(repeats, dim=2)
            key = key.repeat_interleave(repeats, dim=2)

        _, recurrent_state = self.chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        if recurrent_state is None:
            raise RuntimeError("Independent Delta prefix kernel returned no final state")
        return packed_conv_tail, hf_conv_state, recurrent_state

    def _packed_causal_conv(
        self,
        mixed_qkv: torch.Tensor,
        prefix_conv_tail: torch.Tensor,
        cu_seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        lengths = (cu_seq_lens[1:] - cu_seq_lens[:-1]).to(dtype=torch.long)
        history_length = prefix_conv_tail.shape[-1]
        projected_segments = torch.split(mixed_qkv.squeeze(0), lengths.tolist(), dim=-1)
        extended_segments = [
            torch.cat((prefix_conv_tail.squeeze(0), segment), dim=-1)
            for segment in projected_segments
        ]
        extended = torch.cat(extended_segments, dim=-1).transpose(0, 1).contiguous()
        extended = extended.unsqueeze(0).transpose(1, 2)
        extended_lengths = lengths + history_length
        seq_idx = torch.repeat_interleave(
            torch.arange(lengths.numel(), device=mixed_qkv.device, dtype=torch.int32),
            extended_lengths,
        ).unsqueeze(0)
        convolved_extended = self.causal_conv1d_fn(
            x=extended,
            weight=self.conv1d.weight.squeeze(1),
            bias=self.conv1d.bias,
            activation=self.activation,
            seq_idx=seq_idx,
        )
        convolved_segments = torch.split(
            convolved_extended.squeeze(0), extended_lengths.tolist(), dim=-1
        )
        return torch.cat(
            [segment[:, history_length:] for segment in convolved_segments], dim=-1
        ).unsqueeze(0)

    def _forward_packed(
        self,
        hidden_states: torch.Tensor,
        cu_seq_lens_q: torch.Tensor,
    ) -> torch.Tensor:
        if not qwen3_5.is_fast_path_available:
            raise RuntimeError(
                "Packed Independent Delta prefix requires FLA and causal-conv1d fast kernels"
            )
        if hidden_states.shape[0] != 1:
            raise ValueError(
                "Packed Independent Delta prefix expects batch dimension 1; "
                f"got {tuple(hidden_states.shape)}"
            )
        if cu_seq_lens_q.ndim != 1 or cu_seq_lens_q.numel() < 2:
            raise ValueError("cu_seq_lens_q must be one-dimensional with at least two entries")
        cu_seq_lens_q = cu_seq_lens_q.to(device=hidden_states.device, dtype=torch.int32)
        if int(cu_seq_lens_q[0]) != 0 or int(cu_seq_lens_q[-1]) != hidden_states.shape[1]:
            raise ValueError("cu_seq_lens_q must span exactly all packed hidden states")
        if bool(((cu_seq_lens_q[1:] - cu_seq_lens_q[:-1]) <= 0).any()):
            raise ValueError("Packed Independent Delta prefix does not support empty sequences")

        prefix_tail, _, prefix_recurrent = self._compute_prefix_states(hidden_states.dtype)
        num_sequences = cu_seq_lens_q.numel() - 1
        initial_state = prefix_recurrent.expand(num_sequences, -1, -1, -1).contiguous()

        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2).contiguous()
        mixed_qkv = self._packed_causal_conv(mixed_qkv, prefix_tail, cu_seq_lens_q)
        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv,
            [self.key_dim, self.key_dim, self.value_dim],
            dim=-1,
        )
        total_tokens = hidden_states.shape[1]
        query = query.reshape(1, total_tokens, -1, self.head_k_dim)
        key = key.reshape(1, total_tokens, -1, self.head_k_dim)
        value = value.reshape(1, total_tokens, -1, self.head_v_dim)
        z = self.in_proj_z(hidden_states).reshape(1, total_tokens, -1, self.head_v_dim)
        beta = self.in_proj_b(hidden_states).sigmoid()
        a = self.in_proj_a(hidden_states)
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            repeats = self.num_v_heads // self.num_k_heads
            query = query.repeat_interleave(repeats, dim=2)
            key = key.repeat_interleave(repeats, dim=2)

        core_attn_out, _ = self.chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seq_lens_q,
        )
        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(1, total_tokens, -1)
        return self.out_proj(core_attn_out)

    def _forward_from_prefix_cache(
        self,
        hidden_states: torch.Tensor,
        hf_conv_state: torch.Tensor,
        recurrent_state: torch.Tensor,
        attention_mask: torch.Tensor | None,
        **kwargs,
    ) -> tuple[torch.Tensor, _FunctionalDeltaCache]:
        batch_size = hidden_states.shape[0]
        working_cache = _FunctionalDeltaCache(
            self.layer_idx,
            hf_conv_state.expand(batch_size, -1, -1).clone(),
            recurrent_state.expand(batch_size, -1, -1, -1).clone(),
        )
        output = super().forward(
            hidden_states=hidden_states,
            cache_params=working_cache,
            attention_mask=attention_mask,
            **kwargs,
        )
        return output, working_cache

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params=None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ):
        has_state = cache_params is not None and cache_params.has_previous_state(self.layer_idx)
        if has_state:
            return super().forward(
                hidden_states=hidden_states,
                cache_params=cache_params,
                attention_mask=attention_mask,
                **kwargs,
            )

        cu_seq_lens_q = kwargs.get("cu_seq_lens_q")
        if cu_seq_lens_q is not None:
            if cache_params is not None:
                raise ValueError("Packed Independent Delta prefix does not support cache_params")
            if attention_mask is not None:
                raise ValueError("Packed Independent Delta prefix expects attention_mask=None")
            return self._forward_packed(hidden_states, cu_seq_lens_q)

        _, hf_conv_state, recurrent_state = self._compute_prefix_states(hidden_states.dtype)
        token_mask = None
        if attention_mask is not None:
            token_mask = attention_mask[:, -hidden_states.shape[1] :].to(torch.bool)

        if token_mask is None or bool(token_mask.all()):
            output, working_cache = self._forward_from_prefix_cache(
                hidden_states,
                hf_conv_state,
                recurrent_state,
                attention_mask,
                **kwargs,
            )
            if cache_params is not None:
                layer_cache = working_cache.layers[self.layer_idx]
                cache_params.update_conv_state(layer_cache.conv_states, self.layer_idx)
                cache_params.update_recurrent_state(layer_cache.recurrent_states, self.layer_idx)
            return output

        # Padding tokens must occur before the prefix semantically.  Run each
        # compact valid sequence from the same prefix state, then scatter its
        # outputs (and, when requested, final cache) back to batch layout.
        restored = hidden_states.new_zeros(hidden_states.shape)
        final_conv_states = []
        final_recurrent_states = []
        for batch_idx in range(hidden_states.shape[0]):
            positions = token_mask[batch_idx].nonzero(as_tuple=False).flatten()
            if positions.numel() == 0:
                final_conv_states.append(hf_conv_state[0])
                final_recurrent_states.append(recurrent_state[0])
                continue
            compact = hidden_states[batch_idx : batch_idx + 1, positions]
            compact_output, compact_cache = self._forward_from_prefix_cache(
                compact,
                hf_conv_state,
                recurrent_state,
                None,
                **kwargs,
            )
            restored[batch_idx, positions] = compact_output[0]
            layer_cache = compact_cache.layers[self.layer_idx]
            final_conv_states.append(layer_cache.conv_states[0])
            final_recurrent_states.append(layer_cache.recurrent_states[0])
        if cache_params is not None:
            cache_params.update_conv_state(torch.stack(final_conv_states), self.layer_idx)
            cache_params.update_recurrent_state(torch.stack(final_recurrent_states), self.layer_idx)
        return restored


class Qwen3_5IndependentDeltaKVPrefixForConditionalGeneration(
    Qwen3_5ForConditionalGeneration
):
    """Qwen3.5 conditional generation with independent Delta-space prefixes."""

    def __init__(self, config):
        original_cls = qwen3_5.Qwen3_5GatedDeltaNet
        qwen3_5.Qwen3_5GatedDeltaNet = Qwen3_5IndependentDeltaKVPrefixGatedDeltaNet
        try:
            super().__init__(config)
        finally:
            qwen3_5.Qwen3_5GatedDeltaNet = original_cls


__all__ = [
    "Qwen3_5IndependentDeltaKVPrefixForConditionalGeneration",
    "Qwen3_5IndependentDeltaKVPrefixGatedDeltaNet",
]
