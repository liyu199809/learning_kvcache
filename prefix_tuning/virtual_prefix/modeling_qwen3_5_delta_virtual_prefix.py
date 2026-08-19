"""Qwen3.5 with layer-specific virtual tokens for every DeltaNet mixer.

For a new sequence, linear-attention layer ``l`` consumes
``[P_l, X_l]`` where ``P_l`` is an independently learned ``[M, hidden]``
continuous-token table. Only the outputs for ``X_l`` leave the mixer. Decode
uses the convolution/recurrent cache produced by prefill and does not prepend
the virtual tokens again.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import torch
from transformers import DynamicCache
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen3_5
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    Qwen3_5GatedDeltaNet,
)


@dataclass
class _PrefixLayout:
    """How to project the extended mixer output back to the user layout."""

    valid_positions: list[torch.Tensor] | None
    user_starts: list[int] | None
    original_length: int
    prefix_length: int


class _FunctionalDeltaCache:
    """Minimal per-layer cache that replaces tensors instead of mutating them."""

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


class Qwen3_5DeltaVirtualPrefixGatedDeltaNet(Qwen3_5GatedDeltaNet):
    """Stock Qwen3.5 GDN preceded by M learned continuous tokens per layer."""

    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        self._delta_prefix_config = config
        self.num_virtual_tokens = int(getattr(config, "delta_prefix_num_virtual_tokens", 0))
        if self.num_virtual_tokens <= 0:
            raise ValueError("text_config.delta_prefix_num_virtual_tokens must be positive")
        # These tokens live at the input of the frozen GDN mixer. The decoder's
        # real hidden states have already passed input_layernorm at this point.
        self.prefix_tokens = torch.nn.Parameter(
            torch.zeros(self.num_virtual_tokens, self.hidden_size)
        )

    def _prepend_virtual_tokens(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, _PrefixLayout]:
        batch_size, seq_len, hidden_size = hidden_states.shape
        prefix = self.prefix_tokens.to(hidden_states.dtype)

        if attention_mask is None:
            extended = torch.cat(
                [prefix.unsqueeze(0).expand(batch_size, -1, -1), hidden_states],
                dim=1,
            )
            return extended, None, _PrefixLayout(None, None, seq_len, self.num_virtual_tokens)

        if attention_mask.ndim != 2:
            raise ValueError(
                "Delta virtual prefix expects the 2-D linear-attention mask; "
                f"got shape {tuple(attention_mask.shape)}"
            )
        token_mask = attention_mask[:, -seq_len:].to(device=hidden_states.device, dtype=torch.bool)
        if bool(token_mask.all()):
            extended = torch.cat(
                [prefix.unsqueeze(0).expand(batch_size, -1, -1), hidden_states],
                dim=1,
            )
            extended_mask = torch.ones(
                batch_size,
                seq_len + self.num_virtual_tokens,
                device=attention_mask.device,
                dtype=attention_mask.dtype,
            )
            return extended, extended_mask, _PrefixLayout(None, None, seq_len, self.num_virtual_tokens)

        # Compact every sample to [zero-padding, P_l, valid user tokens]. The
        # leading zeros start from the all-zero GDN state and occur before P_l,
        # so neither the depthwise convolution nor the recurrence can let
        # padding modify the virtual prefix or the final cache.
        extended = hidden_states.new_zeros(
            batch_size,
            seq_len + self.num_virtual_tokens,
            hidden_size,
        )
        extended_mask = attention_mask.new_zeros(batch_size, seq_len + self.num_virtual_tokens)
        valid_positions: list[torch.Tensor] = []
        user_starts: list[int] = []
        for batch_idx in range(batch_size):
            positions = token_mask[batch_idx].nonzero(as_tuple=False).flatten()
            num_valid = int(positions.numel())
            pad_len = seq_len - num_valid
            prefix_start = pad_len
            user_start = prefix_start + self.num_virtual_tokens
            extended[batch_idx, prefix_start:user_start] = prefix
            extended_mask[batch_idx, prefix_start:user_start] = 1
            if num_valid:
                extended[batch_idx, user_start : user_start + num_valid] = hidden_states[
                    batch_idx, positions
                ]
                extended_mask[batch_idx, user_start : user_start + num_valid] = 1
            valid_positions.append(positions)
            user_starts.append(user_start)
        return extended, extended_mask, _PrefixLayout(
            valid_positions,
            user_starts,
            seq_len,
            self.num_virtual_tokens,
        )

    @staticmethod
    def _remove_virtual_outputs(output: torch.Tensor, layout: _PrefixLayout) -> torch.Tensor:
        if layout.valid_positions is None:
            return output[:, layout.prefix_length :]

        restored = output.new_zeros(output.shape[0], layout.original_length, output.shape[-1])
        for batch_idx, positions in enumerate(layout.valid_positions):
            num_valid = int(positions.numel())
            if num_valid:
                start = layout.user_starts[batch_idx]
                restored[batch_idx, positions] = output[batch_idx, start : start + num_valid]
        return restored

    def _compute_packed_prefix_states(
        self,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute differentiable convolution and recurrent states for one prefix.

        The recurrent state can be broadcast to all packed sequences directly.
        causal-conv1d does not allow ``seq_idx`` and ``initial_states`` at the
        same time, so callers seed the packed convolution by prepending only
        the final ``kernel_size - 1`` projected prefix values per segment.
        """

        prefix = self.prefix_tokens.to(dtype).unsqueeze(0)
        mixed_qkv = self.in_proj_qkv(prefix).transpose(1, 2).contiguous()
        history_length = self.conv_kernel_size - 1
        conv_tail = torch.nn.functional.pad(
            mixed_qkv,
            (max(history_length - mixed_qkv.shape[-1], 0), 0),
        )[:, :, -history_length:]

        convolved = self.causal_conv1d_fn(
            x=mixed_qkv,
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
        beta = self.in_proj_b(prefix).sigmoid()
        a = self.in_proj_a(prefix)
        g = -self.A_log.float().exp() * torch.nn.functional.softplus(a.float() + self.dt_bias)
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
            raise RuntimeError("Delta virtual prefix kernel did not return its final recurrent state")
        return conv_tail, recurrent_state

    def _packed_causal_conv(
        self,
        mixed_qkv: torch.Tensor,
        prefix_conv_tail: torch.Tensor,
        cu_seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Run causal convolution over packed segments with prefix context."""

        lengths = (cu_seq_lens[1:] - cu_seq_lens[:-1]).to(dtype=torch.long)
        history_length = prefix_conv_tail.shape[-1]
        projected_segments = torch.split(mixed_qkv.squeeze(0), lengths.tolist(), dim=-1)
        extended_segments = [
            torch.cat((prefix_conv_tail.squeeze(0), segment), dim=-1)
            for segment in projected_segments
        ]
        # causal-conv1d accepts [B, C, T], but its seq_idx kernel requires the
        # underlying channel-last stride layout (stride(C) == 1).
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
        convolved_segments = torch.split(convolved_extended.squeeze(0), extended_lengths.tolist(), dim=-1)
        return torch.cat(
            [segment[:, history_length:] for segment in convolved_segments],
            dim=-1,
        ).unsqueeze(0)

    def _forward_packed(
        self,
        hidden_states: torch.Tensor,
        cu_seq_lens_q: torch.Tensor,
    ) -> torch.Tensor:
        """Apply one shared differentiable prefix state to every packed segment."""

        if not qwen3_5.is_fast_path_available:
            raise RuntimeError(
                "Packed Delta virtual prefix requires flash-linear-attention and causal-conv1d fast kernels"
            )
        if hidden_states.shape[0] != 1:
            raise ValueError(
                "Packed Delta virtual prefix expects hidden_states batch dimension 1; "
                f"got {tuple(hidden_states.shape)}"
            )
        if cu_seq_lens_q.ndim != 1 or cu_seq_lens_q.numel() < 2:
            raise ValueError(
                "Packed Delta virtual prefix expects one-dimensional cu_seq_lens_q with at least two entries"
            )
        cu_seq_lens_q = cu_seq_lens_q.to(device=hidden_states.device, dtype=torch.int32)
        if int(cu_seq_lens_q[0]) != 0 or int(cu_seq_lens_q[-1]) != hidden_states.shape[1]:
            raise ValueError(
                "Packed Delta virtual prefix cu_seq_lens_q must span exactly all hidden states; "
                f"got first={int(cu_seq_lens_q[0])}, last={int(cu_seq_lens_q[-1])}, "
                f"tokens={hidden_states.shape[1]}"
            )
        if bool(((cu_seq_lens_q[1:] - cu_seq_lens_q[:-1]) <= 0).any()):
            raise ValueError("Packed Delta virtual prefix does not support empty sequences")

        prefix_conv_tail, prefix_recurrent_state = self._compute_packed_prefix_states(hidden_states.dtype)
        num_sequences = cu_seq_lens_q.numel() - 1
        initial_state = prefix_recurrent_state.expand(num_sequences, -1, -1, -1).contiguous()

        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2).contiguous()
        mixed_qkv = self._packed_causal_conv(mixed_qkv, prefix_conv_tail, cu_seq_lens_q)
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
        g = -self.A_log.float().exp() * torch.nn.functional.softplus(a.float() + self.dt_bias)
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params=None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ):
        has_state = cache_params is not None and cache_params.has_previous_state(self.layer_idx)
        if has_state:
            # Decode or chunked continuation: the prefill cache already contains
            # the effect of P_l, so applying it again would be incorrect.
            return super().forward(
                hidden_states=hidden_states,
                cache_params=cache_params,
                attention_mask=attention_mask,
                **kwargs,
            )

        cu_seq_lens_q = kwargs.get("cu_seq_lens_q")
        if cu_seq_lens_q is not None:
            if cache_params is not None:
                raise ValueError("Packed Delta virtual prefix training does not support cache_params")
            if attention_mask is not None:
                raise ValueError("Packed Delta virtual prefix expects attention_mask=None")
            return self._forward_packed(hidden_states, cu_seq_lens_q)

        token_mask = None
        if attention_mask is not None:
            token_mask = attention_mask[:, -hidden_states.shape[1] :].to(torch.bool)

        if token_mask is None or bool(token_mask.all()):
            # Execute the causal sequence [P_l; X_l] as two consecutive
            # chunks. This is mathematically identical to concatenation, keeps
            # the user chunk's FLA tiling unchanged, and matches the native
            # vLLM implementation. The intermediate cache remains in the
            # autograd graph, so gradients flow from X_l back through P_l.
            prefix_cache = DynamicCache(config=self._delta_prefix_config)
            prefix = self.prefix_tokens.to(hidden_states.dtype)
            prefix = prefix.unsqueeze(0).expand(hidden_states.shape[0], -1, -1)
            super().forward(
                hidden_states=prefix,
                cache_params=prefix_cache,
                attention_mask=None,
                **kwargs,
            )
            prefix_layer_cache = prefix_cache.layers[self.layer_idx]
            working_cache = _FunctionalDeltaCache(
                self.layer_idx,
                prefix_layer_cache.conv_states.clone(),
                prefix_layer_cache.recurrent_states.clone(),
            )
            output = super().forward(
                hidden_states=hidden_states,
                cache_params=working_cache,
                attention_mask=attention_mask,
                **kwargs,
            )
            if cache_params is not None:
                final_layer_cache = working_cache.layers[self.layer_idx]
                cache_params.update_conv_state(final_layer_cache.conv_states, self.layer_idx)
                cache_params.update_recurrent_state(final_layer_cache.recurrent_states, self.layer_idx)
            return output

        extended, extended_mask, layout = self._prepend_virtual_tokens(hidden_states, attention_mask)
        output = super().forward(
            hidden_states=extended,
            cache_params=cache_params,
            attention_mask=extended_mask,
            **kwargs,
        )
        return self._remove_virtual_outputs(output, layout)


class Qwen3_5DeltaVirtualPrefixForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    """Qwen3.5 conditional generation with independent GDN virtual tokens."""

    def __init__(self, config):
        original_cls = qwen3_5.Qwen3_5GatedDeltaNet
        qwen3_5.Qwen3_5GatedDeltaNet = Qwen3_5DeltaVirtualPrefixGatedDeltaNet
        try:
            super().__init__(config)
        finally:
            qwen3_5.Qwen3_5GatedDeltaNet = original_cls


__all__ = [
    "Qwen3_5DeltaVirtualPrefixForConditionalGeneration",
    "Qwen3_5DeltaVirtualPrefixGatedDeltaNet",
]
