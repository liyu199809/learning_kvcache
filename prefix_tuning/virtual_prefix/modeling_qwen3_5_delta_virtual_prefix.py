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

        if kwargs.get("cu_seq_lens_q") is not None:
            raise NotImplementedError(
                "Packed/padding-free Transformers training needs rebuilt cu_seqlens "
                "after inserting per-layer virtual tokens; use padded batches for now."
            )

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
