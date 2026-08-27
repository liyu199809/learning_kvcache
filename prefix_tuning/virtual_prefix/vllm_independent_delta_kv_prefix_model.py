"""Native vLLM Qwen3.5 model for independent Delta K/V/beta/a prefixes."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from vllm.forward_context import get_forward_context
from vllm.model_executor.model_loader.weight_utils import sharded_weight_loader
from vllm.model_executor.models import qwen3_5
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    Qwen3_5GatedDeltaNet,
)
from vllm.model_executor.models.qwen3_next import fused_gdn_gating
from vllm.model_executor.utils import set_weight_attrs


class Qwen3_5IndependentDeltaKVPrefixGatedDeltaNet(Qwen3_5GatedDeltaNet):
    """Seed native GDN request states from direct projected-space parameters."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_virtual_tokens = int(
            getattr(self.config, "independent_delta_prefix_num_virtual_tokens", 0)
        )
        if self.num_virtual_tokens <= 0:
            raise ValueError(
                "text_config.independent_delta_prefix_num_virtual_tokens must be positive"
            )

        local_key_dim = self.key_dim // self.tp_size
        local_value_dim = self.value_dim // self.tp_size
        local_value_heads = self.num_v_heads // self.tp_size
        self.prefix_k = torch.nn.Parameter(
            torch.zeros(self.num_virtual_tokens, local_key_dim)
        )
        self.prefix_v = torch.nn.Parameter(
            torch.empty(self.num_virtual_tokens, local_value_dim)
        )
        self.prefix_beta_logits = torch.nn.Parameter(
            torch.zeros(self.num_virtual_tokens, local_value_heads)
        )
        self.prefix_a = torch.nn.Parameter(
            torch.zeros(self.num_virtual_tokens, local_value_heads)
        )
        for parameter in (
            self.prefix_k,
            self.prefix_v,
            self.prefix_beta_logits,
            self.prefix_a,
        ):
            set_weight_attrs(parameter, {"weight_loader": sharded_weight_loader(1)})

        self.register_buffer("_cached_prefix_conv_state", None, persistent=False)
        self.register_buffer("_cached_prefix_recurrent_state", None, persistent=False)
        self._cached_prefix_key = None
        self._prefix_cache_misses = 0

    def invalidate_independent_delta_prefix_cache(self) -> None:
        self._cached_prefix_key = None
        self._cached_prefix_conv_state = None
        self._cached_prefix_recurrent_state = None

    def _get_prefix_states(
        self,
        conv_state_dtype: torch.dtype,
        recurrent_state_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (
            self.prefix_k._version,
            self.prefix_v._version,
            self.prefix_beta_logits._version,
            self.prefix_a._version,
            self.prefix_k.device,
            conv_state_dtype,
            recurrent_state_dtype,
        )
        if key != self._cached_prefix_key:
            conv_state, recurrent_state = self._compute_prefix_states(
                conv_state_dtype, recurrent_state_dtype
            )
            self._cached_prefix_conv_state = conv_state.detach()
            self._cached_prefix_recurrent_state = recurrent_state.detach()
            self._cached_prefix_key = key
            self._prefix_cache_misses += 1
        return self._cached_prefix_conv_state, self._cached_prefix_recurrent_state

    def _compute_prefix_states(
        self,
        conv_state_dtype: torch.dtype,
        recurrent_state_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        local_key_dim = self.key_dim // self.tp_size
        prefix_q = self.prefix_k.new_zeros(self.num_virtual_tokens, local_key_dim)
        mixed_qkv = torch.cat((prefix_q, self.prefix_k, self.prefix_v), dim=-1)

        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )
        conv_input = mixed_qkv.transpose(0, 1).unsqueeze(0)
        convolved = F.conv1d(
            conv_input,
            conv_weights.unsqueeze(1),
            bias=self.conv1d.bias,
            padding=self.conv_kernel_size - 1,
            groups=conv_input.shape[1],
        )[:, :, : self.num_virtual_tokens]
        convolved = self.act(convolved).squeeze(0).transpose(0, 1).contiguous()

        query, key, value = self.rearrange_mixed_qkv(convolved)
        # fused_gdn_gating returns the same g/beta parameterization as HF:
        # beta=sigmoid(b), g=-exp(A_log)*softplus(a+dt_bias).
        g, beta = fused_gdn_gating(
            self.A_log,
            self.prefix_a.contiguous(),
            self.prefix_beta_logits.contiguous(),
            self.dt_bias,
        )
        initial_state = torch.zeros(
            1,
            self.num_v_heads // self.tp_size,
            self.head_v_dim,
            self.head_k_dim,
            device=self.prefix_k.device,
            dtype=recurrent_state_dtype,
        )
        cu_seqlens = torch.tensor(
            [0, self.num_virtual_tokens], device=self.prefix_k.device, dtype=torch.long
        )
        _, final_recurrent_state = self.chunk_gated_delta_rule(
            q=query,
            k=key,
            v=value,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=True,
        )

        history_length = self.conv_kernel_size - 1
        final_conv_state = mixed_qkv[-history_length:].transpose(0, 1).contiguous()
        return (
            final_conv_state.to(conv_state_dtype),
            final_recurrent_state.squeeze(0).to(recurrent_state_dtype),
        )

    def _forward_core(self, mixed_qkv, b, a, core_attn_out):
        context = get_forward_context()
        metadata_by_layer = context.attn_metadata
        if metadata_by_layer is None:
            return super()._forward_core(mixed_qkv, b, a, core_attn_out)
        metadata = metadata_by_layer[self.prefix]
        has_initial_state = metadata.has_initial_state
        if metadata.num_prefills <= 0 or has_initial_state is None or bool(has_initial_state.all()):
            return super()._forward_core(mixed_qkv, b, a, core_attn_out)

        state_indices = metadata.non_spec_state_indices_tensor
        if state_indices is None or state_indices.shape[0] != has_initial_state.shape[0]:
            raise RuntimeError("Independent Delta prefix metadata/state-index mismatch")
        new_state_indices = state_indices[~has_initial_state]
        if bool((new_state_indices < 0).any()):
            raise RuntimeError("Independent Delta prefix received an invalid cache state index")

        self_kv_cache = self.kv_cache[context.virtual_engine]
        conv_state = self_kv_cache[0].transpose(-1, -2)
        recurrent_state = self_kv_cache[1]
        prefix_conv, prefix_recurrent = self._get_prefix_states(
            conv_state.dtype, recurrent_state.dtype
        )
        conv_state[new_state_indices] = 0
        conv_state[new_state_indices, :, : prefix_conv.shape[-1]] = prefix_conv.unsqueeze(0)
        recurrent_state[new_state_indices] = prefix_recurrent.unsqueeze(0)

        metadata.has_initial_state = torch.ones_like(has_initial_state)
        try:
            return super()._forward_core(mixed_qkv, b, a, core_attn_out)
        finally:
            metadata.has_initial_state = has_initial_state


class Qwen3_5IndependentDeltaKVPrefixForConditionalGeneration(
    Qwen3_5ForConditionalGeneration
):
    """Native vLLM Qwen3.5 with independent Delta-space prefixes."""

    def __init__(self, *, vllm_config, prefix: str = "model"):
        original_cls = qwen3_5.Qwen3_5GatedDeltaNet
        qwen3_5.Qwen3_5GatedDeltaNet = Qwen3_5IndependentDeltaKVPrefixGatedDeltaNet
        try:
            super().__init__(vllm_config=vllm_config, prefix=prefix)
        finally:
            qwen3_5.Qwen3_5GatedDeltaNet = original_cls

    def invalidate_independent_delta_prefix_caches(self) -> None:
        for module in self.modules():
            if isinstance(module, Qwen3_5IndependentDeltaKVPrefixGatedDeltaNet):
                module.invalidate_independent_delta_prefix_cache()

    def load_weights(self, weights):
        loaded = super().load_weights(weights)
        expected = {
            name
            for name, _ in self.named_parameters()
            if name.endswith((".prefix_k", ".prefix_v", ".prefix_beta_logits", ".prefix_a"))
        }
        missing = expected.difference(loaded)
        if missing:
            raise RuntimeError(
                "vLLM did not load all Independent Delta prefix tensors: "
                + ", ".join(sorted(missing))
            )
        self.invalidate_independent_delta_prefix_caches()
        return loaded


__all__ = ["Qwen3_5IndependentDeltaKVPrefixForConditionalGeneration"]
