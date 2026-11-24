"""Joint prefix rollout for the vLLM CUDA backends."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fla.ops import fused_recurrent_gated_delta_rule_packed_decode
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.utils import set_weight_attrs
from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states

from .joint_prefix_attention import (
    compiled_gdn_anchored_attention, compiled_gdn_state_rows_attention, compiled_gdn_token_null_attention,
)
from .vllm_independent_delta_kv_prefix_model import Qwen3_5GatedDeltaNet


class JointPrefixGatedDeltaNet(Qwen3_5GatedDeltaNet):
    """Add the prefix to the native GDN readout before gated RMSNorm."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.tp_size != 1:
            raise ValueError("Joint prefix rollout requires tensor parallel size 1")
        config = kwargs.get("config", args[0] if args else None)
        count = config.independent_delta_prefix_num_virtual_tokens
        self.prefix_mode = getattr(config, "independent_delta_prefix_method", "state_rows_joint_softmax")
        self.prefix_state_key = torch.nn.Parameter(
            torch.empty(1, self.num_v_heads, count, self.head_k_dim)
        )
        self.prefix_state_value = torch.nn.Parameter(
            torch.zeros(1, self.num_v_heads, count, self.head_v_dim)
        )
        self.prefix_state_temperature = torch.nn.Parameter(torch.ones(self.num_v_heads))
        self.prefix_state_bias = torch.nn.Parameter(torch.zeros(self.num_v_heads))
        if self.prefix_mode == "token_null_additive":
            self.prefix_null_proj = torch.nn.Parameter(
                torch.zeros(self.num_v_heads, config.hidden_size, dtype=torch.float32),
                requires_grad=False,
            )
            set_weight_attrs(self.prefix_null_proj, {"weight_loader": default_weight_loader})
        for parameter in (
            self.prefix_state_key, self.prefix_state_value,
            self.prefix_state_temperature, self.prefix_state_bias,
        ):
            set_weight_attrs(parameter, {"weight_loader": default_weight_loader})
        self.register_buffer("_joint_key", None, persistent=False)
        self.register_buffer("_joint_value", None, persistent=False)
        self._post_conv_query = None
        # Newer vLLM prefills normalize and unpack Q inside fused_post_conv_prep.
        # Cosine attention can use the normalized query passed to the recurrence.
        self.chunk_gated_delta_rule.register_forward_pre_hook(
            self._capture_prefill_query, with_kwargs=True,
        )

    def forward(self, hidden_states, *args, **kwargs):
        if self.prefix_mode != "token_null_additive":
            return super().forward(hidden_states, *args, **kwargs)
        self._null_projection = F.linear(hidden_states.float(), self.prefix_null_proj.float())
        try:
            return super().forward(hidden_states, *args, **kwargs)
        finally:
            self._null_projection = None

    def _capture_prefill_query(self, module, args, kwargs):
        self._post_conv_query = kwargs["q"]

    @torch.no_grad()
    def refresh_joint_prefix(self):
        key = F.normalize(self.prefix_state_key.float(), dim=-1)
        value = self.prefix_state_value.float()
        if self._joint_key is None:
            self._joint_key = key
            self._joint_value = value.clone()
        else:
            self._joint_key.copy_(key)
            self._joint_value.copy_(value)

    def rearrange_mixed_qkv(self, mixed_qkv):
        query, key, value = super().rearrange_mixed_qkv(mixed_qkv)
        if query is not None:
            self._post_conv_query = query
        return query, key, value

    def _forward_core_decode_non_spec(self, mixed_qkv, b, a, core_attn_out,
                                      attn_metadata, virtual_engine=None):
        # Keep vLLM's packed decode kernel and expose the convolution output
        # to the same post-processing used for prefill.
        count = attn_metadata.num_actual_tokens
        indices = attn_metadata.non_spec_state_indices_tensor[:count]
        if virtual_engine is None:
            from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first

            conv_state, recurrent_state = self.kv_cache
            if not is_conv_state_dim_first():
                conv_state = conv_state.transpose(-1, -2)
        else:
            conv_state, recurrent_state = self.kv_cache[virtual_engine]
            conv_state = conv_state.transpose(-1, -2)
        convolved = causal_conv1d_update(
            mixed_qkv[:count], conv_state,
            self.conv1d.weight.squeeze(1), self.conv1d.bias, self.activation,
            conv_state_indices=indices, validate_data=False,
        )
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=convolved, a=a[:count], b=b[:count], A_log=self.A_log,
            dt_bias=self.dt_bias, scale=self.head_k_dim**-0.5,
            initial_state=recurrent_state, out=core_attn_out[:count].unsqueeze(1),
            ssm_state_indices=indices, use_qk_l2norm_in_kernel=True,
        )
        self._post_conv_query = convolved[:, :self.key_dim].reshape(
            1, count, self.num_k_heads, self.head_k_dim
        )

    def _forward_core(self, mixed_qkv, b, a, core_attn_out):
        metadata = get_forward_context().attn_metadata
        if metadata is None:
            try:
                return super()._forward_core(mixed_qkv, b, a, core_attn_out)
            finally:
                self._post_conv_query = None
        metadata = metadata[self.prefix]
        if metadata.spec_sequence_masks is not None:
            raise ValueError("Joint prefix rollout does not support speculative decoding")
        try:
            super()._forward_core(mixed_qkv, b, a, core_attn_out)
            query = self._post_conv_query
        finally:
            self._post_conv_query = None
        if query is None:
            raise RuntimeError("Native GDN did not expose its post-convolution query")
        if query.shape[2] != self.num_v_heads:
            query = query.repeat_interleave(self.num_v_heads // query.shape[2], dim=2)
        readout = core_attn_out[:metadata.num_actual_tokens]
        if self.prefix_mode == "token_null_additive":
            null = self._null_projection[:metadata.num_actual_tokens].transpose(0, 1)[None, :, :, None]
            contribution, _, _ = compiled_gdn_token_null_attention()(
                query.transpose(1, 2), self._joint_key, self._joint_value, self.prefix_state_bias, null,
            )
            readout.add_(contribution[0].transpose(0, 1).to(readout.dtype))
        elif self.prefix_mode == "anchored_additive":
            contribution, _ = compiled_gdn_anchored_attention()(
                query.transpose(1, 2), self._joint_key, self._joint_value, self.prefix_state_bias,
            )
            readout.add_(contribution[0].transpose(0, 1).to(readout.dtype))
        else:
            joint = compiled_gdn_state_rows_attention()(
                readout.transpose(0, 1).unsqueeze(0), query.transpose(1, 2),
                self._joint_key, self._joint_value,
                self.prefix_state_temperature, self.prefix_state_bias, (-80.0, 80.0),
            )
            readout.copy_(joint[0].transpose(0, 1))


class JointPrefixFlashAttention(FlashAttentionImpl):
    """Merge paged causal attention with a shared, separately cached prefix."""

    def forward(self, layer, query, key, value, kv_cache, attn_metadata,
                output=None, output_scale=None, output_block_scale=None):
        if attn_metadata is None:
            return output.zero_()
        if (attn_metadata.use_cascade or self.dcp_world_size != 1
                or output_scale is not None or output_block_scale is not None):
            raise ValueError("Joint prefix requires ordinary paged FlashAttention")
        from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func

        prefix = self.prefix_layer()
        metadata = attn_metadata
        count = metadata.num_actual_tokens
        live_query = query[:count]
        mode = getattr(prefix.config, "residual_attention_prefix_mode", "joint")
        anchored = mode in ("anchored_additive", "token_null_additive")
        key_cache, value_cache = kv_cache.unbind(0)
        descale_shape = (metadata.query_start_loc.shape[0] - 1, self.num_kv_heads)
        ctx_result = flash_attn_varlen_func(
            q=live_query, k=key_cache, v=value_cache,
            cu_seqlens_q=metadata.query_start_loc, max_seqlen_q=metadata.max_query_len,
            seqused_k=metadata.seq_lens, max_seqlen_k=metadata.max_seq_len,
            softmax_scale=self.scale, causal=metadata.causal,
            block_table=metadata.block_table, scheduler_metadata=metadata.scheduler_metadata,
            num_splits=metadata.max_num_splits, fa_version=self.vllm_flash_attn_version,
            q_descale=layer._q_scale.expand(descale_shape),
            k_descale=layer._k_scale.expand(descale_shape),
            v_descale=layer._v_scale.expand(descale_shape),
            return_softmax_lse=not anchored,
        )
        # Flattening queries is valid here because every request reads the same
        # immutable prefix and noncausal attention treats query rows independently.
        pref, pref_lse = flash_attn_varlen_func(
            q=live_query, k=prefix._cached_prefix_key, v=prefix._cached_prefix_value,
            cu_seqlens_q=prefix._joint_offsets * count, max_seqlen_q=count,
            cu_seqlens_k=prefix._joint_key_offsets, max_seqlen_k=prefix.num_prefix_tokens,
            softmax_scale=self.scale, causal=False, fa_version=self.vllm_flash_attn_version,
            return_softmax_lse=True,
        )
        pref_lse = (pref_lse + prefix.prefix_logit_bias.float()[:, None]).contiguous()
        if mode == "token_null_additive":
            pref_lse = pref_lse - prefix._null_projection[:count].transpose(0, 1)
        if anchored:
            weight = pref_lse.sigmoid().transpose(0, 1).unsqueeze(-1)
            output[:count].copy_((ctx_result.float() + weight * pref.float()).to(output.dtype))
        else:
            ctx, ctx_lse = ctx_result
            merge_attn_states(output[:count], pref, pref_lse, ctx, ctx_lse.contiguous())
        return output
