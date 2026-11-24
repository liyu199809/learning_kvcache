"""Native vLLM Qwen3.5 model with Delta and residual attention prefixes."""

from __future__ import annotations

from importlib.metadata import version
from weakref import ref

import torch
import torch.nn.functional as F
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models import qwen3_5
from vllm.model_executor.models.qwen3_next import Qwen3NextAttention
from vllm.model_executor.utils import set_weight_attrs

from .vllm_residual_prefix_attention import ResidualPrefixAttentionMixin

from .vllm_independent_delta_kv_prefix_model import (
    Qwen3_5IndependentDeltaKVPrefixForConditionalGeneration,
)


class Qwen3_5ResidualSoftPrefixAttention(ResidualPrefixAttentionMixin, Qwen3NextAttention):
    """Native self-attention plus a shared non-causal prefix-only branch."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.joint_prefix = bool(getattr(self.config, "residual_attention_prefix_joint_softmax", False))
        self.token_null_prefix = getattr(self.config, "residual_attention_prefix_mode", "joint") == "token_null_additive"
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
        if self.joint_prefix:
            from .vllm_joint_prefix import FlashAttentionImpl, JointPrefixFlashAttention

            impl = self.attn.impl
            if (type(impl) is not FlashAttentionImpl or impl.kv_cache_dtype.startswith("fp8")
                    or impl.vllm_flash_attn_version != 3):
                raise ValueError("Joint prefix rollout requires FlashAttention 3 and an unquantized KV cache")
            self.prefix_logit_bias = torch.nn.Parameter(torch.zeros(self.num_heads))
            set_weight_attrs(self.prefix_logit_bias, {"weight_loader": default_weight_loader})
            if self.token_null_prefix:
                self.prefix_null_proj = torch.nn.Parameter(
                    torch.zeros(self.num_heads, self.hidden_size, dtype=torch.float32),
                    requires_grad=False,
                )
                set_weight_attrs(self.prefix_null_proj, {"weight_loader": default_weight_loader})
            self.register_buffer("_joint_offsets", None, persistent=False)
            self.register_buffer("_joint_key_offsets", None, persistent=False)
            impl.__class__ = JointPrefixFlashAttention
            impl.prefix_layer = ref(self)

    def invalidate_residual_attention_prefix_cache(self) -> None:
        self._cached_prefix_key = None
        self._cached_prefix_value = None
        self._cached_prefix_projection_key = None

    @torch.no_grad()
    def refresh_residual_attention_prefix_cache(self) -> None:
        """Project the prefix tables once, outside of forward.

        Forward then only reads these buffers, so it never mutates module state
        and can run under torch.compile / CUDA graphs.  Existing buffers are
        updated in place so captured graphs see refreshed values after a
        weight sync.
        """
        dtype = self.qkv_proj.weight.dtype
        prefix_key, prefix_value = self._compute_prefix_key_value(dtype)
        same_layout = (
            self._cached_prefix_key is not None
            and self._cached_prefix_key.shape == prefix_key.shape
            and self._cached_prefix_key.dtype == prefix_key.dtype
            and self._cached_prefix_key.device == prefix_key.device
        )
        if same_layout:
            self._cached_prefix_key.copy_(prefix_key)
            self._cached_prefix_value.copy_(prefix_value)
        else:
            self._cached_prefix_key = prefix_key.detach().contiguous()
            self._cached_prefix_value = prefix_value.detach().contiguous()
        if self.joint_prefix and self._joint_offsets is None:
            self._joint_offsets = torch.arange(2, device=prefix_key.device, dtype=torch.int32)
            self._joint_key_offsets = self._joint_offsets * self.num_prefix_tokens
        self._cached_prefix_projection_key = (
            self.prefix_key_tokens._version,
            self.prefix_value_tokens._version,
            self.prefix_key_tokens.device,
            dtype,
        )
        self._prefix_cache_refreshed = True
        self._prefix_projection_cache_misses += 1

    def _split_qkv(self, qkv: torch.Tensor):
        if self.attn_output_gate:
            return qkv.split([self.q_size * 2, self.kv_size, self.kv_size], dim=-1)
        return qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

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

        if self.token_null_prefix:
            self._null_projection = F.linear(hidden_states.float(), self.prefix_null_proj.float())
            try:
                attn_output = self.attn(q, k, v)
            finally:
                self._null_projection = None
        else:
            attn_output = self.attn(q, k, v)
        if not self.joint_prefix:
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
        config = vllm_config.model_config.hf_config.text_config
        self.joint_prefix = bool(getattr(config, "residual_attention_prefix_joint_softmax", False))
        if self.joint_prefix:
            if version("vllm").split("+", 1)[0] not in {"0.18.0", "0.22.0"}:
                raise ValueError("Joint prefix rollout requires vLLM 0.18.0 or 0.22.0")
            if not vllm_config.model_config.disable_cascade_attn:
                raise ValueError("Set disable_cascade_attn=True for joint prefix rollout")
        original_cls = qwen3_5.Qwen3NextAttention
        qwen3_5.Qwen3NextAttention = Qwen3_5ResidualSoftPrefixAttention
        try:
            super().__init__(vllm_config=vllm_config, prefix=prefix)
        finally:
            qwen3_5.Qwen3NextAttention = original_cls

    def _delta_net_class(self):
        if self.joint_prefix:
            from .vllm_joint_prefix import JointPrefixGatedDeltaNet

            return JointPrefixGatedDeltaNet
        return super()._delta_net_class()

    def invalidate_residual_attention_prefix_caches(self) -> None:
        for module in self.modules():
            if isinstance(module, Qwen3_5ResidualSoftPrefixAttention):
                module.invalidate_residual_attention_prefix_cache()

    def refresh_residual_attention_prefix_caches(self) -> None:
        for module in self.modules():
            if isinstance(module, Qwen3_5ResidualSoftPrefixAttention):
                module.refresh_residual_attention_prefix_cache()

    def load_joint_prefix_weight_bucket(self, weights):
        return super().load_weights(weights, partial=True)

    def refresh_joint_prefix_caches(self):
        from .vllm_joint_prefix import JointPrefixGatedDeltaNet

        self.refresh_residual_attention_prefix_caches()
        for module in self.modules():
            if isinstance(module, JointPrefixGatedDeltaNet):
                module.refresh_joint_prefix()

    def load_weights(self, weights):
        loaded = super().load_weights(weights)
        suffixes = (".prefix_key_tokens", ".prefix_value_tokens")
        if self.joint_prefix:
            suffixes += (".prefix_logit_bias", ".prefix_state_key", ".prefix_state_value",
                         ".prefix_state_temperature", ".prefix_state_bias", ".prefix_null_proj")
        expected = {
            name
            for name, _ in self.named_parameters()
            if name.endswith(suffixes)
        }
        missing = expected.difference(loaded)
        if missing:
            raise RuntimeError(
                "vLLM did not load all residual attention prefix tensors: "
                + ", ".join(sorted(missing))
            )
        if self.joint_prefix:
            self.refresh_joint_prefix_caches()
        else:
            self.refresh_residual_attention_prefix_caches()
        return loaded


__all__ = ["Qwen3_5HybridDeltaResidualAttentionPrefixForConditionalGeneration"]
