"""TP=1 SGLang adapter preserving the deployed hybrid prefix equations.

Register through SGLANG_EXTERNAL_MODEL_PACKAGE. The server overrides only the
architecture label to the native Qwen3.5 label so SGLang selects its GDN pools.
The original checkpoint and all 112 prefix tensors remain unchanged.
"""
import logging

import torch
import torch.nn.functional as F
from sglang.srt.models import qwen3_5 as native
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.layers.attention.fla.chunk import chunk_gated_delta_rule
from sglang.srt.utils import set_weight_attrs

log = logging.getLogger(__name__)


class PrefixGDN(native.Qwen3_5GatedDeltaNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.attn_tp_size != 1:
            raise ValueError("The hybrid prefix adapter currently requires TP=1")
        self.num_virtual_tokens = self.config.independent_delta_prefix_num_virtual_tokens
        shapes = {"prefix_k": self.key_dim, "prefix_v": self.value_dim,
                  "prefix_beta_logits": self.num_v_heads, "prefix_a": self.num_v_heads}
        for name, width in shapes.items():
            parameter = torch.nn.Parameter(torch.zeros(self.num_virtual_tokens, width))
            set_weight_attrs(parameter, {"weight_loader": default_weight_loader})
            self.register_parameter(name, parameter)
        self.register_buffer("prefix_conv", None, persistent=False)
        self.register_buffer("prefix_recurrent", None, persistent=False)

    @torch.no_grad()
    def prepare_prefix(self):
        dtype = self.conv1d.weight.dtype
        k, v = self.prefix_k.to(dtype), self.prefix_v.to(dtype)
        mixed = torch.cat((torch.zeros_like(k), k, v), dim=-1)
        weights = self.conv1d.weight.view(self.conv_dim, self.conv_kernel_size)
        x = F.silu(F.conv1d(mixed.T.unsqueeze(0), weights.unsqueeze(1),
                           bias=self.conv1d.bias, padding=self.conv_kernel_size - 1,
                           groups=self.conv_dim)[:, :, :self.num_virtual_tokens])
        q, k, v = x.squeeze(0).T.contiguous().split(
            [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.reshape(1, -1, self.num_k_heads, self.head_k_dim)
        k = k.reshape(1, -1, self.num_k_heads, self.head_k_dim)
        v = v.reshape(1, -1, self.num_v_heads, self.head_v_dim)
        g = -self.A_log.float().exp() * F.softplus(
            self.prefix_a.float() + self.dt_bias.float())
        beta = self.prefix_beta_logits.float().sigmoid()
        state = torch.zeros(1, self.num_v_heads, self.head_v_dim, self.head_k_dim,
                            device=k.device, dtype=torch.float32)
        chunk_gated_delta_rule(q=q, k=k, v=v, g=g.unsqueeze(0), beta=beta.unsqueeze(0),
                              initial_state=state,
                              initial_state_indices=torch.zeros(1, device=k.device, dtype=torch.int32),
                              cu_seqlens=torch.tensor([0, self.num_virtual_tokens],
                                                      device=k.device, dtype=torch.int32),
                              use_qk_l2norm_in_kernel=True)
        self.prefix_conv = mixed[-(self.conv_kernel_size - 1):].T.contiguous()
        self.prefix_recurrent = state[0]
        if not torch.isfinite(self.prefix_recurrent).all():
            raise RuntimeError(f"Nonfinite prefix state at layer {self.layer_id}")

    def forward(self, hidden_states, forward_batch):
        if forward_batch.forward_mode.is_extend():
            lengths = forward_batch.extend_prefix_lens
            fresh = lengths == 0
            if bool(fresh.any()):
                backend = get_attn_backend().linear_attn_backend
                indices = backend.forward_metadata.mamba_cache_indices[fresh]
                if bool((indices < 0).any()):
                    raise RuntimeError("Invalid GDN prefix state slot")
                cache = backend.req_to_token_pool.mamba2_layer_cache(self.layer_id)
                conv, recurrent = cache.conv[0], cache.temporal
                conv[indices] = 0
                conv[indices, :, :self.prefix_conv.shape[-1]] = self.prefix_conv.to(conv.dtype)
                recurrent[indices] = self.prefix_recurrent.to(recurrent.dtype)
                # The native convolution must consume the seeded history.
                # Restore lengths before any other layer observes this batch.
                forward_batch.extend_prefix_lens = lengths.clamp_min(1)
                try:
                    return super().forward(hidden_states, forward_batch)
                finally:
                    forward_batch.extend_prefix_lens = lengths
        return super().forward(hidden_states, forward_batch)


class PrefixAttention(native.Qwen3_5AttentionDecoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.attn_tp_size != 1:
            raise ValueError("The hybrid prefix adapter currently requires TP=1")
        self.num_prefix_tokens = self.config.residual_attention_prefix_num_virtual_tokens
        for name in ("prefix_key_tokens", "prefix_value_tokens"):
            parameter = torch.nn.Parameter(torch.zeros(self.num_prefix_tokens, self.hidden_size))
            set_weight_attrs(parameter, {"weight_loader": default_weight_loader})
            self.register_parameter(name, parameter)
        self.register_buffer("prefix_key", None, persistent=False)
        self.register_buffer("prefix_value", None, persistent=False)

    @torch.no_grad()
    def prepare_prefix(self):
        dtype = self.qkv_proj.weight.dtype
        key_qkv, _ = self.qkv_proj(self.prefix_key_tokens.to(dtype))
        value_qkv, _ = self.qkv_proj(self.prefix_value_tokens.to(dtype))
        sizes = [self.q_size * (2 if self.attn_output_gate else 1), self.kv_size, self.kv_size]
        key = key_qkv.split(sizes, -1)[1]
        value = value_qkv.split(sizes, -1)[2]
        key = self.k_norm(key.reshape(-1, self.num_kv_heads, self.head_dim))
        rotary_dim = self.rotary_emb.rotary_dim
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(
            0, rotary_dim, 2, device=key.device, dtype=torch.float32) / rotary_dim))
        positions = torch.arange(-self.num_prefix_tokens, 0, device=key.device, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        cos, sin = freqs.cos().to(dtype).unsqueeze(1), freqs.sin().to(dtype).unsqueeze(1)
        rotated, passed = key[..., :rotary_dim], key[..., rotary_dim:]
        if self.rotary_emb.is_neox_style:
            first, second = rotated.chunk(2, -1)
            rotated = torch.cat((first * cos - second * sin, second * cos + first * sin), -1)
        else:
            even, odd = rotated[..., 0::2], rotated[..., 1::2]
            rotated = torch.stack((even * cos - odd * sin, odd * cos + even * sin), -1).flatten(-2)
        key = torch.cat((rotated, passed), -1)
        value = value.reshape(-1, self.num_kv_heads, self.head_dim)
        groups = self.num_heads // self.num_kv_heads
        self.prefix_key = key.repeat_interleave(groups, 1).transpose(0, 1).unsqueeze(0).contiguous()
        self.prefix_value = value.repeat_interleave(groups, 1).transpose(0, 1).unsqueeze(0).contiguous()

    def self_attention(self, positions, hidden_states, forward_batch):
        q, k, v, gate = self.forward_prepare_cuda_fused(positions, hidden_states)
        output = self.attn(q, k, v, forward_batch)
        prefix = F.scaled_dot_product_attention(
            q.reshape(-1, self.num_heads, self.head_dim).transpose(0, 1).unsqueeze(0),
            self.prefix_key, self.prefix_value, dropout_p=0.0, is_causal=False,
            scale=self.scaling).squeeze(0).transpose(0, 1).reshape(-1, self.q_size)
        output = output + prefix
        if self.attn_output_gate:
            output = native.fused_sigmoid_mul(output, gate, inplace=True)
        return self.o_proj(output)[0]


class Qwen3_5ForConditionalGeneration(native.Qwen3_5ForConditionalGeneration):
    def __init__(self, *args, **kwargs):
        gdn = native.Qwen3_5GatedDeltaNet
        attention = native.ALL_DECODER_LAYER_TYPES["attention"]
        native.Qwen3_5GatedDeltaNet = PrefixGDN
        native.ALL_DECODER_LAYER_TYPES["attention"] = PrefixAttention
        try:
            super().__init__(*args, **kwargs)
        finally:
            native.Qwen3_5GatedDeltaNet = gdn
            native.ALL_DECODER_LAYER_TYPES["attention"] = attention

    def load_weights(self, weights):
        loaded = super().load_weights(weights)
        endings = (".prefix_k", ".prefix_v", ".prefix_beta_logits", ".prefix_a",
                   ".prefix_key_tokens", ".prefix_value_tokens")
        expected = {name for name, _ in self.named_parameters() if name.endswith(endings)}
        missing = expected.difference(loaded)
        if len(expected) != 112 or missing:
            raise RuntimeError(f"Hybrid prefix loading failed: expected={len(expected)}, missing={missing}")
        for module in self.modules():
            if isinstance(module, (PrefixGDN, PrefixAttention)):
                module.prepare_prefix()
        log.warning("HYBRID_PREFIX_LOADED tensors=%d numel=%d GDN=24 attention=8",
                    len(expected), sum(p.numel() for n, p in self.named_parameters() if n in expected))
        return loaded


EntryClass = Qwen3_5ForConditionalGeneration
