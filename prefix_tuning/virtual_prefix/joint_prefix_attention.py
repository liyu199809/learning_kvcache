"""Reference attention for prefix and state competition."""

import math
from functools import lru_cache

import torch
import torch.nn.functional as F


def gdn_state_rows_attention(readout, query, prefix_key, prefix_value, temperature, bias, *, clamp):
    """Add the prefix contribution before the gated RMSNorm.

    Queries and state readouts are [batch, heads, tokens, dim]. The prefix
    tensors are broadcastable [batch, heads, prefix tokens, dim].
    """
    return _gdn_state_rows_attention(
        readout, query, F.normalize(prefix_key.float(), dim=-1), prefix_value.float(),
        temperature, bias, clamp,
    )


def _gdn_state_rows_attention(readout, query, normalized_key, prefix_value, temperature, bias, clamp):
    state_logits = readout.float() * temperature.float()[None, :, None, None]
    state_logits = state_logits + bias.float()[None, :, None, None]
    state_log_mass = torch.logsumexp(state_logits, dim=-1, keepdim=True).clamp(*clamp)
    prefix_logits = F.normalize(query.float(), dim=-1) @ normalized_key.transpose(-1, -2)
    prefix_logits = prefix_logits * math.sqrt(query.shape[-1])
    weights = torch.softmax(torch.cat((state_log_mass, prefix_logits), dim=-1), dim=-1)
    return readout + (weights[..., 1:] @ prefix_value).to(readout.dtype)


@lru_cache(maxsize=1)
def compiled_gdn_state_rows_attention():
    return torch.compile(_gdn_state_rows_attention, fullgraph=True)


def anchored_prefix_attention(query, key, value, bias, *, scale):
    """Return the additive prefix contribution and its mass against a zero logit."""
    groups = query.shape[1] // key.shape[1]
    key = key.repeat_interleave(groups, dim=1)
    value = value.repeat_interleave(groups, dim=1)
    logits = (query @ key.transpose(-1, -2)).float() * scale
    return _anchored_read(logits, value, bias)


def _anchored_read(logits, value, bias):
    logits = logits + bias.float()[None, :, None, None]
    prefix_lse = torch.logsumexp(logits, dim=-1, keepdim=True)
    lse = torch.logaddexp(prefix_lse, torch.zeros_like(prefix_lse))
    weights = (logits - lse).exp()
    return weights.to(value.dtype) @ value, prefix_lse.sigmoid()


def gdn_anchored_attention(query, normalized_key, value, bias):
    logits = F.normalize(query.float(), dim=-1) @ normalized_key.transpose(-1, -2)
    return _anchored_read(logits * math.sqrt(query.shape[-1]), value, bias)


@lru_cache(maxsize=1)
def compiled_gdn_anchored_attention():
    return torch.compile(gdn_anchored_attention, fullgraph=True, dynamic=True)


def _token_null_read(scores, value, bias, null_projection):
    raw_lse = torch.logsumexp(scores, dim=-1, keepdim=True)
    shift = bias.float()[None, :, None, None] - null_projection.float()
    lse = raw_lse + shift
    weights = (scores + shift - torch.logaddexp(lse, torch.zeros_like(lse))).exp()
    return weights @ value.float(), lse.sigmoid(), raw_lse


def token_null_attention(query, key, value, bias, null_projection, *, scale):
    groups = query.shape[1] // key.shape[1]
    key = key.repeat_interleave(groups, dim=1)
    value = value.repeat_interleave(groups, dim=1)
    return _token_null_read((query.float() @ key.float().transpose(-1, -2)) * scale,
                            value, bias, null_projection)


def gdn_token_null_attention(query, normalized_key, value, bias, null_projection):
    scores = F.normalize(query.float(), dim=-1) @ normalized_key.transpose(-1, -2)
    return _token_null_read(scores * math.sqrt(query.shape[-1]), value, bias, null_projection)


@lru_cache(maxsize=1)
def compiled_gdn_token_null_attention():
    return torch.compile(gdn_token_null_attention, fullgraph=True, dynamic=True)


def full_attention_with_prefix(query, key, value, prefix_key, prefix_value, prefix_bias,
                               *, context_mask=None, dropout=0.0, scale=None):
    """Joint causal attention over negative-position prefix and user K/V."""
    batch, heads, query_length, head_dim = query.shape
    key_length = key.shape[-2]
    groups = heads // key.shape[1]
    if groups > 1:
        key = key.repeat_interleave(groups, dim=1)
        value = value.repeat_interleave(groups, dim=1)
        prefix_key = prefix_key.repeat_interleave(groups, dim=1)
        prefix_value = prefix_value.repeat_interleave(groups, dim=1)

    if context_mask is None:
        query_positions = torch.arange(key_length - query_length, key_length, device=query.device)
        key_positions = torch.arange(key_length, device=query.device)
        context_mask = key_positions <= query_positions[:, None]
    if context_mask.dtype == torch.bool:
        context_mask = query.new_zeros(context_mask.shape).masked_fill(~context_mask, -torch.inf)
    context_mask = context_mask.to(dtype=query.dtype).expand(batch, heads, query_length, key_length)
    prefix_length = prefix_key.shape[-2]
    prefix_mask = prefix_bias.to(query.dtype).view(1, heads, 1, 1)
    prefix_mask = prefix_mask.expand(batch, heads, query_length, prefix_length)
    return F.scaled_dot_product_attention(
        query,
        torch.cat((prefix_key.expand(batch, -1, -1, head_dim), key), dim=-2),
        torch.cat((prefix_value.expand(batch, -1, -1, head_dim), value), dim=-2),
        attn_mask=torch.cat((prefix_mask, context_mask), dim=-1),
        dropout_p=dropout,
        scale=scale,
    )


@lru_cache(maxsize=64)
def _prefix_block_mask(query_length, key_length, device):
    from torch.nn.attention.flex_attention import create_block_mask

    offset = key_length - query_length

    def causal_mask(batch, head, query_index, key_index):
        return key_index <= query_index + offset

    return create_block_mask(
        causal_mask, B=None, H=None, Q_LEN=query_length, KV_LEN=key_length, device=device,
    )


@lru_cache(maxsize=1)
def _compiled_flex_attention():
    from torch.nn.attention.flex_attention import flex_attention

    return torch.compile(flex_attention, fullgraph=True, dynamic=False)


def flex_full_attention_with_prefix(query, key, value, prefix_key, prefix_value, prefix_bias,
                                    *, scale=None):
    """Joint causal attention with block masking and native grouped-query K/V."""
    batch = query.shape[0]
    prefix_length = prefix_key.shape[-2]
    joint_key = torch.cat((prefix_key.expand(batch, -1, -1, -1), key), dim=-2)
    joint_value = torch.cat((prefix_value.expand(batch, -1, -1, -1), value), dim=-2)
    mask = _prefix_block_mask(query.shape[-2], joint_key.shape[-2], query.device)

    def score_with_prefix_bias(score, batch_index, head, query_index, key_index):
        return score + torch.where(key_index < prefix_length, prefix_bias[head], 0.0)

    # Large default tiles can exceed H100 shared memory at head dimension 256.
    # Force the general kernel so the tile is also valid for short GQA queries.
    kernel_options = None
    if query.shape[-1] == 256:
        kernel_options = {
            "BLOCK_M": 32, "BLOCK_N": 32, "num_stages": 1,
            "FORCE_USE_FLEX_ATTENTION": True,
        }
    return _compiled_flex_attention()(
        query, joint_key, joint_value, score_mod=score_with_prefix_bias,
        block_mask=mask, scale=scale, enable_gqa=query.shape[1] != key.shape[1],
        kernel_options=kernel_options,
    )
