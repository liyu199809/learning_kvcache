import math

import torch

from prefix_tuning.virtual_prefix.joint_prefix_attention import (
    full_attention_with_prefix,
    gdn_state_rows_attention,
)


def test_gdn_state_mass_matches_joint_softmax():
    query = torch.tensor([[[[1.0, 0.0]]]])
    readout = torch.tensor([[[[0.3, -0.2]]]])
    key = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    value = torch.tensor([[[[2.0, 1.0], [1.0, -2.0]]]])
    temp = torch.tensor([2.0])
    bias = torch.tensor([0.4])
    prefix_logits = torch.tensor([[[[math.sqrt(2.0), 0.0]]]])
    weights = torch.cat((temp * readout + bias, prefix_logits), dim=-1).softmax(dim=-1)
    actual = gdn_state_rows_attention(readout, query, key, value, temp, bias, clamp=(-80, 80))
    torch.testing.assert_close(actual, readout + weights[..., 2:] @ value)
    suppressed = gdn_state_rows_attention(
        readout, query, key, value, temp, torch.tensor([60.0]), clamp=(-80, 80),
    )
    torch.testing.assert_close(suppressed, readout)


def test_full_attention_joint_weights_and_causal_cache():
    q = torch.zeros(1, 2, 2, 2)
    k = torch.zeros(1, 1, 2, 2)
    v = torch.tensor([[[[2.0, 2.0], [5.0, 5.0]]]])
    prefix_k = torch.zeros(1, 1, 1, 2)
    prefix_v = torch.tensor([[[[8.0, 8.0]]]])
    bias = torch.tensor([0.0, -torch.inf])
    result = full_attention_with_prefix(q, k, v, prefix_k, prefix_v, bias)
    expected = torch.tensor([[[[5.0, 5.0], [5.0, 5.0]], [[2.0, 2.0], [3.5, 3.5]]]])
    torch.testing.assert_close(result, expected)
    decode = full_attention_with_prefix(q[..., -1:, :], k, v, prefix_k, prefix_v, bias)
    torch.testing.assert_close(decode, expected[..., -1:, :])
