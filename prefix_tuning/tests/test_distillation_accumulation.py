"""A split distillation batch must have the same loss and gradient as the full batch."""
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from verl.trainer.distillation import losses
from verl.utils import tensordict_utils as tu


@pytest.mark.parametrize('batch_size', [24, 96])
@pytest.mark.parametrize('dp_size', [1, 3])
@pytest.mark.parametrize('mode', ['token-mean', 'seq-mean-token-mean', 'seq-mean-token-sum', 'seq-mean-token-sum-norm'])
def test_accumulated_distillation_matches_full_batch(monkeypatch, batch_size, dp_size, mode):
    # Unequal response lengths distinguish global token weighting from averaging
    # the local micro-batch means. The stale info must also be refreshed.
    mask = torch.arange(7)[None, :] < (torch.arange(batch_size) % 7 + 1)[:, None]
    weights = torch.linspace(0.1, 0.9, batch_size * 7, dtype=torch.float64).reshape(batch_size, 7)
    config = SimpleNamespace(loss_agg_mode=mode, loss_scale_factor=7,
                             global_batch_info={'dp_size': 9, 'batch_num_tokens': 1, 'global_batch_size': 2})
    distill_config = SimpleNamespace(distillation_loss=SimpleNamespace(
        loss_mode='synthetic', loss_max_clamp=None, use_policy_gradient=False, use_task_rewards=False))
    monkeypatch.setattr(losses, 'get_distillation_loss_fn',
                        lambda name: lambda **kwargs: (kwargs['model_output']['per_token_loss'], {}))

    reference_parameter = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    reference_tokens = weights * reference_parameter.square()
    sums = (reference_tokens * mask).sum(-1)
    if mode == 'token-mean':
        reference = sums.sum() / mask.sum()
    elif mode == 'seq-mean-token-mean':
        reference = (sums / mask.sum(-1)).mean()
    else:
        reference = sums.mean()
        if mode == 'seq-mean-token-sum-norm':
            reference = reference / 7
    reference.backward()

    parameter = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    accumulated = 0.0
    for index in range(batch_size):
        data = TensorDict({'response_mask': mask[index:index + 1]}, batch_size=[1])
        tu.assign_non_tensor(data, dp_size=dp_size, batch_num_tokens=int(mask.sum()), global_batch_size=batch_size)
        result, _ = losses.distillation_ppo_loss(
            config=config, distillation_config=distill_config,
            model_output={'per_token_loss': weights[index:index + 1] * parameter.square()}, data=data)
        result.backward()
        accumulated += float(result.detach())
    # Sum contributions from all ranks, then apply FSDP's DP gradient average.
    assert accumulated / dp_size == pytest.approx(float(reference.detach()), rel=1e-10)
    assert float(parameter.grad) / dp_size == pytest.approx(float(reference_parameter.grad), rel=1e-10)
