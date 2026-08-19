"""CPU tests for FSDP generic trainable parameter selection."""

from types import SimpleNamespace

import pytest
import torch

from verl.workers.engine.fsdp.transformer_impl import FSDPEngine


def _engine(**overrides):
    engine = FSDPEngine.__new__(FSDPEngine)
    config = {
        "trainable_param_patterns": [r"prefix_tokens"],
        "rollout_sync_trainable_only": True,
        "expected_trainable_param_count": 1,
        "expected_trainable_numel": 8,
    }
    config.update(overrides)
    engine.model_config = SimpleNamespace(**config)
    engine._is_lora = False
    engine._trainable_param_names = set()
    engine.rank = 0
    return engine


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torch.nn.Linear(4, 4, bias=False)
        self.prefix_tokens = torch.nn.Parameter(torch.zeros(2, 4))


def test_selects_and_freezes_exact_parameters():
    model = _Model()
    engine = _engine()
    engine._configure_trainable_parameters(model)
    assert engine._trainable_param_names == {"prefix_tokens"}
    assert model.prefix_tokens.requires_grad
    assert not model.base.weight.requires_grad


def test_count_validation_fails_closed():
    with pytest.raises(ValueError, match="expected 2"):
        _engine(expected_trainable_param_count=2)._configure_trainable_parameters(_Model())


def test_sync_only_requires_patterns():
    with pytest.raises(ValueError, match="requires non-empty"):
        _engine(trainable_param_patterns=[])._configure_trainable_parameters(_Model())
