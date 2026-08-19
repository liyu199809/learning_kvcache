"""CPU tests for generic trainable-only FSDP checkpoints."""

from collections import namedtuple
from types import SimpleNamespace

import pytest
import torch

from verl.utils.checkpoint.checkpoint_manager import BaseCheckpointManager
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager

_LoadResult = namedtuple("_LoadResult", ["missing_keys", "unexpected_keys"])
PREFIX_NAME = "model.language_model.layers.0.linear_attn.prefix_tokens"


@pytest.fixture(autouse=True)
def _patch_dist(monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)


class _Config:
    text_config = SimpleNamespace(delta_prefix_num_virtual_tokens=2)

    def save_pretrained(self, path):
        return None


class _InnerModel:
    def __init__(self):
        self.base = torch.nn.Parameter(torch.tensor([3.0]))
        self.prefix = torch.nn.Parameter(torch.arange(8, dtype=torch.float32).reshape(2, 4))
        self.config = _Config()

    def state_dict(self):
        return {"base.weight": self.base, PREFIX_NAME: self.prefix}

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if "base.weight" in state_dict:
            self.base.data.copy_(state_dict["base.weight"])
        if PREFIX_NAME in state_dict:
            self.prefix.data.copy_(state_dict[PREFIX_NAME])
        missing = [] if strict else [name for name in ("base.weight", PREFIX_NAME) if name not in state_dict]
        unexpected = [name for name in state_dict if name not in ("base.weight", PREFIX_NAME)]
        return _LoadResult(missing, unexpected)

    def can_generate(self):
        return False


class _WrappedModel:
    def __init__(self):
        self._fsdp_wrapped_module = _InnerModel()
        self.config = self._fsdp_wrapped_module.config
        self.can_generate = self._fsdp_wrapped_module.can_generate

    def state_dict(self):
        return self._fsdp_wrapped_module.state_dict()

    def load_state_dict(self, state_dict, strict=True, assign=False):
        return self._fsdp_wrapped_module.load_state_dict(state_dict, strict=strict, assign=assign)

    def named_buffers(self):
        return {}.items()


def _manager(model, config):
    return FSDPCheckpointManager(
        model=model,
        optimizer=None,
        processing_class=None,
        checkpoint_config=config,
        trainable_param_names={PREFIX_NAME},
        base_model_path="/models/base",
    )


def test_save_trainable_only_property():
    manager = BaseCheckpointManager(_WrappedModel(), object(), checkpoint_config={"save_trainable_only": True})
    assert manager.should_save_trainable_only is True


def test_save_and_restore_trainable_only(tmp_path):
    source = _WrappedModel()
    checkpoint = tmp_path / "checkpoint"
    _manager(
        source,
        {"save_trainable_only": True, "save_contents": ["model"]},
    ).save_checkpoint(str(checkpoint), global_step=7)

    state = torch.load(checkpoint / "model_world_size_1_rank_0.pt", weights_only=False)
    assert set(state) == {PREFIX_NAME}
    metadata = __import__("json").loads((checkpoint / "trainable_only_meta.json").read_text())
    assert metadata["parameter_count"] == 1
    assert metadata["total_numel"] == 8
    assert metadata["num_virtual_tokens"] == 2

    target = _WrappedModel()
    target._fsdp_wrapped_module.base.data.fill_(99.0)
    target._fsdp_wrapped_module.prefix.data.zero_()
    _manager(target, {"load_contents": ["model"]}).load_checkpoint(str(checkpoint))
    assert target._fsdp_wrapped_module.base.item() == 99.0
    torch.testing.assert_close(target._fsdp_wrapped_module.prefix, source._fsdp_wrapped_module.prefix)


def test_trainable_only_metadata_name_mismatch_fails(tmp_path):
    source = _WrappedModel()
    checkpoint = tmp_path / "checkpoint"
    _manager(
        source,
        {"save_trainable_only": True, "save_contents": ["model"]},
    ).save_checkpoint(str(checkpoint), global_step=1)
    manager = FSDPCheckpointManager(
        model=_WrappedModel(),
        optimizer=None,
        checkpoint_config={"load_contents": ["model"]},
        trainable_param_names={"wrong.name"},
    )
    with pytest.raises(ValueError, match="parameter names do not match"):
        manager.load_checkpoint(str(checkpoint))
