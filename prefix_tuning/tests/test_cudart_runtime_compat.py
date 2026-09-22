"""Regression test for TileLang/vLLM CUDA runtime library selection."""

import os

import pytest
import torch

from prefix_tuning.virtual_prefix.cuda_runtime_compat import (
    prefer_real_cudart_over_stubs,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_vllm_ignores_tilelang_cudart_stub() -> None:
    real_cudart_path = prefer_real_cudart_over_stubs()
    assert real_cudart_path is not None
    assert "_stub" not in os.path.basename(real_cudart_path)

    pytest.importorskip("tilelang")

    from vllm.distributed.device_communicators.cuda_wrapper import CudaRTLibrary

    library = CudaRTLibrary()
    assert "_stub" not in os.path.basename(library.lib._name)
    assert hasattr(library.lib, "cudaDeviceReset")
