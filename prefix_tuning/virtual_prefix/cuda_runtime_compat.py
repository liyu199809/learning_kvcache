"""Compatibility guards for CUDA libraries loaded alongside TileLang."""

from __future__ import annotations

import os


def prefer_real_cudart_over_stubs() -> str | None:
    """Make vLLM ignore TileLang's link-time CUDA runtime stub.

    PyTorch has already loaded the real CUDA runtime when this function runs.
    vLLM 0.22 otherwise returns the first ``libcudart`` entry in
    ``/proc/self/maps``; after importing TileLang that entry can be
    ``libcudart_stub.so``, which lacks runtime functions such as
    ``cudaDeviceReset``.
    """
    from vllm.distributed.device_communicators import cuda_wrapper

    saved_path = getattr(cuda_wrapper, "_kvcache_real_cudart_path", None)
    if saved_path is not None:
        return saved_path

    original_find_loaded_library = cuda_wrapper.find_loaded_library
    real_cudart_path = original_find_loaded_library("libcudart")
    if real_cudart_path is None:
        return None
    if "_stub" in os.path.basename(real_cudart_path):
        raise RuntimeError(
            "TileLang CUDA stubs were loaded before the real CUDA runtime"
        )

    def find_loaded_library_without_cudart_stub(lib_name: str) -> str | None:
        selected_path = original_find_loaded_library(lib_name)
        if (
            lib_name == "libcudart"
            and selected_path is not None
            and "_stub" in os.path.basename(selected_path)
        ):
            return real_cudart_path
        return selected_path

    cuda_wrapper.find_loaded_library = find_loaded_library_without_cudart_stub
    cuda_wrapper._kvcache_real_cudart_path = real_cudart_path
    return real_cudart_path
