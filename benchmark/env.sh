#!/usr/bin/env bash
# Source this file before direct Python/CLI calls, or use benchmark/run.sh.
BENCHMARK_PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export PROJECT_PYTHON="${PROJECT_PYTHON:-$BENCHMARK_PROJECT_ROOT/benchmark/.venv/bin/python}"
export PATH="$BENCHMARK_PROJECT_ROOT/benchmark/runtime/docker:$PATH"
export DOCKER_HOST="${DOCKER_HOST:-unix://$BENCHMARK_PROJECT_ROOT/benchmark/runtime/docker.sock}"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}localhost,127.0.0.1,6.5.85.147"
export no_proxy="$NO_PROXY"
export HF_HOME="$BENCHMARK_PROJECT_ROOT/benchmark/runtime/hf-cache"
export XDG_CACHE_HOME="$BENCHMARK_PROJECT_ROOT/benchmark/runtime/cache"
export UV_CACHE_DIR="$BENCHMARK_PROJECT_ROOT/benchmark/runtime/uv-cache"
export UV_PYTHON_INSTALL_DIR="$BENCHMARK_PROJECT_ROOT/benchmark/runtime/python"
export TIKTOKEN_CACHE_DIR="$BENCHMARK_PROJECT_ROOT/benchmark/runtime/tiktoken-cache"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

export LITELLM_LOCAL_MODEL_COST_MAP=True
