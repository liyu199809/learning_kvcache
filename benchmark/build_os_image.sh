#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
cd "$BENCHMARK_PROJECT_ROOT"
# Ubuntu 22.04 keeps the official Dockerfile's /etc/apt/sources.list layout.
docker image inspect ubuntu:22.04 >/dev/null
docker tag ubuntu:22.04 ubuntu:latest
DOCKER_BUILDKIT=0 docker build --network host \
    --build-arg http_proxy --build-arg https_proxy --build-arg no_proxy \
    -f benchmark/opd-evolver/reference/LifelongAgentBench/scripts/dockerfile/os_interaction/default \
    -t local-os/default \
    benchmark/opd-evolver/reference/LifelongAgentBench/scripts/dockerfile/os_interaction
