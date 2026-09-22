#!/usr/bin/env bash
# Build and verify the H200 KV-cache environment without creating symlinks.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}
BASE_PYTHON=${BASE_PYTHON:-/usr/local/miniconda3/bin/python}
CONDA_BIN=${CONDA_BIN:-/usr/local/miniconda3/bin/conda}
ENV_NAME=${ENV_NAME:-kvcache}
ENV_PREFIX=${ENV_PREFIX:-/usr/local/miniconda3/envs/$ENV_NAME}
PIP_INDEX_URL=${PIP_INDEX_URL:-https://mirrors.jd.com/pypi/web/simple}
CAUSAL_CONV1D_VERSION=${CAUSAL_CONV1D_VERSION:-1.6.1}
CAUSAL_CONV1D_MAX_JOBS=${CAUSAL_CONV1D_MAX_JOBS:-16}
VERIFY_ONLY=0

usage() {
    cat <<'EOF'
Usage: ./setup_kvcache.sh [--verify-only]

Creates an H200 environment from a copied Python executable and regular
configuration files. It reuses the server image's CUDA/torch/vLLM packages,
installs service dependencies from the JD PyPI mirror, and installs this
checkout's local packages in editable mode.

No symlinks are created. The script checks the environment path after every
installation phase and compares both the Conda env tree and project tree before
and after setup.

Environment overrides:
  PROJECT_ROOT   Project checkout (default: directory containing this script)
  BASE_PYTHON    CUDA-aware base Python (default: /usr/local/miniconda3/bin/python)
  CONDA_BIN      Conda executable used only for activation verification
  ENV_NAME       Conda environment name (default: kvcache)
  ENV_PREFIX     Environment path (default: /usr/local/miniconda3/envs/$ENV_NAME)
  PIP_INDEX_URL  Python package index URL (default: JD mirror)
  CAUSAL_CONV1D_VERSION   causal-conv1d version (default: 1.6.1)
  CAUSAL_CONV1D_MAX_JOBS  CUDA extension build parallelism (default: 16)
EOF
}

while (($#)); do
    case "$1" in
        --verify-only) VERIFY_ONLY=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

CONDA_ROOT=$(dirname -- "$(dirname -- "$CONDA_BIN")")
ENV_PARENT=$(dirname -- "$ENV_PREFIX")
PYTHON="$ENV_PREFIX/bin/python"
BASELINE_LINKS=
FINAL_LINKS=
BASELINE_PROJECT_LINKS=
FINAL_PROJECT_LINKS=
WHEELHOUSE=
CONSTRAINT_FILE=

SERVICE_DEPS=(
    'fastmcp==3.4.3'
    'fastapi-mcp==0.4.0'
    'sqlalchemy==2.0.51'
    'tomli==2.4.1'
    'tomli-w==1.2.0'
    'python-dateutil==2.9.0.post0'
    'protobuf==6.33.6'
    'setuptools==80.10.2'
    'starlette==1.3.1'
    'wandb==0.26.1'
)

cleanup() {
    if [[ -n "$BASELINE_LINKS" && -f "$BASELINE_LINKS" ]]; then
        rm -f -- "$BASELINE_LINKS"
    fi
    if [[ -n "$FINAL_LINKS" && -f "$FINAL_LINKS" ]]; then
        rm -f -- "$FINAL_LINKS"
    fi
    if [[ -n "$BASELINE_PROJECT_LINKS" && -f "$BASELINE_PROJECT_LINKS" ]]; then
        rm -f -- "$BASELINE_PROJECT_LINKS"
    fi
    if [[ -n "$FINAL_PROJECT_LINKS" && -f "$FINAL_PROJECT_LINKS" ]]; then
        rm -f -- "$FINAL_PROJECT_LINKS"
    fi
    if [[ -n "$CONSTRAINT_FILE" && -f "$CONSTRAINT_FILE" ]]; then
        rm -f -- "$CONSTRAINT_FILE"
    fi
    if [[ -n "$WHEELHOUSE" && -d "$WHEELHOUSE" ]]; then
        case "$WHEELHOUSE" in
            /tmp/kvcache-wheels.*) rm -rf -- "$WHEELHOUSE" ;;
            *) echo "Refusing to remove unexpected temporary path: $WHEELHOUSE" >&2 ;;
        esac
    fi
}
trap cleanup EXIT

fail_if_symlink() {
    local path=$1
    if [[ -L "$path" ]]; then
        echo "Refusing symlink path: $path" >&2
        exit 1
    fi
}

assert_no_env_symlinks() {
    local found
    found=$(find "$ENV_PREFIX" -xdev -type l -print -quit 2>/dev/null || true)
    if [[ -n "$found" ]]; then
        echo "Symlink detected in environment: $found" >&2
        exit 1
    fi
}

snapshot_env_links() {
    local destination=$1
    find "$ENV_PARENT" -xdev -type l -print 2>/dev/null | LC_ALL=C sort > "$destination"
}

snapshot_project_links() {
    local destination=$1
    find "$PROJECT_ROOT" -xdev -type l -print 2>/dev/null | LC_ALL=C sort > "$destination"
}

for path in \
    /usr /usr/local "$CONDA_ROOT" "$ENV_PARENT" \
    /home "$HOME" "$PROJECT_ROOT" "$ENV_PREFIX"; do
    fail_if_symlink "$path"
done

[[ -x "$BASE_PYTHON" ]] || { echo "Base Python not found: $BASE_PYTHON" >&2; exit 1; }
[[ -x "$CONDA_BIN" ]] || { echo "Conda not found: $CONDA_BIN" >&2; exit 1; }
[[ -d "$PROJECT_ROOT/verl" ]] || { echo "Invalid project root: $PROJECT_ROOT" >&2; exit 1; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi is unavailable" >&2; exit 1; }

mkdir -p -- "$ENV_PARENT"
BASELINE_LINKS=$(mktemp /tmp/kvcache-links-before.XXXXXX)
FINAL_LINKS=$(mktemp /tmp/kvcache-links-after.XXXXXX)
BASELINE_PROJECT_LINKS=$(mktemp /tmp/kvcache-project-links-before.XXXXXX)
FINAL_PROJECT_LINKS=$(mktemp /tmp/kvcache-project-links-after.XXXXXX)
snapshot_env_links "$BASELINE_LINKS"
snapshot_project_links "$BASELINE_PROJECT_LINKS"

if [[ ! -x "$PYTHON" ]]; then
    if [[ -e "$ENV_PREFIX" ]]; then
        echo "Refusing to overwrite incomplete environment: $ENV_PREFIX" >&2
        exit 1
    fi
    [[ "$VERIFY_ONLY" == 0 ]] || { echo "Environment is missing: $ENV_PREFIX" >&2; exit 1; }
    echo "[1/5] Creating a regular-file Python environment at $ENV_PREFIX"
    BASE_EXECUTABLE=$(readlink -f -- "$BASE_PYTHON")
    BASE_HOME=$(dirname -- "$BASE_EXECUTABLE")
    PYTHON_VERSION=$($BASE_PYTHON -c 'import platform; print(platform.python_version())')
    PYTHON_ABI=$($BASE_PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    mkdir -p -- "$ENV_PREFIX/bin" "$ENV_PREFIX/lib/python$PYTHON_ABI/site-packages"
    install -m 0755 "$BASE_EXECUTABLE" "$PYTHON"
    printf '%s\n' \
        "home = $BASE_HOME" \
        'include-system-site-packages = true' \
        "version = $PYTHON_VERSION" \
        "executable = $BASE_EXECUTABLE" \
        > "$ENV_PREFIX/pyvenv.cfg"
    "$PYTHON" -m pip --version
else
    echo "[1/5] Reusing environment at $ENV_PREFIX"
fi
assert_no_env_symlinks

if [[ "$VERIFY_ONLY" == 0 ]]; then
    echo "[2/5] Configuring Conda activation metadata and pip entrypoints"
    mkdir -p -- "$ENV_PREFIX/conda-meta" "$ENV_PREFIX/etc/conda/activate.d"
    : > "$ENV_PREFIX/conda-meta/history"
    chmod 0644 "$ENV_PREFIX/conda-meta/history"

    for pip_command in pip pip3 pip3.12; do
        pip_entrypoint="$ENV_PREFIX/bin/$pip_command"
        fail_if_symlink "$pip_entrypoint"
        printf '%s\n' \
            '#!/usr/bin/env bash' \
            "exec '$PYTHON' -m pip \"\$@\"" \
            > "$pip_entrypoint"
        chmod 0755 "$pip_entrypoint"
    done

    PIP_ACTIVATE_HOOK="$ENV_PREFIX/etc/conda/activate.d/kvcache-pip.sh"
    fail_if_symlink "$PIP_ACTIVATE_HOOK"
    printf '%s\n' \
        '#!/usr/bin/env bash' \
        'unalias pip 2>/dev/null || true' \
        'hash -r 2>/dev/null || true' \
        > "$PIP_ACTIVATE_HOOK"
    chmod 0644 "$PIP_ACTIVATE_HOOK"
    assert_no_env_symlinks

    echo "[3/5] Downloading and checking service dependency wheels"
    PIP_CONSTRAINT_ARGS=()
    if [[ -f "$PROJECT_ROOT/requirements-freeze.txt" ]]; then
        CONSTRAINT_FILE=$(mktemp /tmp/kvcache-pip-constraints.XXXXXX.txt)
        # B200 used setuptools 81 (H200 vLLM 0.22 requires <81), and its
        # recorded Starlette predates the requirement in FastMCP 3.4.3.
        grep -Ev '^(setuptools|starlette)==' \
            "$PROJECT_ROOT/requirements-freeze.txt" > "$CONSTRAINT_FILE"
        PIP_CONSTRAINT_ARGS=(--constraint "$CONSTRAINT_FILE")
    fi

    WHEELHOUSE=$(mktemp -d /tmp/kvcache-wheels.XXXXXX)
    "$PYTHON" -m pip download \
        --index-url "$PIP_INDEX_URL" \
        "${PIP_CONSTRAINT_ARGS[@]}" \
        --only-binary=:all: \
        --dest "$WHEELHOUSE" \
        "${SERVICE_DEPS[@]}"

    "$BASE_PYTHON" - "$WHEELHOUSE" <<'PY'
import pathlib
import stat
import sys
import zipfile

wheelhouse = pathlib.Path(sys.argv[1])
wheels = sorted(wheelhouse.glob("*.whl"))
if not wheels:
    raise SystemExit("No wheels were downloaded")

for wheel in wheels:
    with zipfile.ZipFile(wheel) as archive:
        for member in archive.infolist():
            mode = member.external_attr >> 16
            if stat.S_IFMT(mode) == stat.S_IFLNK:
                raise SystemExit(f"Wheel contains a symlink entry: {wheel.name}:{member.filename}")
print(f"checked_wheels={len(wheels)} symlink_entries=0")
PY

    "$PYTHON" -m pip install \
        --no-index \
        --find-links "$WHEELHOUSE" \
        "${PIP_CONSTRAINT_ARGS[@]}" \
        --upgrade-strategy only-if-needed \
        "${SERVICE_DEPS[@]}"
    assert_no_env_symlinks

    echo "[4/5] Installing CUDA fast kernels and project packages"
    # The environment includes base site-packages, so a plain pip install can
    # mistake base's ninja metadata for an environment-local executable. Force
    # a regular-file ninja installation into ENV_PREFIX before compiling the
    # causal-conv1d extension required by packed Qwen3.5 training.
    "$PYTHON" -m pip install \
        --ignore-installed \
        --no-deps \
        --no-cache-dir \
        --index-url "$PIP_INDEX_URL" \
        'ninja==1.13.2'
    MAX_JOBS="$CAUSAL_CONV1D_MAX_JOBS" \
    CAUSAL_CONV1D_FORCE_BUILD=TRUE \
    "$PYTHON" -m pip install \
        --no-deps \
        --no-build-isolation \
        --no-cache-dir \
        --index-url "$PIP_INDEX_URL" \
        "causal-conv1d==$CAUSAL_CONV1D_VERSION"
    assert_no_env_symlinks

    "$PYTHON" -m pip install \
        --no-deps \
        --no-build-isolation \
        --editable "$PROJECT_ROOT/verl" \
        --editable "$PROJECT_ROOT/OpenEnv" \
        --editable "$PROJECT_ROOT/OpenEnv/envs/agent_world_model_env" \
        --editable "$PROJECT_ROOT/OpenEnv/envs/env_scaler_env" \
        --editable "$PROJECT_ROOT/OpenEnv/envs/code_judge_env" \
        --editable "$PROJECT_ROOT/prefix_tuning"
    assert_no_env_symlinks
fi

snapshot_env_links "$FINAL_LINKS"
snapshot_project_links "$FINAL_PROJECT_LINKS"
NEW_LINKS=$(comm -13 "$BASELINE_LINKS" "$FINAL_LINKS")
if [[ -n "$NEW_LINKS" ]]; then
    echo "New symlinks detected; refusing the setup result:" >&2
    printf '%s\n' "$NEW_LINKS" >&2
    exit 1
fi
NEW_PROJECT_LINKS=$(comm -13 "$BASELINE_PROJECT_LINKS" "$FINAL_PROJECT_LINKS")
if [[ -n "$NEW_PROJECT_LINKS" ]]; then
    echo "New symlinks detected under the project directory:" >&2
    printf '%s\n' "$NEW_PROJECT_LINKS" >&2
    exit 1
fi

echo "[5/5] Verifying imports, plugin registration, CUDA, and GPU access"
cd "$PROJECT_ROOT"
"$PYTHON" - <<'PY'
import importlib
import importlib.metadata as metadata
import os
import sys

modules = (
    "torch",
    "vllm",
    "transformers",
    "ray",
    "flash_attn",
    "causal_conv1d",
    "verl",
    "openenv",
    "agent_world_model_env",
    "env_scaler_env",
    "code_judge_env",
    "fastmcp",
    "wandb",
    "prefix_tuning.virtual_prefix",
)
loaded = {name: importlib.import_module(name) for name in modules}
importlib.import_module("causal_conv1d_cuda")

torch = loaded["torch"]
vllm = loaded["vllm"]
assert torch.cuda.is_available(), "PyTorch cannot access CUDA"
assert torch.cuda.device_count() > 0, "No CUDA GPUs are visible"
assert str(torch.version.cuda).startswith("13."), torch.version.cuda
assert tuple(map(int, vllm.__version__.split(".")[:2])) >= (0, 22), vllm.__version__

from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen3_5
assert qwen3_5.is_fast_path_available, (
    "Qwen3.5 packed training requires FLA and causal-conv1d fast kernels"
)

project_root = os.path.realpath(os.getcwd())
for package in ("verl", "openenv"):
    package_file = os.path.realpath(loaded[package].__file__)
    assert package_file.startswith(project_root + os.sep), (package, package_file)

plugins = {
    entry.name for entry in metadata.entry_points(group="vllm.general_plugins")
}
assert "qwen35_delta_virtual_prefix" in plugins, sorted(plugins)

print("python", metadata.version("pip"), "at", sys.executable)
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("vllm", vllm.__version__)
print("transformers", loaded["transformers"].__version__)
print("ray", loaded["ray"].__version__)
print("flash_attn", loaded["flash_attn"].__version__)
print("causal_conv1d", metadata.version("causal-conv1d"))
print("qwen3_5_fast_path", qwen3_5.is_fast_path_available)
print("gpus", torch.cuda.device_count(), torch.cuda.get_device_name(0))
print("vllm_plugin", "qwen35_delta_virtual_prefix")
print("kvcache_environment_ready", True)
PY

assert_no_env_symlinks

cat <<EOF

Environment ready. No new symlinks were created.
Activate with:
  source /usr/local/miniconda3/etc/profile.d/conda.sh
  conda activate $ENV_NAME

Direct Python path:
  $PYTHON
EOF
