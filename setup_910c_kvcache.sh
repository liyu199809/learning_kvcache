#!/usr/bin/env bash
# Build and verify the Ascend 910C KV-cache environment without creating symlinks.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}
BASE_PYTHON=${BASE_PYTHON:-/usr/local/miniconda3/bin/python}
CONDA_BIN=${CONDA_BIN:-/usr/local/miniconda3/bin/conda}
CANN_ENV_FILE=${CANN_ENV_FILE:-/usr/local/Ascend/cann-9.1.0/set_env.sh}
ENV_NAME=${ENV_NAME:-kvcache}
ENV_PREFIX=${ENV_PREFIX:-/usr/local/miniconda3/envs/$ENV_NAME}
PIP_INDEX_URL=${PIP_INDEX_URL:-https://mirrors.jd.com/pypi/web/simple}
VERIFY_ONLY=0

usage() {
    cat <<'EOF'
Usage: ./setup_910c_kvcache.sh [--verify-only]

Creates an Ascend 910C environment from a copied Python executable and regular
configuration files. It reuses the server image's CANN/torch_npu/vLLM-Ascend
packages, installs service dependencies from the Tsinghua PyPI mirror, and
installs this checkout's local packages in editable mode.

No symlinks are created. The script checks the environment path after every
installation phase and compares both the Conda env tree and project tree before
and after setup.

Environment overrides:
  PROJECT_ROOT    Project checkout (default: directory containing this script)
  BASE_PYTHON     Base Python with the preinstalled NPU stack
  CONDA_BIN       Conda executable used only for activation verification
  CANN_ENV_FILE   CANN environment setup script
  ENV_NAME        Environment name (default: kvcache)
  ENV_PREFIX      Environment path
  PIP_INDEX_URL   Python package index (default: Tsinghua mirror)
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

ENV_PARENT=$(dirname -- "$ENV_PREFIX")
PYTHON="$ENV_PREFIX/bin/python"
BASELINE_LINKS=
FINAL_LINKS=
BASELINE_PROJECT_LINKS=
FINAL_PROJECT_LINKS=
WHEELHOUSE=

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
    if [[ -n "$WHEELHOUSE" && -d "$WHEELHOUSE" ]]; then
        case "$WHEELHOUSE" in
            /tmp/kvcache-910c-wheels.*) rm -rf -- "$WHEELHOUSE" ;;
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
    /usr /usr/local /usr/local/miniconda3 "$ENV_PARENT" \
    /home /home/liyu.468 "$PROJECT_ROOT" "$ENV_PREFIX"; do
    fail_if_symlink "$path"
done

[[ -x "$BASE_PYTHON" ]] || { echo "Base Python not found: $BASE_PYTHON" >&2; exit 1; }
[[ -x "$CONDA_BIN" ]] || { echo "Conda not found: $CONDA_BIN" >&2; exit 1; }
[[ -f "$CANN_ENV_FILE" ]] || { echo "CANN setup file not found: $CANN_ENV_FILE" >&2; exit 1; }
[[ -d "$PROJECT_ROOT/verl" ]] || { echo "Invalid project root: $PROJECT_ROOT" >&2; exit 1; }
command -v npu-smi >/dev/null || { echo "npu-smi is unavailable" >&2; exit 1; }
[[ $(uname -m) == aarch64 ]] || { echo "Expected aarch64 Ascend host" >&2; exit 1; }

mkdir -p -- "$ENV_PARENT"
BASELINE_LINKS=$(mktemp /tmp/kvcache-910c-links-before.XXXXXX)
FINAL_LINKS=$(mktemp /tmp/kvcache-910c-links-after.XXXXXX)
BASELINE_PROJECT_LINKS=$(mktemp /tmp/kvcache-910c-project-links-before.XXXXXX)
FINAL_PROJECT_LINKS=$(mktemp /tmp/kvcache-910c-project-links-after.XXXXXX)
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
    echo "[2/5] Configuring Conda activation and CANN 9.1"
    mkdir -p -- "$ENV_PREFIX/conda-meta" "$ENV_PREFIX/etc/conda/activate.d"
    : > "$ENV_PREFIX/conda-meta/history"
    printf '%s\n' \
        '#!/usr/bin/env bash' \
        "source '$CANN_ENV_FILE'" \
        'export LD_LIBRARY_PATH="/usr/local/miniconda3/lib:/usr/local/miniconda3/lib64:${LD_LIBRARY_PATH:-}"' \
        > "$ENV_PREFIX/etc/conda/activate.d/910c-cann.sh"
    chmod 0644 "$ENV_PREFIX/conda-meta/history" "$ENV_PREFIX/etc/conda/activate.d/910c-cann.sh"
    assert_no_env_symlinks

    echo "[3/5] Downloading and checking service dependency wheels"
    WHEELHOUSE=$(mktemp -d /tmp/kvcache-910c-wheels.XXXXXX)
    "$PYTHON" -m pip download \
        --index-url "$PIP_INDEX_URL" \
        --only-binary=:all: \
        --dest "$WHEELHOUSE" \
        'fastmcp==3.4.3' \
        'fastapi-mcp==0.4.0' \
        'sqlalchemy==2.0.51' \
        'tomli==2.4.1' \
        'tomli-w==1.2.0' \
        'python-dateutil==2.9.0.post0' \
        'protobuf==6.33.6' \
        'gradio>=6.15.0'

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
        --upgrade-strategy only-if-needed \
        'fastmcp==3.4.3' \
        'fastapi-mcp==0.4.0' \
        'sqlalchemy==2.0.51' \
        'tomli==2.4.1' \
        'tomli-w==1.2.0' \
        'python-dateutil==2.9.0.post0' \
        'protobuf==6.33.6' \
        'gradio>=6.15.0'
    assert_no_env_symlinks

    echo "[4/5] Installing project packages in editable mode"
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

echo "[5/5] Verifying NPU access, project imports, and plugin registration"
# shellcheck disable=SC1090
source "$CANN_ENV_FILE"
cd "$PROJECT_ROOT"
"$PYTHON" - <<'PY'
import importlib
import importlib.metadata as metadata
import os
import sys

modules = (
    "torch",
    "torch_npu",
    "vllm",
    "vllm_ascend",
    "transformers",
    "ray",
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

torch = loaded["torch"]
vllm = loaded["vllm"]
assert hasattr(torch, "npu"), "torch_npu did not register torch.npu"
assert torch.npu.is_available(), "PyTorch cannot access the Ascend NPU"
assert torch.npu.device_count() > 0, "No Ascend NPUs are visible"
assert tuple(map(int, vllm.__version__.split(".")[:2])) >= (0, 23), vllm.__version__

project_root = os.path.realpath(os.getcwd())
for package in ("verl", "openenv"):
    package_file = os.path.realpath(loaded[package].__file__)
    assert package_file.startswith(project_root + os.sep), (package, package_file)

plugins = {entry.name for entry in metadata.entry_points(group="vllm.general_plugins")}
assert "qwen35_delta_virtual_prefix" in plugins, sorted(plugins)

importlib.import_module("verl.workers.rollout.vllm_rollout.vllm_async_server")
importlib.import_module("verl.workers.engine.fsdp.transformer_impl")

value = torch.arange(1024, dtype=torch.float32, device="npu:0").sum()
torch.npu.synchronize()
assert value.item() == 523776.0, value.item()

print("python", sys.executable)
print("torch", torch.__version__, "torch_npu", metadata.version("torch-npu"))
print("vllm", vllm.__version__, "vllm_ascend", metadata.version("vllm-ascend"))
print("transformers", loaded["transformers"].__version__)
print("ray", loaded["ray"].__version__)
print("npu_count", torch.npu.device_count())
print("npu_0", torch.npu.get_device_name(0))
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
