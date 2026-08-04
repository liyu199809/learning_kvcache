#!/usr/bin/env bash
# tau2-bench 环境 bootstrap（一次性）。
#
# tau2 要求 Python >=3.12,<3.14 且强依赖 litellm（会拖动 openai/httpx/tokenizers
# 等），无法复用主 .venv（Python 3.11 + vllm/transformers）。因此为 tau2 建一个
# 独立的 uv venv（Python 3.12），只装 tau2 及其依赖，与主环境完全隔离。
#
# 官方源码固定在 commit 363133a（v1.0.1）；判分逻辑原样使用，不做修改。
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OFFICIAL_ROOT="${TAU2_OFFICIAL_ROOT:-$SCRIPT_DIR/official}"
VENV_DIR="${TAU2_VENV:-$SCRIPT_DIR/.venv}"
TAU2_COMMIT="363133ada1936491fb5bcec33cd62c3518a99f65"
PY_VERSION="${TAU2_PYTHON_VERSION:-3.12}"
REPO_URL="https://github.com/sierra-research/tau2-bench"

command -v uv >/dev/null 2>&1 || { echo "uv 不在 PATH 中，请先安装 uv。" >&2; exit 1; }

# 1) 官方源码：克隆并固定到 pinned commit。
if [[ ! -d "$OFFICIAL_ROOT/.git" ]]; then
    echo "Cloning tau2-bench official source..."
    git clone --quiet "$REPO_URL" "$OFFICIAL_ROOT"
fi
if ! git -C "$OFFICIAL_ROOT" cat-file -e "$TAU2_COMMIT^{commit}" 2>/dev/null; then
    git -C "$OFFICIAL_ROOT" fetch --quiet origin "$TAU2_COMMIT"
fi
git -C "$OFFICIAL_ROOT" checkout --quiet "$TAU2_COMMIT"

# 2) 专用 uv venv（Python 3.12），与主环境隔离。
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo "Creating tau2 venv (Python $PY_VERSION)..."
    uv venv --python "$PY_VERSION" "$VENV_DIR"
fi

# 3) editable 安装官方包（提供 tau2 命令 + 让 DATA_DIR 指向源码 data/）。
#    editable 安装后，tau2 从源码 checkout 解析 data/，无需再设 TAU2_DATA_DIR。
echo "Installing tau2 (editable) into $VENV_DIR ..."
VIRTUAL_ENV="$VENV_DIR" uv pip install --python "$VENV_DIR/bin/python" -e "$OFFICIAL_ROOT"

# 4) 校验：tau2 命令可用、版本、可列出 domain。
"$VENV_DIR/bin/tau2" --help >/dev/null 2>&1 || {
    echo "tau2 命令不可用，安装可能失败。" >&2
    exit 1
}
echo "Official commit: $TAU2_COMMIT"
echo "tau2 venv:       $VENV_DIR ($("$VENV_DIR/bin/python" --version))"
echo "tau2 ready."
