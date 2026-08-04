#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
GORILLA_REPO="${GORILLA_REPO:-$PROJECT_ROOT/benchmark/gorilla}"
OFFICIAL_ROOT="${BFCL_OFFICIAL_ROOT:-$SCRIPT_DIR/official}"
PROJECT_PYTHON="${PROJECT_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
BFCL_COMMIT="cd9429ccf3d4d04156affe883c495b3b047e6b64"
MARKER="$OFFICIAL_ROOT/.bfcl_commit"
MODEL_ALIAS="${BFCL_MODEL_API_NAME:-qwen3.5-4b}"

if [[ ! -d "$GORILLA_REPO/.git" ]]; then
    echo "Missing Gorilla checkout: $GORILLA_REPO" >&2
    echo "Clone https://github.com/ShishirPatil/gorilla.git there or set GORILLA_REPO." >&2
    exit 1
fi

if ! git -C "$GORILLA_REPO" cat-file -e "$BFCL_COMMIT^{commit}" 2>/dev/null; then
    echo "Fetching official BFCL v3 commit..."
    git -C "$GORILLA_REPO" fetch origin "$BFCL_COMMIT"
fi

installed_commit=""
if [[ -f "$MARKER" ]]; then
    installed_commit="$(<"$MARKER")"
fi

if [[ "$installed_commit" != "$BFCL_COMMIT" ]]; then
    rm -rf "$OFFICIAL_ROOT"
    mkdir -p "$OFFICIAL_ROOT"
    git -C "$GORILLA_REPO" archive \
        "$BFCL_COMMIT" \
        berkeley-function-call-leaderboard \
        | tar -x -C "$OFFICIAL_ROOT" --strip-components=1
    printf '%s\n' "$BFCL_COMMIT" >"$MARKER"
fi

if [[ ! -x "$PROJECT_PYTHON" ]]; then
    echo "Project uv environment not found: $PROJECT_PYTHON" >&2
    exit 1
fi

# BFCL v3 pins NumPy 1.x and older vendor SDKs. Run its source through
# PYTHONPATH instead of installing its distribution metadata into the project.
mapfile -t MISSING_SPECS < <("$PROJECT_PYTHON" - <<'PY'
from importlib.util import find_spec

requirements = {
    "anthropic": "anthropic==0.53.0",
    "boto3": "boto3",
    "cohere": "cohere==5.13.3",
    "datamodel_code_generator": "datamodel-code-generator==0.25.7",
    "google.genai": "google-genai==1.24.0",
    "mistralai": "mistralai==1.7.0",
    "overrides": "overrides",
    "qwen_agent": "qwen-agent",
    "soundfile": "soundfile",
    "tree_sitter": "tree-sitter==0.21.3",
    "tree_sitter_java": "tree-sitter-java==0.21.0",
    "tree_sitter_javascript": "tree-sitter-javascript==0.21.4",
    "writerai": "writer-sdk>=2.1.0",
}
for module, requirement in requirements.items():
    try:
        present = find_spec(module) is not None
    except ModuleNotFoundError:
        present = False
    if not present:
        print(requirement)
PY
)
if (( ${#MISSING_SPECS[@]} )); then
    uv pip install --python "$PROJECT_PYTHON" --no-deps "${MISSING_SPECS[@]}"
fi

BFCL_PROJECT_ROOT="$SCRIPT_DIR/artifacts" \
PYTHONPATH="$OFFICIAL_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
"$PROJECT_PYTHON" - <<'PY'
import numpy
import soundfile
import transformers
from bfcl_eval.constants.category_mapping import VERSION_PREFIX

assert VERSION_PREFIX == "BFCL_v3", VERSION_PREFIX
print(
    f"BFCL ready: {VERSION_PREFIX}; "
    f"NumPy {numpy.__version__}; "
    f"Transformers {transformers.__version__}; "
    f"SoundFile {soundfile.__version__}"
)
PY

BFCL_PROJECT_ROOT="$SCRIPT_DIR/artifacts" \
PYTHONPATH="$OFFICIAL_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
"$PROJECT_PYTHON" -m bfcl_eval test-categories >/dev/null

touch "$SCRIPT_DIR/.gitignore"
grep -Fxq "$MODEL_ALIAS" "$SCRIPT_DIR/.gitignore" \
    || printf '%s\n' "$MODEL_ALIAS" >>"$SCRIPT_DIR/.gitignore"

echo "Official commit: $BFCL_COMMIT"
echo "Environment:     $PROJECT_ROOT/.venv"
