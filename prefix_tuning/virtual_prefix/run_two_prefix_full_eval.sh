#!/usr/bin/env bash
set -euo pipefail

# BFCL uses one socket per worker. The host default (1024) is too small for
# two 1024-thread generators plus normal files and keep-alive connections.
ulimit -n "${EVAL_NOFILE_LIMIT:-65536}"

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
EXP="$ROOT/outputs/prefix_composition_3domain_20260917"
source "$ROOT/benchmark/env.sh"
cd "$ROOT"

STAMP=$(date +%Y%m%d_%H%M%S)
OUT=${1:-$EXP/eval/two_prefix_full_$STAMP}
[[ ! -e "$OUT" ]] || { echo "Refusing to overwrite $OUT" >&2; exit 1; }
mkdir -p "$OUT/logs"

run_tau2() {
    local domain=$1 port=$2 alias=$3 task_domain=$4
    local base="$OUT/$domain" endpoint="http://127.0.0.1:$port/v1"
    mkdir -p "$base"
    bash benchmark/run.sh --benchmark tau2 --tau2-domain "$task_domain" \
      --tau2-max-steps 200 --llm-max-completion-tokens 16384 \
      --tau2-judge-api-key-env BENCHMARK_LOCAL_JUDGE_ONLY \
      --openai-base-url "$endpoint" --model "$alias" --concurrency 128 \
      --output-dir "$base/tau2_$task_domain" \
      > "$OUT/logs/${domain}_tau2_${task_domain}.log" 2>&1
}

run_bfcl() {
    local domain=$1 port=$2 alias=$3 model_path=$4
    local base="$OUT/$domain" endpoint="http://127.0.0.1:$port/v1"
    mkdir -p "$base"
    bash benchmark/run.sh --benchmark bfcl_v3 --bfcl-model-path "$model_path" \
      --bfcl-categories all --openai-base-url "$endpoint" --model "$alias" \
      --concurrency 1024 --output-dir "$base/bfcl_v3" \
      > "$OUT/logs/${domain}_bfcl.log" 2>&1
}

run_code() {
    local domain=$1 port=$2 alias=$3 dataset=$4
    local base="$OUT/$domain" endpoint="http://127.0.0.1:$port/v1"
    mkdir -p "$base"
    if [[ "$dataset" == livecodebench ]]; then
        bash benchmark/run.sh --benchmark livecodebench --lcb-release v6 \
          --code-execution-backend nsjail --code-thinking off --code-pass-k 1 \
          --code-eval-workers 16 --code-memory 4g --code-eval-timeout 7200 \
          --llm-max-completion-tokens 16384 --openai-base-url "$endpoint" --model "$alias" \
          --concurrency 512 --output-dir "$base/livecodebench" \
          > "$OUT/logs/${domain}_livecodebench.log" 2>&1
    else
        bash benchmark/run.sh --benchmark "$dataset" \
          --code-execution-backend nsjail --code-thinking off --code-pass-k 1 \
          --code-eval-workers 16 --code-memory 4g --code-eval-timeout 7200 \
          --llm-max-completion-tokens 16384 --openai-base-url "$endpoint" --model "$alias" \
          --concurrency 512 --output-dir "$base/$dataset" \
          > "$OUT/logs/${domain}_${dataset}.log" 2>&1
    fi
}

STATE="$EXP/eval/active_prefix_eval_services.json"
[[ -f "$STATE" ]] || { echo "Missing service state $STATE" >&2; exit 1; }

python - "$OUT/run_manifest.json" <<'PY'
import datetime, json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "scope": {
        "humaneval+": 164,
        "mbpp+": 378,
        "livecodebench_v6": 1055,
        "tau2": {"airline": 50, "retail": 114, "telecom": 114},
        "bfcl_v3": 4441,
    },
    "models": ["awm", "envscaler"],
    "code_pass_k": 1,
    "code_eval_workers": 16,
    "nofile_limit": 65536,
    "concurrency": {"code_each": 512, "tau2_each_domain": 128, "bfcl": 1024},
    "benchmarks_run_concurrently": True,
    "tau2_trials": 1,
    "thinking": False,
}, indent=2) + "\n")
PY

pids=()
for spec in "awm 18140 prefix-awm" "envscaler 18142 prefix-envscaler"; do
    read -r domain port alias <<< "$spec"
    model_path="$EXP/prefixes/$domain"
    for task_domain in airline retail telecom; do
        run_tau2 "$domain" "$port" "$alias" "$task_domain" & pids+=("$!")
    done
    run_bfcl "$domain" "$port" "$alias" "$model_path" & pids+=("$!")
    for dataset in humaneval+ mbpp+ livecodebench; do
        run_code "$domain" "$port" "$alias" "$dataset" & pids+=("$!")
    done
done

fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
python - "$OUT/run_status.json" "$fail" <<'PY'
import datetime, json, sys
from pathlib import Path
path, failed = Path(sys.argv[1]), bool(int(sys.argv[2]))
path.write_text(json.dumps({
    "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "status": "failed" if failed else "complete",
}, indent=2) + "\n")
PY
echo "Evaluation artifacts: $OUT"
exit "$fail"
