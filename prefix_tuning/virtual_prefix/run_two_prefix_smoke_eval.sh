#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
EXP="$ROOT/outputs/prefix_composition_3domain_20260917"
source "$ROOT/benchmark/env.sh"
cd "$ROOT"
STAMP=$(date +%Y%m%d_%H%M%S)
OUT=${1:-$EXP/eval/two_prefix_smoke_$STAMP}
[[ ! -e "$OUT" ]] || { echo "Refusing to overwrite $OUT" >&2; exit 1; }
mkdir -p "$OUT/logs"

python - "$OUT/bfcl_ids.json" <<'PY'
import json, sys
from pathlib import Path
ids = {}
for path in sorted(Path("benchmark/bfcl_v3/official/bfcl_eval/data").glob("BFCL_v3*.json")):
    with path.open() as handle:
        ids[path.stem.removeprefix("BFCL_v3_")] = [json.loads(next(handle))["id"]]
Path(sys.argv[1]).write_text(json.dumps(ids, indent=2) + "\n")
PY

run_interactive() {
    local domain=$1 port=$2 alias=$3 model_path=$4
    local base="$OUT/$domain" endpoint="http://127.0.0.1:$port/v1"
    mkdir -p "$base"
    for task_domain in airline retail telecom; do
        bash benchmark/run.sh --benchmark tau2 --tau2-domain "$task_domain" --tau2-num-tasks 2 \
          --tau2-max-steps 50 --llm-max-completion-tokens 4096 \
          --tau2-judge-api-key-env BENCHMARK_LOCAL_JUDGE_ONLY \
          --openai-base-url "$endpoint" --model "$alias" --concurrency 1 \
          --output-dir "$base/tau2_$task_domain" > "$OUT/logs/${domain}_tau2_${task_domain}.log" 2>&1
    done
    bash benchmark/run.sh --benchmark bfcl_v3 --bfcl-model-path "$model_path" \
      --bfcl-categories all --bfcl-run-ids-file "$OUT/bfcl_ids.json" \
      --openai-base-url "$endpoint" --model "$alias" --concurrency 1 \
      --output-dir "$base/bfcl_v3" > "$OUT/logs/${domain}_bfcl.log" 2>&1
}

run_code() {
    local domain=$1 port=$2 alias=$3
    local base="$OUT/$domain" endpoint="http://127.0.0.1:$port/v1"
    mkdir -p "$base"
    for dataset in humaneval+ mbpp+; do
        bash benchmark/run.sh --benchmark "$dataset" --max-tasks 2 \
          --code-execution-backend nsjail --code-thinking off --code-pass-k 1 \
          --code-eval-workers 2 --code-memory 4g --code-eval-timeout 600 \
          --llm-max-completion-tokens 16384 --openai-base-url "$endpoint" --model "$alias" \
          --concurrency 1 --output-dir "$base/$dataset" > "$OUT/logs/${domain}_${dataset}.log" 2>&1
    done
    bash benchmark/run.sh --benchmark livecodebench --task-ids 1873_A,1873_B,abc387_b \
      --lcb-release v6 --code-execution-backend nsjail --code-thinking off --code-pass-k 1 \
      --code-eval-workers 2 --code-memory 4g --code-eval-timeout 600 \
      --llm-max-completion-tokens 16384 --openai-base-url "$endpoint" --model "$alias" \
      --concurrency 1 --output-dir "$base/livecodebench" > "$OUT/logs/${domain}_livecodebench.log" 2>&1
}

STATE="$EXP/eval/active_prefix_eval_services.json"
[[ -f "$STATE" ]] || { echo "Missing service state $STATE" >&2; exit 1; }
run_interactive awm 18140 prefix-awm "$EXP/prefixes/awm" & p1=$!
run_code awm 18141 prefix-awm & p2=$!
run_interactive envscaler 18142 prefix-envscaler "$EXP/prefixes/envscaler" & p3=$!
run_code envscaler 18143 prefix-envscaler & p4=$!
fail=0
for pid in $p1 $p2 $p3 $p4; do wait "$pid" || fail=1; done
python - "$OUT/run_status.json" "$fail" <<'PY'
import datetime, json, sys
from pathlib import Path
path, failed = Path(sys.argv[1]), bool(int(sys.argv[2]))
path.write_text(json.dumps({"completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                            "status": "failed" if failed else "complete"}, indent=2) + "\n")
PY
echo "Evaluation artifacts: $OUT"
exit "$fail"
