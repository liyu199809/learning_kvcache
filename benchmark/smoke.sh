#!/usr/bin/env bash
# Full inference+scoring smoke. MODE=generation skips only Docker-dependent work.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
cd "$BENCHMARK_PROJECT_ROOT"
MODE=${MODE:-full}
ENDPOINT=${OPENAI_BASE_URL:-http://127.0.0.1:18081/v1}
MODEL=${MODEL:-qwen3.5-4b-benchmark}
MODEL_PATH=${MODEL_PATH:-/home/liyu.468/models/Qwen3.5-4B}
OUTPUT_ROOT=${OUTPUT_ROOT:-$BENCHMARK_PROJECT_ROOT/outputs/benchmark_smoke_$(date +%Y%m%d_%H%M%S)}
[[ "$MODE" == full || "$MODE" == generation ]] || { echo 'MODE must be full or generation' >&2; exit 2; }
if [[ "$MODE" == full ]]; then
    docker run --rm --network none ubuntu:22.04 /bin/true || {
        echo 'Docker cannot run an isolated container. Configure a usable Docker runtime before full scoring.' >&2
        exit 1
    }
    docker image inspect mysql:8.0 local-os/default >/dev/null
fi
[[ ! -e "$OUTPUT_ROOT" ]] || { echo "Refusing to overwrite $OUTPUT_ROOT" >&2; exit 1; }
mkdir -p "$OUTPUT_ROOT/logs"
COMMON=(--openai-base-url "$ENDPOINT" --model "$MODEL" --concurrency 1)
for domain in airline retail telecom; do
    bash benchmark/run.sh --benchmark tau2 --tau2-domain "$domain" --tau2-num-tasks 2 \
        --tau2-max-steps 50 --llm-max-completion-tokens 4096 \
        --tau2-judge-api-key-env BENCHMARK_LOCAL_JUDGE_ONLY "${COMMON[@]}" \
        --output-dir "$OUTPUT_ROOT/tau2_$domain" > "$OUTPUT_ROOT/logs/tau2_$domain.log" 2>&1
done
"$PROJECT_PYTHON" - "$OUTPUT_ROOT/bfcl_ids.json" <<'PY'
import json,sys
from pathlib import Path
ids={}
for p in sorted(Path('benchmark/bfcl_v3/official/bfcl_eval/data').glob('BFCL_v3*.json')):
    with p.open() as f: row=json.loads(next(f))
    ids[p.stem.removeprefix('BFCL_v3_')]=[row['id']]
Path(sys.argv[1]).write_text(json.dumps(ids,indent=2))
PY
bash benchmark/run.sh --benchmark bfcl_v3 --bfcl-model-path "$MODEL_PATH" \
    --bfcl-categories all --bfcl-run-ids-file "$OUTPUT_ROOT/bfcl_ids.json" "${COMMON[@]}" \
    --output-dir "$OUTPUT_ROOT/bfcl_v3" > "$OUTPUT_ROOT/logs/bfcl_v3.log" 2>&1
CODE_EXTRA=()
[[ "$MODE" == full ]] || CODE_EXTRA+=(--code-generate-only)
for dataset in humaneval+ mbpp+ livecodebench; do
    SELECT=(--max-tasks 2)
    [[ "$dataset" != livecodebench ]] || SELECT=(--task-ids 1873_A,abc387_b)
    bash benchmark/run.sh --benchmark "$dataset" "${SELECT[@]}" "${CODE_EXTRA[@]}" \
        --lcb-release v6 --code-thinking off --llm-max-completion-tokens 16384 \
        --code-eval-workers 2 "${COMMON[@]}" --output-dir "$OUTPUT_ROOT/$dataset" \
        > "$OUTPUT_ROOT/logs/$dataset.log" 2>&1
done
if [[ "$MODE" == full ]]; then
    benchmark/coding/.venv/bin/python benchmark/coding/smoke.py > "$OUTPUT_ROOT/logs/coding_controls.log" 2>&1
    for kind in db os; do
        STEPS=6; [[ "$kind" != os ]] || STEPS=8
        bash benchmark/run.sh --benchmark "lifelong_$kind" --data-dir benchmark/LifelongAgentBench \
            --tasks 0-1 --max-steps "$STEPS" --llm-max-completion-tokens 2048 "${COMMON[@]}" \
            --output-dir "$OUTPUT_ROOT/lifelong_$kind" > "$OUTPUT_ROOT/logs/lifelong_$kind.log" 2>&1
    done
fi
printf 'Completed mode=%s. Artifacts: %s\n' "$MODE" "$OUTPUT_ROOT"
