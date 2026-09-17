#!/usr/bin/env bash
# Score the seven saved B200 generations; this starts no model/GPU service.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
cd "$BENCHMARK_PROJECT_ROOT"
SOURCE_ROOT=${SOURCE_ROOT:-$BENCHMARK_PROJECT_ROOT/outputs/benchmark_bringup_20260917/smoke}
OUTPUT_ROOT=${OUTPUT_ROOT:-$BENCHMARK_PROJECT_ROOT/outputs/coding_nsjail_$(date +%Y%m%d_%H%M%S)}
[[ ! -e "$OUTPUT_ROOT" ]] || { echo "Refusing to overwrite $OUTPUT_ROOT" >&2; exit 2; }
mkdir -p "$OUTPUT_ROOT/logs"
for name in humaneval+ mbpp+ livecodebench livecodebench_v6_new; do
    case "$name" in
        humaneval+) dataset=humaneval+; ids=HumanEval/0,HumanEval/1 ;;
        mbpp+) dataset=mbpp+; ids=Mbpp/2,Mbpp/3 ;;
        livecodebench) dataset=livecodebench; ids=1873_A,1873_B ;;
        livecodebench_v6_new) dataset=livecodebench; ids=abc387_b ;;
    esac
    bash benchmark/run.sh --benchmark "$dataset" --lcb-release v6 --task-ids "$ids" \
        --code-samples "$SOURCE_ROOT/${name}_generate/samples.jsonl" \
        --code-execution-backend nsjail --code-eval-workers 2 --code-memory 4g \
        --code-eval-timeout 300 --code-timeout 6 --code-n-samples 1 --code-pass-k 1 \
        --model qwen3.5-4b-benchmark --openai-base-url http://127.0.0.1:18081/v1 \
        --code-thinking off --llm-max-completion-tokens 16384 \
        --output-dir "$OUTPUT_ROOT/$name" > "$OUTPUT_ROOT/logs/$name.log" 2>&1
done
printf 'Seven saved generations scored. Results: %s\n' "$OUTPUT_ROOT"
