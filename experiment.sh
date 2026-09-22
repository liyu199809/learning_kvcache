#!/usr/bin/env bash
# Single public launcher for the three-domain prefix experiment.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXP=${EXPERIMENT_ROOT:-$ROOT/outputs/prefix_composition_3domain_20260917}
INTERNAL="$ROOT/prefix_tuning/virtual_prefix/internal"
PYTHON=${KVCACHE_PYTHON:-/usr/local/miniconda3/envs/kvcache/bin/python}

usage() {
    cat <<'EOF'
Usage:
  ./experiment.sh tools {start|stop|restart|status} [awm|envscaler|taco ...]
  ./experiment.sh train all [--awm-resume CHECKPOINT]
  ./experiment.sh train {awm|envscaler|taco|mixed} [Hydra overrides ...]
  ./experiment.sh train-smoke {awm|envscaler|taco|mixed} [Hydra overrides ...]
  ./experiment.sh serve {start|stop|restart|status}
  ./experiment.sh eval [all|awm|envscaler|taco|base] [OUTPUT_DIR]
  ./experiment.sh status

Environment overrides:
  EXPERIMENT_ROOT, KVCACHE_PYTHON, EVAL_NOFILE_LIMIT
  EVAL_TEMPERATURE, EVAL_TOP_P, EVAL_CODE_TOP_K, EVAL_TAU2_CONCURRENCY
  Full training defaults: USE_WANDB=1, WANDB_MODE=online,
  VAL_BEFORE_TRAIN=True, TEST_FREQ=20
EOF
}

activate_kvcache() {
    if [[ -f /usr/local/miniconda3/etc/profile.d/conda.sh ]]; then
        # shellcheck disable=SC1091
        source /usr/local/miniconda3/etc/profile.d/conda.sh
        conda activate kvcache
    fi
}

configure_full_training() {
    # Keep the public entrypoint authoritative for observability and validation.
    # Explicit caller overrides are still honored for controlled experiments.
    export USE_WANDB=${USE_WANDB:-1}
    export WANDB_MODE=${WANDB_MODE:-online}
    export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}
    export TEST_FREQ=${TEST_FREQ:-20}
}

run_tools() {
    local action=${1:-status}
    shift || true
    local services=("$@")
    [[ ${#services[@]} -gt 0 ]] || services=(awm envscaler taco)
    local service failed=0
    for service in "${services[@]}"; do
        echo "[$service] $action"
        case "$service" in
            awm) bash "$INTERNAL/awm_service.bash" "$action" tool || failed=1 ;;
            envscaler) bash "$INTERNAL/envscaler_service.bash" "$action" || failed=1 ;;
            taco) bash "$INTERNAL/taco_service.bash" "$action" || failed=1 ;;
            *) echo "Unknown tool service: $service" >&2; failed=1 ;;
        esac
    done
    return "$failed"
}

model_port() {
    case "$1" in
        awm) echo 18140 ;; envscaler) echo 18142 ;; taco) echo 18144 ;; base) echo 18146 ;;
    esac
}

model_alias() {
    case "$1" in
        awm) echo prefix-awm ;; envscaler) echo prefix-envscaler ;;
        taco) echo prefix-taco ;; base) echo base-qwen3.5-4b ;;
    esac
}

model_path() {
    case "$1" in
        awm|envscaler|taco) echo "$EXP/prefixes/$1" ;;
        base) echo /home/liyu.468/models/Qwen3.5-4B ;;
    esac
}

check_model_service() {
    local model=$1 port alias payload
    port=$(model_port "$model")
    alias=$(model_alias "$model")
    payload=$(curl --noproxy "*" -fsS "http://127.0.0.1:$port/v1/models")
    "$PYTHON" - "$alias" "$payload" <<'PY'
import json, sys
alias, payload = sys.argv[1:]
models = [item["id"] for item in json.loads(payload).get("data", [])]
if alias not in models:
    raise SystemExit(f"expected {alias!r}, got {models!r}")
PY
}

run_eval() {
    local selection=${1:-all}
    local stamp out eval_temperature eval_top_p eval_code_top_k eval_tau2_concurrency
    shift || true
    stamp=$(date +%Y%m%d_%H%M%S)
    out=${1:-$EXP/eval/formal_$stamp}
    eval_temperature=${EVAL_TEMPERATURE:-0.0}
    eval_top_p=${EVAL_TOP_P:-1.0}
    eval_code_top_k=${EVAL_CODE_TOP_K:-}
    eval_tau2_concurrency=${EVAL_TAU2_CONCURRENCY:-2}
    [[ ! -e "$out" ]] || { echo "Refusing to overwrite $out" >&2; return 1; }

    local models
    if [[ "$selection" == all ]]; then
        models=(awm envscaler taco base)
    else
        case "$selection" in awm|envscaler|taco|base) ;; *) usage; return 2 ;; esac
        models=("$selection")
    fi
    local model
    for model in "${models[@]}"; do check_model_service "$model"; done

    ulimit -n "${EVAL_NOFILE_LIMIT:-65536}"
    # shellcheck disable=SC1091
    source "$ROOT/benchmark/env.sh"
    mkdir -p "$out/logs"
    "$PYTHON" - "$out/run_manifest.json" "$selection" "${models[*]}" \
      "$eval_temperature" "$eval_top_p" "${eval_code_top_k:-null}" \
      "$eval_tau2_concurrency" <<'PY'
import datetime, json, sys
from pathlib import Path
path, selection, models, temperature, top_p, top_k, tau2_concurrency = sys.argv[1:]
Path(path).write_text(json.dumps({
    "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "selection": selection,
    "models": models.split(),
    "scope": {"humaneval+": 164, "mbpp+": 378, "livecodebench_v6": 1055,
              "bfcl_v3": 4441, "tau2": {"airline": 50, "retail": 114}},
    "code": {"thinking": True, "max_tokens": 16384, "api_timeout": 1800,
             "generation_concurrency": 32, "score_workers": 16,
             "nsjail_memory_per_process": "64g"},
    "bfcl_concurrency": 128,
    "tau2_concurrency_per_domain": int(tau2_concurrency),
    "tau2_scheduling": "sequential_models_and_domains",
    "sampling": {"temperature": float(temperature), "top_p": float(top_p),
                 "code_top_k": None if top_k == "null" else int(top_k)},
    "tau2_user_and_judge": "external_openai_compatible",
    "phases": ["code_by_dataset", "bfcl", "tau2"],
}, indent=2) + "\n")
PY

    run_code() {
        local m=$1 dataset=$2 port alias endpoint base
        port=$(model_port "$m"); alias=$(model_alias "$m")
        endpoint="http://127.0.0.1:$port/v1"; base="$out/$m"
        mkdir -p "$base"
        local extra=()
        [[ "$dataset" == livecodebench ]] && extra=(--lcb-release v6)
        local sampling=(--temperature "$eval_temperature" --top-p "$eval_top_p")
        [[ -z "$eval_code_top_k" ]] || sampling+=(--code-top-k "$eval_code_top_k")
        bash "$ROOT/benchmark/run.sh" --benchmark "$dataset" "${extra[@]}" \
          --code-execution-backend nsjail --code-thinking on --code-pass-k 1 \
          --code-eval-workers 16 --code-memory 64g --code-eval-timeout 7200 \
          --code-api-timeout 1800 --llm-max-completion-tokens 16384 \
          --openai-base-url "$endpoint" --model "$alias" --concurrency 32 \
          "${sampling[@]}" \
          --output-dir "$base/$dataset" >"$out/logs/${m}_${dataset}.log" 2>&1
    }

    run_bfcl() {
        local m=$1 port alias endpoint
        port=$(model_port "$m"); alias=$(model_alias "$m")
        endpoint="http://127.0.0.1:$port/v1"
        bash "$ROOT/benchmark/run.sh" --benchmark bfcl_v3 \
          --bfcl-model-path "$(model_path "$m")" --bfcl-categories all \
          --openai-base-url "$endpoint" --model "$alias" --concurrency 128 \
          --temperature "$eval_temperature" \
          --output-dir "$out/$m/bfcl_v3" >"$out/logs/${m}_bfcl.log" 2>&1
    }

    run_tau2() {
        local m=$1 domain=$2 port alias endpoint
        port=$(model_port "$m"); alias=$(model_alias "$m")
        endpoint="http://127.0.0.1:$port/v1"
        bash "$ROOT/benchmark/run.sh" --benchmark tau2 --tau2-domain "$domain" \
          --tau2-max-steps 200 --llm-max-completion-tokens 16384 \
          --tau2-require-external-user --tau2-require-external-judge \
          --openai-base-url "$endpoint" --model "$alias" \
          --concurrency "$eval_tau2_concurrency" \
          --temperature "$eval_temperature" --top-p "$eval_top_p" \
          --output-dir "$out/$m/tau2_$domain" >"$out/logs/${m}_tau2_${domain}.log" 2>&1
    }

    local fail=0 dataset pid domain
    local pids=()
    for dataset in humaneval+ mbpp+ livecodebench; do
        pids=()
        for model in "${models[@]}"; do run_code "$model" "$dataset" & pids+=("$!"); done
        for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
    done
    pids=()
    for model in "${models[@]}"; do run_bfcl "$model" & pids+=("$!"); done
    for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
    for model in "${models[@]}"; do
        for domain in airline retail; do
            run_tau2 "$model" "$domain" || fail=1
        done
    done
    "$PYTHON" - "$out/run_status.json" "$fail" <<'PY'
import datetime, json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "status": "failed" if int(sys.argv[2]) else "complete",
}, indent=2) + "\n")
PY
    echo "Evaluation artifacts: $out"
    return "$fail"
}

show_status() {
    echo "== evaluation services =="
    "$PYTHON" "$INTERNAL/eval_services.py" status || true
    echo "== training =="
    if [[ -f "$EXP/recovery_status.json" ]]; then
        echo "active recovery:"
        cat "$EXP/recovery_status.json"
        if [[ -f "$EXP/queue_status.json" ]]; then
            echo "original queue:"
            cat "$EXP/queue_status.json"
        fi
    elif [[ -f "$EXP/queue_status.json" ]]; then
        cat "$EXP/queue_status.json"
    else
        echo "no queue status"
    fi
    echo "== latest evaluation =="
    if [[ -f "$EXP/eval/active_four_model_rerun.json" ]]; then
        "$PYTHON" - "$EXP/eval/active_four_model_rerun.json" <<'PY'
import json, os, sys
data = json.load(open(sys.argv[1]))
print(data["run_root"])
for name, job in data.get("jobs", {}).items():
    state = "running" if os.path.exists(f"/proc/{job['pid']}") else "finished"
    print(f"  {name}: {state} (pid={job['pid']})")
PY
        return
    fi
    local latest
    latest=$(find "$EXP/eval" -mindepth 2 -maxdepth 2 -name run_manifest.json -printf '%T@ %h\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2- || true)
    if [[ -n "$latest" ]]; then
        echo "$latest"
        [[ -f "$latest/run_status.json" ]] && cat "$latest/run_status.json" || echo "running"
    else
        echo "none"
    fi
}

command=${1:-help}
shift || true
case "$command" in
    tools)
        activate_kvcache
        run_tools "$@"
        ;;
    train)
        activate_kvcache
        configure_full_training
        target=${1:-all}; shift || true
        if [[ "$target" == all ]]; then
            exec "$PYTHON" "$INTERNAL/training_queue.py" "$@"
        fi
        exec env RUN_MODE=full bash "$INTERNAL/train_domain.bash" "$target" "$@"
        ;;
    train-smoke)
        activate_kvcache
        target=${1:?train-smoke requires a domain}; shift
        exec env RUN_MODE=smoke bash "$INTERNAL/train_domain.bash" "$target" "$@"
        ;;
    _train-domain)
        activate_kvcache
        exec bash "$INTERNAL/train_domain.bash" "$@"
        ;;
    serve)
        activate_kvcache
        exec "$PYTHON" "$INTERNAL/eval_services.py" "${1:-status}"
        ;;
    eval)
        activate_kvcache
        run_eval "$@"
        ;;
    status)
        show_status
        ;;
    help|-h|--help)
        usage
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
