#!/usr/bin/env bash
# 2026-09-22/23 三域 Hybrid Prefix 实验：训练、导出、组合、四卡串行评测。
# 仅编排已有仓库；不下载/修改代码、模型、数据，不安装依赖。
# 历史代码：2bec7de54c6afa3fd6185293e9331ed63ea2c774（tau2 info.git_commit）。
# 训练：旧 V0 同步路径，不能用当前 partial rollout + 新 distillation loss 替代。
# 基座/teacher: Qwen3.5-4B；GDN prefix=512，attention prefix=64。
# 3 actor/rollout + 1 teacher，FSDP2，batch/mini-batch=24，micro-batch/GPU=1。
# 1 epoch；AWM/EnvScaler 67 步，TACO 66 步；lr=1e-5；forward-KL top64。
# 原始数据目录见 HISTORICAL_EXP；需同时具备 awm_data、envscaler_data、
# data/deepcoder_taco/177913a7bd43791646ef6a43645caa3c871ab3db 工具环境数据。
# 已准备的 hybrid-G512-A64 初始化权重应直接复用，避免重新随机初始化。
# 历史评测：vLLM 0.22.1；Python 3.12；TP=1/DP=4；代码评测 Python 3.11。
# requirements-freeze.txt 中 vLLM=0.18.0 与实跑日志不一致，不能当作精确环境锁。
# 评测记录：eval/serial_dp4_20260923_161445（三域）；..._234527（组合）。
# HumanEval/MBPP 展示 metrics.base['pass@1']，不是 Plus 分数；LCB v6 1055题。
# 生成失败保留在分母里。tau2 seed=300（历史官方默认），每题1 trial，最多200步。
# 图中组合 Airline=39/50 (78%)，现存 results.json=34/50 (68%)，尚未找到78%的来源。
# 其余23个数与截图一致；随机采样、外部模拟器可能使重跑分数变化。
# 截图顺序: HumanEval / MBPP / LCB / BFCL / Airline / Retail
# AWM:       89.63 / 83.86 / 37.44 / 67.70 / 76.00 / 81.58
# EnvScaler: 90.24 / 84.39 / 37.35 / 68.12 / 82.00 / 81.58
# TACO:      88.41 / 83.07 / 46.45 / 67.73 / 76.00 / 83.33
# 组合:      91.46 / 84.66 / 45.69 / 67.92 / 68.00(现存记录) / 85.96
#
# 合作者：先准备匹配历史版本的代码/环境/权重/数据，再按 --help 设置路径。
# 快速复评使用已有 prefixes 和组合模型；完整重训输出到新的 EXPERIMENT_ROOT。
set -euo pipefail

usage() {
cat <<'EOF'
用法：bash reproduce_20260923.sh {check|train|compose|eval|all} [模型]
  check           检查依赖、历史代码和资产路径，不启动训练/评测
  train [all|awm|envscaler|taco]   三域依次训练并导出；all 后自动组合
  compose         按 AWM → EnvScaler → TACO 顺序拼接三个已导出 prefix
  eval [all|awm|envscaler|taco|composed]  每模型完整跑六项基准，模型间串行
  all             训练三域、导出、组合，然后评测四个模型

# 1. 修改为合作者自己的路径（此脚本应放在仓库根目录或设置 REPO_ROOT）
export REPO_ROOT=/home/liyu.468/learning_kvcache
export KVCACHE_PYTHON=/usr/local/miniconda3/envs/kvcache/bin/python
export HISTORICAL_EXP=$REPO_ROOT/outputs/prefix_composition_3domain_20260922_4gpu
export INITIAL_MODEL=$HISTORICAL_EXP/prepared/hybrid-G512-A64
export TRAIN_DATA=$HISTORICAL_EXP/data
export TEACHER_MODEL=/home/liyu.468/models/Qwen3.5-4B

# 2a. 最快：直接使用旧权重复评（结果写入新的时间戳目录）
EXPERIMENT_ROOT=$HISTORICAL_EXP bash reproduce_20260923.sh eval all

# 2b. 完整重训：必须使用新目录，不覆盖旧实验
EXPERIMENT_ROOT=$REPO_ROOT/outputs/reproduce_20260923_new bash reproduce_20260923.sh all

# 只看将执行的命令，无副作用，不要求 GPU/依赖/权重存在
DRY_RUN=1 bash reproduce_20260923.sh all

可选：GPUS=0,1,2,3；USE_WANDB=0（默认只改变日志方式）；
      EVAL_OUTPUT=/新目录；PORT=18140；DP_RPC_PORT=18150；
      EVAL_LIBSTDCXX=/usr/lib/x86_64-linux-gnu/libstdc++.so.6。
外部 tau2 user/judge：deepseek-v4-pro-ga-260813，需在仓库 .env 或环境变量
提供 ARK_API_KEY；脚本不含密钥。四张 GPU 需供本实验独占使用。
历史源码应已准备好；脚本不会自动 git checkout 或回退修复。
EOF
}
ACTION=${1:---help}
SELECTION=${2:-all}
case "$ACTION" in -h|--help|help) usage; exit 0 ;; check|train|compose|eval|all) ;; *) usage; exit 2 ;; esac
REPO_ROOT=${REPO_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}
HISTORICAL_EXP=${HISTORICAL_EXP:-$REPO_ROOT/outputs/prefix_composition_3domain_20260922_4gpu}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-$REPO_ROOT/outputs/reproduce_20260923_$(date +%Y%m%d_%H%M%S)}
INITIAL_MODEL=${INITIAL_MODEL:-$HISTORICAL_EXP/prepared/hybrid-G512-A64}
TRAIN_DATA=${TRAIN_DATA:-$HISTORICAL_EXP/data}
TEACHER_MODEL=${TEACHER_MODEL:-$(dirname "$REPO_ROOT")/models/Qwen3.5-4B}
KVCACHE_PYTHON=${KVCACHE_PYTHON:-/usr/local/miniconda3/envs/kvcache/bin/python}
GPUS=${GPUS:-0,1,2,3}
PORT=${PORT:-18140}
DP_RPC_PORT=${DP_RPC_PORT:-18150}
EVAL_OUTPUT=${EVAL_OUTPUT:-$EXPERIMENT_ROOT/eval/reproduce_$(date +%Y%m%d_%H%M%S)}
EVAL_LIBSTDCXX=${EVAL_LIBSTDCXX:-/usr/lib/x86_64-linux-gnu/libstdc++.so.6}
DRY_RUN=${DRY_RUN:-0}
BASE_COMMIT=2bec7de54c6afa3fd6185293e9331ed63ea2c774
export REPO_ROOT EXPERIMENT_ROOT KVCACHE_PYTHON
export PYTHON_BIN="$KVCACHE_PYTHON" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export NO_PROXY="localhost,127.0.0.1,${NO_PROXY:-}" no_proxy="localhost,127.0.0.1,${NO_PROXY:-}"
export PYTHONPATH="$REPO_ROOT/verl:$REPO_ROOT:$REPO_ROOT/OpenEnv/src:$REPO_ROOT/OpenEnv/envs${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"
fail() { echo "ERROR: $*" >&2; exit 1; }
run() { printf '+ '; printf '%q ' "$@"; printf '\n'; [[ "$DRY_RUN" == 1 ]] || "$@"; }
need() { [[ -e "$1" ]] || fail "缺少 $1"; }
source_check() {
  [[ "$DRY_RUN" == 1 ]] && return 0
  git cat-file -e "$BASE_COMMIT^{commit}" 2>/dev/null || fail "仓库需包含历史 commit $BASE_COMMIT"
  local paths=(benchmark prefix_tuning/virtual_prefix/vllm_plugin.py)
  if [[ "$1" == train ]]; then paths+=(verl prefix_tuning rollout experiment.sh); fi
  git diff --quiet "$BASE_COMMIT" -- "${paths[@]}" || fail "相关代码与历史版本不同。请使用独立的 $BASE_COMMIT checkout；仅设置 TRAINER_MODE=sync 不会还原历史 loss。"
}
common_check() {
  [[ "$DRY_RUN" == 1 ]] && return 0
  [[ "$GPUS" =~ ^[0-9]+,[0-9]+,[0-9]+,[0-9]+$ ]] || fail 'GPUS 需指定4张卡'
  [[ -x "$KVCACHE_PYTHON" ]] || fail "Python 不存在：$KVCACHE_PYTHON"
  command -v nvidia-smi >/dev/null || fail '需要 NVIDIA CUDA 环境'
  "$KVCACHE_PYTHON" - <<'PY'
import importlib.metadata as m
assert m.version('vllm') == '0.22.1', '历史实跑为 vLLM 0.22.1'
assert any(p.value == 'prefix_tuning.virtual_prefix.vllm_plugin:register'
           for p in m.entry_points(group='vllm.general_plugins')), '缺少 hybrid prefix vLLM plugin'
print('vLLM 0.22.1 / prefix plugin OK')
PY
}
train_check() {
  source_check train; common_check
  [[ "$DRY_RUN" == 1 ]] && return 0
  need "$INITIAL_MODEL/config.json"; need "$TEACHER_MODEL/config.json"; need "$TRAIN_DATA/val.parquet"
  local domain
  for domain in awm envscaler taco; do need "$TRAIN_DATA/sources/$domain/train.parquet"; done
  [[ ! -e "$EXPERIMENT_ROOT/checkpoints" ]] || fail "重训目录已有 checkpoints：$EXPERIMENT_ROOT"
}
compose() {
  run env CUDA_VISIBLE_DEVICES= "$KVCACHE_PYTHON" -m prefix_tuning.virtual_prefix.compose_hybrid_prefixes \
    --input "$EXPERIMENT_ROOT/prefixes/awm" --input "$EXPERIMENT_ROOT/prefixes/envscaler" \
    --input "$EXPERIMENT_ROOT/prefixes/taco" --output "$EXPERIMENT_ROOT/models/awm_envscaler_taco"
}
train() {
  train_check
  local domains=(awm envscaler taco) domain steps thinking assistant_turns user_turns tool_length
  if [[ "$SELECTION" != all ]]; then
    case "$SELECTION" in awm|envscaler|taco) domains=("$SELECTION") ;; *) fail '未知训练域' ;; esac
  fi
  # 验证集包含三个域，因此三个工具服务都需启动；沿用仓库的CPU工具服务管理。
  run bash "$REPO_ROOT/experiment.sh" tools start awm envscaler taco
  for domain in "${domains[@]}"; do
    steps=67; thinking=False; assistant_turns=20; user_turns=19; tool_length=800
    if [[ "$domain" == envscaler ]]; then assistant_turns=17; user_turns=16; tool_length=4000; fi
    if [[ "$domain" == taco ]]; then steps=66; thinking=True; assistant_turns=1; user_turns=1; fi
    run env CUDA_VISIBLE_DEVICES="$GPUS" ASCEND_RT_VISIBLE_DEVICES="$GPUS" \
      ACTOR_GPUS=3 TEACHER_GPUS=1 TRAINER_MODE=sync RUN_MODE=full \
      ENABLE_THINKING="$thinking" MAX_ASSISTANT_TURNS="$assistant_turns" \
      MAX_USER_TURNS="$user_turns" MAX_TOOL_RESPONSE_LENGTH="$tool_length" \
      MODEL="$INITIAL_MODEL" TEACHER_MODEL="$TEACHER_MODEL" DATA_ROOT="$TRAIN_DATA" \
      RUN_NAME="${domain}_4gpu" CHECKPOINT_DIR="$EXPERIMENT_ROOT/checkpoints/${domain}_4gpu" \
      TRAIN_BATCH_SIZE=24 PPO_MINI_BATCH_SIZE=24 DISTILLATION_TOPK=64 DISTILLATION_TOPK_CHUNK_SIZE=512 \
      TOTAL_EPOCHS=1 TOTAL_TRAINING_STEPS= RESUME_MODE=disable \
      MAX_PROMPT_LENGTH=28672 MAX_RESPONSE_LENGTH=16384 MAX_MODEL_LEN=49152 \
      PPO_MAX_TOKEN_LEN_PER_GPU=49152 MAX_NUM_BATCHED_TOKENS=49152 \
      ACTOR_LR=1e-5 ROLLOUT_TEMPERATURE=1.0 TEACHER_TEMPERATURE=1.0 \
      ROLLOUT_ENFORCE_EAGER=True TEACHER_ENFORCE_EAGER=True \
      GPU_MEMORY_UTILIZATION=0.65 NUM_WORKERS=24 FILTER_WORKERS=4 DATALOADER_WORKERS=4 \
      USE_WANDB="${USE_WANDB:-0}" VAL_BEFORE_TRAIN=True TEST_FREQ=20 SAVE_FREQ=20 \
      RAY_ADDRESS=local RAY_TMPDIR="/tmp/kvcache-reproduce-${domain}-$$" \
      bash "$REPO_ROOT/experiment.sh" train "$domain" \
      trainer.use_v1=False +ray_kwargs.ray_init.address=local ray_kwargs.ray_init.num_cpus=28 \
      +ray_kwargs.ray_init.num_gpus=4 +ray_kwargs.ray_init.include_dashboard=False \
      +ray_kwargs.ray_init.object_store_memory=8589934592
    run env CUDA_VISIBLE_DEVICES= "$KVCACHE_PYTHON" \
      "$REPO_ROOT/prefix_tuning/virtual_prefix/export_hybrid_delta_residual_attention_prefix_checkpoint.py" \
      --checkpoint "$EXPERIMENT_ROOT/checkpoints/${domain}_4gpu/global_step_$steps/actor" \
      --prepared-model "$INITIAL_MODEL" --output "$EXPERIMENT_ROOT/prefixes/$domain"
  done
  [[ "$SELECTION" != all ]] || compose
}
SERVER_PID=
stop_server() {
  [[ -n "$SERVER_PID" ]] || return 0
  # 只回收此脚本创建的独立服务进程组，不使用 pkill/ray stop。
  kill -TERM -- "-$SERVER_PID" 2>/dev/null || true
  local i
  for ((i=0; i<60; i++)); do kill -0 -- "-$SERVER_PID" 2>/dev/null || break; sleep 1; done
  if kill -0 -- "-$SERVER_PID" 2>/dev/null; then kill -KILL -- "-$SERVER_PID" 2>/dev/null || true; fi
  wait "$SERVER_PID" 2>/dev/null || true
  SERVER_PID=
}
trap stop_server EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
wait_ready() {
  "$KVCACHE_PYTHON" - "$PORT" "$1" "$SERVER_PID" <<'PY'
import json,os,sys,time,urllib.request
port,alias,pid=sys.argv[1:]; opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
for _ in range(300):
    os.kill(int(pid),0)
    try:
        with opener.open(f'http://127.0.0.1:{port}/v1/models',timeout=5) as r: data=json.load(r)
        if alias in [x['id'] for x in data['data']]: break
    except Exception: pass
    time.sleep(3)
else: raise SystemExit('vLLM startup timed out; see service log')
PY
}
bench() {
  local suite=$1; shift
  local cmd=(env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 bash "$REPO_ROOT/benchmark/run.sh" --benchmark "$suite" \
    --openai-base-url "http://127.0.0.1:$PORT/v1" --model "$alias" --temperature "$temp" \
    --top-p 1.0 --output-dir "$EVAL_OUTPUT/$name/$out_suite" "$@")
  if [[ "$DRY_RUN" == 1 ]]; then run "${cmd[@]}"; else run "${cmd[@]}" 2>&1 | tee "$EVAL_OUTPUT/logs/${name}_${out_suite}.log"; fi
}
evaluate() {
  source_check eval; common_check
  local names=(awm envscaler taco awm_envscaler_taco) name alias temp model suite out_suite domain
  case "$SELECTION" in all) ;; composed) names=(awm_envscaler_taco) ;; awm|envscaler|taco) names=("$SELECTION") ;; *) fail '未知评测模型' ;; esac
  if [[ "$DRY_RUN" != 1 ]]; then
    command -v setsid >/dev/null || fail '需要 Linux setsid'
    for suite in .venv/bin/python coding/.venv/bin/python tau2/.venv/bin/python runtime/bin/nsjail; do need "$REPO_ROOT/benchmark/$suite"; done
    [[ ! -e "$EVAL_OUTPUT" ]] || fail "评测输出已存在：$EVAL_OUTPUT"
    if [[ -f "$REPO_ROOT/.env" ]]; then set -a; source "$REPO_ROOT/.env"; set +a; fi
    [[ -n "${ARK_API_KEY:-}" ]] || fail 'tau2 需要 ARK_API_KEY'
    for name in "${names[@]}"; do
      model="$EXPERIMENT_ROOT/prefixes/$name"
      [[ "$name" != awm_envscaler_taco ]] || model="$EXPERIMENT_ROOT/models/$name"
      need "$model/config.json"
    done
    mkdir -p "$EVAL_OUTPUT/logs"
    ulimit -n 65536
  fi
  for name in "${names[@]}"; do
    temp=0.0; alias="prefix-$name"; model="$EXPERIMENT_ROOT/prefixes/$name"
    [[ "$name" != taco ]] || temp=0.6
    if [[ "$name" == awm_envscaler_taco ]]; then
      temp=0.6; alias=prefix-awm-envscaler-taco; model="$EXPERIMENT_ROOT/models/$name"
    fi
    local server=(env CUDA_VISIBLE_DEVICES="$GPUS" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
      OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 \
      HF_HUB_OFFLINE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn LD_PRELOAD="$EVAL_LIBSTDCXX" \
      "$KVCACHE_PYTHON" -m vllm.entrypoints.openai.api_server --model "$model" \
      --served-model-name "$alias" --host 127.0.0.1 --port "$PORT" --trust-remote-code \
      --model-impl vllm --dtype bfloat16 --tensor-parallel-size 1 --data-parallel-size 4 \
      --data-parallel-size-local 4 --data-parallel-backend mp --data-parallel-rpc-port "$DP_RPC_PORT" \
      --gpu-memory-utilization 0.80 --max-model-len 65536 --max-num-seqs 512 \
      --max-num-batched-tokens 131072 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
      --reasoning-parser qwen3 --language-model-only --generation-config vllm \
      --default-chat-template-kwargs '{"enable_thinking":false}')
    if [[ "$DRY_RUN" == 1 ]]; then run "${server[@]}"; else
      setsid "${server[@]}" >"$EVAL_OUTPUT/logs/${name}_service.log" 2>&1 & SERVER_PID=$!
      wait_ready "$alias"
    fi
    for suite in humaneval+ mbpp+ livecodebench; do
      out_suite=$suite; local extra=()
      [[ "$suite" != livecodebench ]] || extra=(--lcb-release v6)
      bench "$suite" ${extra[@]+"${extra[@]}"} --code-execution-backend nsjail --code-thinking on \
        --code-pass-k 1 --code-n-samples 1 --code-seed 42 --code-eval-workers 8 \
        --code-memory 64g --code-eval-timeout 7200 --code-api-timeout 1800 \
        --llm-max-completion-tokens 16384 --concurrency 16
    done
    out_suite=bfcl_v3
    bench bfcl_v3 --bfcl-model-path "$model" --bfcl-categories all --concurrency 24
    for domain in airline retail; do
      out_suite=tau2_$domain
      bench tau2 --tau2-domain "$domain" --tau2-split base --tau2-num-trials 1 \
        --tau2-max-steps 200 --tau2-require-external-user --tau2-require-external-judge \
        --tau2-user-model openai/deepseek-v4-pro-ga-260813 \
        --tau2-user-base-url https://ark.cn-beijing.volces.com/api/v3 --tau2-user-api-key-env ARK_API_KEY \
        --tau2-judge-model openai/deepseek-v4-pro-ga-260813 \
        --tau2-judge-base-url https://ark.cn-beijing.volces.com/api/v3 --tau2-judge-api-key-env ARK_API_KEY \
        --tau2-user-temperature 0.0 --tau2-user-top-p 1.0 \
        --tau2-judge-temperature 0.0 --tau2-judge-top-p 1.0 \
        --llm-max-completion-tokens 16384 --concurrency 2
    done
    stop_server
  done
  echo "评测输出：$EVAL_OUTPUT"
}
case "$ACTION" in
  check) train_check; echo '训练检查通过；评测另需 benchmark 三个独立环境、NsJail、ARK_API_KEY 和已导出 prefix。' ;;
  train) train ;;
  compose) compose ;;
  eval) evaluate ;;
  all) SELECTION=all; train; evaluate ;;
esac
