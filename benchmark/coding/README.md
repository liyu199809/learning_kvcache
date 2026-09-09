# Coding benchmarks

接入统一入口 `python -m benchmark.eval.run_eval`，通过 OpenAI-compatible API
评测已有模型服务。代码、依赖环境、LCB 缓存均位于 `benchmark/coding/`。

| `--benchmark` | 数据与判分 |
|---|---|
| `livecodebench` | 官方 `code_generation_lite`，公开 + 私有测试，官方 LCB codegen evaluator |
| `humaneval+` | EvalPlus HumanEval+ v0.1.10，164 题 |
| `mbpp+` | EvalPlus MBPP+ v0.2.0，378 道经过筛选的题目 |

HumanEval+/MBPP+ 一次生成同时报告 `metrics.base` 和 `metrics.plus`。
plus 通过要求 base 和增强测试都通过。这复用了 EvalPlus 0.3.1 官方数据反序列化、
special oracle、超时与正确性检查；不使用训练侧的 dense reward。

## 安装

在仓库根目录执行（A800 已完成安装）：

```bash
bash benchmark/coding/setup.sh
```

使用独立 Python 3.11 环境及 `requirements.lock`，不修改训练环境。
LCB evaluator 固定到 commit `28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24`。
判题依赖运行中的 Docker 和本地 `ubuntu:latest` 镜像；也可通过
`--code-docker-image IMAGE` 选择兼容 glibc 的 Linux 镜像。
判题容器禁网、以非 root 用户运行、只读挂载运行时，限制内存/CPU/进程数，
不挂载仓库、模型或 API 凭据。正式候选代码不会在宿主进程执行。

## 运行

```bash
# 同时输出 HumanEval 和 HumanEval+ 分数
.venv/bin/python -m benchmark.eval.run_eval \
  --benchmark 'humaneval+' --model qwen3.5-4b \
  --openai-base-url http://127.0.0.1:8000/v1 --concurrency 8

# 同时输出 MBPP 和 MBPP+ 分数
.venv/bin/python -m benchmark.eval.run_eval \
  --benchmark 'mbpp+' --model qwen3.5-4b \
  --openai-base-url http://127.0.0.1:8000/v1 --concurrency 8

# LCB v6 官方累计 release：1055 题
.venv/bin/python -m benchmark.eval.run_eval \
  --benchmark livecodebench --lcb-release v6 --model qwen3.5-4b \
  --openai-base-url http://127.0.0.1:8000/v1 --concurrency 8

# LCB v5 官方累计 release：880 题，可继续筛选时间范围
HF_ENDPOINT=https://hf-mirror.com .venv/bin/python -m benchmark.eval.run_eval \
  --benchmark livecodebench --lcb-release v5 \
  --lcb-start-date 2024-08-01 --lcb-end-date 2025-04-30 \
  --model qwen3.5-4b --concurrency 8
```

`--lcb-release` 只接受 `v5` 或 `v6`，分别对应官方累计
`release_v5` 和 `release_v6`；默认 `v6`。v6 原始文件约 4.49 GB。数据固定在 HF revision
`0fe84c3912ea0c4d4a78037083943e8f0c4dd505`，每个文件校验官方 LFS SHA-256。
`--lcb-data-dir DIR` 可指定已下载的 `test*.jsonl`，同样校验版本。
时间范围两端均包含；任务索引是 release/date 筛选后的源顺序。

通用选择参数：`--tasks 0-9`、`--max-tasks 10`、`--task-ids HumanEval/0,HumanEval/1`。
LCB task_id 使用官方 question_id（例如 `abc387_b`）。子集分数明确标记
`subset: true`，不将它当作全数据集分数。

默认每题生成 1 个样本，temperature=0、top_p=1，最大 16384 completion tokens。
`--code-thinking off` 或 `--code-thinking on` 可显式传入 Qwen chat-template 的
`enable_thinking`，未指定时保持服务端默认行为。评测 manifest 记录该设置；
逐题样本额外保存 API 返回的 usage 和 reasoning 字段，以便统计输出长度与截断。
可通过 `--code-top-k`、`--code-min-p`、`--code-presence-penalty`、
`--code-repetition-penalty` 显式配置采样，未指定时保持原服务默认行为。
`--code-seed 42` 设置请求随机种子，多样本时使用 `42 + sample_index`。
这些设置与 thinking 开关独立，并记录在 manifest 中。
调用 `--code-n-samples 10 --temperature 0.2 --code-pass-k 1,5,10`
可报告多样本估计 pass@k（只报告 k <= n 的项目）。`--code-timeout 6`
控制 LCB 单测试超时或 EvalPlus 最小超时，`--code-eval-workers 4` 控制判题并发。
本项目使用 chat 完整解答提示、EvalPlus 官方 sanitizer、LCB 最后一个 Python
代码块提取。比较 leaderboard 时应同时对齐采样、提示、时间窗口和超时参数。

## 产物与离线判分

默认输出 `workspace/eval_logs/<benchmark>/<timestamp>/`，可用 `--output-dir` 指定。

- `manifest.json`：版本/数据 hash、选中 task_id、模型与采样参数。
- `samples.jsonl`：逐样本完整 solution、原始回答、结束原因及请求错误。
- `raw_samples.jsonl`：新增生成任务在代码清洗前立即保存的 API 回答，便于区分推理与后处理故障。
- `results.json`：逐题/逐样本官方判分。
- `summary.json`：base/plus 或 LCB pass@k，以及请求错误数和子集标识。

空回答和请求失败保留在评分分母中；有生成错误时进程返回非零。
输出目录若已包含本项目产物会拒绝覆盖。仅生成用 `--code-generate-only`；
随后使用新输出目录和 `--code-samples /path/to/samples.jsonl` 离线判分。
外部 JSONL 必须提供 `task_id` 和完整 `solution`（不接受只有函数体的 completion），
每个选中任务必须恰好有 `--code-n-samples` 条记录。

## 验证

清洗后的三个 TACO 模型可通过以下入口逐模型评测最后的 step 66（需先用
`verl.model_merger` 导出到 `workspace/eval_models/cleaned_taco_step66/<训练实验名>/`）：

```bash
.venv/bin/python benchmark/coding/run_cleaned_taco_suite.py \
  --thinking off --output-root /absolute/path/to/a-new-evaluation-directory
```

仅 LCB 的 32K 随机采样对照（thinking 独立指定）：

```bash
.venv/bin/python benchmark/coding/run_cleaned_taco_suite.py \
  --lcb-only --thinking off --sampling coding \
  --max-tokens 32768 --max-model-len 65536 \
  --output-root /absolute/path/to/a-new-sampling-evaluation-directory
```

`coding` 参数为 temperature=0.6、top_p=0.95、top_k=20、min_p=0.0、
presence_penalty=0.0、repetition_penalty=1.0，逐请求 seed=42。每题仍只生成一份回答，
属于单次随机采样 pass@1，存在采样波动；不要把单次小幅变化视为统计显著。
选择该参数组不会自动开启 thinking，也不会改变 HumanEval/MBPP 或训练采样默认设置。

此入口使用 DP=8、TP=1，8 张 GPU 共同推理同一个数据集，各数据集推理串行；
官方隔离容器判分在 CPU 上与后续推理重叠执行。`--thinking training` 沿用各轮
Student 模式，`--thinking both` 则对每个模型分别测 on/off。输出目录必须全新，
原始 checkpoint 不修改。权重 SHA-256、服务命令、采样设置保存在 suite manifest
和 events 中。LCB v6 复用相同的 v5 880 题回答并追加新增 175 题，完整官方判分后
同时汇报累计 v6 和新增题子集；不把新增题分数当作 v6 全量分数。
中断后可用 `--resume` 复用已完整判分的模型，启动前会重新校验原始参数和权重
hash；`--models off_off off_on` 可明确选择剩余模型。已有产物不会覆盖；部分完成且
没有请求错误的数据集会保留已有样本，在独立 retry 目录仅补齐缺失任务，不按答案质量重采样。

`sanitize_fast.py` 保留 EvalPlus 0.3.1 的代码筛选规则，仅剪枝不可能胜出的搜索区间、
排除追加后续行也无法修复的语法前缀，并消除重复提取，避免长篇复述导致二次复杂度后处理。
`test_sanitize_fast.py` 与原实现逐字比较提取和清洗结果；`audit_sanitizer.py` 另可对已有
真实回答做等价性审计。本轮 1594 份原实现产物全部一致，报告保存在评测目录的
`sanitizer_equivalence_v2.json`。候选代码的官方测试和隔离判题未改动。

```bash
cd benchmark/coding
RUN_CODE_SANDBOX_TESTS=1 .venv/bin/python -m unittest test_coding -v
cd ../..
benchmark/coding/.venv/bin/python benchmark/coding/smoke.py
```

单元测试覆盖数据版本、私有测试解码、任务选择、代码提取、样本分母和 API 错误。
Docker 测试验证官方 stdin/functional 判分与 base/plus 差异。
`smoke.py` 通过统一入口，对 HumanEval+、MBPP+、LCB v5、LCB v6 各选一题（v6 选取新增题）、各给一份正确和错误解，
检查 pass@1=0.5、pass@2=1；不调用模型，也不占用 GPU。

官方参考：[LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench)、
[LCB 数据集](https://huggingface.co/datasets/livecodebench/code_generation_lite)、
[EvalPlus](https://github.com/evalplus/evalplus/tree/v0.3.1)。
