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
调用 `--code-n-samples 10 --temperature 0.2 --code-pass-k 1,5,10`
可报告多样本估计 pass@k（只报告 k <= n 的项目）。`--code-timeout 6`
控制 LCB 单测试超时或 EvalPlus 最小超时，`--code-eval-workers 4` 控制判题并发。
本项目使用 chat 完整解答提示、EvalPlus 官方 sanitizer、LCB 最后一个 Python
代码块提取。比较 leaderboard 时应同时对齐采样、提示、时间窗口和超时参数。

## 产物与离线判分

默认输出 `workspace/eval_logs/<benchmark>/<timestamp>/`，可用 `--output-dir` 指定。

- `manifest.json`：版本/数据 hash、选中 task_id、模型与采样参数。
- `samples.jsonl`：逐样本完整 solution、原始回答、结束原因及请求错误。
- `results.json`：逐题/逐样本官方判分。
- `summary.json`：base/plus 或 LCB pass@k，以及请求错误数和子集标识。

空回答和请求失败保留在评分分母中；有生成错误时进程返回非零。
输出目录若已包含本项目产物会拒绝覆盖。仅生成用 `--code-generate-only`；
随后使用新输出目录和 `--code-samples /path/to/samples.jsonl` 离线判分。
外部 JSONL 必须提供 `task_id` 和完整 `solution`（不接受只有函数体的 completion），
每个选中任务必须恰好有 `--code-n-samples` 条记录。

## 验证

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
