# Qwen3.5-4B 原生基线与自进化模型代码评测对比

评测时间：2026-09-05（Asia/Shanghai）

原生模型：`/mnt/storage/disk1/verl_data/base_model/Qwen3.5-4B`

## 结果总表

下表均为 pass@1 百分比；括号内是相对原生 `Qwen3.5-4B` 基线的变化（百分点）。

| 模型 | HumanEval base | HumanEval+ | MBPP base | MBPP+ | LCB v5 | LCB v6 |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B 原生 | 90.24 | 86.59 | 83.60 | 69.31 | 45.23 | 43.51 |
| AWM step67 | 89.02 (-1.22) | 84.15 (-2.44) | **85.45 (+1.85)** | **73.02 (+3.71)** | 46.82 (+1.59) | **44.45 (+0.94)** |
| EnvScaler step67 | **93.90 (+3.66)** | **87.80 (+1.21)** | 85.19 (+1.59) | 72.75 (+3.44) | **46.93 (+1.70)** | 44.36 (+0.85) |
| TACO step67 | 88.41 (-1.83) | 84.76 (-1.83) | 81.75 (-1.85) | 69.31 (+0.00) | 40.91 (-4.32) | 38.58 (-4.93) |
| Mixed step203 | 86.59 (-3.65) | 81.71 (-4.88) | 82.54 (-1.06) | 67.99 (-1.32) | 43.18 (-2.05) | 41.42 (-2.09) |

## 结论

- **EnvScaler 是唯一在 6 个指标上全部超过原生基线的训练模型**：HumanEval+ +1.21pp、MBPP+ +3.44pp、LCB v5 +1.70pp、LCB v6 +0.85pp。
- AWM 在 MBPP+ 上增益最大（+3.71pp），LCB v5/v6 也有增益，但 HumanEval+ 退化 2.44pp。
- TACO 在 MBPP+ 与基线持平，HumanEval+ 和 LCB 均退化，尤其 LCB v6 -4.93pp。
- Mixed 在这 6 个代码指标上均低于原生基线。
- 因此，如果目标是综合代码能力，这四组中 **EnvScaler step67** 的训练增益最稳定；如果更看重 MBPP+，AWM step67 最高。

## 空最终答案

`empty_final_answer` 表示模型在 16K token 内只产生了 reasoning，没有 final content；已按未通过计入 pass@1，不是丢样或 API 错误。

| 模型 | HumanEval+ | MBPP+ | LCB v5 | LCB v6 |
|---|---:|---:|---:|---:|
| Qwen3.5-4B 原生 | 5 | 16 | 459 | 567 |
| AWM step67 | 2 | 7 | 440 | 552 |
| EnvScaler step67 | 1 | 6 | 439 | 551 |
| TACO step67 | 8 | 13 | 497 | 620 |
| Mixed step203 | 11 | 18 | 471 | 583 |

## 评测口径

- 所有模型使用同一生成参数：`temperature=0`、`top_p=1`、最大生成 16384 tokens、pass@1。
- HumanEval/MBPP 使用 EvalPlus 官方 base 和 plus 测试集。
- LCB v5 是官方累计 release，880 题；LCB v6 是官方累计 release，1055 题（v5 880 题 + v6 新增 175 题）。
- LiveCodeBench dataset revision：`0fe84c3912ea0c4d4a78037083943e8f0c4dd505`；evaluator commit：`28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24`。
- 所有代码都由官方 scorer 在无网络、非 root、只读 Docker 沙箱中执行。

## 原生基线结果目录

`workspace/eval_logs/eval_main/qwen3.5-4b-base`

目录中保留了 EvalPlus 的 `summary.json` / `results.json`、LCB v5/v6 完整合并 samples，以及最终官方评分。
