# 四个自进化模型评测汇总

评测时间：2026-09-05（Asia/Shanghai）

## 模型

| 简称 | checkpoint / serving name |
|---|---|
| AWM | `full-final-awm-step67` / `opsd-awm-step67` |
| EnvScaler | `full-final-envscaler-step67` / `opsd-envscaler-step67` |
| TACO | `full-final-taco-step67` / `opsd-taco-step67` |
| Mixed | `full-final-mixed-step203` / `opsd-mixed-step203` |

## 总表

下表均为百分比；粗体表示该列最高值。

| 模型 | BFCL v3 Overall | tau2 Airline | tau2 Retail | HumanEval base | HumanEval+ | MBPP base | MBPP+ | LCB v5 | LCB v6 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AWM | 40.37 | 66.00 | 57.89 | 89.02 | 84.15 | **85.45** | **73.02** | 46.82 | **44.45** |
| EnvScaler | 44.42 | 70.00 | **62.28** | **93.90** | **87.80** | 85.19 | 72.75 | **46.93** | 44.36 |
| TACO | 48.34 | **72.00** | 57.02 | 88.41 | 84.76 | 81.75 | 69.31 | 40.91 | 38.58 |
| Mixed | **52.64** | 66.00 | 54.39 | 86.59 | 81.71 | 82.54 | 67.99 | 43.18 | 41.42 |

## BFCL v3 分项

| 模型 | Overall | Non-live | Live | Multi-turn |
|---|---:|---:|---:|---:|
| AWM | 40.37 | 28.92 | 75.70 | 16.50 |
| EnvScaler | 44.42 | 39.23 | 76.77 | 17.25 |
| TACO | 48.34 | 49.95 | **77.57** | 17.50 |
| Mixed | **52.64** | **64.08** | 76.10 | **17.75** |

BFCL 使用仓库内 `benchmark/bfcl_v3` 适配器和 pinned 的官方 BFCL v3 runtime，结果是官方 CSV scorer 输出。

## tau2-bench

| 模型 | Airline | Retail |
|---|---:|---:|
| AWM | 33/50 (66.00%) | 66/114 (57.89%) |
| EnvScaler | 35/50 (70.00%) | **71/114 (62.28%)** |
| TACO | **36/50 (72.00%)** | 65/114 (57.02%) |
| Mixed | 33/50 (66.00%) | 62/114 (54.39%) |

配置：`max_steps=200`，并发 16，agent/user 使用同一被测模型，temperature 0，top_p 1，最大生成 4096 tokens；Retail 使用项目配置的 Ark judge。

## EvalPlus 与 LiveCodeBench

- HumanEval/MBPP 使用 EvalPlus 官方 base 和 plus 测试集，表中为 pass@1。
- LiveCodeBench 使用固定 revision `0fe84c3912ea0c4d4a78037083943e8f0c4dd505` 和 evaluator commit `28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24`。
- LCB v5 是官方累计 release，880 题；LCB v6 是官方累计 release，1055 题（包含 v5 的 880 题与 v6 新增 175 题）。
- 所有代码在无网络、非 root、只读 Docker 沙箱中由官方 scorer 执行。

### 空最终答案数

| 模型 | HumanEval+ | MBPP+ | LCB v5 | LCB v6 |
|---|---:|---:|---:|---:|
| AWM | 2 | 7 | 440 | 552 |
| EnvScaler | 1 | 6 | 439 | 551 |
| TACO | 8 | 13 | 497 | 620 |
| Mixed | 11 | 18 | 471 | 583 |

上述生成错误在超时重试后均为 `empty_final_answer`：模型在 16K token 内只产生了 reasoning，没有 final content。它们已按未通过计入 pass@1，不是丢样。

## 结论

- Mixed 在 BFCL v3 明显最强，优势主要来自 non-live 函数调用。
- EnvScaler 在 HumanEval+ / MBPP+ 和 tau2 Retail 上整体最强，并且 LCB v5 最高。
- TACO 在 tau2 Airline 最高，但 LCB 的空 final 比例也最高。
- AWM 的 LCB v6 最高，与 EnvScaler 差距很小（0.09 个百分点）。

## 结果目录

- `workspace/eval_logs/eval_main/full-final-awm-step67`
- `workspace/eval_logs/eval_main/full-final-envscaler-step67`
- `workspace/eval_logs/eval_main/full-final-taco-step67`
- `workspace/eval_logs/eval_main/full-final-mixed-step203`

每个目录下保留 BFCL CSV，tau2 `results.json`，EvalPlus `summary.json` / `results.json`，以及 LCB v5/v6 的合并 samples 和最终官方评分。
