# Split-4 Cluster / Merge 模型工具使用评测

最后更新：2026-08-28 14:46:42 CST

> 当前进度：5/5 个模型完成全部评测。表中的 `—` 表示该项尚未形成完整、可校验的正式产物。

## 1. 实验对象

| 模型 | Prefix 结构 | 训练 checkpoint / 合并来源 | 推理模型 |
|---|---:|---|---|
| Cluster 0 (step 35) | G512/A64 | `checkpoints/self_evolver_opsd/qwen3_5_4b_awm_opsd_hybrid_prefix_split4_cluster0/global_step_35/actor` | `/mnt/storage/disk1/verl_data/base_model/Qwen3.5-4B-HybridDeltaResidualAttentionPrefix-split4-trained-eval/cluster0-step35` |
| Cluster 1 (step 44) | G512/A64 | `checkpoints/self_evolver_opsd/qwen3_5_4b_awm_opsd_hybrid_prefix_split4_cluster1/global_step_44/actor` | `/mnt/storage/disk1/verl_data/base_model/Qwen3.5-4B-HybridDeltaResidualAttentionPrefix-split4-trained-eval/cluster1-step44` |
| Cluster 2 (step 66) | G512/A64 | `checkpoints/self_evolver_opsd/qwen3_5_4b_awm_opsd_hybrid_prefix_split4_cluster2/global_step_66/actor` | `/mnt/storage/disk1/verl_data/base_model/Qwen3.5-4B-HybridDeltaResidualAttentionPrefix-split4-trained-eval/cluster2-step66` |
| Cluster 3 (step 57) | G512/A64 | `checkpoints/self_evolver_opsd/qwen3_5_4b_awm_opsd_hybrid_prefix_split4_cluster3/global_step_57/actor` | `/mnt/storage/disk1/verl_data/base_model/Qwen3.5-4B-HybridDeltaResidualAttentionPrefix-split4-trained-eval/cluster3-step57` |
| Merged split-4 model | G2048/A256 | `four latest cluster checkpoints concatenated in cluster0/1/2/3 order` | `/mnt/storage/disk1/verl_data/base_model/Qwen3.5-4B-HybridDeltaResidualAttentionPrefix-G2048-A256-step203-split4-trained-merged` |

四个 cluster checkpoint 均聚合为 BF16 prefix-only HF 评测目录；dense base 权重通过同文件系统硬链接复用。Merged 模型按 cluster0 → cluster1 → cluster2 → cluster3 的顺序拼接四段 prefix。

## 2. 统一评测口径

| Benchmark | 数据与设置 |
|---|---|
| BFCL v3 | 官方 `all`，17 类、4441 条；temperature=0；并发 64；官方 generate → evaluate → aggregate |
| tau2-bench | Airline `base` 50 条 + Retail `base` 114 条；1 trial；max steps=50；agent/user max tokens=4096；thinking off；并发 32；Retail NL judge=`deepseek-v4-pro-ga-260813` |
| LifelongDB | `test` 500 条；max steps=6；max tokens=2048；MySQL 8.0；并发 16 |
| LifelongOS | `test` 500 条；max steps=8；max tokens=2048；shell timeout=20s；并发 16 |

推理统一使用 vLLM 0.18.0、BF16、8×A800 data parallel；每次只加载一个模型，跑完该模型的全部 benchmark 后再切换模型。tau2 是单次 self-play trial，user simulator 使用与被测 agent 相同的模型，因此结果包含模拟轨迹的单次运行波动。

## 3. 主结果

### 3.1 BFCL v3

| 模型 | Non-Live AST | Live | Multi-Turn | Hall. / Irrelevance | Overall |
|---|---:|---:|---:|---:|---:|
| Cluster 0 (step 35) | 79.65 | 78.28 | 46.75 | 83.47 | 68.69 |
| Cluster 1 (step 44) | 79.38 | 78.10 | 45.25 | 82.62 | 67.98 |
| Cluster 2 (step 66) | 79.98 | 78.23 | 47.62 | 83.41 | 69.06 |
| Cluster 3 (step 57) | 80.27 | 77.97 | 45.63 | 83.49 | 68.44 |
| Merged split-4 model | 80.31 | 77.92 | 47.25 | 83.34 | 68.95 |

单位：%。`Hall.` 是官方 Overall CSV 的 Irrelevance Detection。

### 3.2 tau2-bench

| 模型 | Airline Pass@1 | Retail Pass@1 | Overall Pass@1 |
|---|---:|---:|---:|
| Cluster 0 (step 35) | 62.00 (31/50) | 55.26 (63/114) | 57.32 (94/164) |
| Cluster 1 (step 44) | 60.00 (30/50) | 52.63 (60/114) | 54.88 (90/164) |
| Cluster 2 (step 66) | 58.00 (29/50) | 53.51 (61/114) | 54.88 (90/164) |
| Cluster 3 (step 57) | 58.00 (29/50) | 53.51 (61/114) | 54.88 (90/164) |
| Merged split-4 model | 62.00 (31/50) | 54.39 (62/114) | 56.71 (93/164) |

### 3.3 LifelongAgentBench

| 模型 | LifelongDB Pass@1 | LifelongOS Pass@1 | Macro Avg | Runtime errors |
|---|---:|---:|---:|---:|
| Cluster 0 (step 35) | 86.00 (430/500) | 44.20 (221/500) | 65.10 | 0 |
| Cluster 1 (step 44) | 86.00 (430/500) | 43.40 (217/500) | 64.70 | 0 |
| Cluster 2 (step 66) | 87.60 (438/500) | 44.60 (223/500) | 66.10 | 0 |
| Cluster 3 (step 57) | 84.80 (424/500) | 41.80 (209/500) | 63.30 | 0 |
| Merged split-4 model | 87.20 (436/500) | 42.00 (210/500) | 64.60 | 0 |

## 4. 完整性检查与原始产物

| 模型 | BFCL 17/17 | tau2 Airline 50/50 | tau2 Retail 114/114 | LifelongDB 500/500 | LifelongOS 500/500 | 总状态 |
|---|---:|---:|---:|---:|---:|---|
| Cluster 0 (step 35) | ✓ | ✓ | ✓ | ✓ | ✓ | 完成 |
| Cluster 1 (step 44) | ✓ | ✓ | ✓ | ✓ | ✓ | 完成 |
| Cluster 2 (step 66) | ✓ | ✓ | ✓ | ✓ | ✓ | 完成 |
| Cluster 3 (step 57) | ✓ | ✓ | ✓ | ✓ | ✓ | 完成 |
| Merged split-4 model | ✓ | ✓ | ✓ | ✓ | ✓ | 完成 |

所有原始结果统一位于 `workspace/eval_logs/split4_cluster_merge_20260828`：每个模型目录下包含 `bfcl_v3/all/result`、`bfcl_v3/all/score`、`tau2/*/results.json`、`lifelong_*/test/summary.csv` 和逐任务轨迹；vLLM 与 runner 日志保存在同一模型目录。

完整性判定要求：BFCL 生成与 score 各 17 个类别文件；tau2 task ID 数量完整且无重复；两个 Lifelong 子集各 500 个唯一 task ID。

## 5. 结果摘要

- BFCL Overall 最佳：Cluster 2 (step 66)（69.06%）。Merged 相对最佳 cluster：-0.11 pp。
- tau2 Overall 最佳：Cluster 0 (step 35)（57.32%）。Merged 相对最佳 cluster：-0.61 pp。
- Lifelong Macro Avg 最佳：Cluster 2 (step 66)（66.10%）。Merged 相对最佳 cluster：-1.50 pp。
