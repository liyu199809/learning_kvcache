# Qwen3.5-4B Tool-Use Benchmark 实验记录

最后更新：2026-08-22（Asia/Shanghai；当日勘误见第 8 节）

本文汇总 Qwen3.5-4B 原生模型、全参数微调（Full-FT）step 200、LoRA step 200、Independent Delta KV Prefix step 203（常规 LR 与 largelr 两个 run）以及 Hybrid Delta + Residual Attention Prefix（step 180/200/203）在 BFCL v3、tau2-bench 与 LifelongAgentBench 上的正式结果。主结果表参考论文表格形式组织：同一 benchmark 下按方法逐行对比，最佳值以 **粗体** 标出。

> 口径提示：tau2/Lifelong 是单次 trial 的 Pass@1，BFCL 是单次生成的官方准确率；均非多次运行均值。**2026-08-21 勘误**：08-13 评测时 vLLM 端点实际加载的是 Full-FT step 200 的 merged 权重（该导出完成于 08-13 11:50，评测 12:01 启动），当时被误记为"Base"；`Base native (08-20)` 才是唯一一次真正的原生模型评测。因此 08-13 与 08-20 两行是**两个不同模型**，其差值是方法差异，不是复现波动。tau2 的 Retail judge 配置跨日期不同，因此 tau2 表按 judge 配置分组，粗体表示同组最佳。

## 1. 实验对象

| Method | 权重/Checkpoint | 训练或合并形式 | 正式评测日期 |
|---|---|---|---:|
| Full-FT (step 200, 08-13) | `checkpoints/self_evolver_opsd/qwen3_5_4b_awm_opsd_fix_context/global_step_200/actor/huggingface_merged/` | FSDP2 world size 6 全参数微调（`run_qwen3_5_4b_awm_opsd_full.sh`），step 200 导出 merged HF 权重；08-13 评测时端点加载的是该权重，产物当时被误标为 "Base" | 2026-08-13 |
| Base native (08-20) | `/mnt/storage/disk1/verl_data/base_model/Qwen3.5-4B` | 原生 dense 权重（2026-06-07 下载后未改动），唯一的原生基线评测 | 2026-08-20 |
| LoRA r64/a128 (step 200) | `checkpoints/self_evolver_opsd/qwen3_5_4b_awm_opsd_lora_r64_a128/global_step_200/actor/` | FSDP2 world size 6；先导出 PEFT adapter，再 `merge_and_unload(safe_merge=True)` 合入 base model | 2026-08-17 |
| Independent ΔKV Prefix m2048 (step 203) | `checkpoints/self_evolver_opsd/qwen3_5_4b_awm_opsd_independent_delta_kv_prefix_m2048/global_step_203/actor/` | FSDP2 world size 6；CPU 聚合为 96 个 BF16 prefix tensors，共 305,135,616 个 prefix 参数；prefix LR `5e-6` | 2026-08-20 |
| Independent ΔKV Prefix m2048 largelr (step 203) | `checkpoints/self_evolver_opsd/qwen3_5_4b_awm_opsd_independent_delta_kv_prefix_largelr/global_step_203/actor/` | 同上，唯一差异是 prefix LR 提高到 `5e-5`（`run_qwen3_5_4b_awm_opsd_independent_delta_kv_prefix_m2048.sh` 默认 `ACTOR_LR=5e-5`，10× 于常规 run）；同为 96 个 BF16 prefix tensors、305,135,616 参数 | 2026-08-21 |
| Independent ΔKV Prefix m2048 largelr (step 180) | `checkpoints/self_evolver_opsd/qwen3_5_4b_awm_opsd_independent_delta_kv_prefix_largelr/global_step_180/actor/` | 同一 largelr run 的 step 180 checkpoint，用于检查训练后期是否退化（与 step 203 同口径全量评测） | 2026-08-21 |
| Hybrid Δ + Residual Attn Prefix G2048/A256 (step 180/200/203) | `checkpoints/self_evolver_opsd/qwen3_5_4b_awm_opsd_hybrid_delta_residual_attention_prefix_g2048_a256/global_step_{180,200,203}/actor/` | FSDP2 world size 6，prefix LR `5e-6`（`run_qwen3_5_4b_awm_opsd_hybrid_delta_residual_attention_prefix_g2048_a256.sh`）；在 Independent ΔKV Prefix（24 个线性层，G=2048）之上给 8 个 full-attention 层各加 key/value prefix（A=256）。CPU 聚合为 96 个 ΔKV tensors + 16 个 attention tensors，共 315,621,376 个 prefix 参数 | 2026-08-22 |

LoRA 合并模型位于 `checkpoints/self_evolver_opsd/qwen3_5_4b_awm_opsd_lora_r64_a128/global_step_200/actor/huggingface_merged/`。两个 Prefix 部署模型分别位于 `/mnt/storage/disk1/verl_data/base_model/Qwen3.5-4B-IndependentDeltaKVPrefix-step203/` 与 `/mnt/storage/disk1/verl_data/base_model/Qwen3.5-4B-IndependentDeltaKVPrefix-largelr-step203/`；Hybrid Prefix 的三个部署模型位于 `/mnt/storage/disk1/verl_data/base_model/Qwen3.5-4B-HybridDeltaResidualAttentionPrefix-G2048-A256-step{180,200,203}/`（prepared 模型 `Qwen3.5-4B-HybridDeltaResidualAttentionPrefix-G2048-A256` 复用 base 权重）。prefix 类方法的“合并”是聚合 FSDP prefix 分片并复用 base 权重，推理时由模型实现显式注入 prefix KV，不会把 prefix 数值折叠进 dense 权重。

## 2. 实验设置

### 2.1 推理服务

| 项目 | 设置 |
|---|---|
| GPU | 8 × NVIDIA A800-SXM4-80GB（每卡 81920 MiB） |
| vLLM | 0.18.0，OpenAI-compatible API |
| 并行 | Data Parallel = 8，Tensor Parallel = 1，API server count = 1 |
| 精度 | BF16 |
| 上下文 | max model length = 262144 |
| 显存利用率 | 0.85 |
| Tool calling | `--enable-auto-tool-choice`，reasoning parser=`qwen3`，tool-call parser=`qwen3_coder` |
| Thinking | tau2/Lifelong 显式关闭；服务 smoke test 的 reasoning 字段为空 |
| 其他环境 | Python 3.11.15，PyTorch 2.10.0+cu128，CUDA runtime 12.8，Docker 29.1.3 |

2026-08-20 原生评测、08-21 largelr 评测当天各 step 服务依次启停；08-22 Hybrid 三个 step 服务依次启停后，当前保留运行的是 Hybrid step 203 服务：

- Endpoint：`http://127.0.0.1:8000/v1`
- Served model：`qwen3.5-4b-hybrid-delta-residual-prefix-step203`
- 加载权重：`/mnt/storage/disk1/verl_data/base_model/Qwen3.5-4B-HybridDeltaResidualAttentionPrefix-G2048-A256-step203`
- 日志：`workspace/eval_logs/vllm_server/hybrid_prefix_g2048_a256_step203_vllm.log`
- Smoke test：普通文本精确返回 `OK`；tool call 正确生成 `get_weather({"city":"北京"})`

（历史：08-20 原生服务 `qwen3.5-4b`，日志 `vllm_server/qwen35_4b_native_rerun_20260820_vllm.log`；08-21 largelr step 203/180 服务日志 `vllm_server/prefix_largelr_step{203,180}_vllm.log`；08-22 Hybrid step 180/200 服务日志 `vllm_server/hybrid_prefix_g2048_a256_step{180,200}_vllm.log`。vllm 服务日志统一归档在 `workspace/eval_logs/vllm_server/`。）

### 2.2 Benchmark 配置

| Benchmark | 数据范围 | 推理限制 | 并发 | 判分方式 |
|---|---|---|---:|---|
| BFCL v3 | 官方 `all`，17 个类别，共 4441 条生成任务 | temperature=0；context=262144 | 64 | 官方 generate → evaluate → aggregate 管线；使用 `Qwen/Qwen3-4B-FC` handler |
| tau2 Airline | official `base` split，50 tasks，1 trial | max steps=50；agent/user max tokens=4096；temperature=0；top_p=1；thinking off | 32 | 环境 DB、communicate checks 与 action checks |
| tau2 Retail | official `base` split，114 tasks，1 trial | 同 Airline | 32 | 环境 DB + NL assertions；judge 配置见下文 |
| LifelongDB | `test`，500 tasks | max steps=6；max completion tokens=2048；MySQL 8.0 | 16 | 数据库最终状态与提交结果精确判分 |
| LifelongOS | `test`，500 tasks | max steps=8；max completion tokens=2048；shell timeout=20s；LLM step timeout=180s | 16 | Ubuntu 容器内隐藏检查脚本 |

### 2.3 tau2 的 simulator 与 judge

tau2 不是固定 prompt 的静态问答：user simulator 也由模型生成，因此这里始终让 user simulator 使用与被测 agent 相同的模型。Retail 的 NL-assertion judge 分为两组：

| Judge 组 | 方法 | Retail NL judge |
|---|---|---|
| Historical local | Full-FT (08-13)、LoRA step 200 | 本地被测模型，thinking off |
| Ark DeepSeek | Base native (08-20)、Prefix step 203（含 largelr）、Hybrid Prefix（08-22） | 方舟 `deepseek-v4-pro-ga-260813`，thinking off |

运行入口会从仓库 `.env` 重新加载 `ARK_API_KEY`。2026-08-20 两次运行的配置输出均显示实际使用方舟 judge，而不是本地 fallback。由于 judge 不同，Retail 只能在同一 judge 组内做严格一些的相对比较；所有 tau2 结果也不能直接与使用官方 GPT-4.1 simulator/judge 的 leaderboard 横比。

### 2.4 可比性限制

1. BFCL 官方 score 文件中的模型标签统一为 `Qwen3-4B (FC)`，实际请求模型由 vLLM endpoint 和 served model ID 决定。旧基线产物仅凭标签不能证明与其它 benchmark 使用完全相同的 checkpoint，因此 BFCL 跨日期差值属于产物级比较。
2. BFCL Overall 是官方按类别聚合的分数，不是对全部底层判分单元做简单微平均。
3. tau2 只有 1 trial，且 user simulator 随被测模型变化；这既包含模型能力差异，也包含 self-play 轨迹波动。
4. 08-13 与 08-20 两行是不同模型（Full-FT step 200 vs 原生权重），且 tau2 judge 配置不同，其差值不能解释为复现波动；原生基线目前只有 08-20 一次，没有重复运行可估计波动幅度。
5. Lifelong 的 skill 标签可重叠，因此分技能统计不能相加得到总样本数。

## 3. 主结果

### 3.1 BFCL v3 leaderboard

`Hall.` 对应 BFCL 汇总文件中的 `Irrelevance Detection`，表示不应调用工具时的识别能力。

| Method | Non-Live | Live | Multi-Turn | Hall. | Overall |
|---|---:|---:|---:|---:|---:|
| Full-FT (08-13) | 75.37 | 76.28 | **51.12** | 80.97 | 67.59 |
| Base native (08-20) | 81.03 | 77.34 | 44.50 | 81.86 | 67.63 |
| LoRA r64/a128 (step 200) | 80.15 | 76.63 | 49.88 | 83.36 | **68.89** |
| Independent ΔKV Prefix m2048 (step 203) | **81.87** | 78.05 | 45.88 | 83.75 | 68.60 |
| Independent ΔKV Prefix m2048 largelr (step 203) | 78.33 | 73.66 | 43.75 | 75.09 | 65.19 |
| Independent ΔKV Prefix m2048 largelr (step 180) | 78.33 | 74.01 | 46.25 | 78.76 | 66.50 |
| Hybrid Δ+ResAttn Prefix G2048/A256 (step 180) | 79.81 | **78.23** | 46.38 | **83.77** | 68.65 |
| Hybrid Δ+ResAttn Prefix G2048/A256 (step 200) | 80.17 | 78.14 | 46.25 | 83.02 | 68.62 |
| Hybrid Δ+ResAttn Prefix G2048/A256 (step 203) | 80.12 | 77.74 | 47.25 | 83.41 | **68.89** |

单位：%。Prefix（5e-6）的 Independent 变体在 Non-Live 最好（81.87），Hybrid 变体在 Live（78.23）与 Hallucination/Irrelevance（83.77）最好；BFCL Overall 最高为 LoRA 与 Hybrid step 203 并列（68.89）；Full-FT 的 Multi-Turn 最高（51.12，比原生 Base 的 44.50 高 6.62 pp）。largelr 两个 checkpoint 全面低于常规 LR run：step 203 Overall 65.19% 为主表最低，step 180 回升到 66.50% 但仍低于常规 run 与原生 Base。Hybrid 三个 checkpoint 的 BFCL Overall 在 68.62-68.89 之间，训练后期稳定无退化。

### 3.2 tau2-bench

粗体表示同一 judge 组内最佳。Overall 为 Airline 与 Retail 按样本数加权的 Pass@1。

| Judge 组 | Method | Airline Pass@1 | Retail Pass@1 | Overall Pass@1 |
|---|---|---:|---:|---:|
| Historical local | Full-FT (08-13) | **68.00** (34/50) | 53.51 (61/114) | 57.93 (95/164) |
| Historical local | LoRA r64/a128 (step 200) | 64.00 (32/50) | **60.53** (69/114) | **61.59** (101/164) |
| Ark DeepSeek | Base native (08-20) | 58.00 (29/50) | 55.26 (63/114) | 56.10 (92/164) |
| Ark DeepSeek | Independent ΔKV Prefix m2048 (step 203) | 68.00 (34/50) | 57.89 (66/114) | **60.98** (100/164) |
| Ark DeepSeek | Independent ΔKV Prefix m2048 largelr (step 203) | 66.00 (33/50) | 50.88 (58/114) | 55.49 (91/164) |
| Ark DeepSeek | Independent ΔKV Prefix m2048 largelr (step 180) | **70.00** (35/50) | 46.49 (53/114) | 53.66 (88/164) |
| Ark DeepSeek | Hybrid Δ+ResAttn Prefix (step 180) | 64.00 (32/50) | 54.39 (62/114) | 57.32 (94/164) |
| Ark DeepSeek | Hybrid Δ+ResAttn Prefix (step 200) | 62.00 (31/50) | **58.77** (67/114) | 59.76 (98/164) |
| Ark DeepSeek | Hybrid Δ+ResAttn Prefix (step 203) | 66.00 (33/50) | 52.63 (60/114) | 56.71 (93/164) |

在同日、同 judge 配置下，Prefix 相对原生 Base 多通过 8 条：Airline +5，Retail +3，Overall +4.88 pp。largelr 与常规 run 同组同日可比：step 203 Overall 掉 5.49 pp 且低于原生 Base；step 180 的 Airline 70.00 是全表最高，但 Retail 46.49 是全表最低，Overall 53.66 反而更差——largelr 两个 checkpoint 的 tau2 Overall 均明显低于常规 run。Hybrid 与常规 run 同 LR、同 judge 组但跨日（08-22 vs 08-20）：三步 Overall 57.32/59.76/56.71，均未超过常规 run 的 60.98；Retail 在 step 200 达到全表最高的 58.77，但 Airline 相对疲软（最高 66.00）。注意 Full-FT 与原生 Base 分属不同 judge 组，tau2 上不能直接横比。

### 3.3 LifelongAgentBench

两个子集各 500 条。Macro Avg 是 DB 与 OS 通过率的简单平均，仅用于紧凑汇总。

| Method | LifelongDB Pass@1 | LifelongOS Pass@1 | Macro Avg |
|---|---:|---:|---:|
| Full-FT (08-13) | 85.00 (425/500) | 42.80 (214/500) | 63.90 |
| Base native (08-20) | 85.20 (426/500) | 44.20 (221/500) | 64.70 |
| LoRA r64/a128 (step 200) | 83.20 (416/500) | **46.60** (233/500) | 64.90 |
| Independent ΔKV Prefix m2048 (step 203) | 86.80 (434/500) | 42.40 (212/500) | 64.60 |
| Independent ΔKV Prefix m2048 largelr (step 203) | 84.00 (420/500) | 43.40 (217/500) | 63.70 |
| Independent ΔKV Prefix m2048 largelr (step 180) | 85.20 (426/500) | **46.60** (233/500) | **65.90** |
| Hybrid Δ+ResAttn Prefix G2048/A256 (step 180) | 86.40 (432/500) | 43.60 (218/500) | 65.00 |
| Hybrid Δ+ResAttn Prefix G2048/A256 (step 200) | **87.60** (438/500) | 41.80 (209/500) | 64.70 |
| Hybrid Δ+ResAttn Prefix G2048/A256 (step 203) | 86.00 (430/500) | 42.00 (210/500) | 64.00 |

LifelongDB 最好为 Hybrid step 200 的 87.60（Independent 常规 run 的 86.80 次之）；LifelongOS 由 LoRA 与 largelr step 180 并列最高（46.60），Macro Avg 最高为 largelr step 180 的 65.90。largelr 从 step 180 到 step 203 两个子集同步下滑（DB -1.20、OS -3.20 pp）；Hybrid 的 DB 在 step 200 达峰后 step 203 回落 1.60 pp，OS 三步稳定在 41.80-43.60。所有 Lifelong 正式运行均为 0 runtime errors。

## 4. 细分诊断

### 4.1 BFCL single-turn 与 multi-turn

下表选取最能区分方法的 Non-Live AST 和 Multi-Turn 子类。粗体为列最佳。

| Method | NL Simple | NL Multiple | NL Parallel | NL Parallel Multi | MT Base | MT Miss Func | MT Miss Param | MT Long Context |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full-FT (08-13) | 57.67 | 94.00 | 62.50 | 78.50 | **62.50** | **52.50** | 35.00 | **54.50** |
| Base native (08-20) | 66.83 | 92.50 | 74.00 | 86.00 | 54.50 | 40.00 | 33.50 | 50.00 |
| LoRA r64/a128 (step 200) | 63.83 | 92.50 | 72.00 | 84.50 | 60.00 | 51.00 | **37.50** | 51.00 |
| Independent ΔKV Prefix m2048 (step 203) | **66.92** | 93.50 | **75.00** | 86.00 | 56.00 | 43.50 | 36.00 | 48.00 |
| Independent ΔKV Prefix m2048 largelr (step 203) | 61.33 | 92.50 | **75.00** | 84.50 | 51.50 | 45.00 | 30.50 | 48.00 |
| Independent ΔKV Prefix m2048 largelr (step 180) | 63.83 | 92.50 | 72.00 | 85.00 | 53.50 | 46.50 | 34.50 | 50.50 |
| Hybrid Δ+ResAttn Prefix (step 180) | 66.25 | 93.00 | 74.50 | 85.50 | 57.00 | 45.50 | 35.50 | 47.50 |
| Hybrid Δ+ResAttn Prefix (step 200) | 66.67 | **94.50** | 74.00 | 85.50 | 57.00 | 45.50 | 36.50 | 46.00 |
| Hybrid Δ+ResAttn Prefix (step 203) | 65.50 | 93.50 | **75.00** | **86.50** | 55.00 | 47.50 | 36.50 | 50.00 |

主要模式很清楚：LoRA/Prefix 改善了多数 single-turn AST 指标，但没有稳定改善 multi-turn。`Miss Param` 都偏低，是最一致的 BFCL 短板；largelr step 203 把该短板进一步放大到 30.50，且 Hall.（75.09）相比常规 run（83.75）明显退化；step 180 部分回升（Miss Param 34.50、Hall. 78.76）但仍低于常规 run，说明更大 LR 让"不该调工具时别调"的行为变差、且后期继续恶化。Hybrid 三个 checkpoint 的 single-turn AST 全面达到或接近全表最佳（NL Multiple 94.50、NL Parallel Multi 86.50 为列最高），multi-turn 仍低于 Full-FT。

### 4.2 tau2 终止与判分构成

| Judge 组 | Method | Airline 正常/上限 | Airline DB | Retail 正常/上限 | Retail DB | Retail NL |
|---|---|---:|---:|---:|---:|---:|
| Historical local | Full-FT (08-13) | 46 / 4 | 34/46 (73.91%) | 106 / 8 | 63/106 (59.43%) | 46/57 (80.70%) |
| Historical local | LoRA step 200 | 43 / 7 | 33/43 (76.74%) | 107 / 7 | 70/107 (65.42%) | 49/57 (85.96%) |
| Ark DeepSeek | Base native (08-20) | 47 / 3 | 30/47 (63.83%) | 109 / 5 | 64/109 (58.72%) | 48/58 (82.76%) |
| Ark DeepSeek | Prefix step 203 | 48 / 2 | 35/48 (72.92%) | 111 / 3 | 69/111 (62.16%) | 49/57 (85.96%) |
| Ark DeepSeek | Prefix largelr step 203 | 46 / 4 | 33/46 (71.74%) | 106 / 8 | 63/106 (59.43%) | 46/57 (80.70%) |
| Ark DeepSeek | Prefix largelr step 180 | 48 / 2 | 35/48 (72.92%) | 108 / 6 | 59/108 (54.63%) | 42/58 (72.41%) |
| Ark DeepSeek | Hybrid Prefix step 180 | 46 / 4 | 33/46 (71.74%) | 106 / 8 | 64/106 (60.38%) | 48/55 (87.27%) |
| Ark DeepSeek | Hybrid Prefix step 200 | 46 / 4 | 32/46 (69.57%) | 109 / 5 | 69/109 (63.30%) | 47/58 (81.03%) |
| Ark DeepSeek | Hybrid Prefix step 203 | 45 / 5 | 34/45 (75.56%) | 109 / 5 | 67/109 (61.47%) | 44/59 (74.58%) |

同日对比中，Prefix 的优势同时来自更少的 max-step 终止和更高的 DB match。原生 Base 的 Airline write-action match 为 27/47（57.45%），Prefix 为 33/48（68.75%）；Retail 分别为 108/162（66.67%）与 106/166（63.86%），说明 Retail 最终差异不能只用 write-action 命中率解释。largelr step 203 的 write-action match 为 Airline 32/49（65.31%）、Retail 94/157（59.87%），step 180 为 Airline 31/48（64.58%）、Retail 90/154（58.44%），均低于常规 run，且 largelr 的 Retail DB/NL 同步下滑（step 180 的 Retail NL 72.41% 为全表最低）。Hybrid 的 write-action match 明显更高：Airline 三步 30/45（66.67%）、34/48（70.83%）、34/47（72.34%），Retail 三步 106/156（67.95%）、107/160（66.88%）、102/158（64.56%）；其 Retail NL 在 step 180 达到 87.27%（全表最高），但 step 203 回落到 74.58%，tau2 跨步波动主要来自 Retail。

### 4.3 Lifelong 轨迹形态

| Method | DB 平均 steps | DB 失败末动作（submit / execute / 其它） | DB 失败用满 6 steps | OS 平均 steps | OS 失败末动作（finish / execute） | OS 失败用满 8 steps |
|---|---:|---:|---:|---:|---:|---:|
| Full-FT (08-13) | 2.604 | 58 / 17 / 0 | 17 | 6.308 | 142 / 144 | 178 |
| Base native (08-20) | 2.596 | 46 / 28 / 0 | 34 | 6.418 | 116 / 163 | 202 |
| LoRA step 200 | 2.628 | 51 / 30 / 3 | 36 | 6.454 | 124 / 143 | 183 |
| Prefix step 203 | 2.660 | 42 / 24 / 0 | 33 | 6.410 | 124 / 164 | 205 |
| Prefix largelr step 203 | 2.500 | 63 / 17 / 0 | 21 | 6.010 | 151 / 132 | 172 |
| Prefix largelr step 180 | 2.480 | 56 / 18 / 0 | 20 | 6.020 | 155 / 112 | 154 |
| Hybrid Prefix step 180 | 2.688 | 45 / 23 / 0 | 31 | 6.384 | 120 / 162 | 199 |
| Hybrid Prefix step 200 | 2.668 | 39 / 23 / 0 | 32 | 6.414 | 126 / 165 | 206 |
| Hybrid Prefix step 203 | 2.702 | 44 / 26 / 0 | 34 | 6.448 | 125 / 165 | 206 |

DB 失败通常分为两类：已经 `submit` 但结果或最终数据库状态不匹配，以及达到上限仍停留在 `execute`。OS 失败也分为两类：未在上限前调用 `finish`，或调用 `finish` 后隐藏检查未通过。

### 4.4 Lifelong 弱技能

LifelongDB：

| Method | `delete` | `subquery_nested` | `table_alias` | `where_nested_conditions` |
|---|---:|---:|---:|---:|
| Full-FT (08-13) | 50.00 (28/56) | 53.25 (41/77) | 63.30 (69/109) | 62.71 (37/59) |
| Base native (08-20) | 53.57 (30/56) | 55.84 (43/77) | 63.30 (69/109) | 66.10 (39/59) |
| LoRA step 200 | 50.00 (28/56) | 53.25 (41/77) | 61.47 (67/109) | 62.71 (37/59) |
| Prefix step 203 | 64.29 (36/56) | **64.94** (50/77) | **69.72** (76/109) | **71.19** (42/59) |
| Prefix largelr step 203 | 50.00 (28/56) | 57.14 (44/77) | 60.55 (66/109) | 64.41 (38/59) |
| Prefix largelr step 180 | **66.07** (37/56) | 62.34 (48/77) | 68.81 (75/109) | 59.32 (35/59) |
| Hybrid Prefix step 180 | 58.93 (33/56) | 64.94 (50/77) | 66.97 (73/109) | 67.80 (40/59) |
| Hybrid Prefix step 200 | 60.71 (34/56) | **67.53** (52/77) | **72.48** (79/109) | 66.10 (39/59) |
| Hybrid Prefix step 203 | 57.14 (32/56) | 59.74 (46/77) | 66.06 (72/109) | 69.49 (41/59) |

LifelongOS：

| Method | `gpasswd` | `chgrp` | `addgroup` | `useradd` |
|---|---:|---:|---:|---:|
| Full-FT (08-13) | **14.29** (5/35) | 20.39 (31/152) | 24.58 (29/118) | 28.13 (27/96) |
| Base native (08-20) | 8.57 (3/35) | 24.34 (37/152) | 25.42 (30/118) | **32.29** (31/96) |
| LoRA step 200 | 8.57 (3/35) | **26.97** (41/152) | 24.58 (29/118) | 29.17 (28/96) |
| Prefix step 203 | 5.71 (2/35) | 23.03 (35/152) | 23.73 (28/118) | 26.04 (25/96) |
| Prefix largelr step 203 | 8.57 (3/35) | 23.03 (35/152) | 23.73 (28/118) | 30.21 (29/96) |
| Prefix largelr step 180 | 8.57 (3/35) | 25.00 (38/152) | **29.66** (35/118) | 31.25 (30/96) |
| Hybrid Prefix step 180 | 5.71 (2/35) | 23.03 (35/152) | 22.03 (26/118) | 28.12 (27/96) |
| Hybrid Prefix step 200 | 5.71 (2/35) | 22.37 (34/152) | 22.88 (27/118) | 23.96 (23/96) |
| Hybrid Prefix step 203 | 5.71 (2/35) | 20.39 (31/152) | 22.03 (26/118) | 26.04 (25/96) |

Prefix 对 DB 的 DELETE、嵌套查询和复杂 WHERE 有稳定收益；largelr step 180 在 `delete`/`table_alias`/`addgroup` 上接近或超过常规 run，说明高 LR 的技能收益一度存在，但到 step 203 又回落（step 203 的 DB 弱技能被抹平到 Full-FT/LoRA 水平）。Hybrid step 200 的 `subquery_nested`（67.53）与 `table_alias`（72.48）为全表最高，DB 弱技能收益与常规 run 相当或更好；其 OS 弱技能与其它 prefix 变体一样疲软（`gpasswd` 三步均为 5.71）。OS 的用户/组/权限管理整体没有对应提升。OS 后续训练更适合加入 `id`、`getent`、`stat`、`readlink` 等执行后验证轨迹，并强调在最后一步调用 `finish`。

## 5. 结论

1. **Prefix 系方法最适合当前的数据库与 single-turn function-calling 目标。** Independent 变体取得最佳 BFCL Non-Live（81.87），Hybrid 变体取得最佳 Live（78.23）、Hall.（83.77）与最佳 LifelongDB（87.60，step 200），两者在同 judge 的 tau2 对比中均明显超过原生 Base；Hybrid 的 BFCL Overall（68.89，step 203）追平 LoRA 的全表最高。
2. **LoRA 的优势集中在 LifelongOS，BFCL Overall 已被追平。** 其 LifelongOS 46.60% 与 largelr step 180 并列最高；BFCL Overall 68.89% 与 Hybrid step 203 并列最高。
3. **Multi-turn 是 LoRA/Prefix 相对 Full-FT 的短板。** 两者的 BFCL Multi-Turn 均未超过 Full-FT 的 51.12（原生 Base 为 44.50），提示全参数更新对多轮能力有帮助而 LoRA/Prefix 未能复制；`Miss Param` 尤其弱。
4. **08-13 的"Base"实为 Full-FT step 200，原生基线只有 08-20 一次。** 原先"两次 Base 差值=复现波动"的解读作废：两者 BFCL Overall 仅差 0.04 pp，但那是 Full-FT 与原生模型恰好接近，不代表评测稳定；Multi-Turn 上两者相差 6.62 pp 属于方法差异。原生模型目前没有重复评测。
5. **下一轮训练建议拆分目标。** DB/单轮 tool use 可优先沿用 Prefix；OS/多步骤 agent 可针对权限管理、执行后验证和 finish 时机单独构造数据，而不是只扩大通用 function-calling 数据。
6. **largelr（5e-5）训练后期确认退化，但问题不止"训多了"。** 同一 largelr run，step 180 -> step 203 在 5 个 benchmark 家族中 4 个下滑：BFCL Overall 66.50 -> 65.19、Hall. 78.76 -> 75.09、LifelongDB 85.20 -> 84.00、LifelongOS 46.60 -> 43.40（tau2 Airline 也 -4.00 pp，仅 Retail +4.39 pp），且训练侧 on-policy score 平稳（~0.80）、distill loss 缓慢下降并无告警，符合"held-out 退化而训练指标无感"的过拟合特征。但更关键的是：即使取较优的 step 180，largelr 的 BFCL Overall（66.50）、tau2 Overall（53.66）、LifelongDB（85.20）仍全面低于 5e-6 常规 run 的 step 203（68.60 / 60.98 / 86.80），仅 LifelongOS（46.60）和 Macro Avg（65.90）占优。结论：5e-5 的主要问题是 LR 本身偏大，后期过拟合是次生问题；prefix LR 应保持 5e-6 量级，或配合更早停止（< step 180）/更强正则再试高 LR。
7. **Hybrid（ΔKV + full-attention residual prefix）在 5e-6 下训练稳定，是当前最均衡的 prefix 变体。** 三个 checkpoint（180/200/203）的 BFCL Overall 稳定在 68.62-68.89（step 203 追平 LoRA 全表最高），single-turn AST 子类达到或接近全表最佳，LifelongDB 在 step 200 创全表新高 87.60，tau2 Retail step 200 亦为全表最高（58.77）；与同 LR 的 Independent 常规 run 相比，tau2 Overall 略低（59.76 vs 60.98）、LifelongOS 无优势。给 full-attention 层加 prefix 的增益集中在 single-turn AST 与 DB 技能，未改善 multi-turn 与 OS。

## 6. 正式产物

服务器根目录：`/mnt/storage/disk3/self_evolver`

### 6.1 Full-FT step 200（08-13，曾误记为 Base）

- 评测时端点实际加载的模型：`checkpoints/self_evolver_opsd/qwen3_5_4b_awm_opsd_fix_context/global_step_200/actor/huggingface_merged/`
- 训练脚本：`verl/examples/on_policy_distillation_trainer/run_qwen3_5_4b_awm_opsd_full.sh`（FSDP2 world size 6，run name `qwen3_5_4b_awm_opsd_fix_context`）
- tau2 Airline 主运行：`workspace/eval_logs/tau2/airline/qwen35_4b_thinkoff_full_20260813_130315/results.json`
- tau2 Airline task 42 补跑：`workspace/eval_logs/tau2/airline/qwen35_4b_thinkoff_retry42_20260813_1319/results.json`
- tau2 Retail：`workspace/eval_logs/tau2/retail/qwen35_4b_thinkoff_full_clean_20260813_1319/results.json`
- BFCL v3：`workspace/eval_logs/bfcl_v3/all/20260813_120100/`
- LifelongDB：`workspace/eval_logs/lifelong_db/test/qwen35_4b_full_20260813_130944/`
- LifelongOS：`workspace/eval_logs/lifelong_os/test/qwen35_4b_full_clean_20260813_132859/`

### 6.2 Base native (08-20)

- 评测权重：`/mnt/storage/disk1/verl_data/base_model/Qwen3.5-4B`（原生，未改动）
- tau2 Airline：`workspace/eval_logs/tau2/airline/native_qwen35_4b_rerun_20260820/results.json`
- BFCL v3：`workspace/eval_logs/bfcl_v3/all/native_qwen35_4b_rerun_20260820/`
- LifelongDB：`workspace/eval_logs/lifelong_db/test/native_qwen35_4b_rerun_20260820/`
- LifelongOS：`workspace/eval_logs/lifelong_os/test/native_qwen35_4b_rerun_20260820/`

### 6.3 LoRA step 200

- 合并模型：`checkpoints/self_evolver_opsd/qwen3_5_4b_awm_opsd_lora_r64_a128/global_step_200/actor/huggingface_merged/`
- tau2 Airline 主运行：`workspace/eval_logs/tau2/airline/lora_r64_a128_step200_full_20260817_1120/results.json`
- tau2 Airline task 25 补跑：`workspace/eval_logs/tau2/airline/lora_r64_a128_step200_retry25_20260817_1142/results.json`
- tau2 Retail 主运行：`workspace/eval_logs/tau2/retail/lora_r64_a128_step200_full_20260817_1120/results.json`
- tau2 Retail task 23 补跑：`workspace/eval_logs/tau2/retail/lora_r64_a128_step200_retry23_20260817_1142/results.json`
- BFCL v3：`workspace/eval_logs/bfcl_v3/all/lora_r64_a128_step200_20260817_1120/`
- LifelongDB：`workspace/eval_logs/lifelong_db/test/lora_r64_a128_step200_full_20260817_1120/`
- LifelongOS：`workspace/eval_logs/lifelong_os/test/lora_r64_a128_step200_full_20260817_1120/`

### 6.4 Independent ΔKV Prefix step 203

- 输入 checkpoint：`checkpoints/self_evolver_opsd/qwen3_5_4b_awm_opsd_independent_delta_kv_prefix_m2048/global_step_203/actor/`
- CPU 合并模型：`/mnt/storage/disk1/verl_data/base_model/Qwen3.5-4B-IndependentDeltaKVPrefix-step203/`
- 合并清单：`/mnt/storage/disk1/verl_data/base_model/Qwen3.5-4B-IndependentDeltaKVPrefix-step203/prefix_merge_manifest.json`
- tau2 Airline：`workspace/eval_logs/tau2/airline/independent_delta_kv_prefix_m2048_step203_20260820/results.json`
- tau2 Retail：`workspace/eval_logs/tau2/retail/independent_delta_kv_prefix_m2048_step203_20260820/results.json`
- BFCL v3：`workspace/eval_logs/bfcl_v3/all/independent_delta_kv_prefix_m2048_step203_20260820/`
- LifelongDB：`workspace/eval_logs/lifelong_db/test/independent_delta_kv_prefix_m2048_step203_20260820/`
- LifelongOS：`workspace/eval_logs/lifelong_os/test/independent_delta_kv_prefix_m2048_step203_20260820/`

### 6.5 Independent ΔKV Prefix largelr step 203

- 输入 checkpoint：`checkpoints/self_evolver_opsd/qwen3_5_4b_awm_opsd_independent_delta_kv_prefix_largelr/global_step_203/actor/`
- 训练脚本：`verl/examples/on_policy_distillation_trainer/run_qwen3_5_4b_awm_opsd_independent_delta_kv_prefix_m2048.sh`（`ACTOR_LR=5e-5`，run name `qwen3_5_4b_awm_opsd_independent_delta_kv_prefix_largelr`）
- CPU 合并模型：`/mnt/storage/disk1/verl_data/base_model/Qwen3.5-4B-IndependentDeltaKVPrefix-largelr-step203/`
- 合并清单：`/mnt/storage/disk1/verl_data/base_model/Qwen3.5-4B-IndependentDeltaKVPrefix-largelr-step203/prefix_merge_manifest.json`
- tau2 Airline：`workspace/eval_logs/tau2/airline/independent_delta_kv_prefix_largelr_step203_20260821/results.json`
- tau2 Retail：`workspace/eval_logs/tau2/retail/independent_delta_kv_prefix_largelr_step203_20260821/results.json`
- BFCL v3：`workspace/eval_logs/bfcl_v3/all/independent_delta_kv_prefix_largelr_step203_20260821/`
- LifelongDB：`workspace/eval_logs/lifelong_db/test/independent_delta_kv_prefix_largelr_step203_20260821/`
- LifelongOS：`workspace/eval_logs/lifelong_os/test/independent_delta_kv_prefix_largelr_step203_20260821/`

### 6.6 Independent ΔKV Prefix largelr step 180（过拟合复查）

- 输入 checkpoint：`checkpoints/self_evolver_opsd/qwen3_5_4b_awm_opsd_independent_delta_kv_prefix_largelr/global_step_180/actor/`
- CPU 合并模型：`/mnt/storage/disk1/verl_data/base_model/Qwen3.5-4B-IndependentDeltaKVPrefix-largelr-step180/`（含 `prefix_merge_manifest.json`）
- 训练日志（用于核对 on-policy score / distill loss 曲线）：`wandb/run-20260820_123834-3dcqorfr/files/output.log`
- tau2 Airline：`workspace/eval_logs/tau2/airline/independent_delta_kv_prefix_largelr_step180_20260821/results.json`
- tau2 Retail：`workspace/eval_logs/tau2/retail/independent_delta_kv_prefix_largelr_step180_20260821/results.json`
- BFCL v3：`workspace/eval_logs/bfcl_v3/all/independent_delta_kv_prefix_largelr_step180_20260821/`
- LifelongDB：`workspace/eval_logs/lifelong_db/test/independent_delta_kv_prefix_largelr_step180_20260821/`
- LifelongOS：`workspace/eval_logs/lifelong_os/test/independent_delta_kv_prefix_largelr_step180_20260821/`

### 6.7 Hybrid Δ + Residual Attention Prefix G2048/A256（step 180/200/203）

- 输入 checkpoint：`checkpoints/self_evolver_opsd/qwen3_5_4b_awm_opsd_hybrid_delta_residual_attention_prefix_g2048_a256/global_step_{180,200,203}/actor/`（run 内仅这三个 step 含完整 actor 权重，其余 step 目录只有 `data.pt`）
- 训练脚本：`verl/examples/on_policy_distillation_trainer/run_qwen3_5_4b_awm_opsd_hybrid_delta_residual_attention_prefix_g2048_a256.sh`（prefix LR `5e-6`，FSDP2 world size 6）
- CPU 导出模型：`/mnt/storage/disk1/verl_data/base_model/Qwen3.5-4B-HybridDeltaResidualAttentionPrefix-G2048-A256-step{180,200,203}/`（96 个 ΔKV tensors + 16 个 attention tensors，共 315,621,376 个 prefix 参数，各含 `prefix_merge_manifest.json`）
- vLLM 服务日志：`workspace/eval_logs/vllm_server/hybrid_prefix_g2048_a256_step{180,200,203}_vllm.log`
- tau2 Airline：`workspace/eval_logs/tau2/airline/hybrid_delta_residual_attention_prefix_g2048_a256_step{180,200,203}_20260822/results.json`
- tau2 Retail：`workspace/eval_logs/tau2/retail/hybrid_delta_residual_attention_prefix_g2048_a256_step{180,200,203}_20260822/results.json`
- BFCL v3：`workspace/eval_logs/bfcl_v3/all/hybrid_delta_residual_attention_prefix_g2048_a256_step{180,200,203}_20260822/`
- LifelongDB：`workspace/eval_logs/lifelong_db/test/hybrid_delta_residual_attention_prefix_g2048_a256_step{180,200,203}_20260822/`
- LifelongOS：`workspace/eval_logs/lifelong_os/test/hybrid_delta_residual_attention_prefix_g2048_a256_step{180,200,203}_20260822/`

每个 Lifelong 目录包含 `summary.csv` 与 `trajectories/*.json`；tau2 `results.json` 包含任务、完整对话、reward breakdown 和终止原因；BFCL `result/` 保存生成结果，`score/` 保存逐类别判分与官方汇总 CSV。

## 7. 运行与补跑记录

- Full-FT (08-13) tau2 Airline 主运行缺 task 42，后以 concurrency=1 补跑；最终 50 个 task ID 完整且无重复。
- Full-FT (08-13) LifelongOS 的早期运行 `qwen35_4b_full_20260813_132600` 因镜像 apt 索引问题被排除；修复后从头得到 `full_clean` 500 条正式结果。
- LoRA tau2 主运行分别缺 Airline task 25 与 Retail task 23，均以 concurrency=1 补跑；聚合时按 task ID 去重，最终 50/50 与 114/114 完整。
- Prefix 和 Base native (08-20) 的 BFCL、tau2、LifelongDB、LifelongOS 均一次完整结束，无 task 级补跑。
- Prefix largelr (08-21) 的 BFCL、tau2（Airline 50/50、Retail 114/114 完整）、LifelongDB、LifelongOS 均一次完整结束，无 task 级补跑，Lifelong 两个子集 0 runtime errors。
- Prefix largelr step 180 复查（08-21 下午）同样四项全量一次完整结束：tau2 50/50 与 114/114 完整，Lifelong 两个子集 500/500、0 runtime errors，无补跑。
- Hybrid Prefix（08-22）三个 step 各自四项全量一次完整结束（step 180/200/203 依次评测，每个 step 重启一次 vLLM 服务）：tau2 均 50/50 与 114/114 完整，Lifelong 两个子集均 500/500、0 runtime errors，无补跑。
- BFCL multi-turn 中的空响应或解析失败由官方 runner 捕获并按失败计分，没有人工修改生成或 score。

## 8. 版本信息

- **2026-08-21 勘误**：核实 08-13 评测的 vLLM 端点实际加载的是全参数微调 checkpoint `qwen3_5_4b_awm_opsd_fix_context/global_step_200/actor/huggingface_merged/`（该导出完成于 08-13 11:50，BFCL 评测 12:01 启动），而非原生 `/mnt/storage/disk1/verl_data/base_model/Qwen3.5-4B`（自 06-07 下载后未改动）。全文原 "Base (08-13)" 改为 "Full-FT (step 200, 08-13)"，"Base rerun (08-20)" 改为 "Base native (08-20)"；数值本身不变，仅口径与解读更正。
- 2026-08-21 largelr 评测时 self_evolver HEAD：`c2ae941cdfb40c49645a762fc80140e65d613c17`（同 08-20），worktree 含既有本地修改；vLLM/评测管线与 08-20 完全一致，仅更换加载权重与 served model ID。
- 2026-08-22 Hybrid 评测时 self_evolver HEAD 同为 `c2ae941cdfb40c49645a762fc80140e65d613c17`（worktree 含 hybrid prefix 实现的本地未提交文件）；vLLM/评测管线与之前完全一致，仅更换加载权重与 served model ID。Hybrid 使用专用导出脚本 `prefix_tuning/virtual_prefix/export_hybrid_delta_residual_attention_prefix_checkpoint.py`（统一 merger 尚未收录该类型）。
- 2026-08-20 评测时 self_evolver HEAD：`c2ae941cdfb40c49645a762fc80140e65d613c17`，worktree 含既有本地修改。
- 2026-08-13 报告记录的 self_evolver base commit：`39a7cead4374dbf4744d4e0ea59f93b62e130939`。
- tau2 official commit：`363133ada1936491fb5bcec33cd62c3518a99f65`。
- LifelongAgentBench 上游快照记录：`d6f19b42eb358d9150379f0c68c2985c5a867520`。
