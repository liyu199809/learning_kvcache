# Benchmark 实验报告（qwen3.5-4b 基线 + LoRA step 200）

- 日期：基线 2026-08-13；LoRA 追加评测 2026-08-17（Asia/Shanghai）
- 基线模型服务：tau2/Lifelong 使用 `qwen3.5-4b`；BFCL 基线产物标签为 `Qwen3-4B (FC)`
- LoRA 模型服务：`qwen3.5-4b-lora-r64-a128-step200`，OpenAI-compatible vLLM endpoint `http://127.0.0.1:8000/v1`
- 评测范围：tau2-bench（Airline + Retail，关闭 thinking）、BFCL v3 all、LifelongDB test、LifelongOS test
- tau2/Lifelong 采样：temperature=0，top_p=1，单次 trial（pass@1）；BFCL 使用其产物中的官方 score 汇总

## 1. LoRA step 200 追加评测

### 1.1 Checkpoint 合并与服务

- 输入 checkpoint：`checkpoints/self_evolver_opsd/qwen3_5_4b_awm_opsd_lora_r64_a128/global_step_200/actor/`。
- 该 checkpoint 是 FSDP2、world size 6 的 LoRA-only 分片；LoRA 配置为 rank 64、alpha 128，base model 为 `/mnt/storage/disk1/verl_data/base_model/Qwen3.5-4B`。
- 先用 VERL merger 导出 PEFT adapter，再使用 PEFT `merge_and_unload(safe_merge=True)` 合入 base model。最终模型位于 `checkpoints/self_evolver_opsd/qwen3_5_4b_awm_opsd_lora_r64_a128/global_step_200/actor/huggingface_merged/`，约 8.5 GiB、4,539,265,536 parameters。
- 已用抽样权重验证合并结果满足 `base + (alpha/r) × B@A`（仅有 bfloat16 舍入误差），并通过 Transformers 元数据加载。
- 使用 `start_services.sh start vllm` 启动，served model name 为 `qwen3.5-4b-lora-r64-a128-step200`。评测结束后 vLLM 仍在运行，PID 1281906，port 8000，health=ok。
- 服务 smoke test 已验证普通文本与 tool call 均可用，且 reasoning 字段为空；tau2 的 agent、user simulator 和 judge 均关闭 thinking。

### 1.2 与基线对比

| Benchmark | 子集 | 基线 | LoRA step 200 | 变化 |
|---|---|---:|---:|---:|
| tau2-bench | Airline | 34/50（68.00%） | 32/50（64.00%） | -4.00 pp |
| tau2-bench | Retail | 61/114（53.51%） | 69/114（60.53%） | +7.02 pp |
| tau2-bench | 加权汇总 | 95/164（57.93%） | 101/164（61.59%） | +3.66 pp |
| BFCL v3 | Overall Acc | 67.59% | 68.89% | +1.30 pp |
| LifelongDB | test | 425/500（85.00%） | 416/500（83.20%） | -1.80 pp |
| LifelongOS | test | 214/500（42.80%） | 233/500（46.60%） | +3.80 pp |

LoRA 对 Retail、BFCL 和 LifelongOS 有正向收益，尤其是 Retail 与 OS；Airline 和 LifelongDB 略有回落。按 tau2 两个 domain 的样本数加权，整体提升 3.66 个百分点。

BFCL 对比需要额外谨慎：两次官方 score 文件都显示处理器标签 `Qwen3-4B (FC)`。本次生成请求实际发送到 LoRA endpoint，服务返回的模型根目录也已核对为上述 `huggingface_merged`；但旧 BFCL 产物仅凭标签无法证明与 tau2/Lifelong 基线 checkpoint 完全相同。因此 BFCL 的 +1.30 pp 可作为产物级对比，不应当作严格受控的同基座消融。

### 1.3 tau2-bench（LoRA，thinking 关闭）

配置与基线一致：base split，1 trial，最大 50 steps，主运行 32 并发，agent/user 最大 4096 completion tokens；补跑使用 concurrency=1。

- Airline：32/50，pass@1=64.00%。正常停止 43 条，达到 50-step 上限 7 条；DB reward 33/43（76.74%），communicate checks 5/7（71.43%）；read action 63/65（96.92%），write action 33/43（76.74%）。18 条失败包括 DB only 9、communicate only 1、DB + communicate 1、max steps 7。
- Retail：69/114，pass@1=60.53%。正常停止 107 条，达到 50-step 上限 7 条；DB reward 70/107（65.42%），NL assertions 49/57（85.96%），communicate checks 55/58（94.83%）；read action 331/351（94.30%），write action 112/158（70.89%）。45 条失败包括 DB only 31、DB + NL assertion 6、NL assertion only 1、max steps 7。
- 加权汇总：101/164，pass@1=61.59%。Airline 主运行因单条 HTTP 请求长期挂起保存了 49 条，Retail 主运行保存了 113 条；分别单独补跑 task 25 和 task 23 后，两个 domain 的 task ID 均完整且无重复。task 25 得分 0，task 23 得分 1。

### 1.4 BFCL v3（LoRA）

| 评测组 | 基线 | LoRA step 200 | 变化 |
|---|---:|---:|---:|
| Overall Acc | 67.59% | 68.89% | +1.30 pp |
| Non-Live Overall Acc | 75.37% | 80.15% | +4.78 pp |
| Live Acc | 76.28% | 76.63% | +0.35 pp |
| Multi-Turn Acc | 51.12% | 49.88% | -1.24 pp |

LoRA 的 Non-Live Overall Acc 为 80.15%，其中 AST Summary 从基线 73.17% 提升到 78.21%（+5.04 pp）；Simple 63.83%、Multiple 92.50%、Parallel 72.00%、Parallel Multiple 84.50%，Python/Java/JavaScript Simple 分别为 86.50%/53.00%/52.00%。Live 的 Simple/Multiple/Parallel/Parallel Multiple 分别为 70.54%/76.83%/56.25%/70.83%。Multi-Turn 中 Base 60.00%、Miss Function 51.00%、Miss Parameter 37.50%、Long Context 51.00%；多轮总体相对基线略降，缺参恢复仍是主要短板。

### 1.5 Lifelong（LoRA）

- LifelongDB：416/500，pass@1=83.20%，0 runtime errors。平均 2.628 steps；成功样本 2.308，失败样本 4.214。84 条失败中，最后动作为 `submit` 51 条、`execute` 30 条，另有 3 条未形成标准 action；36 条失败用满 6 steps。较弱技能仍包括 `delete` 28/56（50.00%）、`subquery_nested` 41/77（53.25%）、`table_alias` 67/109（61.47%）、`where_nested_conditions` 37/59（62.71%）。
- LifelongOS：233/500，pass@1=46.60%，0 runtime errors。平均 6.454 steps；成功样本 5.777，失败样本 7.045。267 条失败中，最后仍为 `execute` 143 条、已调用 `finish` 124 条；183 条失败用满 8 steps。较弱技能包括 `gpasswd` 3/35（8.57%）、`addgroup` 29/118（24.58%）、`chgrp` 41/152（26.97%）、`useradd` 28/96（29.17%）。

LoRA 没有改变两个 Lifelong 数据集的主要错误形态：DB 仍受精确提交格式、嵌套查询和数据库状态影响；OS 仍集中在用户/组/权限管理及多步骤任务收尾。但 OS 的绝对通过数增加 19 条，是本轮最明确的泛化收益之一。

## 2. 基线结论汇总（2026-08-13）

| Benchmark | 子集 | 通过/总数 | 分数 | Runtime errors |
|---|---:|---:|---:|---:|
| tau2-bench | Airline | 34/50 | 68.00% | 0 |
| tau2-bench | Retail | 61/114 | 53.51% | 0 |
| tau2-bench | Airline + Retail（加权汇总） | 95/164 | 57.93% | 0 |
| BFCL v3 (`Qwen3-4B FC`) | all（官方综合准确率） | — | 67.59% | — |
| LifelongDB | test | 425/500 | 85.00% | 0 |
| LifelongOS | test | 214/500 | 42.80% | 0 |

`qwen3.5-4b` 在结构化 SQL 任务上表现最好；tau2 多轮工具交互居中；LifelongOS 明显最弱，主要瓶颈是复杂 Linux 状态变更任务以及在步数上限前完成并调用 `finish`。BFCL 的 `Qwen3-4B (FC)` 综合准确率为 67.59%，单轮/Live function calling 较强，多轮 function calling 明显较弱。由于 BFCL 结果的模型标签不同，不应与 tau2/Lifelong 结果视作同一个 checkpoint 的横向能力切片。

## 3. 基线 tau2-bench（thinking 关闭）

配置：base split，1 trial，最大 50 steps，32 并发，agent/user 最大 4096 completion tokens。Agent、user simulator 和 Retail 自然语言断言 judge 均通过 `enable_thinking=false` 关闭 thinking。

### Airline

- 最终：34/50，pass@1=68.00%。
- 正常停止 46 条；达到 50-step 上限 4 条。
- DB match：34/46（73.91%，达到 max steps 的 4 条没有 DB 判分）。
- Communicate checks：7/10（70.00%）。
- 工具动作匹配：read 77/82（93.90%），write 31/48（64.58%）。
- 16 条失败构成：DB only 9；DB + communicate 3；max steps 4。
- 首次全量有任务 42 的 HTTP 请求异常挂起，因此原目录保存 49 条；任务 42 后续单独补跑完成且得分 0，最终 50 条 task ID 完整、无重复。

### Retail

- 最终：61/114，pass@1=53.51%。
- 正常停止 106 条；达到 50-step 上限 8 条。
- DB match：63/106（59.43%）。
- NL assertions：46/57（80.70%）。
- Communicate checks：49/56（87.50%）。
- 工具动作匹配：read 297/322（92.24%），write 111/160（69.38%）。
- 53 条失败构成：DB only 38；DB + NL assertion 5；NL assertion only 2；max steps 8。

### 可比性说明

本次只提供了本地 `qwen3.5-4b` 服务，因此 tau2 的 user simulator 和 Retail NL-assertion judge 也路由到该模型；官方源码默认二者使用 GPT-4.1。故本报告适合本地模型迭代对比，但 Retail 分数不能直接与使用官方 GPT-4.1 simulator/judge 的 leaderboard 数值横向比较。Airline 本次没有 NL assertion，受 judge 差异影响较小，但 user simulator 仍不同。

## 4. 基线 BFCL v3

BFCL 原始 score 目录中的官方模型标签为 `Qwen3-4B (FC)`。总体分数是 BFCL 官方按类别聚合得到的准确率，不是简单的“正确数/全部底层判分单元”微平均，因此汇总表不填写统一分子和分母。

| 评测组 | 官方分数 |
|---|---:|
| Overall Acc | 67.59% |
| Non-Live Overall Acc | 75.37% |
| Live Overall Acc | 76.28% |
| Multi-Turn Overall Acc | 51.12% |

### Non-Live

- AST Summary：73.17%。
- Simple AST：57.67%；其中 Python 83.00%（332/400）、Java 44.00%（44/100）、JavaScript 46.00%（23/50）。
- Multiple AST：94.00%（188/200）。
- Parallel AST：62.50%（125/200）。
- Parallel Multiple AST：78.50%（157/200）。
- Irrelevance Detection：84.17%（202/240）。

### Live

- Overall Acc：76.28%；AST Summary：75.28%。
- Simple AST：70.54%（182/258）。
- Multiple AST：77.02%（811/1053）。
- Parallel AST：50.00%（8/16）。
- Parallel Multiple AST：66.67%（16/24）。
- Irrelevance Detection：77.78%（686/882）。
- Relevance Detection：77.78%（14/18）。

### Multi-Turn

- Overall Acc：51.12%。
- Base：62.50%（125/200）。
- Miss Function：52.50%（105/200）。
- Miss Parameter：35.00%（70/200），是 BFCL 最明显的短板。
- Long Context：54.50%（109/200）。

BFCL 的核心改进方向是 multi-turn 参数缺失恢复、长上下文中的函数状态维护，以及 Java/JavaScript simple function calling。优势集中在 Non-Live Multiple AST 和 Python Simple AST。

## 5. 基线 LifelongDB

配置：test 500 条，最大 6 steps，16 并发，MySQL 8.0，最大 2048 completion tokens；成功由环境精确判分。

- 最终：425/500，pass@1=85.00%，0 runtime errors。
- 平均 steps：全部 2.604；成功样本 2.351；失败样本 4.040。
- 75 条失败中：58 条已 `submit` 但结果/数据库状态不匹配；17 条用满 6 steps 仍停留在 `execute`、未提交。
- 较弱技能（标签可重叠，括号为成功/总数）：`delete` 28/56（50.00%）、`subquery_nested` 41/77（53.25%）、`where_nested_conditions` 37/59（62.71%）、`table_alias` 69/109（63.30%）、`subquery_multiple` 38/57（66.67%）。
- 较强技能：`order_by_multiple_columns_same_direction` 56/57（98.25%）、`having_single_condition_with_aggregate` 96/100（96.00%）、`order_by_single_column` 49/52（94.23%）、`group_by_single_column` 110/117（94.02%）。

主要改进方向是嵌套/多子查询、DELETE，以及提交结果的精确格式与最终数据库状态验证。

## 6. 基线 LifelongOS

配置：test 500 条，最大 8 steps，16 并发，单条 shell 命令超时 20 秒，LLM step 超时 180 秒，最大 2048 completion tokens；成功由容器内隐藏检查脚本判定。

- 最终：214/500，pass@1=42.80%，0 runtime errors。
- 平均 steps：全部 6.308；成功样本 5.528；失败样本 6.892。
- 219/500 条运行到第 8 step；失败样本中 178/286 条运行到第 8 step。
- 286 条失败中：144 条达到上限时最后仍为 `execute`、未调用 `finish`；142 条调用了 `finish`，但隐藏环境检查未通过。
- 最弱技能（标签可重叠）：`gpasswd` 5/35（14.29%）、`chage` 5/32（15.63%）、`chgrp` 31/152（20.39%）、`usermod` 30/129（23.26%）、`addgroup` 29/118（24.58%）、`useradd` 27/96（28.13%）。
- 相对较强技能：`sleep` 35/48（72.92%）、`exit` 30/45（66.67%）、`chsh` 25/43（58.14%）、`rm` 56/102（54.90%）。

核心短板集中在用户/组/权限生命周期管理，以及多步骤任务的收尾验证。建议训练时加入“执行后检查（`stat`/`id`/`getent`/`readlink` 等）再 finish”的轨迹，并针对 user/group 管理命令增加高覆盖样本。

## 7. 运行环境与版本

- 8 × NVIDIA A800-SXM4-80GB（每卡 81920 MiB）
- vLLM 0.18.0，data parallel size=8
- Python 3.11.15
- PyTorch 2.10.0+cu128，CUDA runtime 12.8
- Docker 29.1.3
- self_evolver base commit：`39a7cead4374dbf4744d4e0ea59f93b62e130939`（评测时 worktree 含既有本地改动）
- tau2 official commit：`363133ada1936491fb5bcec33cd62c3518a99f65`
- LifelongAgentBench commit：`d6f19b42eb358d9150379f0c68c2985c5a867520`

本次评测新增两项本地集成调整：tau2 允许通过环境变量覆盖 NL-assertion judge 并由统一入口传入本地模型/think-off 参数；LifelongOS Dockerfile 仅将 Ubuntu apt 源替换为清华镜像以完成依赖安装，任务所需 apt 索引予以保留。

## 8. 正式原始产物

服务器根目录：`/mnt/storage/disk3/self_evolver`

LoRA step 200：

- 合并模型：`checkpoints/self_evolver_opsd/qwen3_5_4b_awm_opsd_lora_r64_a128/global_step_200/actor/huggingface_merged/`
- tau2 Airline 主运行（49 条）：`workspace/eval_logs/tau2/airline/lora_r64_a128_step200_full_20260817_1120/results.json`
- tau2 Airline task 25 补跑：`workspace/eval_logs/tau2/airline/lora_r64_a128_step200_retry25_20260817_1142/results.json`
- tau2 Retail 主运行（113 条）：`workspace/eval_logs/tau2/retail/lora_r64_a128_step200_full_20260817_1120/results.json`
- tau2 Retail task 23 补跑：`workspace/eval_logs/tau2/retail/lora_r64_a128_step200_retry23_20260817_1142/results.json`
- BFCL v3 生成结果与 score：`workspace/eval_logs/bfcl_v3/all/lora_r64_a128_step200_20260817_1120/`
- LifelongDB：`workspace/eval_logs/lifelong_db/test/lora_r64_a128_step200_full_20260817_1120/`
- LifelongOS：`workspace/eval_logs/lifelong_os/test/lora_r64_a128_step200_full_20260817_1120/`

基线：

- tau2 Airline 主运行（49 条）：`workspace/eval_logs/tau2/airline/qwen35_4b_thinkoff_full_20260813_130315/results.json`
- tau2 Airline task 42 补跑：`workspace/eval_logs/tau2/airline/qwen35_4b_thinkoff_retry42_20260813_1319/results.json`
- tau2 Retail：`workspace/eval_logs/tau2/retail/qwen35_4b_thinkoff_full_clean_20260813_1319/results.json`
- BFCL v3 原始生成结果：`workspace/eval_logs/bfcl_v3/all/20260813_120100/result/Qwen_Qwen3-4B-FC/`
- BFCL v3 官方 score 与汇总 CSV：`workspace/eval_logs/bfcl_v3/all/20260813_120100/score/`
- LifelongDB：`workspace/eval_logs/lifelong_db/test/qwen35_4b_full_20260813_130944/`
- LifelongOS：`workspace/eval_logs/lifelong_os/test/qwen35_4b_full_clean_20260813_132859/`

每个 Lifelong 目录包含 `summary.csv` 和逐任务 `trajectories/*.json`；tau2 `results.json` 包含任务、完整对话轨迹、reward breakdown 和终止原因；BFCL `result/` 保存逐条生成结果，`score/` 保存逐类别判分与 `data_overall.csv` 等官方汇总。

## 9. 排除的运行与补跑说明

- tau2 Retail 初次运行因 NL judge 仍指向官方默认 GPT-4.1 而中止；修复为本地 judge 后从头重跑，初次部分结果不计分。
- LifelongOS `qwen35_4b_full_20260813_132600` 在 87 条时发现镜像 apt 索引被清理，造成 `os_90` 初始化无法安装 zsh；该部分运行已排除。恢复 apt 索引后先验证 `os_90` 通过，再从头生成上述 `full_clean` 500 条正式结果。
- LoRA tau2 主运行各有一条请求在高并发 BFCL/Lifelong 同时运行期间长期挂起。终止两个 tau2 主进程后，仅对缺失的 Airline task 25 与 Retail task 23 以 concurrency=1 补跑；最终聚合严格按 task ID 去重，50/50 与 114/114 均完整。
