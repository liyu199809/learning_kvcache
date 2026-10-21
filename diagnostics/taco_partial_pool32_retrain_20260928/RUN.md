# TACO partial rollout 全量重训

2026-09-28 在 `gpu-h200:/home/liyu.468/learning_kvcache` 启动独立新训练。原始模型初始化、数据沿用之前三域实验；不恢复旧 checkpoint。已核对实际配置为 batch/mini-batch 24、micro-batch 1、epoch 1、pool 32、resume disable，过滤后仍为 1591 条，trainer 确认总步数 66。三个 student 推理副本已开始处理训练前验证样本。下方为启动记录，训练完成状态以服务器 `status.json` 为准。

| 配置 | 本次设置 |
|---|---|
| 模型初始化 | `outputs/prefix_composition_3domain_20260917/prepared/hybrid-G512-A64` |
| prefix | GDN 512 / attention 64，112 个张量，78,905,344 个可训练参数 |
| actor/rollout | GPU 0–2，共卡 FSDP2 partial rollout |
| teacher | GPU 3，固定 Qwen3.5-4B，privileged append |
| 训练 batch / optimizer mini-batch | 24 / 24 |
| 每 GPU micro-batch | 1，3 卡各累积 8 个后统一更新一次 |
| rollout 池 | 32；完成并打分的 24 条进入训练，保留 8 条，补齐池缺口 |
| Thinking / 输出上限 | True / 16384 |
| 学习率 / 温度 | 1e-5 / 1.0 |
| 训练集 / epoch | 1591 条 TACO / 1 epoch，66 次更新（1584 条，余 7 条沿用 drop-last） |
| 验证集 / 周期 | 原 100 条验证集，训练前验证，每 20 步及最终验证 |
| checkpoint | 每 20 步及最终保存，结束后自动导出并检查 prefix |
| 日志 | 本地 console，当前服务器无 W&B 凭据 |

已保留纯蒸馏全局 loss 归一化修复；相关梯度累积和 partial rollout 测试此前共 34 项通过。本次未启用 batch 96 配置。

服务器实验目录：
`/home/liyu.468/learning_kvcache/outputs/taco_partial_pool32_retrain_20260928_1830`

- `status.json`：监督进程状态、最新更新步与指标、完成或失败原因。
- `launcher.log`：完整训练日志。
- `metrics.jsonl`：逐步指标。
- `checkpoints/taco_3actor1teacher_batch24_pool32/global_step_66/`：预计最终 checkpoint。
- `prefixes/taco/`：训练成功后自动导出的模型。
- `launch_config.json`：启动参数与训练代码 SHA256。

监督进程启动 PID：199298。它只清理该训练自身的子进程，不调用全局 `ray stop`。

查看状态：

```bash
ssh gpu-h200 'cat /home/liyu.468/learning_kvcache/outputs/taco_partial_pool32_retrain_20260928_1830/status.json'
```

查看日志：

```bash
ssh gpu-h200 'tail -n 30 /home/liyu.468/learning_kvcache/outputs/taco_partial_pool32_retrain_20260928_1830/launcher.log'
```

启动脚本：服务器 `diagnostics/taco_partial_pool32_retrain_20260928/train_taco.py`，本目录也保存一份。

## 2026-09-28 20:27 速度快照

本次已完成 12/66 步。比较旧同步训练与新 partial 训练的第 2–12 步，两者均使用 3 actor + 1 teacher、batch 24、16K 输出上限。首步、模型启动和验证均不计入下表。旧全程 61 个普通步的结果几乎相同（656.2 秒/步）。

| 指标 | 之前同步 | 本次 partial |
|---|---:|---:|
| 平均每步 | 656.6 秒 | 485.4 秒 |
| rollout / 轨迹后处理 / teacher 打分阶段 | 627.8 秒 | 456.3 秒 |
| actor 参数更新 | 16.4 秒 | 17.2 秒 |
| 样本吞吐（不含启动与验证） | 131.6 条/小时 | 178.0 条/小时 |
| 平均 response 长度 | 9105 token | 8489 token |

粗比每步耗时下降 26.1%，样本吞吐约 1.35 倍（+35.3%）。按实际生成的 response token 数归一后，token 吞吐提高约 26.1%。生成/取样计时包含 teacher 与后处理，不是纯 decode 速度。

这是历史粗对照：样本完成顺序、输出长度不同，本次还修正了蒸馏梯度归一化；旧日志的 GPU 型号没有核实，不能把全部差异归因于 partial 调度。只测到前 12 步，未证明最终训练效果改善。

原始汇总：`speed_comparison.json`。旧日志：`/home/liyu.468/learning_kvcache/outputs/prefix_composition_3domain_20260922_4gpu/logs/taco_4gpu_full_20260923_022329.log`。训练继续运行，本次比较未暂停、重启或修改训练。
