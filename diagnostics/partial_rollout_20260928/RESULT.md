# H200 共卡 partial rollout 接入记录

后续调整：当前默认已改为固定 rollout 池 32、训练 batch 24。本文保留第一版整批 warmup 和小 batch 验收记录；最新实现与实测见 [32 条池接入记录](../partial_rollout_32_20260928/RESULT.md)。

目标：`gpu-h200:/home/liyu.468/learning_kvcache`。服务器代码已修改；本地 `work/` 是审阅副本，`integration.patch` 包含代码与测试。

已终止 `/opt/occupy_all_gpus.py` 及其子进程，保留平台 Ray 和无关进程。所有验收进程已退出，四张 H200 均恢复为 4 MiB、利用率 0%；没有启动完整训练。

## 实现

默认 V1 `colocate_async`：actor 与 student rollout 共卡，生成、训练分阶段交替，teacher 使用独立卡。先提交一个 warmup batch；每步取完成且 teacher 已打分的轨迹组成 batch，其余暂停，更新 prefix 后续跑。不会把半条轨迹用于训练。

- 默认 `PARTIAL_WARMUP_BATCHES=1`、`PARTIAL_MAX_OFF_POLICY=8`、`PARTIAL_OFF_POLICY_STRATEGY=wait`。允许跨 prefix 版本，旧任务到年龄阈值时等待其完成，默认不丢弃。
- 保存 token、logprob 和工具会话；清除旧 prefix 的 KV/GDN 状态，在新 prefix 下重建。续跑保持原始单次生成预算，修复未显式传 `max_tokens` 时反复获得完整预算的问题。
- 增加生成门控：vLLM 0.22.1 恢复 weights 会提前唤醒调度器，必须等 prefix 同步、KV 恢复、显式 resume 完成后放行。实际 GPU 测试暴露的 CUDA 错误已修复；pause 失败会中止更新。
- warmup 计入总入队预算；最后几步停止补新 batch，收尾既有长尾。恢复队列存在时先计入其 prompt 数。
- 默认 run name 加 `_partial`；`TRAINER_MODE=sync` 恢复原 V0 同步路径。串行队列导出遵循指定 `MODEL`。启动前检查 Python 环境的自定义 vLLM plugin、TransferQueue 和 partial 参数。

保留 FSDP2、G512/A64 hybrid prefix、112 个可训练张量、prefix-only 同步与 checkpoint、固定 teacher 的 top-64 forward-KL。没有修改 loss，没有开启 CUDA Graph。

## 验收

9 个回归测试通过：token/logprob 续接、显式与默认预算、年龄阈值 wait、分阶段恢复门控、pause 失败传播、warmup 总量与收尾。Shell 语法、Python 编译、`git diff --check` 通过。

| 真实 GPU 验收 | 配置 | 结果 |
|---|---|---|
| TACO | 1 actor/rollout + 1 teacher，batch 2，输出上限 4096，3 步 | 完成；轨迹最大跨 2 个 prefix 版本；loss、梯度有限；112 个张量均更新 |
| EnvScaler | 3 actor/rollout + 1 teacher，batch 6，输出上限 4096，3 步 | 完成；训练最多 28 turns；world-size 3 checkpoint 保存、导出成功；112 个张量均更新 |
| 混合验证 | EnvScaler、TACO 各 1 行 | 完成；reward 均为 1，仅用于流程验收 |
| 四卡 checkpoint 恢复 | step 3 → step 4 | 完成；只读取 6 条新样本（24 → 30），仅一批；无额外 warmup；新 checkpoint 的 112 个张量均更新 |

TACO 验收了跨版本续跑与门控；后续 EnvScaler 包含默认生成预算修复；恢复测试包含最新总入队预算修改。这是功能测试，使用小 batch 与 4096 输出上限；正式默认 batch 24、输出上限 16384。全部验收通过（`acceptance.json`）。没有同配置吞吐对照，不能据此承诺加速倍率或质量提升。

**依赖限制：** 当前 TransferQueue 0.1.7 没有队列 checkpoint API。actor、optimizer、dataloader 可恢复；中途进程重启会丢失已从 dataloader 取出、尚未训练的排队轨迹。进程内 partial 续跑不受影响，正常跑完会收尾。有队列 API 的版本（>= 0.1.9）可重新派发在途 episode，仍从原 prompt 重跑，不恢复旧工具 session。

## 启动

在 H200 执行。显式复用已有数据、初始化，使用新的实验目录。完整 100 行验证包含 AWM，需要其工具服务；本次测试已启动 EnvScaler 与 TACO 服务。

```bash
cd /home/liyu.468/learning_kvcache
./experiment.sh tools start awm

DATA_ROOT=/home/liyu.468/learning_kvcache/outputs/prefix_composition_3domain_20260922_4gpu/data \
MODEL=/home/liyu.468/learning_kvcache/outputs/prefix_composition_3domain_20260917/prepared/hybrid-G512-A64 \
EXPERIMENT_ROOT=/home/liyu.468/learning_kvcache/outputs/partial_rollout_full_20260928 \
ACTOR_GPUS=3 TEACHER_GPUS=1 CUDA_VISIBLE_DEVICES=0,1,2,3 \
USE_WANDB=0 \
./experiment.sh train envscaler \
  +ray_kwargs.ray_init.address=local \
  ray_kwargs.ray_init.num_cpus=32 \
  +ray_kwargs.ray_init.num_gpus=4 \
  +ray_kwargs.ray_init.include_dashboard=False \
  +ray_kwargs.ray_init.object_store_memory=8589934592
```

三个域串行训练时，沿用上述数据、初始化、新实验目录，设置 `TRAIN_DEVICES=0,1,2,3 TRAIN_ACTOR_GPUS=3 TRAIN_TEACHER_GPUS=1 ./experiment.sh train all`。串行队列的原硬件默认仍为 8 卡。

受控性能对照添加 `TRAINER_MODE=sync`，使用另一个新目录，保持 batch、输出预算、数据、teacher、GPU 布局相同。比较整轮时间、生成 token/s、版本跨度与验证结果。

服务器产物在 `outputs/partial_rollout_h200_20260928/`：`smoke_metrics.json`、`prefix_verification.json`、`acceptance.json`、三个 launcher 日志、checkpoints、导出 prefixes。启动与核验脚本在 `diagnostics/partial_rollout_20260928/`。
