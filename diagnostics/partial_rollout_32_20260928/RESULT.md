# 固定训练 batch 24，rollout 池 32

已修改 `gpu-h200:/home/liyu.468/learning_kvcache`。正式训练之前就默认 batch 24；先前 batch 2/6 是功能 smoke。此次将整批 warmup 的 48 条初始队列改为有明确上限的 32 条池。

每次消费 24 条完整且 teacher 已打分的轨迹，保留未消费的 8 条，仅补齐池中缺口；不会每轮另投 32 条而导致队列积压。actor 和 rollout 继续共卡，未完成任务在 prefix 更新后续跑。PPO mini-batch、有效训练 batch 都保持 24。

| 阶段 | 新投放 | 训练 | 剩余 |
|---|---:|---:|---:|
| 第一步 | 32 | 24 | 8 |
| 后续普通步 | 24 | 24 | 8 |
| 最后一步 | 16 | 24 | 0 |

例如 3 步合计读取并训练 72 条，与同步训练的 3 × 24 相同。最后收尾减少补样本，而不减少训练 batch。

- 默认 `PARTIAL_ROLLOUT_TARGET=32`（batch 24 时）。`PARTIAL_ROLLOUT_TARGET=36` 是 1.5 倍，普通步剩 12 条，最后补 12 条。
- rollout 池包含未完成和已完成待训练的 prompt groups；只有完成且打分的轨迹可进入训练。
- 每步记录 `training/partial/` 下的 `submitted_prompts`、`consumed_prompts`、`queued_prompts`、`remaining_admission_budget`。
- 数据读取 chunk 自动为 1，随后合成所需的 32/24/16 条。这只改变读取粒度；epoch 和 optimizer 步数仍按训练 batch 24 计算。
- `rollout_target_size=null` 保留原整批 warmup 路径。默认使用新 `_partial_pool32` run name，避免自动恢复旧读取粒度的 checkpoint。
- 当前 TransferQueue 0.1.7 的进程重启限制仍然存在；正常暂停与续跑保留样本。

## 验收

18 项测试通过，覆盖 32/36 条池、训练 batch 24、不积压、最后收尾、恢复队列以及旧续跑/预算/恢复门控。

H200 实跑 EnvScaler 已通过：3 actor/rollout + 1 teacher，训练 batch 24、PPO mini-batch 24、rollout 池 32、输出上限 16384、3 次真实更新。数据、模型初始化沿用之前实验，未启用验证。实际新投放为 32/24/16，消费为 24/24/24，剩余为 8/8/0；72 条均被训练。第三步轨迹最大跨 2 个 prefix 版本，112 个 prefix 张量均更新且有限；world-size 3 checkpoint 已保存、导出。最终结果记录在 `acceptance.json`。四张 GPU 均已释放（4 MiB、0%）。

| 实测更新步 | 生成/取样阶段 | 完整更新步 |
|---|---:|---:|
| 1 | 185.0 秒 | 276.8 秒 |
| 2 | 155.1 秒 | 236.1 秒 |
| 3（收尾） | 137.0 秒 | 173.3 秒 |

剔除首步后平均 204.7 秒/步；旧 EnvScaler 记录平均 260.2 秒/步。两者均 batch 24、3+1 GPU 布局、16K 输出上限。粗比每步耗时下降 21.3%，训练样本吞吐约 1.27 倍（+27.1%）；生成/取样阶段均值 146.1 秒，旧值 231.1 秒。这是不同主机的历史对照，只有两步参与新均值，不能隔离卡型、样本与编译差异。含首步三步均值为 228.7 秒，Ray/模型启动与验证不计入上述 step 时间。

本次训练、保存均已完成；我在执行期间覆盖了仍由 bash 读取的 launcher 文件，导致训练结束后 shell 额外读取尾部参数并报一次 command not found。现有脚本另做了完整入口执行检查：仅用诊断 stub 替代 GPU Python 入口，验证 batch/池/Ray 参数均被正确传递，shell 正常退出。该检查不启动 GPU，原始日志保留这一现场覆盖错误。没有把 stub 检查当成真实 GPU 训练。

日志与产物：服务器 `outputs/partial_rollout_32_h200_20260928/`。
启动脚本：[start_full_batch.py](start_full_batch.py)。它只跑 3 步验收，未启动完整一轮训练。

## 使用

沿用已有 H200 启动命令和 3+1 GPU 布局，正式训练 batch 默认就是 24，rollout 池默认就是 32，无需缩小 batch。要用 36 条池：

```bash
cd /home/liyu.468/learning_kvcache
./experiment.sh tools start awm

PARTIAL_ROLLOUT_TARGET=36 TRAIN_BATCH_SIZE=24 PPO_MINI_BATCH_SIZE=24 \
ACTOR_GPUS=3 TEACHER_GPUS=1 CUDA_VISIBLE_DEVICES=0,1,2,3 \
DATA_ROOT=/home/liyu.468/learning_kvcache/outputs/prefix_composition_3domain_20260922_4gpu/data \
MODEL=/home/liyu.468/learning_kvcache/outputs/prefix_composition_3domain_20260917/prepared/hybrid-G512-A64 \
EXPERIMENT_ROOT=/home/liyu.468/learning_kvcache/outputs/partial_pool36_full_20260928 \
USE_WANDB=0 ./experiment.sh train envscaler \
  +ray_kwargs.ray_init.address=local \
  ray_kwargs.ray_init.num_cpus=64 \
  +ray_kwargs.ray_init.num_gpus=4 \
  +ray_kwargs.ray_init.include_dashboard=False \
  +ray_kwargs.ray_init.object_store_memory=8589934592
```

切换为新调度使用新实验目录或默认新 run name；旧 dataloader checkpoint 的读取 batch 大小不同。`TRAINER_MODE=sync` 可跑同步对照。
