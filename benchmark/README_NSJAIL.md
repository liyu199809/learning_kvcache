# 本地 NsJail 代码判分（2026-09-17）

项目：`gpu-b200:/home/liyu.468/learning_kvcache`。

本轮只补齐 HumanEval+、MBPP+、LiveCodeBench 的本地执行判分。模型答案沿用此前生成结果，不启动 GPU 模型服务。tau2/BFCL 保持此前已经完成的流程；Lifelong DB/OS 的有状态环境适配仍待后续完成。

## 实现

- `benchmark/runtime/bin/nsjail`：项目内编译的 NsJail 3.6，源码固定为 `f78475530b46d0186111a9096b30725f816b55fe`；Kafel 固定为 `76d0f41bf3eb5c4008713d64b9767b461a9129a3`。
- `benchmark/coding/nsjail_backend.py`：把原有 `score.py` 和官方评分依赖放入 NsJail 执行；没有修改 EvalPlus 或 LCB 的评分算法。
- `benchmark/coding/nsjail.policy`：限制网络 socket、重新挂载、创建 namespace、提权与进程内存检查等系统调用。
- 统一入口新增 `--code-execution-backend nsjail`，原有 Docker 后端保留，默认不自动切换；NsJail 失败时也不会退回裸进程执行。
- `benchmark/setup_nsjail.sh` / `benchmark/build_nsjail.sh`：下载、解包和编译均在项目内，不向系统或训练 conda 环境安装软件包。

## 实际隔离方式

外层容器不允许在新 user namespace 中重新挂载 `/proc`，因此使用 NsJail 的 root supervisor 模式，在执行 Python 前切换到本轮独立的非 root UID，删除全部 capabilities，并启用 `no_new_privs`。

保留 mount、PID、network、IPC、UTS namespace；不创建或改写 cgroup。只有 Python、依赖、官方评分代码以只读方式挂入，另有本次输出目录及临时文件系统；项目、模型、GPU 和宿主网络不暴露给候选代码。

限制为最多 1–4 个 CPU、64 个进程/线程、每进程地址空间限制、文件大小限制、墙上时间限制。运行优先级为 nice 19。`--code-memory` 在此后端表示每进程的 `RLIMIT_AS`，并非 Docker 的进程树总内存硬限额，二者不能混称。

## 验证证据

实际隔离探针已经通过：非 root UID、零 capabilities、`no_new_privs`、seccomp 生效；项目和 GPU 不可见；INET socket 被拒绝；只读评分文件不能修改；内存超限失败；超时后没有残留子进程。

发现并修复了一项会导致假零分的问题：EvalPlus 会把栈限制设为 4 GiB，NsJail 默认栈硬限制更小，导致官方子进程在执行答案前退出。现已将栈硬限制与配置的内存限额对齐，通过 `EVALPLUS_MAX_MEMORY_BYTES` 使用官方支持的配置入口；启动评分器时保留 8 MiB 的栈软限制，避免普通线程各自预留 4 GiB 地址空间而启动失败。包装器同时检查异常退出的官方 worker，避免把该错误记成模型答案超时。

官方真实题目控制测试和 7 份模型答案的最终结果见本报告末尾验收记录。

## 复跑命令

以下命令在远端项目根目录执行，无需启动模型服务：

```bash
source benchmark/env.sh

# 验证本地隔离
benchmark/coding/.venv/bin/python benchmark/coding/verify_nsjail.py \
  --output-dir outputs/nsjail_isolation_check

# 官方真实题目，每组包含一份正确和一份故意错误答案
benchmark/coding/.venv/bin/python benchmark/coding/smoke.py \
  --execution-backend nsjail \
  --output-dir outputs/nsjail_official_controls

# 重放此前已生成的 7 份模型答案，结果写到新目录
bash benchmark/score_coding_smoke.sh
```

评分任意已有答案时，统一入口用法为：

```bash
bash benchmark/run.sh --benchmark humaneval+ \
  --task-ids HumanEval/0,HumanEval/1 --code-samples <samples.jsonl> \
  --code-execution-backend nsjail --code-eval-workers 2 \
  --code-memory 4g --code-eval-timeout 300 --code-pass-k 1 \
  --output-dir <新的输出目录>
```

每轮保存任务/数据版本、NsJail 二进制和策略哈希、实际挂载配置、stdout/stderr、逐题结果及汇总。全量评测应先分批运行，避免把所有 LCB 隐藏测试一次装入内存。

产物目录：`outputs/benchmark_nsjail_20260917/`。此前错误探测保留在 `controls/`、`controls_verified/` 和相应日志中，不应作为有效模型分数使用。没有调用 E2B，没有新增云沙箱服务费用。

## 验收记录

`controls_final/` 的四组真实题目验收全部通过：HumanEval+、MBPP+、LCB v5、LCB v6。每组各一份正确答案、一份故意错误答案，逐份结果均为 `[通过, 失败]`，因此 pass@1 = 0.5、pass@2 = 1.0；EvalPlus 基础测试和扩展测试均符合预期。这些控制值用于确认判分行为，不是模型成绩。

`isolation_final/verification.json` 保存最终配置的实际隔离验证；内存限制和超时清理均通过，父子进程超时清理耗时约 2.38 秒。另有两项 NsJail 测试通过（LCB 标准输入/函数调用正确与错误答案、官方 worker 崩溃不记成零分）；既有 coding 测试 11 项通过、2 项 Docker 专用测试跳过，runner 测试 3 项通过。脚本语法和 `git diff --check` 通过，原有 `score.py` 与修改前逐字一致。

重放结果保存在 `generated/`。其中 `1873_B` 的失败已定位为模型答案执行 `prod // a[i]` 时未处理输入中的零，触发 `ZeroDivisionError`；评分进程正常结束，属于答案错误。

| 已保存模型答案 | 题目 | 通过数 |
|---|---|---:|
| HumanEval+ | HumanEval/0、HumanEval/1 | 2/2，基础和扩展测试均通过 |
| MBPP+ | Mbpp/2、Mbpp/3 | 2/2，基础和扩展测试均通过 |
| LCB v6 中已有题目 | 1873_A、1873_B | 1/2 |
| LCB v6 新增题目 | abc387_b | 1/1 |

以上均为之前 `qwen3.5-4b-benchmark` 生成的一次采样答案，关闭思考、输出上限 16384 tokens；本轮只做执行和判分。7 份答案中 6 份通过，样本太少，不代表完整数据集准确率，也不能据此判断 prefix 组合有效。

机器可读验收汇总：`outputs/benchmark_nsjail_20260917/validation_summary.json`。本轮完成代码 benchmark 的小样本推理结果重放与官方判分；Lifelong DB/OS 的有状态环境、全量评测、prefix 训练与组合实验仍待后续阶段完成。未启动或停止 GPU 任务。
