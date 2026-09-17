# gpu-b200 benchmark 验证记录（2026-09-17）

项目：`/home/liyu.468/learning_kvcache`。本轮目标是打通基础模型的推理与验证流程，不是测定 prefix 组合效果或发布全量 benchmark 成绩。

## 已准备

| 数据/环境 | 状态 |
|---|---|
| tau2-bench | 官方固定 commit；独立 Python 3.12 环境；官方数据可读取 |
| BFCL-v3 | 官方固定 commit；17 类、4,441 题；独立 Python 3.11 客户端 |
| Lifelong DB / OS | 各 500 题；全部通过官方数据结构构造；执行依赖已安装 |
| HumanEval+ / MBPP+ | 官方 v0.1.10 / v0.2.0，164 / 378 题；EvalPlus 0.3.1 |
| LiveCodeBench | 固定 HF revision；v5 880 题、v6 1,055 题；全部 6 个数据文件哈希验证 |

Lifelong 官方公开集只有 `train` split，当前 `test.jsonl` 仅是现有评测入口的别名，不能声称是独立留出集。本轮没有将这些任务加入训练。

环境分别位于 `benchmark/.venv`、`benchmark/tau2/.venv`、`benchmark/coding/.venv`。基础推理使用现有 `kvcache` conda 环境；没有升级其训练依赖。所有数据、缓存、日志、镜像和临时评测文件均放在项目目录下。

## 实际验证

| 项目 | 验证结果 |
|---|---|
| tau2 Airline | 2 条完整对话与官方判分；reward 1 / 0 |
| tau2 Retail | 2 条完整对话与官方判分；reward 1 / 0 |
| tau2 Telecom | 2 条完整对话与官方判分；均达到 50 步上限，reward 0 |
| BFCL-v3 | 17 类各 1 题推理与官方评分完成；13 题判对、4 题判错，无推理请求异常 |
| HumanEval+ / MBPP+ | 各 2 条完整代码生成，正常 stop，无请求错误；执行判分待 Docker |
| LCB | 1873_A / 1873_B 及 v6 新增题 abc387_b 共 3 题完整生成，正常 stop、无请求错误；执行判分待 Docker |
| Lifelong DB / OS | 各 500 条结构校验通过；真实数据库/OS 回合待 Docker |

tau2 的用户模拟器与可用的自然语言断言 judge 使用本地模型，旨在验证闭环；不能与使用指定官方 judge 的正式成绩直接比较。BFCL 使用现有 Qwen3-4B-FC handler 对接 Qwen3.5-4B，也仅用于流程验证。正式 prefix 对比须固定模板、模型、judge、预算与任务集合。

基础模型为 `/home/liyu.468/models/Qwen3.5-4B`。临时独立 vLLM 服务使用 GPU 3、显存比例 0.08、32,768 上下文、关闭 thinking、原生 tool parser，服务别名 `qwen3.5-4b-benchmark`。现有训练 teacher API 实测每次只生成 1 token，不能用于 benchmark；早期失败探测保留为诊断，不计入成功验证。

## 修复与检查

- 补齐 BFCL API 模型名补丁，区分本地 tokenizer 路径和服务模型名。
- 修复 BFCL / coding smoke 默认解释器错误，避免依赖不存在的项目根 `.venv`。
- tau2 启动器实际应用 judge 配置；即使上游退出 0，也检查缺失回合与 `infrastructure_error`，避免假成功。
- BFCL 按 ID 生成的小样本需要同步筛选官方题目与答案；新增单独包装器，保留官方评分逻辑，并明确标注 subset smoke。
- 统一交互评测存在错误时返回非零退出码。
- tau2 状态回归测试 3 项通过；BFCL ID 筛选、缺失 ID、重复 ID 检查通过；coding 测试 11 项通过，2 项 Docker 测试未运行。
- 客户端与 tau2 环境依赖检查通过。coding 使用项目原有最小依赖锁；不安装未使用的 Anthropic、Google、stop-sequencer 生成器扩展，官方评分器导入检查通过。

## 尚未完成及原因

Docker CLI、独立 daemon、Ubuntu 22.04 和 MySQL 8.0 镜像已准备。项目文件系统不支持镜像层所需特殊文件，镜像存储改为项目路径下的 tmpfs；但服务器容器的 cgroup 挂载只读，启动评测容器失败。

因此尚未完成：HumanEval+/MBPP+/LCB 官方代码执行判分、正确/错误控制样本判分、Lifelong DB/OS 完整回合与奖励验证，以及 `local-os/default` 镜像构建。没有在宿主机直接执行模型生成的代码或 OS 操作作为替代。

自动审批拒绝了在独立挂载命名空间内将 cgroup 挂载改为可写的尝试，理由是未明确授权的安全边界变更且影响范围不确定。该命令没有执行；后续需要可用 Docker 主机/接口，或用户明确授权后重新审核该方案。

## 复跑入口（远端项目根目录）

```bash
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate kvcache
source benchmark/env.sh
python benchmark/serve_b200.py start
python benchmark/serve_b200.py status
MODE=generation bash benchmark/smoke.sh
```

`MODE=generation` 运行 tau2/BFCL 推理与评分、代码生成，跳过 Docker 依赖部分。默认输出新建在 `outputs/benchmark_smoke_<时间>/`，拒绝覆盖现有目录。

Docker 可运行后，在支持项目数据路径挂载的运行环境配置 `DOCKER_HOST`，构建 `local-os/default`，再运行：

```bash
bash benchmark/build_os_image.sh
MODE=full bash benchmark/smoke.sh
```

远程 Docker 主机若看不到本项目绝对路径，还需要同步评测环境与数据或调整挂载配置，不能仅替换接口地址。

完整流程包含原有 `benchmark/coding/smoke.py` 的官方真实题目正确/错误控制：HumanEval+、MBPP+、LCB v5/v6 各一题，预期 pass@1=0.5、pass@2=1。此控制目前尚未执行成功。

环境重建入口：`bash benchmark/setup_b200.sh`；现有网络对 GitHub/HF/DockerHub 有限制，首次数据和镜像已通过本地下载或官方签名 CDN 转存，重建脚本不承诺绕过网络限制。

所有运行证据：`outputs/benchmark_bringup_20260917/`，包括 `smoke/`、`logs/`、数据清单、进程记录和修改前代码备份。训练进程及用户其它代码改动未停止或覆盖。临时 benchmark vLLM 服务在验证后已关闭，复跑时按上述入口启动。
