# Self-Evolver

**Agent 自我进化训练框架：On-Policy Self-Distillation（OPSD）全流程**

学生模型（Qwen3.5-4B）在三类 Agent 环境中做 on-policy rollout；教师模型以**特权视角**（专家建议 / 标准答案注入上下文）对同一条 token 序列给出蒸馏信号；两者通过 verl 的 GRPO + forward-KL 蒸馏联合训练，最终用标准 benchmark 评测。环境服务、轨迹精炼、数据集构建、训练、评测的完整闭环都在这一个仓库里。

## Pipeline 总览

```
              ┌────────────────────────────────────────────────────────┐
              │  OpenEnv 环境服务（CPU 进程，随用随起）                   │
              │   AWM       :8899   1000 个工具环境 / 10000 任务         │
              │   EnvScaler :8900   191 个 checklist 环境               │
              │   CodeJudge :8901   DeepCoder/TACO 编程题判题            │
              └────────────────────────────────────────────────────────┘
                                      │
  vLLM :8000（学生 Qwen3.5-4B）        ▼
        └────────────► rollout/refine.py
                        学生 rollout → verify → 教师给一条 # Advice → 重试
                                      │   traj_data/*_refine_full.jsonl
                                      ▼
                      rollout/build_refine_opsd_dataset.py
                      （质量过滤 + 分层采样 → 三源混合数据集）
                                      │   traj_data/opsd_mixed_1625_v1/{train,val}.parquet
                                      ▼
                      verl On-Policy Distillation（GRPO + forward_kl_topk）
                      自蒸馏：教师 = 同一模型 + ground truth 特权上下文
                                      │   checkpoints/
                                      ▼
                      eval_main.py：合并 checkpoint → vLLM → 评测
                      BFCL v3 · tau2-bench · LifelongAgentBench
```

三个训练数据视图共享同一套 verl 行协议（`messages` + `tools` + `env_config` 路由到对应环境服务）：

| 视图 | 数据源 | 环境 | 验证方式 |
|---|---|---|---|
| `awm` | `Snowflake/AgentWorldModel-1K` | AWM 工具环境（SQLite 状态 + MCP 工具） | Python verifier / LLM 裁判 |
| `envscaler` | `XXHStudyHard/EnvScaler-191-Env` + `EnvScaler-RL-Scenario` | EnvScaler checklist 环境 | 确定性 dense 评分 |
| `taco` | `agentica-org/DeepCoder-Preview-Dataset`（taco 子集，pinned revision） | CodeJudge 代码判题 | 单测通过率 |

## 仓库结构

| 目录 | 说明 |
|---|---|
| `rollout/` | **自研核心**：环境交互与 refine 循环（`refine.py`）、数据集构建（`build_refine_opsd_dataset.py`、`refine2swift.py`）、verl agent loop（`verl_awm_agent_loop.py`）、ms-swift 插件（`awm_opsd_plugin.py`、`awm_scheduler_plugin.py`） |
| `verl/` | verl（RL/蒸馏训练框架）fork，含 `examples/on_policy_distillation_trainer/` 全部训练脚本与自蒸馏/特权上下文支持 |
| `OpenEnv/` | OpenEnv（统一 RL 环境框架）fork，含 AWM / EnvScaler / CodeJudge 三个环境服务 |
| `ms-swift/` | ms-swift fork，`swift/rl_core/data.py` 增加了 `teacher_prompt`/teacher-view（OPSD 数据协议） |
| `benchmark/` | 评测 harness：BFCL v3、tau2-bench、LifelongAgentBench（db/os）适配器与统一入口 |
| `prefix_tuning/` | Qwen3.5 DeltaNet 虚拟前缀调优（独立研究支线，见其自身 README） |
| 顶层脚本 | 见下文「快速开始」「服务与常用变量速查」 |

> 四个上游 fork（verl、ms-swift、OpenEnv 等）以普通目录形式合并在**同一条 git 历史**里，
> 没有 submodule——`git clone` 即全量，按上文 editable 安装即可使用。

## 环境要求

- **GPU**：8 × 80GB（开发机为 8×A800-80G，CUDA 12.8）。`RUN_MODE=smoke` 冒烟只需 3 卡。
- **Python**：3.11（用 [uv](https://docs.astral.sh/uv/) 管理环境）。
- **磁盘**：≥ 60 GB 空闲（模型 ~8G + 环境数据 ~1.7G + 数据集 + checkpoint）。
- **API key**：生成训练数据阶段需要火山方舟 `ARK_API_KEY`（教师模型 / LLM 裁判）。训练与纯 RL 冒烟不需要。
- 国内网络建议设置 HF 镜像：`export HF_ENDPOINT=https://hf-mirror.com`（本仓库 `uv.toml` 已将 PyPI 指向清华镜像）。

## 快速开始

以下命令均在仓库根目录执行。

### 1. 安装 Python 环境

```bash
# 安装 uv（已装可跳过）
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone <本仓库地址> && cd self_evolver
uv venv --python 3.11 .venv
source .venv/bin/activate

# 375 个锁定版本（torch 2.10.0 + cu12.8 运行时 + vllm 0.18.0 等）
uv pip install -r requirements-freeze.txt

# 仓库内的 fork 包以 editable 方式安装（--no-deps：依赖已全部在 freeze 里，
# 避免 uv 重新解析版本、偏离锁定环境）
uv pip install --no-deps -e verl -e ms-swift -e OpenEnv \
    -e OpenEnv/envs/agent_world_model_env -e prefix_tuning
```

> **flash-attn 编译提示**：`flash-attn==2.8.3.post1` 在 PyPI 上只有源码包，安装时会本地编译
> （约 30–60 分钟，需要 nvcc 与 g++，可用 `MAX_JOBS=32` 控制并行度）。机器上需装有与
> torch 匹配的 CUDA 工具链（本仓库开发环境为 CUDA 12.8 + gcc 11）。vLLM 自带注意力
> kernel，flash-attn 主要被 ms-swift 训练路径使用——只想跑 verl 路径的话，可以先从
> freeze 里去掉这一行再安装，之后需要时单独补装。

### 2. 配置密钥

```bash
cp .env.example .env
# 编辑 .env，填入 ARK_API_KEY（生成数据集阶段必需）
wandb login   # 训练日志上报需要（或把 WANDB_API_KEY 写进 .env）
```

### 3. 下载模型与数据

```bash
# ① 基座模型 Qwen3.5-4B（~8GB）。之后所有脚本都通过 $MODEL 引用它
./hfd.sh Qwen/Qwen3.5-4B --local-dir ./models/Qwen3.5-4B
export MODEL=$PWD/models/Qwen3.5-4B      # 建议写进 ~/.bashrc

# ② DeepCoder/TACO 数据（pinned revision + sha256 校验，~800MB）
bash verl/examples/on_policy_distillation_trainer/download_deepcoder_taco.sh

# ③ AWM（~830MB）与 EnvScaler（~62MB）数据：无需手动下载，
#    对应环境服务首次启动时自动从 HuggingFace 拉取并缓存到
#    awm_data/ 与 envscaler_data/
```

### 4. 启动环境服务

| 命令 | 服务 | 端口 |
|---|---|---|
| `./start_services.sh start` | vLLM 学生模型 + AWM 环境 | 8000 / 8899 |
| `./start_envscaler_service.sh start` | EnvScaler 环境 | 8900 |
| `nohup ./start_code_judge_service.sh > /tmp/code_judge.log 2>&1 &` | CodeJudge（TACO 判题） | 8901 |

```bash
./start_services.sh start                 # 首次会自动下载 AWM 数据，耐心等待
./start_envscaler_service.sh start        # 首次会自动下载 EnvScaler 数据
nohup ./start_code_judge_service.sh > /tmp/code_judge.log 2>&1 &
./start_services.sh status                # 查看健康状态；logs vllm / logs tool 看日志
```

### 5. 冒烟验证（纯 RL，不需要任何 API key）

跑一个真实的 GRPO optimizer step 验证「环境 + 模型 + 训练框架」整条链路：

```bash
./start_services.sh stop vllm    # 释放 GPU（环境服务保持运行）

bash verl/examples/on_policy_distillation_trainer/run_qwen3_5_4b_envscaler_rl_smoke.sh      # 2 卡
bash verl/examples/on_policy_distillation_trainer/run_qwen3_5_4b_deepcoder_taco_rl_smoke.sh # 2 卡
```

### 6.（可选）从头重建 OPSD 训练数据集

训练数据集 `traj_data/opsd_mixed_1625_v1/`（~108MB，train/val parquet + manifest）
**已随仓库分发**，clone 后直接进入第 7 步即可训练。本节仅当需要重新生成数据时使用。

流程两步：**refine**（学生 rollout + 教师建议，产出 jsonl）→ **build**（过滤采样，产出 parquet）。
refine 需要学生 vLLM 服务、三个环境服务和教师 API（`ARK_API_KEY`），全程可断点续跑（`--resume`）。

```bash
./start_services.sh restart vllm   # refine 需要学生推理服务

# ① AWM：1000 环境 × 全部任务（约 1 万条记录，数小时）
python -m rollout.refine --dataset awm --no-only-infer \
    --output-jsonl traj_data/awm_refine_full.jsonl \
    --report traj_data/awm_refine_full.json --resume

# ② EnvScaler：两个版本（学生工具预算 8 轮 / 16 轮）
python -m rollout.refine --dataset envscaler --no-only-infer --student-max-iterations 8 \
    --output-jsonl traj_data/envscaler_refine_full.jsonl \
    --report traj_data/envscaler_refine_full.json --resume
python -m rollout.refine --dataset envscaler --no-only-infer --student-max-iterations 16 \
    --concurrency 128 \
    --output-jsonl traj_data/envscaler_refine_16iter_full.jsonl \
    --report traj_data/envscaler_refine_16iter_full.json --resume

# ③ DeepCoder/TACO
python -m rollout.refine --dataset deepcoder-taco --no-only-infer \
    --output-jsonl traj_data/deepcoder_taco_refine_full.jsonl --resume

# ④ 过滤 + 分层采样 → traj_data/opsd_mixed_1625_v1/{train,val}.parquet
#    （需要三个环境服务都在线，用于拉取各 scenario 的工具 schema）
python -m rollout.build_refine_opsd_dataset --model "$MODEL"
```

数据集规模（`traj_data/opsd_mixed_1625_v1/manifest.json`）：每源 1625 条训练（共 4875）+ 100 条验证（EnvScaler/TACO 各 50）。

### 7. OPSD 训练（一键）

```bash
./start_services.sh stop vllm    # 训练需要全部 8 卡：6 actor + 2 teacher

# 冒烟（3 卡、单个 optimizer step）
RUN_MODE=smoke bash verl/examples/on_policy_distillation_trainer/run_all_qwen3_5_4b_opsd_dataset_views.sh

# 正式训练：依次跑 awm / envscaler / taco / mixed 四个视图
bash verl/examples/on_policy_distillation_trainer/run_all_qwen3_5_4b_opsd_dataset_views.sh

# 只跑一个视图（awm|envscaler|taco|mixed），或只做预检
bash verl/examples/on_policy_distillation_trainer/run_qwen3_5_4b_opsd_dataset_view.sh mixed
PREFLIGHT_ONLY=1 bash verl/examples/on_policy_distillation_trainer/run_qwen3_5_4b_opsd_dataset_view.sh mixed
```

- 训练日志实时写入 `logs/<run_name>_<mode>_<时间戳>.log`，指标上报 W&B 项目 `self_evolver_opsd_3way`。
- checkpoint 存 `checkpoints/self_evolver_opsd_3way/<run_name>/`，rollout 轨迹存 `rollout_trajs/<run_name>/`；断点自动续训（`RESUME_MODE=auto`）。
- 自蒸馏机制：`distillation.self_distillation=True`，教师是同一模型，唯一区别是 ground truth（`reward_model.ground_truth` 列）被注入教师上下文（`privileged_mode=append`），学生与教师在**同一条 on-policy token 序列**上计算 forward-KL(topk)。

### 8. 评测

`eval_main.py` 一条命令完成「合并 checkpoint → 起 vLLM → 跑 benchmark → 清理临时模型」：

```bash
.venv/bin/python eval_main.py \
    --checkpoint checkpoints/self_evolver_opsd_3way/qwen3_5_4b_opsd_mixed_1625/global_step_100 \
    --model-name mixed-step100 \
    --benchmarks bfcl,tau2,lifelong
```

| benchmark | 别名 | 前置准备 |
|---|---|---|
| BFCL v3 | `bfcl` | `git clone https://github.com/ShishirPatil/gorilla.git benchmark/gorilla` 后运行 `bash benchmark/bfcl_v3/setup.sh` |
| tau2-bench | `tau2`（airline + retail） | `bash benchmark/tau2/setup.sh`（自动建独立 Python 3.12 venv，pinned v1.0.1） |
| LifelongAgentBench | `lifelong`（db + os） | 数据已内置；`lifelong_db` 需本机 Docker（每任务一个 `mysql:8.0` 容器） |
| 全部 | `all` | 以上全部 |

评测期间 root `.env` 里的 `ARK_API_KEY` 用于 tau2 retail 的 NL 裁判。结果输出在 `workspace/eval_logs/`。
也可以对任意在跑的 vLLM 服务单独评测：`python -m benchmark.eval.run_eval --benchmark lifelong_db --openai-base-url http://127.0.0.1:8000/v1 --model qwen3.5-4b ...`。

## 服务与常用变量速查

```bash
./start_services.sh {start|stop|restart} [vllm|tool]   # vllm=推理服务, tool=AWM 环境
./start_services.sh {status|logs {vllm|tool}}
./start_envscaler_service.sh {start|stop|restart|status|logs}
```

| 变量 | 默认 | 说明 |
|---|---|---|
| `MODEL` | `/mnt/storage/disk1/verl_data/base_model/Qwen3.5-4B` | 基座模型路径（**新环境务必设置**） |
| `RUN_MODE` | `full` | `smoke`=3 卡 1 step 冒烟 |
| `VIEWS` | `awm envscaler taco mixed` | `run_all_...sh` 要跑的视图 |
| `TRAIN_GPUS` / `NPROC` | 全部 8 卡 | run_opsd.sh（ms-swift 路径）的 GPU 拓扑 |
| `ARK_API_KEY` | — | 火山方舟 key（教师 / 裁判 / teacher rollout） |
| `HF_ENDPOINT` | — | 设为 `https://hf-mirror.com` 走国内镜像 |
| `AWM_DATA_DIR` / `ENVSCALER_DATA_DIR` / `DEEPCODER_TACO_DATA_DIR` | `awm_data/` / `envscaler_data/` / `data/deepcoder_taco/<rev>` | 环境数据缓存位置 |

## 其他工作流

- **教师基线**：用 Ark 上的强模型跑 AWM，测教师 pass@1 —— `MODE=smoke ./run_teacher_rollout.sh`（`smoke|medium|full`）。
- **ms-swift 训练路径**（GRPO + colocate vLLM + LoRA 自蒸馏，单命令）：`MODEL=$MODEL ./run_opsd.sh`。与 verl 路径的区别：LoRA 微调、教师特权信息来自数据集 `teacher_prompt` 列而非 ground truth 注入。
- **前缀调优支线**：`prefix_tuning/`（Qwen3.5 DeltaNet 虚拟前缀），训练与合并脚本见 `prefix_tuning/virtual_prefix/` 及 `verl/examples/on_policy_distillation_trainer/run_qwen3_5_4b_awm_opsd_*prefix*.sh`。
- **单元测试**：`python -m pytest rollout/tests -q`。

## 常见问题

- **GRPO batch 整除规则**（改 batch/GPU 数时）：`generation_batch_size = per_device_train_batch_size × NPROC × grad_accum` 必须能被 `num_generations` 整除（见 `run_opsd.sh` 头部注释）。
- **训练起不来先看预检**：`PREFLIGHT_ONLY=1` 会校验数据集 schema、环境服务健康、GPU 数、wandb 登录，不占 GPU。
- **显存不足**：降低 `GPU_MEMORY_UTILIZATION`（默认 0.65）或 `MAX_MODEL_LEN`；smoke 模式默认 0.40。
- **长上下文**：AWM 多轮轨迹较长，full 训练 `MAX_PROMPT_LENGTH=28672 + MAX_RESPONSE_LENGTH=16384`，需要 80G 卡；小卡请用 smoke 配置。
- **HF 下载慢/失败**：`export HF_ENDPOINT=https://hf-mirror.com`；TACO 下载脚本也读取该变量。
