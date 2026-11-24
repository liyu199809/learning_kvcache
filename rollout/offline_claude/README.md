# 910c3 离线轨迹采集

在远端仓库根目录运行。三个数据集共用本机 vLLM 的 Qwen3-4B，每次尝试独立 reset OpenEnv 并创建全新的 Claude Code session。AWM 和 EnvScaler 把当题业务工具通过 stdio MCP 暴露给 Claude Code。TACO 使用仓库已有的「输出 Python → 外部 CodeJudge」协议。采集器调用各环境自己的 code verifier。

启动模型服务（前台运行，如果端口已有该服务可直接采集）：

```bash
bash rollout/offline_claude/serve.sh
```

在另一个远端终端按需独立执行：

```bash
cd /home/liyu.468/learning_kvcache
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate kvcache
export CLAUDE_BIN=/root/.local/opt/kvcache-claude-2.1.273/package/claude

python rollout/offline_claude/collect.py awm e_commerce_33 0
python rollout/offline_claude/collect.py envscaler env_164_rl 38
python rollout/offline_claude/collect.py taco taco 102
```

可以在 scenario 后给多个零起始的 task index。同一题重跑会创建一个新目录，保留之前的尝试。每个目录位于 `outputs/offline_claude/<domain>/<scenario>-<index>-<session_id>/`，包含：

- `episode.json`：题目、session ID、模型、终止原因、答案、原始 verifier 结果。
- `events.jsonl`：每行一条标准消息（system、user、assistant、tool），包含 reasoning_content、工具调用及按 ID 配对的返回。
- `tools.json` ：当题工具 schema。
- `stderr.log`：CLI 诊断信息。

`scored` 表示 verifier 已给分，是否解对要看对应环境的 reward 和 verify_result。默认每题 1200 秒，覆盖环境初始化、推理及评分。采集器不设轮次或工具调用次数上限。异常会留下已采到的记录；重试时按 session ID 区分。这里采集的是推理事件，不包含模型服务端逐 token 的概率。

Qwen3.5-4B 批量采集使用 `serve_q35_dp16.sh`，单机 DP=16、TP=1。批次输出目录为 `outputs/q35_dp16_session_experience/`，任务并发 32。

批量采集默认使用 `--teacher-session`。每题创建一个 Claude Code 会话。每次学生提交后由环境评分；未满分时，GLM-5.3-joybuilder 读取本题的会话和公开评分摘要，必要时调用诊断工具，再把自然语言建议作为同一会话的下一条 user 消息发送给学生。提示词要求 teacher 指明如何继续检查或修复，避免给出题目要求学生自行找到的答案。AWM 和 EnvScaler 延续业务状态。TACO 对每次提交独立调用 CodeJudge。整题共享 1200 秒，包括所有学生推理、teacher 调用和评分。满分、超时或执行错误时结束。

完整会话按顺序写入 `events.jsonl`。`episode.json` 的 `rounds` 保留每次提交、评分和 teacher 的指令及工具记录；`instruction_message_index` 对应轨迹中指令的位置（从零开始）。批次索引中的 `reward` 为首轮分数，`last_verified_reward` 为最近一次评分，`verified_rounds` 为已评分次数。历史的独立重试实验仍可用 `--teacher-retry` 单独运行。

teacher 的评分反馈只包含是否超时、有限的数值得分和通过数等整数计数。工具签名、学生本身的工具调用和返回可见。参考解、隐藏用例、verifier 代码和自由文本错误不进入 teacher 请求。采集器直接访问 `http://llm-gw.jd.local/v1`，凭据从 `JD_LLM_API_KEY` 或当前用户的 `~/.config/kvcache-learning/jd-teacher.key` 读取。在历史独立重试模式中，teacher 请求失败后，可在原单题命令后添加 `--resume-failure <首轮目录>` 复用已有的首轮轨迹。

AWM 和 EnvScaler 的 teacher 在独立环境副本中 reset 原题并重放有返回的学生调用。它可以从本题对学生公开、且名称以 get、list、search、find、lookup、count、describe 或 check 加下划线开头的业务工具中选择查询，最多实际尝试三次。与原轨迹不同的重放结果计入 `replay.different_results`。TACO 的诊断工具只对学生已提交的 Python 做语法检查。每次探查的工具名、参数和模型所见返回保存在当轮 `teacher.teacher_tool_calls` 中。
