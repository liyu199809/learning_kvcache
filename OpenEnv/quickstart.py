"""
AWM Quick Start —— 通过 OpenEnv 跑通一个完整的 agent episode。

流程:  reset → list_tools → call_tool → verify → done
这是 agentic RL 训练循环的核心原语 (rollout 的每一步)。

前置:
  1. AWM server 已在 http://localhost:8899 运行
  2. 子进程依赖已装 (sqlalchemy / fastapi-mcp / mcp-agent / openai)

运行:
  cd OpenEnv
  PYTHONPATH=src:envs .venv/bin/python quickstart.py
"""
import asyncio
from agent_world_model_env import AWMEnv
from openenv.core.env_server.mcp_types import CallToolAction

BASE_URL = "http://localhost:8899"
SCENARIO = "e_commerce_33"
TASK_IDX = 0


async def main():
    async with AWMEnv(base_url=BASE_URL) as env:

        # 1. RESET —— 加载场景+任务,启动隔离的 MCP 子进程
        result = await env.reset(scenario=SCENARIO, task_idx=TASK_IDX)
        obs = result.observation
        print("=" * 64)
        print("1. RESET")
        print("=" * 64)
        print(f"  scenario    : {obs.scenario}")
        print(f"  task        : {obs.task}")
        print(f"  num_tools   : {obs.num_tools}")
        print(f"  has_verifier: {obs.has_verifier}")

        # 2. LIST_TOOLS —— 发现该场景可用的 MCP 工具
        tools = await env.list_tools()
        print("\n" + "=" * 64)
        print(f"2. LIST_TOOLS  (共 {len(tools)} 个, 展示前 5)")
        print("=" * 64)
        for t in tools[:5]:
            print(f"  - {getattr(t, 'name', '?')}")

        # 3. CALL_TOOL —— 调真实工具 (任务第一步: 搜索耳机)
        obs = await env.call_tool("search_products", query="wireless noise cancelling headphones")
        print("\n" + "=" * 64)
        print("3. CALL_TOOL:  search_products(query='wireless noise cancelling headphones')")
        print("=" * 64)
        tr = obs.tool_result
        if isinstance(tr, list):
            print(f"  返回 {len(tr)} 条结果, 前 3 条:")
            for item in tr[:3]:
                print(f"    {str(item)[:140]}")
        else:
            print(f"  result: {str(tr)[:300]}")

        # 4. VERIFY —— 跑验证拿 reward (code 模式, 零成本, 不需要 LLM)
        #    只搜索了没加购物车, 任务未完成, 预期 incomplete
        result = await env.step(CallToolAction(
            tool_name="verify",
            arguments={"verifier_mode": "code", "final_answer": ""},
        ))
        obs = result.observation
        print("\n" + "=" * 64)
        print("4. VERIFY  (verifier_mode='code')")
        print("=" * 64)
        print(f"  reward_type  : {obs.reward_type}")
        print(f"  reward       : {result.reward}")
        print(f"  verify_result: {str(obs.verify_result)[:200]}")

        # 5. DONE —— 结束 episode, 保留 session 看 artifact
        result = await env.step(CallToolAction(
            tool_name="done",
            arguments={"keep_session": True},
        ))
        obs = result.observation
        print("\n" + "=" * 64)
        print("5. DONE")
        print("=" * 64)
        print(f"  done            : {result.done}")
        print(f"  trajectory_path : {obs.trajectory_path}")
        print(f"  session_dir     : {obs.session_dir}")
        print("\n✅ Quick Start 完成! trajectory 已保存, 可离线分析。")


if __name__ == "__main__":
    asyncio.run(main())
