"""LifelongAgentBench / DB 任务的原生工具 schema（手写，OpenAI function 格式）。

两个工具与 env 内部 action 一一对应：
  - execute(command): 对已初始化的 MySQL 执行一条 SQL
  - submit(answer?):  提交最终答案。SELECT 任务评估最近一次 execute 的输出；
                      INSERT/UPDATE/DELETE 任务评估最终表状态（answer 可省）。

系统提示（DB_NATIVE_SYSTEM_PROMPT）绝大部分与官方 LifelongAgentBench 的
TASK_REQUIREMENT_DICT[DB_BENCH] 保持一致，仅把"动作格式"从官方的
`Action: Operation`/`Action: Answer` 文本协议替换为 execute/submit 工具调用，
附两个工具调用示例，并提醒模型最后一轮必须调用 submit。
"""
from __future__ import annotations

DB_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "execute",
            "description": (
                "Execute exactly ONE SQL statement against the initialized MySQL "
                "database and return its raw output. Do NOT put multiple "
                "semicolon-separated statements in one call."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "A single SQL statement to execute.",
                    }
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": (
                "Commit your final answer. For querying tasks (SELECT), pass the "
                "query result as `answer`; for modifying tasks (INSERT/UPDATE/DELETE), "
                "`answer` can be anything. Only call this when you are sure."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": (
                            "Final answer. For SELECT, the query result like "
                            "[(1, 'John Doe', 'HR'), (2, 'Jane Smith', 'IT'), ...]."
                        ),
                    }
                },
                "required": [],
            },
        },
    },
]


# 绝大部分沿用官方 TASK_REQUIREMENT_DICT[DB_BENCH]，仅把动作格式改为工具调用，
# 附两个工具调用示例，并强调最后一轮必须 submit。
DB_NATIVE_SYSTEM_PROMPT = """I will ask you a question, then you should help me operate a MySQL database with SQL to answer the question.
You have to explain the problem and your solution to me and write down your thoughts.
After thinking and explaining thoroughly, every round you can call one of the two tools: `execute` or `submit`.

To operate the database, call the `execute` tool with a single SQL statement in the `command` argument.
Every time you can only execute one SQL statement. Every time you call `execute`, I will execute the SQL for you and give you the output.
If the SQL is not executed successfully, the response will be the error message.
Otherwise, the response will be the raw MySQL response.
For SELECT queries, the response will be the result of the query, such as [(1, 'John Doe', 'HR'), (2, 'Jane Smith', 'IT'), ...], where each tuple represents a row and the elements are the values of the columns in the row.
For SQL such as INSERT, UPDATE, and DELETE, the response will be an empty list [] indicating that the SQL was executed successfully.

If you have obtained the answer by interacting with the database, you MUST commit your final answer by calling the `submit` tool.
For querying tasks, pass the query result as the `answer` argument, e.g. answer="[(1, 'John Doe', 'HR'), (2, 'Jane Smith', 'IT'), ...]".
DO NOT call `submit` unless you are sure about your answer. I expect an accurate and correct answer.
Your answer should be accurate. Your answer must be exactly the same as the correct answer.
If the question is about modifying the database, then after done operation, your answer field can be anything.
If the question is about querying the database, then after done operation, your answer field should be the result of the query.
We note that the column names will not be displayed in the result, and you need to ensure both the orders of the columns and rows are correct.

Here are two examples of how to call the tools:
Example 1 (run a SQL statement): call the `execute` tool with
  {"command": "SELECT name, department FROM employees WHERE salary > 5000;"}
Example 2 (commit the final answer for a SELECT task): call the `submit` tool with
  {"answer": "[(1, 'John Doe', 'HR'), (2, 'Jane Smith', 'IT')]"}

The task will finish once you call `submit` or the number of rounds reaches the limit, and the system will judge whether you pass the task or not.
On the LAST round you MUST call `submit` to commit your answer, otherwise the task will be judged as FAIL."""
