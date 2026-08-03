"""DB 任务的 native function-calling agent。

模型通过 execute/submit 两个原生工具与 MySQL 环境交互（thinking 保留）；
tool_calls 优先走 native 通道，为空时由 native.py 兜底解析 content 里的 XML
tool_call。旧的文本 JSON 动作协议已移除，仅保留 function-calling 单一路径。
"""
from __future__ import annotations

import json
from typing import List, Optional

from .llm_client import AsyncLLM
from .types import Action, BasicInfo, Observation, StepRecord
from .native import native_turn
from .tool_schemas import (
    DB_TOOLS, DB_NATIVE_SYSTEM_PROMPT,
    OS_TOOLS, OS_NATIVE_SYSTEM_PROMPT,
)


class NativeReActAgent:
    """走原生 tool calling 的 DB agent（与 OPSD 训练同源）。

    维护完整多轮 messages；用 llm.chat_message(tools=...) 让模型吐 native
    tool_calls（为空时 native.py 会兜底解析 content 里的 XML tool_call）；
    取第一个动作映射回 {"action","params"} 交给 env.step。thinking 保留。

    修复：末步不禁用工具（始终 tool_choice="auto"），避免模型把 submit 以
    XML 文本吐出而丢答案。
    """

    def __init__(self, llm: AsyncLLM, prompt_type: str = "db",
                 tools: Optional[List[dict]] = None,
                 system_prompt: Optional[str] = None,
                 max_tokens: Optional[int] = None):
        self.llm = llm
        self.prompt_type = prompt_type
        self.tools = tools if tools is not None else DB_TOOLS
        self.system_prompt = system_prompt or DB_NATIVE_SYSTEM_PROMPT
        self.max_tokens = max_tokens
        self.task_instruction = ""
        self.messages: List[dict] = []
        self._pending_tool_call_id: Optional[str] = None
        self.last_debug: Optional[dict] = None

    def reset(self, info: BasicInfo) -> None:
        self.task_instruction = info.instruction
        self.prompt_type = info.prompt_type or self.prompt_type
        # 每个 benchmark 通过 meta_data 注入 native 配置；缺省回退 DB。
        meta = info.meta_data or {}
        self.tools = meta.get("native_tools") or self.tools or DB_TOOLS
        self.system_prompt = (
            meta.get("native_system_prompt") or self.system_prompt or DB_NATIVE_SYSTEM_PROMPT
        )
        # 结算动作名（DB=submit, OS=finish）与提前结算守卫命令（None 表示不加守卫）。
        self.finish_action = str(meta.get("finish_action", "submit")).lower()
        self.guardrail_command = meta.get("guardrail_command", "SHOW TABLES;")
        self.messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": info.instruction},
        ]
        self._pending_tool_call_id = None
        self.last_debug = None

    def observe(self, observation: Observation) -> None:
        """把上一步 env 输出作为 tool 响应回填（配对上一轮 tool_call）。"""
        if self._pending_tool_call_id is None:
            return
        text = observation.get("output")
        if text is None:
            text = observation.get("message") or observation.get("error") or ""
        self.messages.append({
            "role": "tool",
            "tool_call_id": self._pending_tool_call_id,
            "content": str(text),
        })
        self._pending_tool_call_id = None

    async def step(
        self,
        observation: Observation,
        history: List[StepRecord],
        current_step: int = 1,
        max_steps: int = 30,
    ) -> tuple[Action, str, str]:
        self.observe(observation)

        # 最后一轮：注入运行时提醒，强制模型 submit（工具仍保持 auto，不禁用，
        # 避免禁用工具导致模型把 submit 以 XML 文本吐出而丢答案）。
        if current_step >= max_steps:
            self.messages.append({
                "role": "user",
                "content": (
                    f"This is the LAST round. You MUST call the {self.finish_action} "
                    f"tool now to conclude the task. Do not call execute again."
                ),
            })

        input_messages = [dict(m) for m in self.messages]
        turn = await native_turn(
            self.llm, self.messages,
            tools=self.tools, tool_choice="auto", max_tokens=self.max_tokens,
        )
        assistant_msg = self.messages[-1] if self.messages else {}

        raw_response = turn.content or ""
        if turn.reasoning:
            raw_response = f"<think>{turn.reasoning}</think>\n{raw_response}"
        if not raw_response and turn.tool_calls:
            raw_response = json.dumps(turn.tool_calls, ensure_ascii=False)

        action = self._tool_call_to_action(turn.tool_calls, history)

        self.last_debug = {
            "input_messages": input_messages,
            "tool_choice": "auto",
            "assistant_message": assistant_msg,
            "reasoning": turn.reasoning,
            "content": turn.content,
            "tool_calls": turn.tool_calls,
        }
        prompt = json.dumps(input_messages, ensure_ascii=False, default=str)
        return action, raw_response, prompt

    def _tool_call_to_action(self, tool_calls: List[dict],
                             history: List[StepRecord]) -> Action:
        if not tool_calls:
            self._pending_tool_call_id = None
            return {"action": getattr(self, "finish_action", "submit"), "params": {},
                    "_parse_error": "no_tool_call"}
        tc = tool_calls[0]
        self._pending_tool_call_id = tc.get("id") or None
        fn = tc.get("function", {}) or {}
        name = str(fn.get("name", "")).strip().lower()
        raw_args = fn.get("arguments", "") or "{}"
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
        except Exception:
            args = {}
        if not isinstance(args, dict):
            args = {}

        finish_action = getattr(self, "finish_action", "submit")
        guardrail_cmd = getattr(self, "guardrail_command", "SHOW TABLES;")
        if name in (finish_action, "submit", "finish"):
            # 提前结算守卫：尚未 execute 过就结算，且配置了守卫命令时，改为安全探查。
            if guardrail_cmd:
                has_execute = any(
                    isinstance(r.action, dict)
                    and str(r.action.get("action", "")).lower().strip() == "execute"
                    for r in history
                )
                if not has_execute:
                    return {"action": "execute", "params": {"command": guardrail_cmd},
                            "_parse_error": "guardrail_prevent_early_submit"}
            return {"action": finish_action, "params": {k: v for k, v in args.items()}}
        if name == "execute":
            command = str(args.get("command", "")).strip()
            return {"action": "execute", "params": {"command": command}}
        return {"action": name or "Invalid", "params": args,
                "_parse_error": f"unknown_tool:{name}"}
