"""极简 OpenAI 兼容异步 LLM 客户端（对接 vLLM）。

从 opd_evolver/base/engine/async_llm.py 抽取核心逻辑并瘦身：
- 去掉 pricing / cost_monitor / 图像生成；
- 保留对 Qwen3 reasoning-parser 的 reasoning_content 兼容抽取。
"""
from __future__ import annotations

import threading
from typing import Any, Optional

from openai import AsyncOpenAI


def _assistant_message_text(message: Any) -> str:
    """从 chat.completions 的 message 中稳健抽取文本内容。

    兼容三种情况：普通字符串 content、list 形式的 content blocks、
    以及 vLLM reasoning-parser 返回的 reasoning_content 字段。
    """
    raw = getattr(message, "content", None)
    if isinstance(raw, str) and raw.strip():
        return raw
    if isinstance(raw, list):
        chunks: list[str] = []
        for block in raw:
            if isinstance(block, dict):
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    chunks.append(block["text"])
            elif isinstance(block, str):
                chunks.append(block)
        merged = "".join(chunks).strip()
        if merged:
            return merged
    reasoning = getattr(message, "reasoning_content", None)
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning
    if hasattr(message, "model_dump"):
        try:
            extra = message.model_dump()
        except Exception:
            extra = {}
        for key in ("reasoning_content", "reasoning"):
            val = extra.get(key)
            if isinstance(val, str) and val.strip():
                return val
    return raw if isinstance(raw, str) else ""


class AsyncLLM:
    """一个可 await 调用的 LLM 封装：``await llm(prompt) -> str``。"""

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str = "EMPTY",
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_completion_tokens: Optional[int] = None,
        system_msg: Optional[str] = None,
    ):
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.top_p = top_p
        self.max_completion_tokens = max_completion_tokens
        self.system_msg = system_msg
        self.aclient = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._lock = threading.Lock()
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.call_count = 0

    async def __call__(self, prompt: Any, max_tokens: Optional[int] = None) -> str:
        messages: list[dict[str, Any]] = []
        if self.system_msg:
            messages.append({"role": "system", "content": self.system_msg})
        if isinstance(prompt, (str, list)):
            messages.append({"role": "user", "content": prompt})
        else:
            raise ValueError(f"prompt must be str or list, got {type(prompt)}")

        tokens_to_use = max_tokens if max_tokens is not None else self.max_completion_tokens
        sampling = {"temperature": self.temperature, "top_p": self.top_p}
        try:
            create_kwargs: dict[str, Any] = dict(model=self.model, messages=messages, **sampling)
            if tokens_to_use is not None:
                create_kwargs["max_tokens"] = tokens_to_use
            response = await self.aclient.chat.completions.create(**create_kwargs)
        except Exception as e:
            # 调用失败返回空串，由上层解析成 Invalid action，不中断整条 episode。
            print(f"[AsyncLLM] call failed: {type(e).__name__}: {e}")
            return ""

        if response is None or not getattr(response, "choices", None):
            return ""
        usage = getattr(response, "usage", None)
        if usage is not None:
            with self._lock:
                self.total_input_tokens += getattr(usage, "prompt_tokens", 0) or 0
                self.total_output_tokens += getattr(usage, "completion_tokens", 0) or 0
                self.call_count += 1
        return _assistant_message_text(response.choices[0].message)

    async def chat_message(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: Optional[list[dict]] = None,
        tool_choice: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ):
        """返回完整 assistant message（可读 .content / .tool_calls / .reasoning_content）。

        tools 提供时透传给端点以启用 native function-calling。失败返回 None。
        """
        tokens_to_use = max_tokens if max_tokens is not None else self.max_completion_tokens
        sampling = {"temperature": self.temperature, "top_p": self.top_p}
        try:
            create_kwargs: dict[str, Any] = dict(model=self.model, messages=messages, **sampling)
            if tokens_to_use is not None:
                create_kwargs["max_tokens"] = tokens_to_use
            if tools is not None:
                create_kwargs["tools"] = tools
                if tool_choice is not None:
                    create_kwargs["tool_choice"] = tool_choice
            response = await self.aclient.chat.completions.create(**create_kwargs)
        except Exception as e:
            print(f"[AsyncLLM] chat_message failed: {type(e).__name__}: {e}")
            return None
        if response is None or not getattr(response, "choices", None):
            return None
        usage = getattr(response, "usage", None)
        if usage is not None:
            with self._lock:
                self.total_input_tokens += getattr(usage, "prompt_tokens", 0) or 0
                self.total_output_tokens += getattr(usage, "completion_tokens", 0) or 0
                self.call_count += 1
        return response.choices[0].message

    def usage_summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "model": self.model,
                "total_input_tokens": self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
                "call_count": self.call_count,
            }
