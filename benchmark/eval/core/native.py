"""Native function-calling helpers — 移植自 rollout/common.py（解耦，不 import rollout/openenv）。

含 native tool_calls 抽取、reasoning 处理、以及对 Qwen3-coder XML 风格
tool_call 的兜底解析（当 native tool_calls 通道为空但模型把工具调用以
`<function=...><parameter=...>` 文本形式吐在 content 时）。
"""
from __future__ import annotations

import json
import re
import uuid
from typing import Any, NamedTuple, Optional


class LLMTurnNative(NamedTuple):
    content: str              # assistant 文本（可为 ""），已剥离 reasoning
    tool_calls: list[dict]    # 序列化后的 tool_calls（native 或 XML 兜底，可为空）
    reasoning: str            # 抽取到的 reasoning/think（可为 ""）


_THINK_FULL_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINK_PREFIX_RE = re.compile(r"^.*?</think>", re.DOTALL)
_THINK_CAPTURE_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)

# Qwen3-coder XML tool-call 格式：
#   <tool_call>
#   <function=NAME>
#   <parameter=KEY>
#   VALUE
#   </parameter>
#   </function>
#   </tool_call>
_XML_FUNC_RE = re.compile(r"<function=([A-Za-z0-9_]+)>", re.DOTALL)
_XML_PARAM_RE = re.compile(r"<parameter=([A-Za-z0-9_]+)>(.*?)</parameter>", re.DOTALL)


def strip_think(text: str) -> str:
    if not text:
        return ""
    text = _THINK_FULL_RE.sub("", text)
    text = _THINK_PREFIX_RE.sub("", text)
    return text.strip()


def _extract_reasoning(msg: Any) -> Optional[str]:
    """读取 `--reasoning-parser qwen3` 拆出的 reasoning 字段（reasoning_content/reasoning）。"""
    for attr in ("reasoning_content", "reasoning"):
        val = getattr(msg, attr, None)
        if val:
            return str(val).strip()
    return None


def _reasoning_from_content(content: str) -> str:
    if not content:
        return ""
    m = _THINK_CAPTURE_RE.search(content)
    if m:
        return m.group(1).strip()
    if "</think>" in content:
        return content.split("</think>", 1)[0].strip()
    return ""


def serialize_tool_calls(msg: Any) -> Optional[list[dict]]:
    """把 native assistant 消息的 pydantic `tool_calls` 提成可 JSON 序列化的 dict。"""
    tcs = getattr(msg, "tool_calls", None) or []
    if not tcs:
        return None
    out: list[dict] = []
    for tc in tcs:
        fn = getattr(tc, "function", None)
        out.append({
            "id": getattr(tc, "id", "") or "",
            "type": getattr(tc, "type", "function") or "function",
            "function": {
                "name": getattr(fn, "name", "") or "",
                "arguments": getattr(fn, "arguments", "") or "",
            },
        })
    return out


def parse_xml_tool_calls(content: str) -> Optional[list[dict]]:
    """兜底：从 content 里解析 Qwen3-coder XML 风格的 tool_call。

    当 `tool_choice` 使 native 解析失效、或模型把工具调用当文本吐出时使用。
    只取第一个 <function=...> 块（本环境每轮一个动作）。返回与
    serialize_tool_calls 相同的结构，无匹配时返回 None。
    """
    if not content or "<function=" not in content:
        return None
    fm = _XML_FUNC_RE.search(content)
    if not fm:
        return None
    name = fm.group(1)
    # 参数：截取该 function 块内的 <parameter=...>...</parameter>
    block = content[fm.end():]
    next_fn = _XML_FUNC_RE.search(block)
    if next_fn:
        block = block[:next_fn.start()]
    args: dict[str, Any] = {}
    for pm in _XML_PARAM_RE.finditer(block):
        args[pm.group(1)] = pm.group(2).strip()
    return [{
        "id": f"xmlfallback-{uuid.uuid4().hex[:12]}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
    }]


def _prepare_native_messages(messages: list[dict]) -> list[dict]:
    """构造 wire payload：丢弃自记账的 reasoning 字段；仅把最近一轮 assistant 的
    reasoning 以 <think> 重新 inline 回其 content（连续性但不膨胀历史）。不改入参。"""
    last_assistant = -1
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "assistant":
            last_assistant = i
            break
    wire: list[dict] = []
    for i, m in enumerate(messages):
        wm = {k: v for k, v in m.items() if k != "reasoning"}
        if (i == last_assistant and m.get("role") == "assistant"
                and m.get("reasoning")):
            think = f"<think>{m['reasoning']}</think>"
            base = wm.get("content") or ""
            wm["content"] = f"{think}\n{base}" if base else think
        wire.append(wm)
    return wire


async def native_turn(
    llm: Any,
    messages: list[dict],
    *,
    max_tokens: Optional[int] = None,
    tools: Optional[list[dict]] = None,
    tool_choice: str = "auto",
) -> LLMTurnNative:
    """一次 native 轮次：带 `tools` 调用 LLM，把规范的 native assistant 消息追加到
    `messages`，返回 LLMTurnNative。tool_calls 优先取 native 通道，为空时回退到
    content 里的 XML 兜底解析。只保留第一个 tool_call（每轮一个动作，保证 tool 响应配对）。
    """
    wire = _prepare_native_messages(messages)
    msg = await llm.chat_message(wire, tools=tools, tool_choice=tool_choice,
                                 max_tokens=max_tokens)
    if msg is None:
        messages.append({"role": "assistant", "content": ""})
        return LLMTurnNative("", [], "")

    raw_content = getattr(msg, "content", None) or ""
    reasoning = _extract_reasoning(msg)
    if reasoning is None:
        reasoning = _reasoning_from_content(raw_content)
        clean_content = strip_think(raw_content)
    else:
        clean_content = raw_content.strip()

    tool_calls = serialize_tool_calls(msg)
    if not tool_calls:
        # 兜底：模型把工具调用以 XML 文本吐在 content（tool_choice 限制 / 习惯）。
        tool_calls = parse_xml_tool_calls(clean_content) or parse_xml_tool_calls(raw_content)
    if tool_calls:
        tool_calls = tool_calls[:1]

    assistant_msg: dict = {"role": "assistant", "content": clean_content or None}
    if reasoning:
        assistant_msg["reasoning"] = reasoning
    if tool_calls:
        assistant_msg["tool_calls"] = tool_calls
    messages.append(assistant_msg)

    return LLMTurnNative(clean_content, tool_calls or [], reasoning or "")
