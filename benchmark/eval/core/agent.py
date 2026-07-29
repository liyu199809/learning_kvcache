"""通用 ReAct agent 与健壮的 LLM action 解析。

- ``parse_llm_action_response`` 原样移植自 opd_evolver/base/engine/utils.py，
  是让 4B 小模型稳定产出可执行 action 的关键（含 markdown/残缺 JSON 修复、
  SQL 启发式兜底），请勿随意简化。
- prompt 模板按 ``prompt_type`` 选择；首期提供 db(sql) 一份，其余 benchmark
  后续在此登记即可。
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from .llm_client import AsyncLLM
from .types import Action, BasicInfo, Observation, StepRecord

# ---------------------------------------------------------------------------
# Prompt 模板
# ---------------------------------------------------------------------------
# 移植自 opd_evolver/subagents/react_agent.py 的 SQL_PROMPT（db 任务使用）。
SQL_PROMPT = """You are a SQL expert solving natural language to SQL tasks.

==== Progress ====
[Step {current_step}/{max_steps}] Remaining: {remaining_steps} step(s)
{budget_warning}

==== Task ====
{task_instruction}

🚨 CRITICAL: If task shows "DATABASE: <name>", you MUST execute "USE <name>;" first!

==== Schema Exploration ====
Step 1: SHOW DATABASES;  -- list all databases
Step 2: USE <database_name>;  -- REQUIRED: select the database (e.g., USE pets_1;)
Step 3: SHOW TABLES;  -- list tables in selected database
Step 4: DESCRIBE <table>;  -- examine ONE table per execute (not multiple)
Step 5: SELECT * FROM <table> LIMIT 3;  -- sample data

🚨 CRITICAL:
- You MUST execute "USE <database_name>;" before any table queries
- Execute ONLY ONE SQL statement per action (NO semicolon-separated commands)
- Example: ✗ DESCRIBE t1; DESCRIBE t2;  ✓ DESCRIBE t1;  (then next step: DESCRIBE t2;)

==== Critical SQL Patterns ====
BOTH X AND Y → INTERSECT (NOT AND in WHERE)
EITHER X OR Y → UNION or OR in WHERE
NOT / EXCLUDE → EXCEPT or NOT IN subquery
TOP-1 AGGREGATION → GROUP BY key ORDER BY SUM(col) DESC LIMIT 1

==== Output Format ====
Your LAST execute output is the evaluated answer. It must return ONLY the expected columns/values.
For SELECT tasks, submit evaluates the latest query output. For INSERT/UPDATE/DELETE tasks,
submit evaluates the final table state.

==== Knowledge ====
{context}

==== Action Space ====
{action_space}

==== Memory ====
{memory}

==== Current Observation ====
{obs}

==== Actions ====
execute: {{"action": "execute", "params": {{"command": "SQL query"}}, "memory": "findings"}}
submit:  {{"action": "submit", "params": {{}}}}

Reply with ONLY valid JSON.
"""

# 通用兜底模板（未登记的 prompt_type 使用）。
GENERIC_PROMPT = """You are an autonomous agent solving a task step by step.

==== Progress ====
[Step {current_step}/{max_steps}] Remaining: {remaining_steps} step(s)
{budget_warning}

==== Task ====
{task_instruction}

==== Knowledge ====
{context}

==== Action Space ====
{action_space}

==== Memory ====
{memory}

==== Current Observation ====
{obs}

Reply with ONLY a valid JSON object: {{"action": "<name>", "params": {{}}, "memory": "notes"}}.
"""

PROMPT_TEMPLATES: Dict[str, str] = {
    "db": SQL_PROMPT,
    "sql": SQL_PROMPT,
    "generic": GENERIC_PROMPT,
}


def _fix_invalid_escapes(text: str) -> str:
    out: List[str] = []
    in_string = False
    i = 0
    length = len(text)
    while i < length:
        ch = text[i]
        if ch == '"' and (i == 0 or text[i - 1] != "\\"):
            in_string = not in_string
            out.append(ch)
            i += 1
            continue
        if in_string and ch == "\\" and i + 1 < length:
            next_ch = text[i + 1]
            if next_ch in ('"', "\\", "/", "b", "f", "n", "r", "t"):
                out.append(ch)
                out.append(next_ch)
                i += 2
                continue
            if next_ch == "u" and i + 5 < length:
                out.append(text[i : i + 6])
                i += 6
                continue
            out.append(next_ch)
            i += 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _escape_control_chars_in_strings(text: str) -> str:
    out: List[str] = []
    in_string = False
    escape = False
    for ch in text:
        if escape:
            out.append(ch)
            escape = False
            continue
        if ch == "\\":
            out.append(ch)
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            out.append(ch)
            continue
        if in_string:
            if ch == "\n":
                out.append("\\n")
                continue
            if ch == "\r":
                out.append("\\r")
                continue
            if ch == "\t":
                out.append("\\t")
                continue
        out.append(ch)
    return "".join(out)


def parse_llm_action_response(resp: str) -> Dict[str, Any]:
    """稳健解析 LLM 输出为 action dict。移植自 opd_evolver 的同名函数。"""
    try:
        if not resp:
            return {"action": "no_action", "params": {}, "_parse_error": "Empty LLM response"}

        def _extract_block(marker: str) -> Optional[str]:
            start = resp.find(marker)
            if start == -1:
                return None
            start += len(marker)
            end = resp.find("```", start)
            return resp[start : end if end != -1 else None].strip()

        def _extract_balanced(text: str, open_char: str, close_char: str) -> Optional[str]:
            start = text.find(open_char)
            if start == -1:
                return None
            depth = 0
            in_string = False
            escape = False
            for idx in range(start, len(text)):
                ch = text[idx]
                if escape:
                    escape = False
                    continue
                if ch == "\\":
                    escape = True
                    continue
                if ch == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == open_char:
                    depth += 1
                elif ch == close_char:
                    depth -= 1
                    if depth == 0:
                        return text[start : idx + 1].strip()
            return None

        def _truncate_after_balanced_json(text: str) -> Optional[str]:
            obj_start = text.find("{")
            arr_start = text.find("[")
            starts = [idx for idx in (obj_start, arr_start) if idx != -1]
            if not starts:
                return None
            start = min(starts)
            in_string = False
            escape = False
            stack: List[str] = []
            for idx in range(start, len(text)):
                ch = text[idx]
                if escape:
                    escape = False
                    continue
                if ch == "\\":
                    escape = True
                    continue
                if ch == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch in "[{":
                    stack.append(ch)
                elif ch == "}":
                    if stack and stack[-1] == "{":
                        stack.pop()
                        if not stack:
                            return text[start : idx + 1].strip()
                elif ch == "]":
                    if stack and stack[-1] == "[":
                        stack.pop()
                        if not stack:
                            return text[start : idx + 1].strip()
            return text[start:].strip()

        def _repair_partial_json(text: str) -> Optional[str]:
            candidate = text.strip().strip("`")
            if not candidate:
                return None
            action_match = re.search(r'["\']action["\']\s*:', candidate)
            if action_match:
                brace_before = candidate.rfind("{", 0, action_match.start() + 1)
                if brace_before != -1:
                    candidate = candidate[brace_before:]
                else:
                    candidate = "{" + candidate[action_match.start():]
            truncated = _truncate_after_balanced_json(candidate)
            if truncated:
                candidate = truncated
            rebuilt: List[str] = []
            stack: List[str] = []
            in_string = False
            escape = False
            for ch in candidate:
                if escape:
                    rebuilt.append(ch)
                    escape = False
                    continue
                if ch == "\\":
                    rebuilt.append(ch)
                    escape = True
                    continue
                if ch == '"':
                    rebuilt.append(ch)
                    in_string = not in_string
                    continue
                if not in_string and ch in "[{":
                    stack.append(ch)
                    rebuilt.append(ch)
                    continue
                if not in_string and ch == "}":
                    if stack and stack[-1] == "{":
                        stack.pop()
                        rebuilt.append(ch)
                    continue
                if not in_string and ch == "]":
                    if stack and stack[-1] == "[":
                        stack.pop()
                        rebuilt.append(ch)
                    continue
                rebuilt.append(ch)
            repaired = "".join(rebuilt).strip()
            repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
            while stack:
                open_ch = stack.pop()
                repaired += "}" if open_ch == "{" else "]"
            return repaired or None

        def _normalize_action_data(action_data: Any):
            if isinstance(action_data, list):
                if not action_data:
                    return None, "Empty list returned by LLM"
                action_data = action_data[0]
            if not isinstance(action_data, dict):
                return None, "Missing 'action' key or invalid dict"
            if "action" not in action_data and "name" in action_data:
                action_data = {
                    "action": action_data.get("name"),
                    "params": action_data.get("arguments", {}),
                }
            action = action_data.get("action")
            if not isinstance(action, str) or not action.strip():
                return None, "Missing 'action' key or invalid dict"
            normalized: Dict[str, Any] = dict(action_data)
            normalized["action"] = action.strip().lower()
            params = normalized.get("params")
            if params is None:
                params = {}
            elif not isinstance(params, dict):
                params = {"value": params}
            normalized["params"] = params
            if normalized["action"] == "execute" and "command" not in params:
                for field in ("command", "query", "sql"):
                    value = normalized.get(field)
                    if isinstance(value, str) and value.strip():
                        params["command"] = value.strip()
                        break
            return normalized, None

        def _heuristic_fallback_action(text: str) -> Optional[Dict[str, Any]]:
            block_match = re.search(r"```(?:sql|bash|python)?\s*([\s\S]*?)```", text, re.IGNORECASE)
            if block_match:
                block = block_match.group(1).strip()
                if block and not block.startswith("{") and not block.startswith("["):
                    return {
                        "action": "execute",
                        "params": {"command": block},
                        "_parse_error": "heuristic_fallback_code_block",
                    }
            sql_match = re.search(
                r"(?is)\b(select|with|show|describe|desc|use|insert|update|delete|create|drop|alter)\b[\s\S]*?;",
                text,
            )
            if sql_match:
                return {
                    "action": "execute",
                    "params": {"command": sql_match.group(0).strip()},
                    "_parse_error": "heuristic_fallback_sql",
                }
            lowered = text.lower()
            if re.search(r"\b(sql|database|table|query|schema|select|join|where)\b", lowered):
                return {
                    "action": "execute",
                    "params": {"command": "SHOW TABLES;"},
                    "_parse_error": "heuristic_fallback_sql_intent",
                }
            return None

        def _try_json_loads(text: str):
            try:
                return json.loads(text), None
            except Exception as e:
                original_error = f"{type(e).__name__}: {e}"
                for fixer in (
                    _fix_invalid_escapes,
                    _escape_control_chars_in_strings,
                    lambda t: _fix_invalid_escapes(_escape_control_chars_in_strings(t)),
                ):
                    try:
                        fixed = fixer(text)
                        if fixed != text:
                            return json.loads(fixed), None
                    except Exception:
                        pass
                return None, original_error

        candidates: List[str] = []

        def _append(value: Optional[str]) -> None:
            if value is None:
                return
            cleaned = value.strip()
            if cleaned and cleaned not in candidates:
                candidates.append(cleaned)

        _append(_extract_block("```json"))
        if not candidates:
            _append(_extract_block("```"))
        _append(resp.strip())
        _append(_extract_balanced(resp, "{", "}"))
        _append(_extract_balanced(resp, "[", "]"))
        action_match = re.search(r'["\']action["\']\s*:', resp)
        if action_match:
            brace_before = resp.rfind("{", 0, action_match.start() + 1)
            _append(resp[brace_before:] if brace_before != -1 else resp[action_match.start():])
        for cand in list(candidates):
            _append(_repair_partial_json(cand))

        parse_errors: List[str] = []
        for cand in candidates:
            action_data, err = _try_json_loads(cand)
            if action_data is None:
                parse_errors.append(err or "parse_failed")
                continue
            normalized, normalize_err = _normalize_action_data(action_data)
            if normalized is None:
                parse_errors.append(normalize_err or "normalize_failed")
                continue
            return normalized

        heuristic = _heuristic_fallback_action(resp)
        if heuristic is not None:
            return heuristic
        return {
            "action": "Invalid",
            "params": {},
            "_parse_error": "; ".join(parse_errors) if parse_errors else "Unknown parse failure",
        }
    except Exception as e:
        return {"action": "Invalid", "params": {}, "_parse_error": f"{type(e).__name__}: {e}"}


def parse_memory_field(resp: str) -> Optional[str]:
    """尽力从响应中抽取 memory 字段，用于滚动记忆。"""
    try:
        blob = re.search(r"\{[\s\S]*\}", resp)
        if blob:
            obj = json.loads(blob.group(0))
            if isinstance(obj, dict) and obj.get("memory"):
                return str(obj["memory"])
    except Exception:
        pass
    return None


class ReActAgent:
    """通用 ReAct agent：仅保留最近 N 步文本记忆，不做 LLM 记忆压缩。"""

    def __init__(self, llm: AsyncLLM, prompt_type: str = "db", keep_recent: int = 6):
        self.llm = llm
        self.prompt_type = prompt_type
        self.keep_recent = keep_recent
        self.task_instruction = ""
        self.action_space = ""
        self.context = ""
        self._memory: List[str] = []

    def reset(self, info: BasicInfo) -> None:
        self.task_instruction = info.instruction
        self.action_space = info.action_space
        self.prompt_type = info.prompt_type or self.prompt_type
        self._memory = []

    def _memory_text(self) -> str:
        if not self._memory:
            return "None"
        recent = self._memory[-self.keep_recent :]
        return "\n".join(f"{i + 1}. {line}" for i, line in enumerate(recent))

    def _budget_warning(self, remaining: int) -> str:
        finish = "submit"
        if remaining <= 3:
            return f"🚨 CRITICAL: Only {remaining} steps left! Use '{finish}' NOW!"
        if remaining <= 5:
            return f"⚠️ Warning: {remaining} steps remaining. Plan to finish soon."
        return ""

    def _build_prompt(self, observation: Observation, current_step: int, max_steps: int) -> str:
        remaining = max_steps - current_step
        template = PROMPT_TEMPLATES.get(self.prompt_type, GENERIC_PROMPT)
        return template.format(
            task_instruction=self.task_instruction,
            context=self.context or "No prior knowledge available.",
            action_space=self.action_space,
            memory=self._memory_text(),
            obs=observation,
            current_step=current_step,
            max_steps=max_steps,
            remaining_steps=remaining,
            budget_warning=self._budget_warning(remaining),
        )

    async def step(
        self,
        observation: Observation,
        history: List[StepRecord],
        current_step: int = 1,
        max_steps: int = 30,
    ) -> tuple[Action, str, str]:
        prompt = self._build_prompt(observation, current_step, max_steps)
        resp = await self.llm(prompt)
        action = parse_llm_action_response(resp)

        # 守卫：DB/SQL 环境下，尚未 execute 过就 submit，改为安全探查查询。
        if self.prompt_type in ("db", "sql"):
            act_name = str(action.get("action", "")).lower().strip()
            has_execute = any(
                isinstance(r.action, dict)
                and str(r.action.get("action", "")).lower().strip() == "execute"
                for r in history
            )
            if act_name == "submit" and not has_execute:
                action = {
                    "action": "execute",
                    "params": {"command": "SHOW TABLES;"},
                    "_parse_error": "guardrail_prevent_early_submit",
                }

        mem = parse_memory_field(resp)
        # 把 params 里的 memory 抠出来，避免污染 env 参数
        if isinstance(action.get("params"), dict) and "memory" in action["params"]:
            mem = mem or str(action["params"].pop("memory"))
        self._memory.append(f"act={action.get('action')}, note={mem}")
        return action, resp, prompt
