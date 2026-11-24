"""OpenEnv-compatible adapter for RUC-NLPIR/Agent-World executable environments."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import threading
from typing import Any
from urllib.request import Request, urlopen

from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction, Tool
from pydantic import BaseModel, Field


_ENV_LOCK = threading.Lock()


class AgentWorldObservation(BaseModel):
    done: bool = False
    reward: float | None = None
    reward_type: str = "tool_call_ok"
    error: str | None = None
    scenario: str | None = None
    task: str | None = None
    task_idx: int | None = None
    task_id: str | None = None
    tool_result: Any = None
    verify_result: dict[str, Any] | None = None
    num_tools: int | None = None


class AgentWorldListToolsObservation(BaseModel):
    done: bool = False
    error: str | None = None
    tools: list[Tool] = Field(default_factory=list)


def _normalize_schema(params: Any) -> dict[str, Any]:
    if isinstance(params, dict) and params.get("type") == "object":
        out = dict(params)
        out.setdefault("properties", {})
        return out
    if isinstance(params, dict) and "properties" in params:
        return {"type": "object", **params}
    return {"type": "object", "properties": {}}


def _snapshot_db(db_dir: str) -> dict[str, str]:
    snap: dict[str, str] = {}
    root = Path(db_dir)
    if not root.exists():
        return snap
    for path in sorted(root.rglob("*")):
        if path.is_file():
            rel = str(path.relative_to(root))
            try:
                snap[rel] = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                snap[rel] = "<unreadable>"
    return snap


def _diff_db(before: dict[str, str], after: dict[str, str], max_chars: int = 6000) -> dict[str, Any]:
    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    modified = {}
    for k in sorted(set(before) & set(after)):
        if before[k] != after[k]:
            # Summarize JSON changes cleanly when possible
            try:
                b_obj, a_obj = json.loads(before[k]), json.loads(after[k])
                if isinstance(b_obj, list) and isinstance(a_obj, list):
                    b_set = {json.dumps(x, sort_keys=True, ensure_ascii=False) for x in b_obj}
                    a_set = {json.dumps(x, sort_keys=True, ensure_ascii=False) for x in a_obj}
                    new_items = [json.loads(x) for x in list(a_set - b_set)[:8]]
                    del_items = [json.loads(x) for x in list(b_set - a_set)[:8]]
                    modified[k] = {
                        "before_len": len(b_obj),
                        "after_len": len(a_obj),
                        "added_records": new_items,
                        "removed_records": del_items,
                    }
                    continue
            except Exception:
                pass
            modified[k] = {"after_preview": after[k][:1200]}
    diff = {"added_files": added, "removed_files": removed, "modified_files": modified}
    text = json.dumps(diff, ensure_ascii=False)
    if len(text) > max_chars:
        return {"truncated": True, "summary": text[:max_chars]}
    return diff


def _bind_tools(tool_schemas: list[dict[str, Any]], work_dir: str, db_dir: str) -> dict[str, Any]:
    with _ENV_LOCK:
        prev_db = os.environ.get("MCP_DB_DIR")
        prev_cwd = os.getcwd()
        os.environ["MCP_DB_DIR"] = db_dir
        os.chdir(work_dir)
        ns: dict[str, Any] = {"__name__": "agent_world_environment"}
        fns: dict[str, Any] = {}
        try:
            for item in tool_schemas:
                src = item.get("implementation") or ""
                if not src:
                    continue
                try:
                    exec(src, ns)
                except Exception:
                    continue
                fn_name = re.sub(r"[^0-9A-Za-z_]", "_", item["name"])
                fn = ns.get(fn_name)
                if callable(fn):
                    fns[item["name"]] = fn
        finally:
            os.chdir(prev_cwd)
            if prev_db is None:
                os.environ.pop("MCP_DB_DIR", None)
            else:
                os.environ["MCP_DB_DIR"] = prev_db
    return fns


def _invoke_tool(fns: dict[str, Any], name: str, arguments: dict[str, Any], work_dir: str, db_dir: str) -> Any:
    fn = fns.get(name)
    if fn is None:
        return {"error": f"Unknown tool: {name}"}
    with _ENV_LOCK:
        prev_db = os.environ.get("MCP_DB_DIR")
        prev_cwd = os.getcwd()
        os.environ["MCP_DB_DIR"] = db_dir
        os.chdir(work_dir)
        try:
            res = fn(**(arguments or {}))
        except Exception as exc:
            res = {"error": f"{type(exc).__name__}: {exc}"}
        finally:
            os.chdir(prev_cwd)
            if prev_db is None:
                os.environ.pop("MCP_DB_DIR", None)
            else:
                os.environ["MCP_DB_DIR"] = prev_db
    try:
        json.dumps(res, ensure_ascii=False)
        return res
    except TypeError:
        return str(res)


def _load_judge_credentials() -> tuple[str, str, str]:
    base_url = os.environ.get("VERIFIER_BASE_URL") or os.environ.get("TEACHER_BASE_URL") or "https://llmapi.paratera.com/v1"
    model = os.environ.get("VERIFIER_MODEL") or os.environ.get("TEACHER_MODEL") or "GLM-5.3"
    key = os.environ.get("VERIFIER_API_KEY") or os.environ.get("TEACHER_API_KEY") or os.environ.get("JD_LLM_API_KEY")
    if not key:
        for cand in (
            Path.home() / ".config/kvcache-learning/teacher.key",
            Path.home() / ".config/kvcache-learning/jd-teacher.key",
        ):
            if cand.exists():
                key = cand.read_text().strip()
                break
    if not key:
        raise RuntimeError("Verifier/Teacher API key is missing")
    return base_url.rstrip("/"), model, key


def _extract_json_object(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if m:
        return json.loads(m.group(1))
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError(f"Could not parse JSON from judge output: {text[:300]}")


def _call_rubric_judge(
    task: str,
    reference_answer: str,
    success_criteria: list[str],
    ref_chain_outputs: list[dict[str, Any]],
    ref_db_diff: dict[str, Any],
    student_trajectory: list[dict[str, Any]],
    student_db_diff: dict[str, Any],
    final_answer: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    base_url, model, key = _load_judge_credentials()
    system = (
        "You are a strict, objective benchmark verifier. Evaluate whether the student's tool "
        "execution trace, resulting database state, and final answer satisfy EVERY success criterion.\n"
        "Rules:\n"
        "- Ground truth comes ONLY from the verified reference outputs, reference DB diff, and student's actual tool results.\n"
        "- Ignore volatile timestamps, request IDs, or newly generated random IDs unless a criterion specifically requires matching an existing ID.\n"
        "- A criterion passes ONLY if all required facts/values/actions in that criterion are genuinely achieved and accurately reported.\n"
        "- Output valid JSON ONLY matching this exact schema:\n"
        '{"checks": [{"index": 0, "passed": true, "reason": "brief factual justification"}]}'
    )
    # Keep prompt compact while retaining all verifiable facts
    def _trim(obj: Any, limit: int) -> str:
        s = json.dumps(obj, ensure_ascii=False)
        return s if len(s) <= limit else s[:limit] + "...[TRUNCATED]"

    user = (
        f"## Task\n{task}\n\n"
        f"## Success Criteria ({len(success_criteria)} total)\n"
        + "\n".join(f"[{i}] {c}" for i, c in enumerate(success_criteria))
        + f"\n\n## Reference Answer\n{reference_answer[:4000]}\n\n"
        f"## Reference Verified Tool Chain Replay Outputs\n{_trim(ref_chain_outputs, 6000)}\n\n"
        f"## Reference DB Diff\n{_trim(ref_db_diff, 3000)}\n\n"
        f"## Student Tool Calls & Observations ({len(student_trajectory)} calls)\n{_trim(student_trajectory, 12000)}\n\n"
        f"## Student DB Diff\n{_trim(student_db_diff, 3000)}\n\n"
        f"## Student Final Answer\n{final_answer[:6000]}"
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.0,
        "max_tokens": int(os.environ.get("VERIFIER_MAX_TOKENS", "4096")),
    }
    if os.environ.get("AGENT_WORLD_STRUCTURED_VERIFIER") == "1":
        payload["reasoning_effort"] = "low"
        payload["response_format"] = {"type": "json_object"}
    req = Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urlopen(req, timeout=180) as resp:
        raw = json.load(resp)
    msg = raw["choices"][0]["message"]
    content = msg.get("content") or ""
    usage_raw = raw.get("usage") or {}
    usage = {
        "model": model,
        "calls": 1,
        "prompt_tokens": int(usage_raw.get("prompt_tokens") or 0),
        "completion_tokens": int(usage_raw.get("completion_tokens") or 0),
        "reasoning_tokens": int((usage_raw.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0),
        "cached_tokens": int((usage_raw.get("prompt_tokens_details") or {}).get("cached_tokens") or 0),
        "total_tokens": int(usage_raw.get("total_tokens") or 0),
    }
    parsed = _extract_json_object(content)
    return parsed, usage


class AgentWorldEnvironment:
    """Isolated executable environment backed by RUC-NLPIR/Agent-World."""

    def __init__(self, data_dir: str | Path | None = None) -> None:
        self._data_dir = Path(
            data_dir
            or os.environ.get("AGENT_WORLD_DATA_DIR")
            or "/host-ssd/Agent-World/environment_mix"
        )
        self._work_dir: str | None = None
        self._db_dir: str | None = None
        self._scenario: str | None = None
        self._task_idx: int | None = None
        self._task: dict[str, Any] | None = None
        self._tool_schemas: list[dict[str, Any]] = []
        self._fns: dict[str, Any] = {}
        self._initial_db: dict[str, str] = {}
        self._trajectory: list[dict[str, Any]] = []
        self._ref_cache: tuple[list[dict[str, Any]], dict[str, Any]] | None = None
        self._done = False

    def close(self) -> None:
        if self._work_dir and os.path.exists(self._work_dir):
            shutil.rmtree(self._work_dir, ignore_errors=True)
        self._work_dir = None
        self._db_dir = None

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        scenario: str | None = None,
        task_idx: int | None = None,
        **kwargs: Any,
    ) -> AgentWorldObservation:
        del seed, episode_id, kwargs
        self.close()
        if not scenario or task_idx is None:
            return AgentWorldObservation(reward_type="reset_error", error="scenario and task_idx required")
        ckpt_path = self._data_dir / f"{scenario}_step4_checkpoint.json"
        db_src = self._data_dir / scenario
        q_path = self._data_dir / "questions" / f"{scenario}.json"
        if not ckpt_path.exists() or not db_src.is_dir() or not q_path.exists():
            return AgentWorldObservation(
                reward_type="reset_error",
                scenario=scenario,
                task_idx=int(task_idx),
                error=f"Missing checkpoint/db/questions for env_id={scenario}",
            )
        ckpt = json.loads(ckpt_path.read_text(encoding="utf-8"))
        q_data = json.loads(q_path.read_text(encoding="utf-8"))
        tasks = q_data.get("tasks") or []
        idx = int(task_idx)
        if idx < 0 or idx >= len(tasks):
            return AgentWorldObservation(
                reward_type="reset_error",
                scenario=scenario,
                task_idx=idx,
                error=f"task_idx {idx} out of range (0..{len(tasks)-1})",
            )
        tool_schemas = ckpt.get("data", {}).get("ToolDesignAgent", {}).get("tool_schemas", [])
        work_dir = tempfile.mkdtemp(prefix=f"aw_{scenario}_")
        db_dir = os.path.join(work_dir, "database")
        shutil.copytree(db_src, db_dir)
        fns = _bind_tools(tool_schemas, work_dir, db_dir)

        self._work_dir = work_dir
        self._db_dir = db_dir
        self._scenario = scenario
        self._task_idx = idx
        self._task = tasks[idx]
        self._tool_schemas = [t for t in tool_schemas if t["name"] in fns]
        self._fns = fns
        self._initial_db = _snapshot_db(db_dir)
        self._trajectory = []
        self._ref_cache = None
        self._done = False
        return AgentWorldObservation(
            reward_type="reset_ok",
            scenario=scenario,
            task=str(self._task["task"]),
            task_idx=idx,
            task_id=f"{scenario}:{idx}",
            num_tools=len(self._tool_schemas),
        )

    def _replay_verified_chain(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if self._ref_cache is not None:
            return self._ref_cache
        chain = (self._task or {}).get("grading_rubric", {}).get("verified_tool_chain") or []
        ref_dir = tempfile.mkdtemp(prefix=f"aw_ref_{self._scenario}_")
        ref_db = os.path.join(ref_dir, "database")
        try:
            shutil.copytree(self._data_dir / str(self._scenario), ref_db)
            ref_before = _snapshot_db(ref_db)
            ref_fns = _bind_tools(self._tool_schemas, ref_dir, ref_db)
            outputs = []
            for step_i, item in enumerate(chain):
                tname = item.get("tool")
                targs = item.get("args") or {}
                res = _invoke_tool(ref_fns, tname, targs, ref_dir, ref_db)
                outputs.append({"step": step_i, "tool": tname, "args": targs, "result": res})
            ref_diff = _diff_db(ref_before, _snapshot_db(ref_db))
        finally:
            shutil.rmtree(ref_dir, ignore_errors=True)
        self._ref_cache = (outputs, ref_diff)
        return self._ref_cache

    def _verify(self, action: CallToolAction) -> AgentWorldObservation:
        assert self._task is not None and self._db_dir is not None
        final_answer = str((action.arguments or {}).get("final_answer") or "")
        rubric = self._task.get("grading_rubric") or {}
        criteria = list(rubric.get("success_criteria") or [])
        if not criteria:
            return AgentWorldObservation(
                reward=0.0,
                reward_type="no_verifier",
                error="Task has no success_criteria",
            )
        ref_outputs, ref_db_diff = self._replay_verified_chain()
        student_db_diff = _diff_db(self._initial_db, _snapshot_db(self._db_dir))
        try:
            parsed, usage = _call_rubric_judge(
                task=str(self._task["task"]),
                reference_answer=str(self._task.get("reference_answer") or ""),
                success_criteria=criteria,
                ref_chain_outputs=ref_outputs,
                ref_db_diff=ref_db_diff,
                student_trajectory=self._trajectory,
                student_db_diff=student_db_diff,
                final_answer=final_answer,
            )
        except Exception as exc:
            return AgentWorldObservation(
                reward=None,
                reward_type="judge_error",
                error=f"{type(exc).__name__}: {exc}",
            )
        checks = parsed.get("checks") or []
        passed_map = {int(c["index"]): bool(c.get("passed")) for c in checks if isinstance(c, dict) and "index" in c}
        passed_checks = sum(1 for i in range(len(criteria)) if passed_map.get(i, False))
        total_checks = len(criteria)
        reward = 1.0 if passed_checks == total_checks else passed_checks / total_checks
        return AgentWorldObservation(
            done=False,
            reward=reward,
            reward_type="rubric_judge",
            verify_result={
                "passed_checks": passed_checks,
                "total_checks": total_checks,
                "checks": checks,
                "student_db_diff": student_db_diff,
                "verifier_usage": usage,
            },
        )

    def step(self, action: Any, **kwargs: Any) -> AgentWorldObservation | AgentWorldListToolsObservation:
        del kwargs
        if self._done:
            return AgentWorldObservation(done=True, reward_type="episode_already_done", error="Episode already done")
        if isinstance(action, ListToolsAction):
            tools = [
                Tool(
                    name=t["name"],
                    description=t.get("description") or "",
                    input_schema=_normalize_schema(t.get("parameters")),
                )
                for t in self._tool_schemas
            ]
            return AgentWorldListToolsObservation(tools=tools)
        if not isinstance(action, CallToolAction):
            return AgentWorldObservation(reward=0.0, reward_type="invalid_action", error=f"Unsupported action {type(action)}")
        if action.tool_name == "verify":
            return self._verify(action)
        if action.tool_name == "done":
            self._done = True
            return AgentWorldObservation(done=True, reward_type="done")
        assert self._work_dir is not None and self._db_dir is not None
        args = action.arguments or {}
        result = _invoke_tool(self._fns, action.tool_name, args, self._work_dir, self._db_dir)
        self._trajectory.append({"tool": action.tool_name, "arguments": args, "result": result})
        return AgentWorldObservation(reward_type="tool_call_ok", tool_result=result)
