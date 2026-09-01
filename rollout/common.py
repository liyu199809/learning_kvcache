"""
Shared plumbing for rollout and critic:
  - RetryLLM: AsyncOpenAI wrapper with per-call timeout + exponential-backoff
    retry, exposing `chat_message` (chat.completions, returns the full
    assistant message).
  - agent_turn: one LLM turn against a connected env — parse tool_call, append
    the assistant message, execute the tool, append its response. Shared by the
    rollout / student / teacher / critic loops.
  - JsonlCheckpoint: append-only JSONL with async-safe append + resume support.
  - run_jobs: as_completed fan-out over a list of Jobs (anything with .key()
    and async .run()), gated by a Semaphore, with a periodic progress reporter.
  - progress_reporter: prints [done/total, in_flight, ok/fail, rate, eta].
  - parse_tool_call / format_tools: shared with rollout's agent loop.
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
import re
import time
from typing import Any, Awaitable, Callable, Iterable, NamedTuple, Protocol

import json_repair
from openai import AsyncOpenAI


# ---------------------------------------------------------------------------
# 1a. Client-side rate limiter (RPM + TPM sliding-window)
# ---------------------------------------------------------------------------
class RateLimiter:
    """Sliding-window RPM + TPM gate. Every call to `acquire(tokens)` waits
    until BOTH windows have enough budget in the next 60 s, then reserves
    that budget. Windowed against wall time, so leaked budget from expired
    requests is reclaimed automatically."""

    def __init__(self, rpm: int | None, tpm: int | None):
        self._rpm = rpm
        self._tpm = tpm
        self._req_events: list[float] = []             # timestamps
        self._tok_events: list[tuple[float, int]] = []  # (timestamp, tokens)
        self._lock = asyncio.Lock()

    def _prune(self, now: float) -> None:
        cutoff = now - 60.0
        while self._req_events and self._req_events[0] < cutoff:
            self._req_events.pop(0)
        while self._tok_events and self._tok_events[0][0] < cutoff:
            self._tok_events.pop(0)

    async def acquire(self, tokens: int) -> None:
        if self._rpm is None and self._tpm is None:
            return
        while True:
            async with self._lock:
                now = time.monotonic()
                self._prune(now)
                waits: list[float] = []
                if self._rpm is not None and len(self._req_events) >= self._rpm:
                    waits.append(self._req_events[0] + 60.0 - now)
                if self._tpm is not None:
                    used = sum(t for _, t in self._tok_events)
                    if used + tokens > self._tpm:
                        need = used + tokens - self._tpm
                        acc = 0
                        for ts, t in self._tok_events:
                            acc += t
                            if acc >= need:
                                waits.append(ts + 60.0 - now)
                                break
                if not waits:
                    self._req_events.append(now)
                    self._tok_events.append((now, tokens))
                    return
            await asyncio.sleep(max(0.05, min(waits)))

    async def record_tokens(self, tokens: int) -> None:
        """Adjust the last reservation with the true token count once known."""
        if self._tpm is None or not self._tok_events:
            return
        async with self._lock:
            ts, _ = self._tok_events[-1]
            self._tok_events[-1] = (ts, tokens)


def estimate_tokens(text_or_messages) -> int:
    """Cheap heuristic: 1 token ≈ 3 chars. Sufficient for reservation."""
    if isinstance(text_or_messages, str):
        return max(1, len(text_or_messages) // 3)
    total = 0
    for m in text_or_messages or []:
        c = m.get("content", "") if isinstance(m, dict) else ""
        if isinstance(c, str):
            total += len(c)
        elif isinstance(c, list):
            for part in c:
                t = (part.get("text") or "") if isinstance(part, dict) else ""
                total += len(str(t))
    return max(1, total // 3)


def _is_rate_limit_error(e: Exception) -> bool:
    """Detect Ark/OpenAI 429 without a hard dependency on openai's exception
    hierarchy: check class name + message substrings."""
    name = type(e).__name__
    if name in ("RateLimitError", "APIStatusError"):
        return True
    msg = str(e)
    return ("429" in msg) or ("rate limit" in msg.lower()) \
        or ("RateLimitExceeded" in msg) or ("TooManyRequests" in msg)


# ---------------------------------------------------------------------------
# 1b. LLM wrapper (retry + timeout + optional rate limiter)
# ---------------------------------------------------------------------------
class RetryLLM:
    """Thin async wrapper around AsyncOpenAI that shields callers from both
    transient network errors and per-call timeouts. `chat_message()` returns
    the full assistant message (content + tool_calls) through the shared retry
    machinery.

    If a `RateLimiter` is passed, every call reserves capacity beforehand and
    corrects the reservation with the true token usage from the response.
    429 errors get a longer backoff (10s -> 30s -> 90s) than other faults."""

    def __init__(self, base_url: str, api_key: str, model: str,
                 timeout: float = 120.0, retries: int = 3,
                 limiter: "RateLimiter | None" = None):
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self._model = model
        self._timeout = timeout
        self._retries = retries
        self._limiter = limiter

    async def _acquire(self, est_tokens: int) -> None:
        if self._limiter is not None:
            await self._limiter.acquire(est_tokens)

    async def _record_actual(self, resp) -> None:
        if self._limiter is None:
            return
        usage = getattr(resp, "usage", None)
        total = getattr(usage, "total_tokens", None)
        if total is None:
            return
        await self._limiter.record_tokens(int(total))

    async def _call_with_retry(self, coro_fn: Callable[[], Awaitable[Any]],
                                extract: Callable[[Any], str],
                                est_tokens: int) -> str:
        last_err: Exception | None = None
        for attempt in range(self._retries + 1):
            await self._acquire(est_tokens)
            try:
                resp = await asyncio.wait_for(coro_fn(), timeout=self._timeout)
                await self._record_actual(resp)
                return extract(resp)
            except Exception as e:
                last_err = e
                if attempt >= self._retries:
                    raise
                if _is_rate_limit_error(e):
                    delay = 10.0 * (3 ** attempt)  # 10 / 30 / 90
                else:
                    delay = 1.0 * (3 ** attempt)   # 1 / 3 / 9
                await asyncio.sleep(delay)
        raise last_err  # pragma: no cover

    async def chat_message(self, messages: list[dict], temperature: float = 1.0,
                           max_tokens: int = 2048,
                           tools: list | None = None,
                           tool_choice=None,
                           extra_body: dict | None = None):
        """Return the full assistant message object so callers can read both
        `.content` and `.tool_calls` (some models — e.g. seed-2.1-pro — emit
        native tool_calls with content=None even when the system prompt asks
        for an XML tool-call format).

        `tools` / `tool_choice` enable native function-calling: when `tools`
        is provided it is passed straight through to the OpenAI-compatible
        endpoint so the model emits `message.tool_calls`. Both default to None
        (omitted from the request), so existing XML-route callers are
        completely unaffected.

        `extra_body` carries non-standard request fields the OpenAI SDK does
        not model, forwarded via the SDK's extra_body mechanism - e.g.
        {"thinking": {"type": "disabled"}} to switch off an Ark seed/doubao
        reasoning model, or {"chat_template_kwargs": {"enable_thinking":
        False}} for vLLM-served Qwen. Per-call on purpose: one client can
        serve both thinking (teacher advice) and non-thinking (judge) turns."""
        est = estimate_tokens(messages) + max_tokens

        def _create():
            kwargs = dict(
                model=self._model,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=max_tokens,
            )
            if tools is not None:
                kwargs["tools"] = tools
                if tool_choice is not None:
                    kwargs["tool_choice"] = tool_choice
            if extra_body is not None:
                kwargs["extra_body"] = extra_body
            return self._client.chat.completions.create(**kwargs)

        return await self._call_with_retry(
            coro_fn=_create,
            extract=lambda r: r.choices[0].message,
            est_tokens=est,
        )


# ---------------------------------------------------------------------------
# 2. Checkpoint
# ---------------------------------------------------------------------------
class JsonlCheckpoint:
    """Append-only JSONL sink with an async lock so multiple workers can write
    concurrently. Also knows how to enumerate 'done keys' for --resume."""

    def __init__(self, path: str, key_fields: tuple[str, ...] = ("scenario", "task_idx")):
        self.path = path
        self.key_fields = key_fields
        self._lock = asyncio.Lock()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def load_done_keys(self, skip_errors: bool = False) -> set[tuple]:
        done: set[tuple] = set()
        if not os.path.exists(self.path):
            return done
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    if skip_errors and r.get("error"):
                        continue
                    done.add(tuple(r[k] for k in self.key_fields))
                except Exception:
                    pass
        return done

    async def append(self, record: dict) -> None:
        line = json.dumps(record, ensure_ascii=False)
        async with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def read_all(self) -> list[dict]:
        if not os.path.exists(self.path):
            return []
        out: list[dict] = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
        return out

    def clear(self) -> None:
        if os.path.exists(self.path):
            os.remove(self.path)


# ---------------------------------------------------------------------------
# 3. Job protocol + concurrent runner
# ---------------------------------------------------------------------------
class Job(Protocol):
    def key(self) -> tuple: ...
    async def run(self) -> dict: ...


async def progress_reporter(counters: dict, total: int,
                            interval: float = 10.0) -> None:
    start = time.monotonic()
    while True:
        await asyncio.sleep(interval)
        elapsed = time.monotonic() - start
        done = counters.get("done", 0)
        ok = counters.get("ok", 0)
        fail = counters.get("fail", 0)
        pass_ = counters.get("pass", 0)
        in_flight = counters.get("in_flight", 0)
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = (total - done) / rate if rate > 0 else float("inf")
        # `pass` is optional — jobs that report a `success` flag will
        # populate it. When absent, we simply skip printing it so old
        # callers keep the same output.
        extras = f" pass={pass_}" if pass_ or "pass" in counters else ""
        print(
            f"[progress] {done}/{total} done ({done/total:.1%}) | "
            f"in_flight={in_flight} | ok={ok} fail={fail}{extras} | "
            f"elapsed={elapsed:.0f}s | rate={rate:.2f}/s | eta={eta:.0f}s",
            flush=True,
        )


def _summarize_failure(record: dict) -> str:
    """Compact one-liner describing why a job's record is considered a
    failure. Used for eagerly logging each failed record so the operator can
    diagnose without post-hoc analysis."""
    err = (record.get("error") or "").strip()
    if err:
        return err[:300]
    if record.get("success") is False:
        rounds = record.get("rounds_detail") or []
        if rounds:
            last = rounds[-1] or {}
            rt = last.get("verify_reward_type") or "?"
            return f"student did not reach complete (last reward_type={rt})"
        return "student did not reach complete"
    return "unknown failure"


async def run_jobs(jobs: Iterable[Job], concurrency: int,
                   checkpoint: JsonlCheckpoint,
                   progress_interval: float = 10.0,
                   resume: bool = False,
                   resume_skip_errors: bool = True) -> list[dict]:
    """Run every job concurrently (limited by `concurrency`), append each
    result to `checkpoint`, and update in-memory counters for the progress
    reporter. Returns the list of results (in completion order).

    If `resume=True`, skip jobs whose key() is already in the checkpoint.
    With `resume_skip_errors=True` (default), previously-errored records are
    NOT counted as done, so a resumed run retries them.
    """
    jobs = list(jobs)
    if resume:
        done_keys = checkpoint.load_done_keys(skip_errors=resume_skip_errors)
        before = len(jobs)
        jobs = [j for j in jobs if j.key() not in done_keys]
        print(f"[runner] resume: {len(done_keys)} keys already done; "
              f"{before - len(jobs)} skipped; {len(jobs)} remaining.")
    else:
        checkpoint.clear()

    total = len(jobs)
    if total == 0:
        print("[runner] nothing to do")
        return []

    sem = asyncio.Semaphore(concurrency)
    counters: dict = {"in_flight": 0, "done": 0, "ok": 0, "fail": 0, "pass": 0}

    async def _run_one(job: Job) -> dict:
        async with sem:
            counters["in_flight"] += 1
            try:
                record = await job.run()
            except Exception as e:
                record = {
                    **{f: getattr(job, f, None) for f in checkpoint.key_fields},
                    "error": f"job.run raised: {e!r}"[:500],
                    "outcome": "fail",
                }
            finally:
                counters["in_flight"] -= 1
                counters["done"] += 1
                # Classify: an "ok" run means the job pipeline completed
                # without an unrecoverable error (record.error is empty).
                # Independently, "pass" tracks whether the task itself was
                # solved (record.success is truthy). This lets refine
                # distinguish "student failed but pipeline healthy" from
                # "pipeline broke".
                has_error = bool(record.get("error"))
                if has_error:
                    counters["fail"] += 1
                else:
                    counters["ok"] += 1
                if record.get("success"):
                    counters["pass"] += 1
                # Eagerly surface every failing record so the operator does
                # not have to grep the JSONL after the fact.
                if has_error or record.get("success") is False:
                    key_str = ", ".join(
                        f"{f}={getattr(job, f, None)}"
                        for f in checkpoint.key_fields
                    )
                    print(
                        f"[fail] {key_str} :: {_summarize_failure(record)}",
                        flush=True,
                    )
            await checkpoint.append(record)
            return record

    print(f"[runner] launching {total} jobs, concurrency={concurrency}")
    progress_task = asyncio.create_task(
        progress_reporter(counters, total, progress_interval)
    )
    wall_start = time.monotonic()

    pending = [asyncio.create_task(_run_one(j)) for j in jobs]
    results: list[dict] = []
    try:
        for fut in asyncio.as_completed(pending):
            try:
                results.append(await fut)
            except Exception as e:
                print(f"[runner] task raised unexpectedly: {e!r}", flush=True)
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("[runner] interrupt; cancelling remaining tasks...", flush=True)
        for t in pending:
            if not t.done():
                t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        raise
    finally:
        progress_task.cancel()
        try:
            await progress_task
        except asyncio.CancelledError:
            pass

    elapsed = time.monotonic() - wall_start
    print(f"[runner] done: {len(results)} records in {elapsed:.1f}s "
          f"(rate={len(results)/max(elapsed,1e-6):.2f}/s)")
    return results


# ---------------------------------------------------------------------------
# 4. Rollout-specific parsing helpers (kept here so both rollout.py and any
#    critic-side prompt code can reuse them)
# ---------------------------------------------------------------------------
def loads_lenient(text: str) -> Any:
    """Parse JSON, falling back to json_repair for the messy output LLMs
    produce (unquoted keys, trailing commas, over-escaped nested JSON,
    truncated braces). Returns None only when nothing usable can be salvaged."""
    if text is None:
        return None
    if not isinstance(text, str):
        return text
    s = text.strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        try:
            return json_repair.loads(s)
        except Exception:
            return None


def parse_tool_call(content: str) -> dict | None:
    """Parse an XML `<tool_call>{...}</tool_call>` block, using json_repair to
    tolerate malformed JSON (unbalanced braces, over-escaped nested JSON,
    unquoted keys, trailing commas)."""
    if not content:
        return None
    m = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", content, re.DOTALL)
    if not m:
        return None
    data = loads_lenient(m.group(1).strip())
    if isinstance(data, list):
        data = data[0] if data else None
    if not isinstance(data, dict) or "name" not in data:
        return None
    return data


def native_tool_call_to_dict(message) -> dict | None:
    """If the assistant message came back with native OpenAI tool_calls (i.e.
    the model bypassed our XML format), lift the first one into our unified
    dict: {"name": "call_tool", "arguments": {"tool_name": ..., "arguments": ...}}.

    Returns None if there is no native tool_call."""
    tcs = getattr(message, "tool_calls", None) or []
    if not tcs:
        return None
    tc = tcs[0]
    fn = getattr(tc, "function", None)
    name = getattr(fn, "name", None) or ""
    raw_args = getattr(fn, "arguments", "") or ""
    parsed_args = loads_lenient(raw_args) if isinstance(raw_args, str) else raw_args
    if not isinstance(parsed_args, dict):
        parsed_args = {}

    if name in ("list_tools", "call_tool"):
        return {"name": name, "arguments": parsed_args}
    return {"name": "call_tool",
            "arguments": {"tool_name": name, "arguments": parsed_args}}


class ToolCallResult(NamedTuple):
    text: str          # string response fed back to the model
    result: Any        # raw env.step result (None for an unknown tool)


class LLMTurn(NamedTuple):
    content: str            # assistant text (may be "")
    tool_call: dict | None  # parsed tool_call (XML first, then native), or None


class LLMTurnNative(NamedTuple):
    content: str              # assistant text (may = ""), reasoning stripped
    tool_calls: list[dict]    # serialised native tool_calls (may be empty)
    reasoning: str            # extracted reasoning/think (may = "")


_THINK_FULL_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINK_PREFIX_RE = re.compile(r"^.*?</think>", re.DOTALL)


def strip_think(text: str) -> str:
    """Remove reasoning blocks from an assistant message.

    Handles two shapes:
      1. Full pairs: `<think>...</think>` (DeepSeek-R1 / seed-style; can appear
         multiple times or in the middle of content).
      2. Prefix-only: reasoning up to the first `</think>` with no opening
         `<think>` tag — Qwen3 injects `<think>` at the tokenizer level, so
         the generated content starts mid-thought and only emits `</think>`.

    Reasoning helps generate the current turn but must not be carried into the
    conversation history — DeepSeek/seed/Qwen3 all recommend dropping prior
    reasoning, and keeping it wastes context and can degrade later turns."""
    text = _THINK_FULL_RE.sub("", text)
    text = _THINK_PREFIX_RE.sub("", text)
    return text.strip()


async def llm_turn(llm: "RetryLLM", messages: list[dict], *,
                   temperature: float, max_tokens: int) -> LLMTurn:
    """One assistant turn shared by every agent loop: call the LLM, resolve a
    tool_call (XML `<tool_call>` first, then the native tool_calls channel),
    and append the assistant message to `messages` (native calls are serialised
    back to XML so the transcript stays coherent). Reasoning is stripped from
    the appended history message; the raw content is still returned for the
    caller to record / use as the final answer. Executing the tool_call is
    left to the caller, because each loop gates it differently (budgets,
    verify/done rules, trajectory recording)."""
    msg = await llm.chat_message(messages, temperature=temperature,
                                 max_tokens=max_tokens)
    content = msg.content or ""
    tc = parse_tool_call(content)
    if tc is None:
        tc = native_tool_call_to_dict(msg)

    history_content = strip_think(content)
    if history_content:
        messages.append({"role": "assistant", "content": history_content})
    elif tc is not None:
        messages.append({
            "role": "assistant",
            "content": f"<tool_call>{json.dumps(tc, ensure_ascii=False)}</tool_call>",
        })
    else:
        messages.append({"role": "assistant", "content": ""})
    return LLMTurn(content, tc)


# ---------------------------------------------------------------------------
# Native function-calling helpers (used by rollout.refine). These live
# alongside the XML-route helpers above; the XML path is untouched so
# rollout.py / critic.py keep working exactly as before.
# ---------------------------------------------------------------------------
_META_TOOL_NAMES = {"verify", "done", "list_tools"}
_ANY_JSON_TYPES = ["string", "number", "boolean", "object", "array", "null"]


def _normalize_tool_parameters(parameters: dict) -> dict:
    """Repair unconstrained EnvScaler properties for strict tool consumers.

    An empty property schema is valid JSON Schema, but verl's OpenAI schema
    model requires an explicit ``type``. A union of every JSON value type keeps
    the original unconstrained semantics.
    """
    normalized = copy.deepcopy(parameters)
    properties = normalized.get("properties", {})
    if isinstance(properties, dict):
        for schema in properties.values():
            if isinstance(schema, dict) and "type" not in schema:
                schema["type"] = list(_ANY_JSON_TYPES)
    return normalized


def tools_to_openai_schema(tools) -> list[dict]:
    """Convert env `list_tools` tool objects into OpenAI function schemas so a
    model can call the scenario tools natively (by their real names, not via
    the `call_tool` wrapper). Meta tools (verify/done/list_tools) are filtered
    out — verify/done are harness-managed and list_tools is redundant once the
    schemas are supplied up-front. `t.input_schema` is already JSON-Schema
    (properties/required), the exact shape OpenAI expects for `parameters`."""
    schemas: list[dict] = []
    for t in tools:
        if t.name in _META_TOOL_NAMES:
            continue
        params = t.input_schema if isinstance(t.input_schema, dict) else {}
        if not params:
            params = {"type": "object", "properties": {}}
        else:
            params = _normalize_tool_parameters(params)
        schemas.append({
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": params,
            },
        })
    return schemas


def serialize_tool_calls(msg) -> list[dict] | None:
    """Lift a native assistant message's pydantic `tool_calls` into plain,
    JSON-serialisable dicts that can be (a) appended back into the messages
    history verbatim and (b) written to a JSONL trajectory. Returns None when
    the message carries no native tool_calls."""
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


# Match a full <think>...</think> block anywhere in the content, capturing the
# inner reasoning. Used only as a fallback when the server-side reasoning parser
# is not active and the think is still inlined in `content`.
_THINK_CAPTURE_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def _extract_reasoning(msg) -> str | None:
    """Read the reasoning that `--reasoning-parser qwen3` splits into its own
    field. vLLM exposes it as `reasoning_content` (OpenAI-compat) or `reasoning`
    depending on version. Returns the stripped string, or None when the field is
    absent (parser off) so the caller can fall back to inline parsing."""
    for attr in ("reasoning_content", "reasoning"):
        val = getattr(msg, attr, None)
        if val:
            return str(val).strip()
    return None


def _reasoning_from_content(content: str) -> str:
    """Fallback: pull think out of an inline `<think>...</think>` block (used
    when the reasoning parser is off). Returns "" if none is present."""
    if not content:
        return ""
    m = _THINK_CAPTURE_RE.search(content)
    if m:
        return m.group(1).strip()
    # Qwen3 prefix-only shape: text up to the first </think> with no opener.
    if "</think>" in content:
        return content.split("</think>", 1)[0].strip()
    return ""


def _apply_system_override(wire: list[dict], system_text: str) -> list[dict]:
    """Return a copy of `wire` whose leading system message content is replaced
    with `system_text`. If the first message isn't a system message, one is
    prepended. Does not mutate the input (shallow-copies the affected dict)."""
    out = list(wire)
    if out and out[0].get("role") == "system":
        out[0] = {**out[0], "content": system_text}
    else:
        out.insert(0, {"role": "system", "content": system_text})
    return out


def _prepare_native_messages(messages: list[dict]) -> list[dict]:
    """Build the wire payload sent to the model from our stored `messages`.

    Two jobs:
      1. Persisted reasoning is our own bookkeeping field, not part of the
         OpenAI schema — it must never be sent verbatim (the API would reject
         the unknown key). Drop it here.
      2. Continuity: re-inline ONLY the most recent assistant turn's reasoning
         as `<think>...</think>` in that message's content, so the model sees
         its own latest thinking but not the entire reasoning history.

    The input `messages` list is never mutated — a shallow copy of each dict is
    returned."""
    # Index of the last assistant message (the only one that keeps its think).
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


async def llm_turn_native(llm: "RetryLLM", messages: list[dict], *,
                        temperature: float,
                        max_tokens: int,
                        tools: list[dict] = None,tool_choice: str = "auto",
                        system_override: str | None = None,
                        extra_body: dict | None = None) -> LLMTurnNative:
    """Native-function-calling counterpart of `llm_turn`. Calls the LLM with a
    `tools` schema, appends a proper native assistant message (content +
    `tool_calls` with ids, plus a persisted `reasoning` field) to `messages`,
    and returns LLMTurnNative(content, tool_calls). Executing the tool calls and
    appending the paired `role:tool` responses is left to the caller (each loop
    gates it differently).

    Reasoning handling (server run with `--reasoning-parser qwen3`):
      * The model's think is captured and PERSISTED on the assistant message as
        a `reasoning` field so the full transcript (and JSONL) retains it.
      * Only the MOST RECENT assistant turn's think is sent back to the model
        (inlined as `<think>...</think>`); older turns are stripped — see
        `_prepare_native_messages`. This keeps continuity without letting every
        past reasoning block bloat the context.
      * Robust to the parser being off too: if no reasoning field is present,
        an inline `<think>` block in `content` is split out instead.

    `system_override`: when set, the leading system message is replaced with
    this text FOR THIS REQUEST ONLY (wire payload). The stored `messages` are
    left untouched, so the persisted trajectory keeps the original system
    prompt. Used to swap in a "final turn: answer directly, no tool calls"
    prompt so the model doesn't emit literal `<tool_call>` markup as text when
    `tool_choice="none"` disables native tool parsing."""
    wire_messages = _prepare_native_messages(messages)
    if system_override is not None:
        wire_messages = _apply_system_override(wire_messages, system_override)
    msg = await llm.chat_message(wire_messages, temperature=temperature,
                                 max_tokens=max_tokens,
                                 tools=tools, tool_choice=tool_choice,
                                 extra_body=extra_body)
    raw_content = msg.content or ""
    reasoning = _extract_reasoning(msg)
    if reasoning is None:
        # Parser not active (think inline in content) — separate it out so the
        # returned/clean content and the persisted reasoning stay consistent.
        reasoning = _reasoning_from_content(raw_content)
        clean_content = strip_think(raw_content)
    else:
        clean_content = raw_content.strip()

    tool_calls = serialize_tool_calls(msg)

    assistant_msg: dict = {"role": "assistant"}
    # content must be present (may be null) alongside tool_calls per the API.
    assistant_msg["content"] = clean_content or None
    if reasoning:
        assistant_msg["reasoning"] = reasoning
    if tool_calls:
        assistant_msg["tool_calls"] = tool_calls
    messages.append(assistant_msg)

    return LLMTurnNative(clean_content, tool_calls or [], reasoning or "")


async def execute_native_tool_call(env, name: str, arguments: dict) -> "ToolCallResult":
    """Execute a natively-named scenario tool against a connected AWMEnv. Unlike
    `execute_tool_call` (which unwraps the `call_tool` envelope), `name` here is
    the real scenario tool name and `arguments` its dict of args. verify/done
    are refused at runtime as a safety net. Shares the same observation-reading
    logic as `execute_tool_call`."""
    from openenv.core.env_server.mcp_types import CallToolAction

    if name in ("verify", "done"):
        return ToolCallResult(
            f"Error: '{name}' is managed by the harness and cannot be called here.",
            None)
    if not isinstance(arguments, dict):
        arguments = {}
    r = await env.step(CallToolAction(tool_name=name, arguments=arguments))
    obs = r.observation
    if hasattr(obs, "tool_result") and obs.tool_result is not None:
        text = (
            json.dumps(obs.tool_result, ensure_ascii=False)
            if not isinstance(obs.tool_result, str)
            else obs.tool_result
        )
    elif hasattr(obs, "error") and obs.error:
        text = f"Error: {obs.error}"
    else:
        text = json.dumps(obs.model_dump(), ensure_ascii=False)
    return ToolCallResult(text, r)


async def execute_tool_call(env, tc: dict) -> "ToolCallResult":
    """Execute a parsed <tool_call> against a connected AWMEnv. Returns a
    ToolCallResult(text, result): `text` is the string response fed back to the
    model; `result` is the raw env.step result (or None for an unknown tool) so
    callers that care about per-step observations (e.g. reward_type) can read
    them. Shared by every agent loop (rollout / student / teacher / critic)."""
    from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction

    name = tc.get("name", "")
    arguments = tc.get("arguments") or {}
    if name == "list_tools":
        r = await env.step(ListToolsAction())
        return ToolCallResult(format_tools(r.observation.tools), r)
    if name == "call_tool":
        tool_name = arguments.get("tool_name", "")
        inner_args = arguments.get("arguments", "{}")
        if isinstance(inner_args, str):
            inner_args = loads_lenient(inner_args)
        if not isinstance(inner_args, dict):
            inner_args = {}
        r = await env.step(CallToolAction(tool_name=tool_name, arguments=inner_args))
        obs = r.observation
        if hasattr(obs, "tool_result") and obs.tool_result is not None:
            text = (
                json.dumps(obs.tool_result, ensure_ascii=False)
                if not isinstance(obs.tool_result, str)
                else obs.tool_result
            )
        elif hasattr(obs, "error") and obs.error:
            text = f"Error: {obs.error}"
        else:
            text = json.dumps(obs.model_dump(), ensure_ascii=False)
        return ToolCallResult(text, r)
    return ToolCallResult(
        f"Error: Unknown tool '{name}'. Use 'list_tools' or 'call_tool'.", None)


def format_tools(tools) -> str:
    lines = [f"Available MCP Tools ({len(tools)} tools):", "=" * 60]
    for i, t in enumerate(tools, 1):
        lines.append(f"{i}. {t.name}")
        lines.append(f"   Description: {t.description}")
        props = t.input_schema.get("properties", {})
        required = t.input_schema.get("required", [])
        if props:
            lines.append("   Parameters:")
            for pname, pinfo in props.items():
                req = " (required)" if pname in required else ""
                lines.append(
                    f"     - {pname}: {pinfo.get('type', 'any')}{req} — {pinfo.get('description', '')}"
                )
        else:
            lines.append("   Parameters: None")
        lines.append("")
    return "\n".join(lines)
