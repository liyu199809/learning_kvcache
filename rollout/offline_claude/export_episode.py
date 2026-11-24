"""Read a normalized episode or convert an older Claude event trace."""
import argparse
from collections import deque
import json
from pathlib import Path

SYSTEM_PROMPT = ("Solve the current independent benchmark task. Each task starts with fresh state. "
                 "Use only the tools offered for this task. For programming questions return Python in a fenced code block.")


def initial_messages(question, thinking):
    return [{"role": "system", "content": SYSTEM_PROMPT if thinking else SYSTEM_PROMPT + " /no_think"},
            {"role": "user", "content": question}]


class MessageStream:
    def __init__(self, session_id, names, *, seen_call_ids=None):
        self.session_id = session_id
        self.names = names
        self.turns = {}
        self.calls = {}
        self.results = {}
        self.seen_call_ids = set() if seen_call_ids is None else seen_call_ids

    def accept(self, event):
        role = event["type"]
        if role not in ("assistant", "user"):
            return False
        if event.get("session_id") != self.session_id:
            raise ValueError("Unexpected session ID")
        message = event["message"]
        blocks = message["content"]
        if role == "user" and event.get("isSynthetic"):
            # The CLI can inject continuation prompts and compaction summaries.
            # Keep the original trajectory and any tool results in this event.
            if isinstance(blocks, str):
                return False
            blocks = [block for block in blocks if block["type"] != "text"]
            if not blocks:
                return False
        if role == "assistant":
            turn = self.turns.setdefault(message["id"], {"role": "assistant", "content": ""})
        for b in blocks:
            kind = b["type"]
            if role == "assistant" and kind == "tool_use":
                name, call_id = b["name"], b["id"]
                call = {"id": call_id, "type": "function",
                        "function": {"name": name, "arguments": b["input"]}}
                reasons = []
                if call_id in self.seen_call_ids:
                    reasons.append("duplicate_call_id")
                if name not in self.names:
                    reasons.append("unknown_tool")
                if reasons:
                    call["invalid_tool_reasons"] = reasons
                self.seen_call_ids.add(call_id)
                self.calls.setdefault(call_id, deque()).append(call)
                turn.setdefault("tool_calls", []).append(call)
            elif role == "user" and kind == "tool_result":
                call_id = b["tool_use_id"]
                pending = self.calls.get(call_id)
                if not pending:
                    raise ValueError(f"Unmatched tool result {call_id}")
                call = pending.popleft()
                value = b["content"]
                if isinstance(value, list):
                    if any(part["type"] != "text" for part in value):
                        raise ValueError("Unsupported non-text tool result")
                    value = "\n".join(part["text"] for part in value)
                result = {"role": "tool", "name": call["function"]["name"],
                          "tool_call_id": call_id, "content": value}
                if reasons := call.get("invalid_tool_reasons"):
                    result["invalid_tool_reasons"] = reasons
                    if b.get("is_error"):
                        result["is_error"] = True
                self.results[id(call)] = result
            elif role == "assistant" and kind == "thinking":
                turn["reasoning_content"] = turn.get("reasoning_content", "") + b["thinking"]
            elif role == "assistant" and kind == "text":
                turn["content"] += b["text"]
            else:
                raise ValueError(f"Unexpected {role} content block {kind}")
        return True

    def messages(self):
        for turn in self.turns.values():
            yield turn
            for call in turn.get("tool_calls", []):
                if id(call) in self.results:
                    yield self.results[id(call)]

    def unknown_tool_streak(self, previous=0):
        """Count consecutive assistant tool batches whose calls are all unknown."""
        streak = previous
        for turn in self.turns.values():
            calls = turn.get("tool_calls")
            if not calls:
                continue
            if not all("unknown_tool" in call.get("invalid_tool_reasons", ()) for call in calls):
                streak = 0
            elif all(id(call) in self.results for call in calls):
                streak += 1
        return streak

    def check_complete(self):
        if any("duplicate_call_id" not in call.get("invalid_tool_reasons", ())
               for pending in self.calls.values() for call in pending):
            raise ValueError("Missing tool result in completed episode")


def conversation(directory):
    episode = json.loads((directory / "episode.json").read_text())
    if episode["status"] != "scored":
        raise ValueError(f"Episode status is {episode['status']}")
    schemas = json.loads((directory / "tools.json").read_text())
    tools = [{"type": "function", "function": {
        "name": "mcp__episode__" + tool["name"], "description": tool["description"],
        "parameters": tool["inputSchema"]}} for tool in schemas]
    with (directory / "events.jsonl").open() as stream:
        if episode.get("event_format") == "messages-v1":
            messages = [json.loads(line) for line in stream]
        else:
            events = MessageStream(episode["session_id"], {t["function"]["name"] for t in tools})
            for line in stream:
                events.accept(json.loads(line))
            events.check_complete()
            messages = initial_messages(episode["task"], episode["thinking_enabled"]) + list(events.messages())
    metadata = {k: episode[k] for k in ("domain", "scenario", "task_idx", "session_id", "finish")}
    metadata["reward"] = episode["verifier"]["reward"]
    return {"metadata": metadata, "tools": tools, "messages": messages}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path, help="Episode directory or its events.jsonl")
    args = parser.parse_args()
    directory = args.episode.parent if args.episode.is_file() else args.episode
    print(json.dumps(conversation(directory), ensure_ascii=False))
