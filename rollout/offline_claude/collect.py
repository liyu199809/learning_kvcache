"""Collect independent Claude Code episodes against local OpenEnv tasks."""
import argparse
import asyncio
from contextlib import closing
from difflib import SequenceMatcher
import hashlib
import json
import os
import multiprocessing
import signal
from pathlib import Path
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from itertools import chain
from urllib.request import urlopen

if __package__:
    from .export_episode import MessageStream, initial_messages
else:
    from export_episode import MessageStream, initial_messages


from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONTEXT_HEADROOM = 2048
UNKNOWN_TOOL_BATCH_LIMIT = 3


def write(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    tmp.replace(path)


def make_env(domain):
    if domain == "awm":
        os.environ.setdefault("AWM_DATA_DIR", str(ROOT / "awm_data"))
        from agent_world_model_env.server.awm_environment import AWMEnvironment
        return AWMEnvironment()
    if domain == "envscaler":
        os.environ.setdefault("ENVSCALER_DATA_DIR", str(ROOT / "envscaler_data"))
        from env_scaler_env.server.env_scaler_environment import EnvScalerEnvironment
        return EnvScalerEnvironment()
    if domain == "agent_world":
        if __package__:
            from .agent_world_env import AgentWorldEnvironment
        else:
            from agent_world_env import AgentWorldEnvironment
        return AgentWorldEnvironment()
    from code_judge_env.server.config import DATA_DIR
    from code_judge_env.server.data_loader import TacoDataLoader
    from code_judge_env.server.code_judge_environment import CodeJudgeEnvironment
    return CodeJudgeEnvironment(TacoDataLoader(DATA_DIR))


def execute(args, env, record, question, trace, after_answer=None):
    session_id = record["session_id"]
    with urlopen(args.endpoint.rstrip("/") + "/v1/models", timeout=10) as response:
        models = json.load(response)["data"]
    model_window = next((m.get("max_model_len") for m in models if m["id"] == args.model), None)
    if type(model_window) is not int or model_window <= 0:
        raise ValueError(f"vLLM did not report max_model_len for {args.model}")
    window = model_window - CONTEXT_HEADROOM
    if window <= 0:
        raise ValueError(f"Model window {model_window} is too small for {CONTEXT_HEADROOM} reserved tokens")
    record.update(model_context_window=model_window, context_window=window,
                  context_headroom_tokens=CONTEXT_HEADROOM, compactions=[])
    write(trace / "episode.json", record)
    advertised = env.step(ListToolsAction())
    if getattr(advertised, "error", None):
        raise RuntimeError(advertised.error)
    schemas = {t.name: {"name": t.name, "description": t.description or "", "inputSchema": t.input_schema}
               for t in advertised.tools if t.name not in ("verify", "done", "__list_scenarios__")}
    write(trace / "tools.json", list(schemas.values()))

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            req = json.loads(self.rfile.readline())
            try:
                if req["op"] == "list":
                    value = list(schemas.values())
                else:
                    if req["op"] != "call" or req["name"] not in schemas:
                        raise ValueError("unknown episode tool")
                    obs = env.step(CallToolAction(tool_name=req["name"], arguments=req["arguments"]))
                    value = obs.error or obs.tool_result
                reply = {"value": value}
            except Exception as exc:
                reply = {"error": str(exc)}
            self.wfile.write((json.dumps(reply) + "\n").encode())

    output_override = os.environ.get("AGENT_WORLD_STUDENT_MAX_OUTPUT_TOKENS") if getattr(args, "multiturn_pilot", False) else None
    output_limit = int(output_override) if output_override else (8192 if args.thinking else 2048)
    if output_override:
        if not 0 < output_limit < model_window:
            raise ValueError("Student output override must fit in the model context window")
        record.update(student_max_output_tokens=output_limit, student_completion_events=[])
    config_dir = args.state / "claude"
    config_dir.mkdir(exist_ok=True)
    runtime = {**os.environ, "CLAUDE_CONFIG_DIR": str(config_dir), "ANTHROPIC_BASE_URL": args.endpoint,
               "ANTHROPIC_API_KEY": "local", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
               "CLAUDE_CODE_MAX_CONTEXT_TOKENS": str(window),
               "DISABLE_COMPACT": "0", "DISABLE_AUTO_COMPACT": "0",
               "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(output_limit),
               "MAX_THINKING_TOKENS": "6144" if args.thinking else "0",
               "ANTHROPIC_DEFAULT_HAIKU_MODEL": args.model}
    with tempfile.TemporaryDirectory() as td, socketserver.UnixStreamServer(td + "/env", Handler) as server:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        mcp = {"mcpServers": {"episode": {"command": sys.executable, "args": [str(HERE / "mcp_bridge.py"), td + "/env"]}}} if schemas else {"mcpServers": {}}
        initial = initial_messages(question, args.thinking)
        names = {"mcp__episode__" + name for name in schemas}
        seen_call_ids = set()
        messages = MessageStream(session_id, names, seen_call_ids=seen_call_ids)
        path = trace / "events.jsonl"

        def save_messages():
            # Publish a complete snapshot even if the wall-clock deadline interrupts a tool call.
            tmp = path.with_suffix(".tmp")
            invalid = []
            with tmp.open("w") as output:
                for message_index, message in enumerate(chain(initial, messages.messages())):
                    print(json.dumps(message, ensure_ascii=False), file=output)
                    for call_index, call in enumerate(message.get("tool_calls", ())):
                        if reasons := call.get("invalid_tool_reasons"):
                            invalid.append({"message_index": message_index, "call_index": call_index,
                                            "call_id": call["id"], "name": call["function"]["name"],
                                            "reasons": reasons})
            tmp.replace(path)
            if invalid or "invalid_tool_calls" not in record:
                record["invalid_tool_calls"] = len(invalid)
                record["invalid_tool_call_events"] = invalid
            if invalid:
                write(trace / "episode.json", record)

        save_messages()
        cmd = [args.claude, "-p", "--bare", "--setting-sources", "", "--tools", "",
               "--strict-mcp-config", "--mcp-config", json.dumps(mcp), "--model", args.model,
               "--session-id", session_id, "--no-session-persistence",
               "--output-format", "stream-json", "--verbose", "--permission-mode", "dontAsk",
               "--system-prompt", initial[0]["content"]]
        if after_answer:
            cmd += ["--input-format", "stream-json"]
        if output_override:
            cmd += ["--include-partial-messages"]
        if schemas:
            cmd += ["--allowedTools", *("mcp__episode__" + name for name in schemas)]
        try:
            result = None
            with (trace / "stderr.log").open("w") as err, subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err,
                text=True, cwd=args.state, env=runtime
            ) as proc:
                try:
                    earlier_unknown_batches = 0
                    def send_user(text):
                        packet = {"type": "user", "session_id": session_id, "parent_tool_use_id": None,
                                  "message": {"role": "user", "content": text}}
                        proc.stdin.write(json.dumps(packet) + "\n")
                        proc.stdin.flush()

                    if after_answer:
                        send_user(question)
                    else:
                        proc.stdin.write(question)
                        proc.stdin.close()
                    for line in proc.stdout:
                        event = json.loads(line)
                        if messages.accept(event):
                            save_messages()
                            if (after_answer and getattr(args, "multiturn_pilot", False) and event["type"] == "user"
                                    and messages.unknown_tool_streak(earlier_unknown_batches) >= UNKNOWN_TOOL_BATCH_LIMIT):
                                record.update(status="stalled", end_reason="invalid_tool_loop",
                                              finish="stalled", unknown_tool_batch_streak=UNKNOWN_TOOL_BATCH_LIMIT)
                                write(trace / "episode.json", record)
                                proc.kill()
                                return None, "stalled"
                        elif output_override and event["type"] == "stream_event":
                            stream_event = event.get("event", {})
                            stop_reason = stream_event.get("delta", {}).get("stop_reason")
                            if stream_event.get("type") == "message_delta" and stop_reason:
                                record["student_completion_events"].append({
                                    "stop_reason": stop_reason,
                                    "output_tokens": (stream_event.get("usage") or {}).get("output_tokens"),
                                })
                        elif event["type"] == "system" and event.get("subtype") == "init":
                            if schemas:
                                servers = event.get("mcp_servers") or []
                                record["mcp_servers"] = servers
                                write(trace / "episode.json", record)
                                if any(s.get("name") == "episode" and s.get("status") == "failed"
                                       for s in servers):
                                    raise RuntimeError("Episode MCP server failed to connect")
                        elif event["type"] == "system" and event.get("subtype") == "compact_boundary":
                            metadata = event["compact_metadata"]
                            record["compactions"].append({k: metadata[k] for k in
                                ("trigger", "pre_tokens", "post_tokens", "duration_ms") if k in metadata})
                            write(trace / "episode.json", record)
                        elif event["type"] == "result":
                            result = event
                            if event.get("usage") or event.get("modelUsage"):
                                record.setdefault("student_usage_rounds", []).append({
                                    "usage": event.get("usage"),
                                    "modelUsage": event.get("modelUsage"),
                                })
                                write(trace / "episode.json", record)
                            if after_answer and not proc.stdin.closed:
                                if event.get("session_id") != session_id or event.get("is_error"):
                                    raise RuntimeError(f"Claude Code result: {event.get('subtype')}")
                                messages.check_complete()
                                earlier_unknown_batches = messages.unknown_tool_streak(earlier_unknown_batches)
                                initial.extend(messages.messages())
                                messages = MessageStream(session_id, names, seen_call_ids=seen_call_ids)
                                instruction = after_answer(event["result"], len(initial))
                                if instruction is None:
                                    proc.stdin.close()
                                else:
                                    send_user(instruction)
                                    initial.append({"role": "user", "content": instruction})
                                    save_messages()
                                    result = None
                except BaseException:
                    proc.kill()
                    raise
            if not result or result.get("session_id") != session_id or proc.returncode or result.get("is_error"):
                detail = result.get("subtype") if result else "no result"
                raise RuntimeError(f"Claude Code exited {proc.returncode}: {detail}; see stderr.log")
            messages.check_complete()
            return result["result"], "completed"
        finally:
            server.shutdown()


def record_pilot_round(record, trace, step):
    with (trace / "events.jsonl").open() as stream:
        messages = [json.loads(line) for line in stream]
    previous = record["rounds"][-2] if len(record["rounds"]) > 1 else None
    start = previous["message_count"] if previous else 0
    def calls(sequence):
        return [{**call["function"], "id": call["id"]} for message in sequence
                for call in message.get("tool_calls", ()) if not call.get("invalid_tool_reasons")]
    def fingerprint(call):
        return call["name"], json.dumps(call["arguments"], sort_keys=True, ensure_ascii=False)
    seen = {fingerprint(call) for call in calls(messages[:start])}
    current = calls(messages[start:step["message_count"]])
    step.update(tool_calls=current, new_tool_call=any(fingerprint(call) not in seen for call in current))
    if previous:
        previous["after_tool_calls"] = current
        previous["after_score"] = step["verifier"].get("reward")
        if previous["after_score"] is not None and previous["verifier"].get("reward") is not None:
            stalled = previous["after_score"] <= previous["verifier"]["reward"] and not step["new_tool_call"]
            record["stalled_rounds"] = record.get("stalled_rounds", 0) + 1 if stalled else 0


def run_episode(args, index, trace, record, advice, expected_hash):
    os.setsid()
    pilot = getattr(args, "multiturn_pilot", False)
    if pilot:
        record["end_reason"] = "env_error"
        record["teacher_trace_only"] = True
    try:
        with closing(make_env(args.domain)) as env:
            obs = env.reset(scenario=args.scenario, task_idx=index)
            if obs.error or not obs.task:
                raise RuntimeError(str(obs.error or "empty task"))
            record.update(task=obs.task, task_id=getattr(obs, "task_id", None),
                          task_sha256=hashlib.sha256(obs.task.encode()).hexdigest())
            write(trace / "episode.json", record)
            if expected_hash and record["task_sha256"] != expected_hash:
                raise ValueError("Task changed on reset")
            if pilot:
                record["end_reason"] = "runtime_error"
            question = obs.task
            if advice:
                question += "\n\n[Expert advice from the previous attempt]\n" + advice
            def verify(environment, answer):
                return environment.step(CallToolAction(tool_name="verify", arguments={
                    "verifier_mode": "code", "final_answer": answer}))

            def after_answer(answer, message_count):
                if pilot:
                    record["end_reason"] = "grader_error"
                if args.domain == "taco":
                    # CodeJudge closes on verify. Grade each submission in its own instance.
                    with closing(make_env("taco")) as grader:
                        reset = grader.reset(scenario=args.scenario, task_idx=index)
                        if reset.error or reset.task != obs.task:
                            raise ValueError("CodeJudge task mismatch")
                        verdict = verify(grader, answer)
                else:
                    verdict = verify(env, answer)
                grade = verdict.model_dump(mode="json")
                step = {"message_count": message_count, "verifier": grade, "answer": answer}
                record["rounds"].append(step)
                record.update(answer=answer, verifier=grade)
                if pilot:
                    record_pilot_round(record, trace, step)
                write(trace / "episode.json", record)
                if verdict.error or verdict.reward_type in {"no_verifier", "judge_error", "server_error", "invalid_args"}:
                    raise RuntimeError(f"Verification failed: {verdict.reward_type}")
                if verdict.reward is None:
                    raise RuntimeError("Verifier returned no score")
                if verdict.reward >= 1:
                    if pilot:
                        record["end_reason"] = "full_score"
                    return None
                if pilot and record.get("stalled_rounds", 0) >= 2:
                    record["end_reason"] = "stalled"
                    return None
                if args.domain != "taco" and verdict.done:
                    if pilot:
                        record["end_reason"] = "env_error"
                    raise RuntimeError("Environment closed during intermediate verification")
                if __package__:
                    from .teacher_retry import teacher_advice
                else:
                    from teacher_retry import teacher_advice
                if pilot:
                    record["end_reason"] = "runtime_error"
                started = time.monotonic()
                feedback = asyncio.run(teacher_advice(record, trace, in_session=True))
                if pilot:
                    preceding = record["rounds"][-2].get("teacher", {}).get("advice") if len(record["rounds"]) > 1 else None
                    feedback["wall_seconds"] = round(time.monotonic() - started, 3)
                    step["advice_similarity"] = (
                        SequenceMatcher(None, preceding.casefold(), feedback["advice"].casefold(), autojunk=False).ratio()
                        if preceding else None
                    )
                instruction = feedback["advice"]
                step.update(teacher=feedback, instruction_message_index=message_count)
                write(trace / "episode.json", record)
                return instruction

            if args.teacher_session:
                record["rounds"] = []
                record["teacher_mode"] = "same_session"
                answer, finish = execute(args, env, record, question, trace, after_answer)
                if finish == "stalled":
                    return
                record.update(status="verify_error" if record["verifier"].get("error") else "scored",
                              finish=finish)
            else:
                answer, finish = execute(args, env, record, question, trace)
                verdict = verify(env, answer)
                record.update(status="verify_error" if verdict.error else "scored", finish=finish,
                              answer=answer, verifier=verdict.model_dump(mode="json"))
    except Exception as exc:
        record.update(status="error", error=str(exc))
    finally:
        write(trace / "episode.json", record)


def attempt(args, index, trace=None, advice=None, expected_hash=None):
    session_id = str(uuid.uuid4())
    trace = trace or args.state / args.domain / f"{args.scenario}-{index}-{session_id}"
    trace.mkdir(parents=True)
    path = trace / "episode.json"
    record = {"capture_level": "eval", "event_format": "messages-v1", "domain": args.domain,
              "scenario": args.scenario, "task_idx": index, "session_id": session_id, "model": args.model,
              "endpoint": args.endpoint, "timeout_seconds": args.timeout, "thinking_enabled": args.thinking,
              "claude_version": args.claude_version, "status": "started"}
    write(path, record)
    worker = multiprocessing.get_context("fork").Process(target=run_episode, args=(args, index, trace, record, advice, expected_hash))
    worker.start()
    try:
        worker.join(args.timeout)
        timed_out = worker.is_alive()
    finally:
        # Include reset, tool execution and verification in the same deadline.
        # Kill the worker with its CLI and MCP children, including after an interrupted collector.
        try:
            os.killpg(worker.pid, signal.SIGKILL)
        except ProcessLookupError:
            if worker.is_alive():
                worker.kill()
        worker.join()
    record = json.loads(path.read_text())
    if timed_out:
        if getattr(args, "multiturn_pilot", False):
            record["end_reason"] = "timeout"
        record.update(status="timeout", finish="timeout", error=f"Episode exceeded {args.timeout}s")
        write(path, record)
    elif record["status"] == "started":
        record.update(status="error", error=f"Episode worker exited {worker.exitcode}")
        write(path, record)
    for tmp in trace.glob("*.tmp"):
        tmp.unlink()
    return trace, record


def collect(args, index):
    if args.resume_failure:
        trace = args.resume_failure.resolve()
        record = json.loads((trace / "episode.json").read_text())
        expected = (args.domain, args.scenario, index, args.model, args.thinking)
        actual = tuple(record[k] for k in ("domain", "scenario", "task_idx", "model", "thinking_enabled"))
        if actual != expected or trace.parent != args.state / args.domain:
            raise ValueError("Existing episode does not match this task, model, thinking setting or output root")
    else:
        trace, record = attempt(args, index)
    summary = {"trace": str(trace), "status": record["status"], "finish": record.get("finish"),
               "error": record.get("error"), "reward": record.get("verifier", {}).get("reward")}
    failed = ((record["status"] == "scored" and summary["reward"] is not None and summary["reward"] < 1)
              or (record["status"] == "timeout" and (trace / "events.jsonl").exists()))
    rounds = record.get("rounds", [])
    if rounds:
        summary.update(reward=rounds[0]["verifier"].get("reward"), verified_rounds=len(rounds),
                       last_verified_reward=rounds[-1]["verifier"].get("reward"))
    ok = record["status"] == "scored"
    if args.teacher_retry and failed:
        if __package__:
            from .teacher_retry import refine_failure
        else:
            from teacher_retry import refine_failure
        import fcntl
        with (trace / "pair.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            existing = trace / "pair.json"
            if args.resume_failure and existing.exists() and json.loads(existing.read_text())["status"] != "teacher_error":
                raise ValueError("Existing pair has already advanced beyond a failed teacher request")
            pair = refine_failure(args, index, trace, record, attempt, write)
        summary.update(pair=str(trace / "pair.json"), pair_status=pair["status"],
                       retry_reward=pair.get("retry_reward"))
        ok = pair["status"] == "paired"
    print(json.dumps(summary), flush=True)
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("domain", choices=["awm", "envscaler", "taco", "agent_world"])
    parser.add_argument("scenario")
    parser.add_argument("indices", type=int, nargs="+")
    parser.add_argument("--endpoint", default="http://127.0.0.1:18764")
    parser.add_argument("--model", default="qwen3-4b")
    parser.add_argument("--state", type=Path, default=ROOT / "outputs/offline_claude")
    parser.add_argument("--claude", default=os.environ.get("CLAUDE_BIN", "claude"))
    parser.add_argument("--thinking", action="store_true")
    teacher_mode = parser.add_mutually_exclusive_group()
    teacher_mode.add_argument("--teacher-session", action="store_true", help="Coach the student in the same live session after each failed verification")
    teacher_mode.add_argument("--teacher-retry", action="store_true", help="On task failure, request advice and collect one fresh attempt")
    parser.add_argument("--multiturn-pilot", action="store_true")
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--resume-failure", type=Path, help="Resume the teacher stage from a saved failed attempt")
    args = parser.parse_args()
    if args.multiturn_pilot and (args.domain != "agent_world" or not args.teacher_session):
        parser.error("--multiturn-pilot requires agent_world and --teacher-session")
    if args.resume_failure and (not args.teacher_retry or len(args.indices) != 1):
        parser.error("--resume-failure requires --teacher-retry and exactly one task index")
    args.state = args.state.resolve()
    args.claude_version = subprocess.check_output([args.claude, "--version"], text=True).strip()
    results = [collect(args, index) for index in args.indices]
    sys.exit(0 if all(results) else 1)
