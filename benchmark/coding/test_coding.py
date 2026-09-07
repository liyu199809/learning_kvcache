"""CPU regression tests; RUN_CODE_SANDBOX_TESTS=1 includes official Docker scoring."""
import argparse
import asyncio
import base64
import json
import os
import pickle
import tempfile
import threading
import unittest
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from data import decode_tests, lcb_prompt, lcb_sample, release_files, select_tasks
from run import extract_code, generate, prepare_retry_samples, read_samples, score_in_container


class CodingTests(unittest.TestCase):
    def test_release_ranges(self):
        self.assertEqual(len(release_files("v5")), 5)
        self.assertEqual(len(release_files("v6")), 6)
        for value in ("release_v6", "v4", "v7", "v5_v6"):
            with self.assertRaises(ValueError):
                release_files(value)

    def test_private_formats_and_functional_arguments(self):
        tests = [{"input": "[1,2]\n3", "output": "6", "testtype": "functional"}]
        packed = base64.b64encode(zlib.compress(pickle.dumps(json.dumps(tests)))).decode()
        self.assertEqual(decode_tests(packed), tests)
        row = {"public_test_cases": json.dumps(tests), "private_test_cases": packed,
               "metadata": '{"func_name":"add"}'}
        parsed = json.loads(lcb_sample(row)["input_output"])
        self.assertEqual(parsed["fn_name"], "add")
        self.assertEqual(parsed["inputs"], ["[1,2]\n3", "[1,2]\n3"])

    def test_pickle_globals_rejected(self):
        packed = base64.b64encode(zlib.compress(pickle.dumps(Path("x")))).decode()
        with self.assertRaises(pickle.UnpicklingError):
            decode_tests(packed)

    def test_prompt_never_includes_hidden_tests(self):
        prompt = lcb_prompt({"question_content": "Add numbers", "starter_code": "class Solution: pass",
                             "private_test_cases": "SECRET"})
        self.assertIn("class Solution", prompt)
        self.assertNotIn("SECRET", prompt)

    def test_selection_validates_ids_and_indices(self):
        tasks = [{"task_id": str(i)} for i in range(4)]
        args = argparse.Namespace(tasks="1-3", task_ids="2,3", max_tasks=1)
        self.assertEqual(select_tasks(tasks, args), [tasks[2]])
        args.task_ids = "missing"
        with self.assertRaises(ValueError):
            select_tasks(tasks, args)
        args.task_ids, args.tasks = None, "0-5"
        with self.assertRaises(ValueError):
            select_tasks(tasks, args)

    def test_extraction_ignores_reasoning(self):
        self.assertEqual(extract_code("<think>```python\nwrong\n```</think>```python\nprint(1)\n```"), "print(1)")

    def test_missing_samples_fail_instead_of_shrinking_denominator(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "samples.jsonl"
            path.write_text('{"task_id":"a","solution":"print(1)"}\n')
            with self.assertRaises(ValueError):
                read_samples(path, [{"task_id": "a"}, {"task_id": "b"}],
                             argparse.Namespace(benchmark="livecodebench", code_n_samples=1))

    def test_generation_error_is_preserved(self):
        class Client:
            def __init__(self, **kwargs):
                self.chat = self.completions = self
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass
            async def create(self, **kwargs):
                raise RuntimeError("fixture endpoint unavailable")
        args = argparse.Namespace(concurrency=1, openai_base_url="http://localhost/v1",
                                  step_timeout=1, model="fixture", temperature=0, top_p=1,
                                  llm_max_completion_tokens=100, benchmark="livecodebench", code_n_samples=1)
        with tempfile.TemporaryDirectory() as temp, patch("openai.AsyncOpenAI", Client):
            rows = asyncio.run(generate([{"task_id": "a", "prompt": "p"}], args, Path(temp) / "s.jsonl"))
        self.assertEqual(len(rows), 1)
        self.assertIn("endpoint unavailable", rows[0]["error"])

    def test_retry_reuses_successes_and_selects_errors_or_missing(self):
        rows = [
            {"task_id": "a", "sample_index": 0, "solution": "print(1)", "error": None},
            {"task_id": "b", "sample_index": 0, "solution": "", "error": "timeout"},
        ]
        tasks = [{"task_id": name, "prompt": name} for name in ("a", "b", "c")]
        args = argparse.Namespace(benchmark="livecodebench", code_n_samples=1)
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source.jsonl"
            output = Path(temp) / "output.jsonl"
            source.write_text("".join(json.dumps(row) + "\n" for row in rows))
            kept, jobs = prepare_retry_samples(source, tasks, args, output)
        self.assertEqual([(r["task_id"], r["sample_index"]) for r in kept], [("a", 0)])
        self.assertEqual([(t["task_id"], i) for t, i in jobs], [("b", 0), ("c", 0)])

    def test_openai_compatible_generation(self):
        requests = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_POST(self):
                requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                body = json.dumps({"id": "fixture", "object": "chat.completion", "created": 0,
                                   "model": "fixture", "choices": [{"index": 0, "finish_reason": "stop",
                                   "message": {"role": "assistant", "content": "```python\nprint(1)\n```"}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        args = argparse.Namespace(concurrency=2, openai_base_url=f"http://127.0.0.1:{server.server_port}/v1",
                                  step_timeout=5, model="fixture", temperature=.2, top_p=.9,
                                  llm_max_completion_tokens=123, benchmark="livecodebench", code_n_samples=2)
        try:
            with tempfile.TemporaryDirectory() as temp:
                rows = asyncio.run(generate([{"task_id": "a", "prompt": "question"}], args,
                                            Path(temp) / "s.jsonl"))
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(r["solution"] == "print(1)" and not r["error"] for r in rows))
            self.assertEqual(requests[0]["max_tokens"], 123)
            self.assertEqual(requests[0]["messages"], [{"role": "user", "content": "question"}])
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


@unittest.skipUnless(os.environ.get("RUN_CODE_SANDBOX_TESTS") == "1", "Docker tests opt-in")
class OfficialScoringTests(unittest.TestCase):
    def args(self):
        return argparse.Namespace(code_memory="4g", code_eval_workers=2,
                                  code_docker_image="ubuntu:latest", code_eval_timeout=120)

    def test_lcb_stdin_functional_and_wrong_solution(self):
        samples = [{"input_output": json.dumps({"inputs": ["2\n"], "outputs": ["4\n"]})},
                   {"input_output": json.dumps({"inputs": ["[1,2]\n3"], "outputs": ["6"], "fn_name": "add"})}]
        job = {"dataset": "livecodebench", "task_ids": ["stdin", "functional"], "problems": samples,
               "solutions": [["print(int(input())*2)", "print(0)"],
                             ["class Solution:\n def add(self, xs, x): return sum(xs)+x", "invalid !"]],
               "k": [1, 2], "workers": 2, "timeout": 2, "base_only": False}
        result = score_in_container(job, self.args())
        self.assertAlmostEqual(result["metrics"]["pass@1"], .5)
        self.assertAlmostEqual(result["metrics"]["pass@2"], 1.)

    def test_evalplus_base_and_plus_are_distinct(self):
        problem = {"task_id": "HumanEval/fixture", "prompt": "def add(a, b):\n",
                   "canonical_solution": "    return a+b\n", "entry_point": "add", "atol": 0,
                   "base_input": [[1, 2]], "plus_input": [[-1, 5]]}
        job = {"dataset": "humaneval", "task_ids": [problem["task_id"]], "problems": [problem],
               "solutions": [["def add(a,b): return a+b", "def add(a,b): return 3"]],
               "k": [1, 2], "workers": 2, "timeout": 1, "base_only": False}
        result = score_in_container(job, self.args())
        self.assertEqual(result["metrics"]["base"]["pass@1"], 1.)
        self.assertEqual(result["metrics"]["plus"]["pass@1"], .5)
        self.assertEqual(result["metrics"]["plus"]["pass@2"], 1.)


if __name__ == "__main__":
    unittest.main()
