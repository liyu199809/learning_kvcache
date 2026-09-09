import json
from pathlib import Path
import tempfile
import unittest
import socket
from types import SimpleNamespace

from run_cleaned_taco_suite import Suite, merge_lcb, check_port_available


class MergeTests(unittest.TestCase):
    def test_coding_profile_and_resume_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            args = SimpleNamespace(root=temp, output_root=str(Path(temp) / "output"),
                                   resume=False, port=9400, max_tokens=32768,
                                   max_model_len=65536, lcb_only=True, sampling="coding")
            suite = Suite(args)
            command = suite.common("on_on", "off")
            for flag, value in {"--temperature": "0.6", "--top-p": "0.95", "--code-top-k": "20",
                                "--code-min-p": "0.0", "--code-presence-penalty": "0.0",
                                "--code-repetition-penalty": "1.0", "--code-seed": "42"}.items():
                self.assertEqual(command[command.index(flag) + 1], value)
            manifest = {"max_tokens": 32768, **suite.sampling_config}
            suite.check_sampling(manifest)
            manifest["temperature"] = 0
            with self.assertRaises(AssertionError):
                suite.check_sampling(manifest)

    def test_merge_rejects_mismatched_sampling(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            a, b = self.fixture(root)
            path = b / "manifest.json"
            manifest = json.loads(path.read_text())
            manifest["temperature"] = .6
            path.write_text(json.dumps(manifest))
            with self.assertRaises(AssertionError):
                merge_lcb(a, b, root / "merged.jsonl")

    def test_32k_budget_propagates_to_requests(self):
        with tempfile.TemporaryDirectory() as temp:
            args = SimpleNamespace(root=temp, output_root=str(Path(temp) / "output"),
                                   resume=False, port=9400, max_tokens=32768,
                                   max_model_len=65536, lcb_only=True)
            suite = Suite(args)
            command = suite.common("on_on", "off")
            self.assertEqual(command[command.index("--llm-max-completion-tokens") + 1], "32768")
            self.assertEqual(suite.max_model_len, 65536)
            self.assertTrue(suite.lcb_only)

    def test_port_probe_rejects_listener_but_accepts_closed_connections(self):
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
            listener.listen()
            with self.assertRaises(OSError):
                check_port_available(port)
            with socket.create_connection(("127.0.0.1", port), timeout=2) as client:
                accepted, _ = listener.accept()
                with accepted:
                    accepted.shutdown(socket.SHUT_WR)
                    self.assertEqual(client.recv(1), b"")
        check_port_available(port)

    def fixture(self, root):
        directories = []
        for name, start, end, total in (("v5", 0, 880, 880), ("v6", 880, 1055, 1055)):
            directory = root / name
            directory.mkdir()
            rows = [{"task_id": str(i), "sample_index": 0, "solution": "print(1)"}
                    for i in range(start, end)]
            (directory / "samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            (directory / "manifest.json").write_text(json.dumps({
                "revision": "same", "model": "same", "thinking": "off", "max_tokens": 16384,
                "release": "release_" + name, "available_tasks": total,
                "selected_tasks": [r["task_id"] for r in rows]}))
            directories.append(directory)
        return directories

    def test_merge_exact_cumulative_release_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            a, b = self.fixture(root)
            merged = root / "merged.jsonl"
            merge_lcb(a, b, merged)
            rows = [json.loads(line) for line in merged.read_text().splitlines()]
            self.assertEqual([r["task_id"] for r in rows], [str(i) for i in range(1055)])
            with self.assertRaises(FileExistsError):
                merge_lcb(a, b, merged)

    def test_reject_mismatched_thinking(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            a, b = self.fixture(root)
            path = b / "manifest.json"
            manifest = json.loads(path.read_text())
            manifest["thinking"] = "on"
            path.write_text(json.dumps(manifest))
            with self.assertRaises(AssertionError):
                merge_lcb(a, b, root / "merged.jsonl")

    def test_reject_missing_tasks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            a, b = self.fixture(root)
            path = a / "samples.jsonl"
            path.write_text("\n".join(path.read_text().splitlines()[:-1]))
            with self.assertRaises(AssertionError):
                merge_lcb(a, b, root / "merged.jsonl")


if __name__ == "__main__":
    unittest.main()
