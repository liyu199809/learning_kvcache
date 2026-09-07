from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import eval_main  # noqa: E402


class EvalMainCodingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.args = SimpleNamespace(
            port=8000,
            api_key="EMPTY",
            temperature=0.0,
            top_p=1.0,
            max_completion_tokens=4096,
            coding_max_completion_tokens=16384,
            coding_concurrency=8,
            coding_eval_workers=4,
            coding_n_samples=1,
        )

    def command(self, benchmark: str) -> list[str]:
        return eval_main.benchmark_command(
            benchmark,
            self.args,
            "model",
            Path("/tmp/model"),
            Path("/tmp/output"),
        )

    def test_aliases_expand_to_only_requested_coding_suites(self) -> None:
        self.assertEqual(
            eval_main.expand_benchmarks("coding"),
            [
                "humaneval_plus",
                "mbpp_plus",
                "livecodebench_v5",
                "livecodebench_v6",
            ],
        )
        self.assertEqual(
            eval_main.expand_benchmarks("humaneval+,mbpp+,lcb"),
            [
                "humaneval_plus",
                "mbpp_plus",
                "livecodebench_v5",
                "livecodebench_v6",
            ],
        )

    def test_runner_module_is_present_exactly_once(self) -> None:
        command = eval_main.common_eval_args(self.args, "model")
        self.assertEqual(command.count("benchmark.eval.run_eval"), 1)
        self.assertEqual(command[:3], [str(eval_main.PYTHON), "-m", "benchmark.eval.run_eval"])

    def test_evalplus_command(self) -> None:
        command = self.command("humaneval_plus")
        self.assertEqual(command[command.index("--benchmark") + 1], "humaneval+")
        self.assertEqual(
            command[command.index("--llm-max-completion-tokens") + 1], "16384"
        )
        self.assertEqual(command[command.index("--output-dir") + 1], "/tmp/output/humaneval+")

    def test_lcb_release_commands(self) -> None:
        for version in ("v5", "v6"):
            command = self.command(f"livecodebench_{version}")
            self.assertEqual(command[command.index("--benchmark") + 1], "livecodebench")
            self.assertEqual(command[command.index("--lcb-release") + 1], version)
            self.assertEqual(
                command[command.index("--output-dir") + 1],
                f"/tmp/output/livecodebench/{version}",
            )


if __name__ == "__main__":
    unittest.main()
