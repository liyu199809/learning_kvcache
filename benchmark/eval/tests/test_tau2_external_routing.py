import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from benchmark.eval import tau2


class Tau2ExternalRoutingTests(unittest.TestCase):
    def test_user_and_judge_are_external_without_key_in_argv(self):
        captured = {}

        def fake_run(command, **kwargs):
            captured["command"] = command
            captured["env"] = kwargs["env"]
            return Namespace(returncode=7)

        with tempfile.TemporaryDirectory() as directory, patch.dict(
            "os.environ", {"ARK_API_KEY": "sentinel-secret"}
        ), patch("benchmark.eval.tau2._ensure_ready"), patch(
            "benchmark.eval.tau2._check_server"
        ), patch("benchmark.eval.tau2.subprocess.run", side_effect=fake_run):
            args = Namespace(
                openai_base_url="http://127.0.0.1:18081/v1",
                model="candidate",
                llm_max_completion_tokens=16384,
                api_key="EMPTY",
                temperature=0,
                top_p=1,
                tau2_agent_thinking=False,
                tau2_user_thinking=False,
                tau2_user_model=None,
                tau2_user_base_url=None,
                tau2_user_api_key_env="ARK_API_KEY",
                tau2_require_external_user=True,
                tau2_judge_model=None,
                tau2_judge_base_url=None,
                tau2_judge_api_key_env="ARK_API_KEY",
                tau2_judge_thinking=False,
                tau2_require_external_judge=True,
                tau2_max_steps=200,
                output_dir=directory,
                tau2_domain="retail",
                tau2_split="base",
                tau2_num_trials=1,
                concurrency=8,
                tau2_num_tasks=None,
                task_ids=None,
            )
            self.assertEqual(tau2.run_tau2(args), 7)

        command = captured["command"]
        env = captured["env"]
        serialized = "\n".join(command) + env["TAU2_LLM_NL_ASSERTIONS_ARGS"]
        self.assertNotIn("sentinel-secret", serialized)
        self.assertEqual(env["OPENAI_API_KEY"], "sentinel-secret")
        self.assertEqual(
            command[command.index("--user-llm") + 1],
            tau2.DEFAULT_TAU2_USER_MODEL,
        )
        user_args = json.loads(command[command.index("--user-llm-args") + 1])
        self.assertEqual(user_args["max_tokens"], 4096)
        self.assertNotIn("api_key", user_args)


if __name__ == "__main__":
    unittest.main()
