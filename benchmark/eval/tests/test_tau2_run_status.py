import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from benchmark.eval.tau2 import run_tau2


class Tau2RunStatusTests(unittest.TestCase):
    def run_saved(self, simulations):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, 'results.json').write_text(json.dumps({
                'tasks': [{'id': '0'}], 'simulations': simulations}))
            args = Namespace(openai_base_url='http://127.0.0.1:18081/v1', model='test',
                             llm_max_completion_tokens=256, api_key='EMPTY', temperature=0,
                             top_p=1, tau2_max_steps=5, output_dir=directory,
                             tau2_domain='airline', tau2_split='base', tau2_num_trials=1,
                             concurrency=1, tau2_num_tasks=1, task_ids=None,
                             tau2_judge_api_key_env='BENCHMARK_UNUSED_TEST_KEY')
            with patch('benchmark.eval.tau2._ensure_ready'), \
                 patch('benchmark.eval.tau2._check_server'), \
                 patch('benchmark.eval.tau2.subprocess.run', return_value=Namespace(returncode=0)):
                return run_tau2(args)

    def test_upstream_zero_exit_with_infrastructure_error_is_failure(self):
        self.assertEqual(self.run_saved([{'termination_reason': 'infrastructure_error',
                                         'reward_info': None}]), 1)

    def test_missing_simulation_is_failure(self):
        self.assertEqual(self.run_saved([]), 1)

    def test_valid_zero_reward_is_completed_evaluation(self):
        self.assertEqual(self.run_saved([{'termination_reason': 'agent_stop',
                                         'reward_info': {'reward': 0.0}}]), 0)


if __name__ == '__main__':
    unittest.main()
