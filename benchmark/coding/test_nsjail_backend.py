"""Optional real official LCB scoring through NsJail, including function tasks."""
import argparse
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from nsjail_backend import PROJECT, score_in_nsjail


class WorkerFailureTests(unittest.TestCase):
    def test_official_worker_crash_cannot_be_reported_as_zero_accuracy(self):
        # Real failure observed when EvalPlus tried to raise RLIMIT_STACK:
        # the upstream parent returned 0 and classified crashed children as timeout.
        result = {'returncode': 0, 'scores': {'metrics': {'pass@1': 0}},
                  'stdout': '', 'config': '', 'stderr':
                  'Process Process-1:1:\nTraceback (most recent call last):\n'
                  'ValueError: not allowed to raise maximum limit\n'}
        with tempfile.TemporaryDirectory(dir=PROJECT / 'benchmark/runtime') as output:
            with patch('nsjail_backend.execute', return_value=result):
                with self.assertRaisesRegex(RuntimeError, 'NsJail scoring failed'):
                    score_in_nsjail({}, argparse.Namespace(output_dir=output))


@unittest.skipUnless(os.environ.get('RUN_NSJAIL_TESTS') == '1', 'Real NsJail test opt-in')
class NsjailScoringTests(unittest.TestCase):
    def test_stdin_and_functional_correct_and_wrong_answers(self):
        problems = [
            {'input_output': json.dumps({'inputs': ['2\n'], 'outputs': ['4\n']})},
            {'input_output': json.dumps({'inputs': ['[1,2]\n3'], 'outputs': ['6'], 'fn_name': 'add'})},
        ]
        job = {'dataset': 'livecodebench', 'task_ids': ['stdin', 'functional'],
               'problems': problems, 'solutions': [
                   ['print(int(input())*2)', 'print(0)'],
                   ['class Solution:\n def add(self,xs,x): return sum(xs)+x', 'invalid !']],
               'k': [1, 2], 'workers': 2, 'timeout': 2, 'base_only': False}
        with tempfile.TemporaryDirectory(dir=PROJECT / 'benchmark/runtime') as output:
            args = argparse.Namespace(output_dir=output, code_memory='4g',
                                      code_eval_workers=2, code_eval_timeout=120)
            result = score_in_nsjail(job, args)
        self.assertEqual(result['metrics']['pass@1'], .5)
        self.assertEqual(result['metrics']['pass@2'], 1.)
        self.assertEqual([row['passed'] for row in result['results']], [[True, False], [True, False]])


if __name__ == '__main__':
    unittest.main()
