"""Verify three concurrent two-GPU jobs without GPUs or child processes."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from prefix_tuning.virtual_prefix.internal import training_queue as queue


class QueueTest(unittest.TestCase):
    def test_experiment_root_is_absolute(self):
        self.assertTrue(queue.EXP.is_absolute())

    def test_start_assigns_one_student_and_one_teacher_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            experiment = Path(directory)
            (experiment / "logs").mkdir()
            process = Mock(pid=12345)
            with patch.dict(os.environ, {
                     "VAL_BEFORE_TRAIN": "True", "TEST_FREQ": "20",
                     "USE_WANDB": "1", "WANDB_MODE": "online",
                 }, clear=False), \
                 patch.object(queue, "EXP", experiment), \
                 patch.object(queue.subprocess, "Popen", return_value=process) as popen:
                job = queue.start("envscaler", "b", None)

        args = popen.call_args.args[0]
        env = popen.call_args.kwargs["env"]
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "2,3")
        self.assertEqual(env["ACTOR_GPUS"], "1")
        self.assertEqual(env["TEACHER_GPUS"], "1")
        self.assertEqual(env["RUN_NAME"], "envscaler_2gpu")
        self.assertEqual(env["VAL_BEFORE_TRAIN"], "True")
        self.assertEqual(env["TEST_FREQ"], "20")
        self.assertEqual(env["USE_WANDB"], "1")
        self.assertEqual(env["WANDB_MODE"], "online")
        self.assertIn("+ray_kwargs.ray_init.num_gpus=2", args)
        self.assertEqual(job["gpus"], "2,3")

    def simulate(self, failure=False):
        tick = [0]
        starts, exported, commands = [], [], []

        def start(domain, slot, resume):
            starts.append((domain, slot, resume))
            end = tick[0] + {"awm": 3, "envscaler": 2, "taco": 4}[domain]

            class Process:
                pid = 999999

                def poll(self):
                    return None if tick[0] < end else (1 if failure and domain == "envscaler" else 0)

            return dict(domain=domain, slot=slot, log="fake", process=Process(), children={})

        def advance(_):
            tick[0] += 1
            assert tick[0] < 20, "Scheduler did not terminate"

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(queue, "EXP", Path(directory)), \
                 patch.object(queue, "start", start), \
                 patch.object(queue, "capture"), patch.object(queue, "cleanup"), \
                 patch.object(queue, "export", exported.append), \
                 patch.object(queue.time, "sleep", advance), \
                 patch.object(queue.subprocess, "run", lambda args, **kwargs: commands.append(args)):
                if failure:
                    with self.assertRaises(RuntimeError):
                        queue.run(None)
                else:
                    queue.run(None)
        return starts, exported, commands

    def test_three_domains_start_concurrently(self):
        starts, exported, commands = self.simulate()
        self.assertEqual(starts, [("awm", "a", None), ("envscaler", "b", None), ("taco", "c", None)])
        self.assertEqual(exported, ["envscaler", "awm", "taco"])
        self.assertEqual(len(commands), 4)

    def test_failure_preserves_other_domains(self):
        starts, exported, commands = self.simulate(failure=True)
        self.assertEqual(len(starts), 3)
        self.assertEqual(exported, ["awm", "taco"])
        self.assertEqual(commands, [])


if __name__ == "__main__":
    unittest.main()
