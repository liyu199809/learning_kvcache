"""Verify slot reuse and occupier placement without GPUs or child processes."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from prefix_tuning.virtual_prefix import run_two_slot_prefix_queue as queue


class QueueTest(unittest.TestCase):
    def simulate(self, failure=False):
        tick = [0]
        starts, holders, exported, commands = [], [], [], []

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

        def occupy(slot):
            holders.append(slot)
            return dict(slot=slot)

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(queue, "EXP", Path(directory)), \
                 patch.object(queue, "start", start), \
                 patch.object(queue, "capture"), patch.object(queue, "cleanup"), \
                 patch.object(queue, "export", exported.append), \
                 patch.object(queue, "occupy", occupy), \
                 patch.object(queue.time, "sleep", advance), \
                 patch.object(queue.subprocess, "run", lambda args, **kwargs: commands.append(args)):
                if failure:
                    with self.assertRaises(RuntimeError):
                        queue.run(None)
                else:
                    queue.run(None)
        return starts, holders, exported, commands

    def test_reuse_first_finished_slot_then_occupy_other_slot(self):
        starts, holders, exported, commands = self.simulate()
        self.assertEqual(starts, [("awm", "a", None), ("envscaler", "b", None), ("taco", "b", None)])
        self.assertEqual(holders, ["a"])
        self.assertEqual(exported, ["envscaler", "awm", "taco"])
        self.assertEqual(len(commands), 4)

    def test_failure_does_not_start_holder_or_pending_domain(self):
        starts, holders, exported, commands = self.simulate(failure=True)
        self.assertEqual(len(starts), 2)
        self.assertEqual(holders, [])
        self.assertEqual(exported, ["awm"])
        self.assertEqual(commands, [])


if __name__ == "__main__":
    unittest.main()
