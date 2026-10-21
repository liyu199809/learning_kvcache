#!/usr/bin/env python3
"""CPU load generator: burn N worker processes for a given duration.

Overall utilization ~= workers / nproc. SIGTERM stops gracefully.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import signal
import sys
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Occupy CPU cores with busy loops.")
    parser.add_argument("--workers", type=int, default=64, help="number of full-load worker processes (default: 64)")
    parser.add_argument("--duration", type=float, default=60.0, help="run time in seconds (default: 60)")
    return parser.parse_args()


def worker(stop_event, duration: float) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    deadline = time.monotonic() + duration
    x = 1.0000001
    while not stop_event.is_set() and time.monotonic() < deadline:
        # arithmetic busy loop, no allocations
        for _ in range(10000):
            x = x * 1.0000001 % 10.0
    # touch x so the loop is not optimized away
    sys.stderr.write("") if x < 0 else None


def main() -> int:
    args = parse_args()
    workers = max(1, min(args.workers, mp.cpu_count()))
    print(
        f"Starting {workers} CPU worker(s) on {mp.cpu_count()} cores "
        f"(~{workers / mp.cpu_count():.0%} overall) for {args.duration:.0f}s",
        flush=True,
    )

    ctx = mp.get_context("spawn")
    stop_event = ctx.Event()

    def request_stop(_signum, _frame):
        if not stop_event.is_set():
            print("Stopping all workers...", flush=True)
            stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    procs = [ctx.Process(target=worker, args=(stop_event, args.duration)) for _ in range(workers)]
    for p in procs:
        p.start()

    for p in procs:
        p.join()
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
