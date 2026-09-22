#!/usr/bin/env python3
"""Controllable multi-device memory/compute load generator using PyTorch.

Supports NVIDIA CUDA GPUs and Huawei Ascend NPUs (via torch_npu).
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import queue as queue_module
import signal
import sys
import time
from typing import Any

SUPPORTED_BACKENDS = ("cuda", "npu")


def percentage(value: str) -> float:
    number = float(value)
    if not 0 <= number <= 95:
        raise argparse.ArgumentTypeError("must be between 0 and 95")
    return number


def duty_cycle(value: str) -> float:
    number = float(value)
    if not 0 <= number <= 100:
        raise argparse.ArgumentTypeError("must be between 0 and 100")
    return number


def positive_float(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Occupy memory and optionally generate load on one or more "
            "CUDA GPUs or Ascend NPUs."
        )
    )
    parser.add_argument(
        "--backend",
        choices=("auto", *SUPPORTED_BACKENDS),
        default="auto",
        help="accelerator backend to use: cuda, npu, or auto-detect (default: auto)",
    )
    devices = parser.add_mutually_exclusive_group()
    devices.add_argument(
        "--devices",
        help='comma-separated device indexes, or "all" (default: 0)',
    )
    devices.add_argument(
        "--device",
        type=int,
        help="single device index (kept for backward compatibility)",
    )
    parser.add_argument(
        "--memory-percent",
        type=percentage,
        default=50.0,
        help="percentage of each device's currently free memory to reserve, 0-95 (default: 50)",
    )
    parser.add_argument(
        "--utilization",
        type=duty_cycle,
        default=80.0,
        help="approximate compute duty cycle on each device, 0-100 (default: 80)",
    )
    parser.add_argument(
        "--duration",
        type=positive_float,
        default=60.0,
        help="run time per device in seconds (default: 60)",
    )
    parser.add_argument(
        "--matrix-size",
        type=int,
        default=4096,
        help="square matrix size used for multiplication (default: 4096)",
    )
    parser.add_argument(
        "--interval",
        type=positive_float,
        default=0.2,
        help="work/rest control interval in seconds (default: 0.2)",
    )
    return parser.parse_args()


def parse_devices(spec: str, device_count: int) -> list[int]:
    if spec.strip().lower() == "all":
        return list(range(device_count))

    try:
        selected = [int(item.strip()) for item in spec.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError('--devices must be "all" or comma-separated integers') from exc

    if not selected:
        raise ValueError("no devices were selected")
    if len(selected) != len(set(selected)):
        raise ValueError("duplicate device indexes are not allowed")
    invalid = [index for index in selected if index < 0 or index >= device_count]
    if invalid:
        raise ValueError(
            f"device index(es) {invalid} do not exist; visible indexes are 0-{device_count - 1}"
        )
    return selected


def detect_backend() -> str | None:
    """Return "cuda" or "npu" for the first usable backend, or None."""
    try:
        import torch
    except ImportError:
        return None

    try:
        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # a broken CUDA install must not block NPU detection
        pass

    try:
        import torch_npu  # noqa: F401  # registers the torch.npu backend

        if torch.npu.is_available():
            return "npu"
    except Exception:
        pass

    return None


def load_torch(backend: str) -> tuple[Any, Any]:
    """Import torch (plus torch_npu for NPU) and return (torch, backend_api)."""
    import torch

    if backend == "npu":
        try:
            import torch_npu  # noqa: F401  # registers the torch.npu backend
        except ImportError as exc:
            raise ImportError(
                "torch_npu is not installed. Install an Ascend-enabled PyTorch build first."
            ) from exc

    api = getattr(torch, backend, None)
    if api is None:
        raise ImportError(f"this PyTorch build has no '{backend}' backend")
    return torch, api


def format_bytes(value: int) -> str:
    return f"{value / 1024**3:.2f} GiB"


def report(status_queue: Any, level: str, device_index: int, message: str) -> None:
    status_queue.put((level, device_index, message))


def device_worker(
    backend: str,
    device_index: int,
    memory_percent: float,
    utilization: float,
    duration: float,
    matrix_size: int,
    interval: float,
    stop_event: Any,
    status_queue: Any,
) -> None:
    """Run one independent accelerator context so all selected devices work concurrently."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    allocations: list[Any] = []
    a = b = out = None

    try:
        torch, api = load_torch(backend)

        device = torch.device(f"{backend}:{device_index}")
        api.set_device(device)
        free_bytes, total_bytes = api.mem_get_info(device)
        target_bytes = int(free_bytes * memory_percent / 100)
        report(
            status_queue,
            "info",
            device_index,
            f"{api.get_device_name(device)} | total {format_bytes(total_bytes)} | "
            f"free {format_bytes(free_bytes)} | memory target {format_bytes(target_bytes)}",
        )

        # Allocate in chunks so large memory targets do not require one giant tensor.
        chunk_bytes = 256 * 1024**2
        element_bytes = 4  # float32
        remaining = target_bytes
        while remaining >= element_bytes and not stop_event.is_set():
            this_chunk = min(remaining, chunk_bytes)
            elements = this_chunk // element_bytes
            allocations.append(torch.empty(elements, dtype=torch.float32, device=device))
            remaining -= elements * element_bytes

        if stop_event.is_set():
            return

        allocated = sum(t.numel() * t.element_size() for t in allocations)
        report(status_queue, "info", device_index, f"reserved {format_bytes(allocated)}")

        if utilization > 0:
            # a, b, and out coexist. Keep 256 MiB extra headroom for the runtime
            # (CUDA/CANN) and PyTorch.
            matrix_bytes = 3 * matrix_size * matrix_size * 2  # float16
            current_free, _ = api.mem_get_info(device)
            if current_free < matrix_bytes + 256 * 1024**2:
                raise RuntimeError(
                    "not enough free memory remains for compute matrices; lower "
                    "--memory-percent or --matrix-size"
                )
            a = torch.randn((matrix_size, matrix_size), dtype=torch.float16, device=device)
            b = torch.randn_like(a)
            out = torch.empty_like(a)

        started = time.monotonic()
        deadline = started + duration
        work_seconds = interval * utilization / 100

        while not stop_event.is_set() and time.monotonic() < deadline:
            cycle_start = time.monotonic()
            work_deadline = cycle_start + work_seconds

            if a is not None and b is not None and out is not None:
                while not stop_event.is_set() and time.monotonic() < work_deadline:
                    torch.mm(a, b, out=out)
                    out.mul_(1.0 / matrix_size)
                    api.synchronize(device)
                    a, out = out, a

            # Sleeping to the next interval makes --utilization an approximate
            # duty cycle instead of merely queuing asynchronous kernels.
            sleep_for = min(cycle_start + interval, deadline) - time.monotonic()
            if sleep_for > 0 and not stop_event.is_set():
                stop_event.wait(sleep_for)

        elapsed = time.monotonic() - started
        report(status_queue, "done", device_index, f"stopped after {elapsed:.1f}s")
    except Exception as exc:
        report(status_queue, "error", device_index, f"{type(exc).__name__}: {exc}")
        stop_event.set()
        raise SystemExit(1) from exc
    finally:
        a = b = out = None
        allocations.clear()
        try:
            torch, api = load_torch(backend)
            api.empty_cache()
        except Exception:
            pass


def main() -> int:
    args = parse_args()
    if args.matrix_size < 256:
        print("error: --matrix-size must be at least 256", file=sys.stderr)
        return 2

    if args.backend == "auto":
        backend = detect_backend()
        if backend is None:
            print(
                "error: no usable accelerator backend was found (tried CUDA and Ascend NPU); "
                "check the driver and the PyTorch/torch_npu installation.",
                file=sys.stderr,
            )
            return 1
    else:
        backend = args.backend

    try:
        torch, api = load_torch(backend)
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not api.is_available():
        if backend == "npu":
            print(
                "error: Ascend NPU is unavailable; check the driver/CANN and "
                "torch_npu installation.",
                file=sys.stderr,
            )
        else:
            print(
                "error: CUDA is unavailable; check the NVIDIA driver and PyTorch installation.",
                file=sys.stderr,
            )
        return 1

    device_count = api.device_count()
    device_spec = args.devices if args.devices is not None else str(args.device or 0)
    try:
        selected_devices = parse_devices(device_spec, device_count)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(
        f"Starting {len(selected_devices)} worker(s) on {backend} device(s) "
        f"{','.join(map(str, selected_devices))} for {args.duration:.0f}s; "
        "press Ctrl+C to stop all."
    )

    context = mp.get_context("spawn")
    stop_event = context.Event()
    status_queue = context.Queue()

    def request_stop(_signum: int, _frame: object) -> None:
        if not stop_event.is_set():
            print("Stopping all workers...", flush=True)
            stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    processes = [
        context.Process(
            target=device_worker,
            name=f"{backend}-load-{index}",
            args=(
                backend,
                index,
                args.memory_percent,
                args.utilization,
                args.duration,
                args.matrix_size,
                args.interval,
                stop_event,
                status_queue,
            ),
        )
        for index in selected_devices
    ]

    for process in processes:
        process.start()

    while any(process.is_alive() for process in processes):
        try:
            level, index, message = status_queue.get(timeout=0.2)
            stream = sys.stderr if level == "error" else sys.stdout
            print(f"[{backend}:{index}] {message}", file=stream, flush=True)
        except queue_module.Empty:
            pass

    for process in processes:
        process.join()

    # Print messages that arrived between the last liveness check and join.
    while True:
        try:
            level, index, message = status_queue.get_nowait()
            stream = sys.stderr if level == "error" else sys.stdout
            print(f"[{backend}:{index}] {message}", file=stream, flush=True)
        except queue_module.Empty:
            break

    failed = [
        selected_devices[position]
        for position, process in enumerate(processes)
        if process.exitcode != 0
    ]
    if failed:
        print(f"error: worker failure on {backend} device(s) {failed}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
