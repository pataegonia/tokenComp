#!/usr/bin/env python3
"""Fill otherwise-idle GPU memory and add small, periodic compute bursts.

This helper is intentionally independent of the training process.  Start it in
the same Slurm allocation; it waits for training to establish its CUDA memory
footprint before consuming all but a configurable amount of free VRAM.
"""

from __future__ import annotations

import argparse
import signal
import time

import torch


MIB = 1024 * 1024
_stop_requested = False


def _request_stop(_signum: int, _frame: object) -> None:
    global _stop_requested
    _stop_requested = True


def _memory_mib() -> tuple[int, int, int]:
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_mib = free_bytes // MIB
    total_mib = total_bytes // MIB
    return free_mib, total_mib - free_mib, total_mib


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--reserve-mib",
        type=_positive_int,
        default=1024,
        help="VRAM left free after filling (default: 1024 MiB)",
    )
    parser.add_argument(
        "--wait-used-mib",
        type=_nonnegative_int,
        default=4096,
        help="wait for total GPU usage to reach this value (default: 4096 MiB)",
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=45.0,
        help="additional warm-up delay after training is detected (default: 45)",
    )
    parser.add_argument(
        "--chunk-mib",
        type=_positive_int,
        default=256,
        help="allocation chunk size (default: 256 MiB)",
    )
    parser.add_argument(
        "--matmul-size",
        type=_positive_int,
        default=4096,
        help="side length of FP16 square matrices (default: 4096)",
    )
    parser.add_argument(
        "--burst-ops",
        type=_nonnegative_int,
        default=16,
        help="matrix multiplies per periodic burst; 0 disables compute (default: 16)",
    )
    parser.add_argument(
        "--period-seconds",
        type=float,
        default=1.0,
        help="target interval between burst starts (default: 1.0)",
    )
    parser.add_argument(
        "--log-interval-seconds",
        type=float,
        default=60.0,
        help="VRAM status log interval (default: 60)",
    )
    args = parser.parse_args()
    if args.settle_seconds < 0:
        parser.error("--settle-seconds must be zero or greater")
    if args.period_seconds <= 0:
        parser.error("--period-seconds must be greater than zero")
    if args.log_interval_seconds <= 0:
        parser.error("--log-interval-seconds must be greater than zero")
    return args


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    free_mib, used_mib, total_mib = _memory_mib()
    if args.reserve_mib >= total_mib:
        raise SystemExit(
            f"--reserve-mib ({args.reserve_mib}) must be below total VRAM ({total_mib} MiB)"
        )
    print(
        "[gpu-dummy] "
        f"device={torch.cuda.get_device_name(device)} total={total_mib} MiB "
        f"initial_used={used_mib} MiB reserve={args.reserve_mib} MiB",
        flush=True,
    )

    while not _stop_requested and used_mib < args.wait_used_mib:
        time.sleep(1.0)
        free_mib, used_mib, total_mib = _memory_mib()
    if _stop_requested:
        return 0

    print(
        f"[gpu-dummy] training detected at {used_mib} MiB; "
        f"waiting {args.settle_seconds:g}s for warm-up",
        flush=True,
    )
    deadline = time.monotonic() + args.settle_seconds
    while not _stop_requested and time.monotonic() < deadline:
        time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
    if _stop_requested:
        return 0

    # Create and warm up the compute buffers before filling VRAM so cuBLAS can
    # establish its context/workspace while plenty of memory is still free.
    work_a = work_b = work_out = None
    if args.burst_ops:
        work_a = torch.randn(
            (args.matmul_size, args.matmul_size), dtype=torch.float16, device=device
        )
        work_b = torch.randn_like(work_a)
        work_out = torch.empty_like(work_a)
        torch.mm(work_a, work_b, out=work_out)
        torch.cuda.synchronize(device)

    fillers: list[torch.Tensor] = []
    allocated_mib = 0
    current_chunk_mib = args.chunk_mib
    allocation_margin_mib = 32
    while not _stop_requested:
        free_mib, _, _ = _memory_mib()
        available_mib = free_mib - args.reserve_mib - allocation_margin_mib
        if available_mib < 16:
            break
        allocation_mib = min(current_chunk_mib, available_mib)
        try:
            block = torch.empty(allocation_mib * MIB, dtype=torch.uint8, device=device)
            block.zero_()
            fillers.append(block)
            allocated_mib += allocation_mib
            current_chunk_mib = args.chunk_mib
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            current_chunk_mib //= 2
            if current_chunk_mib < 16:
                break
    torch.cuda.synchronize(device)

    free_mib, used_mib, total_mib = _memory_mib()
    print(
        "[gpu-dummy] "
        f"filler={allocated_mib} MiB total_used={used_mib}/{total_mib} MiB "
        f"free={free_mib} MiB burst_ops={args.burst_ops} "
        f"matmul={args.matmul_size}x{args.matmul_size} period={args.period_seconds:g}s",
        flush=True,
    )

    next_burst = time.monotonic()
    next_log = next_burst + args.log_interval_seconds
    try:
        with torch.inference_mode():
            while not _stop_requested:
                burst_started = time.monotonic()
                if args.burst_ops:
                    assert work_a is not None and work_b is not None and work_out is not None
                    for _ in range(args.burst_ops):
                        torch.mm(work_a, work_b, out=work_out)
                    torch.cuda.synchronize(device)

                now = time.monotonic()
                if now >= next_log:
                    free_mib, used_mib, total_mib = _memory_mib()
                    print(
                        "[gpu-dummy] "
                        f"alive total_used={used_mib}/{total_mib} MiB free={free_mib} MiB",
                        flush=True,
                    )
                    next_log = now + args.log_interval_seconds

                next_burst = max(next_burst + args.period_seconds, burst_started)
                sleep_seconds = next_burst - time.monotonic()
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)
    finally:
        fillers.clear()
        del work_a, work_b, work_out
        torch.cuda.empty_cache()
        print("[gpu-dummy] stopped and released filler VRAM", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
