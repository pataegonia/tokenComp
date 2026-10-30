#!/usr/bin/env python3
"""Flag fresh runs stuck in the low-Score training mode (plan §15).

Healthy and failed .0128 seeds share the train PSNR curve until about 20k. A
healthy run then moves bits into the Score path and its PSNR rises; a failed
run lets the Score bits shrink and trails by about 0.9 dB at 40k (H-OFF s2 and
N-ON s2 against s1, 2026-10-29). This reads only the ``train_psnr`` and
``train_codec_score_bits`` records of each run's TensorBoard files and compares
a step window with reference runs at the same step.

    python scripts/check_nfcgs_train_curve.py --runs RUN_DIR... --references RUN_DIR...
"""
from __future__ import annotations

import argparse
from pathlib import Path
import struct

TAGS = ("train_psnr", "train_codec_score_bits")
EARLY_STEP = 25000  # last step where healthy and failed runs still match


def records(path: Path):
    """Raw TFRecord payloads; a record still being written ends the file."""
    with path.open("rb") as stream:
        while len(header := stream.read(12)) == 12:
            (length,) = struct.unpack("<Q", header[:8])
            data = stream.read(length + 4)[:length]
            if len(data) < length:
                return
            yield data


def read_curves(run_dir: Path, stop: int = 0) -> dict[str, dict[int, float]]:
    from tensorboard.compat.proto.event_pb2 import Event
    from tensorboard.util import tensor_util
    curves = {tag: {} for tag in TAGS}
    for path in sorted(run_dir.glob("tensorboard/**/events.out.tfevents*")):
        for data in records(path):
            # Cheap byte filter first: a run logs ~27 scalars per step.
            if not any(tag.encode() in data for tag in TAGS):
                continue
            event = Event.FromString(data)
            if stop and event.step >= stop:
                break
            for value in event.summary.value:
                if value.tag in curves:
                    curves[value.tag][event.step] = (value.simple_value if value.HasField("simple_value")
                                                     else float(tensor_util.make_ndarray(value.tensor)))
    return curves


def window_mean(curve: dict[int, float], end: int, window: int) -> float | None:
    values = [v for step, v in curve.items() if end - window <= step < end]
    return sum(values) / len(values) if values else None


def summarize(curves, end: int, window: int) -> dict:
    psnr = window_mean(curves["train_psnr"], end, window)
    early = window_mean(curves["train_codec_score_bits"], EARLY_STEP, window)
    late = window_mean(curves["train_codec_score_bits"], end, window)
    return dict(psnr=psnr, score_bits_change=late / early - 1 if early and late and end > EARLY_STEP else None)


def check(runs: list[Path], references: list[Path], at: int, window: int, threshold: float) -> list[dict]:
    rows = []
    for run in runs:
        curves = read_curves(run, at + 1)
        last = max(curves["train_psnr"], default=-1)
        end = at if last >= at else (last + 1) // window * window
        if end < window:
            rows.append(dict(run=run, end=end, verdict="no data"))
            continue
        result = summarize(curves, end, window)
        reference = [summarize(read_curves(ref, end), end, window)["psnr"] for ref in references]
        reference = [v for v in reference if v is not None]
        gap = result["psnr"] - sum(reference) / len(reference) if reference else None
        verdict = "no reference" if gap is None else "FAIL?" if gap < -threshold else "ok"
        rows.append(dict(run=run, end=end, gap=gap, verdict=verdict, **result))
    return rows


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.add_argument("--references", type=Path, nargs="*", default=[],
                   help="healthy runs of the closest cell at the same lambda")
    p.add_argument("--at", type=int, default=40000, help="window end; earlier if a run has not reached it")
    p.add_argument("--window", type=int, default=5000)
    p.add_argument("--threshold", type=float, default=0.5, help="PSNR gap in dB that flags a run")
    args = p.parse_args(argv)
    fmt = lambda v, spec: "n/a" if v is None else format(v, spec)
    print(f"{'run':<60} {'step':>6} {'PSNR':>6} {'gap':>6} {'Score bits':>10}  verdict")
    for row in check(args.runs, args.references, args.at, args.window, args.threshold):
        print(f"{str(row['run'])[-60:]:<60} {row['end']:>6} {fmt(row.get('psnr'), '.2f'):>6} "
              f"{fmt(row.get('gap'), '+.2f'):>6} {fmt(row.get('score_bits_change'), '+.0%'):>10}  {row['verdict']}")


if __name__ == "__main__":
    main()
