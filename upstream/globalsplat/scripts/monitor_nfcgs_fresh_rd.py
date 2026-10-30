#!/usr/bin/env python3
"""Evaluate frozen checkpoints on holdout128, then publish paired convergence evidence."""
from __future__ import annotations

import argparse
import csv
import io
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_nfcgs_fresh_rd as runner
from globalsplat.fresh_rd import (atomic_json, checkpoint_path, directory_lock, file_hash,
    gate1_decision, objective, paired_change, rate_text, read_json, validate_rows, wave_runs)

REFERENCE_JOBS = {"0": "455327", "0.0032": "455329", "0.0064": "447744",
                  "0.0128": "447746", "0.0256": "447900", "0.0512": "455341"}


class CheckpointIdentityChanged(ValueError):
    """A valid evaluation belongs to an earlier version of a checkpoint file."""


def supersede_evaluation(output, checkpoint):
    """Keep the old metrics, then require a fresh evaluation of the new file."""
    output,checkpoint=Path(output),Path(checkpoint)
    if (output.parent.name!="validation" or not output.name.startswith("step")
            or not output.name[4:].isdigit() or output.is_symlink()):
        raise ValueError("refusing to archive an unexpected evaluation path")
    parent=output.parent.resolve()
    archive=output.with_name(output.name+f".superseded_{time.time_ns()}")
    if output.resolve().parent!=parent or archive.resolve().parent!=parent:
        raise ValueError("evaluation archive escaped its validation directory")
    if (output/".lock").exists():
        return False  # Another evaluation still owns this output; retry later.
    old=read_json(output/"complete.json")
    digest=file_hash(checkpoint)
    if old["checkpoint_sha256"]==digest:
        return False  # The file changed again; let the next scan verify it.
    output.rename(archive)
    atomic_json(archive/"superseded.json",dict(checkpoint=str(checkpoint),
        evaluated_sha256=old["checkpoint_sha256"],observed_sha256=digest,
        reason="checkpoint file replaced after evaluation; old metrics retained, not accepted for the new file"))
    print(f"EVALUATION_SUPERSEDED checkpoint={checkpoint} archived={archive}",flush=True)
    return True


def invalidate_gate1(output):
    path=Path(output)/"gate1.json"
    if path.exists():
        path.rename(path.with_name(f"gate1.superseded_{time.time_ns()}.json"))


def reference_checkpoint(rate, root=ROOT):
    return Path(root) / ("outputs/nfcgs_order_context/hilbert/dyadic4_stages4_k5_ste/"
        f"lambda{rate_text(rate).replace('.', 'p')}/job_{REFERENCE_JOBS[rate_text(rate)]}/"
        "checkpoints/nfcgs_main/version_0/step000050000.ckpt")


def summary_row(run, step, rows):
    validate_rows(rows, 128)
    return dict(cell=run["cell"], rate_lambda=run["rate_lambda"], seed=run["seed_index"], step=step,
        KiB=statistics.fmean(r["actual_bytes"] for r in rows) / 1024,
        PSNR=statistics.fmean(r["psnr"] for r in rows), SSIM=statistics.fmean(r["ssim"] for r in rows),
        LPIPS=statistics.fmean(r["lpips"] for r in rows), MSE=statistics.fmean(r["mse"] for r in rows),
        J_h=statistics.fmean(objective(r, run["rate_lambda"]) for r in rows), scene_count=len(rows))


def atomic_csv(path, rows):
    columns = ("cell", "rate_lambda", "seed", "step", "KiB", "PSNR", "SSIM", "LPIPS", "MSE", "J_h", "scene_count")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader(); writer.writerows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, encoding="utf-8", newline="", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(stream.getvalue()); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def completed_rows(output, checkpoint=None):
    output = Path(output)
    done = read_json(output / "complete.json")
    path = output / "nfcgs_main/scene_metrics.json"
    if done["split"] != "validation" or done["scene_count"] != 128 or file_hash(path) != done["scene_metrics_sha256"]:
        raise ValueError("monitor requires a complete, unmodified holdout128 evaluation")
    if checkpoint and file_hash(checkpoint) != done["checkpoint_sha256"]:
        raise CheckpointIdentityChanged(f"checkpoint identity changed after evaluation: {checkpoint}; evaluation={output}")
    rows = read_json(path)
    validate_rows(rows, 128, rate_required=done["mode"] != "native")
    return rows


def evaluate(command, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as stream:
        stream.write(shlex.join(command) + "\n"); stream.flush()
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    return result.returncode == 0


def refresh_gate1(output, runs):
    changes = []
    for run in runs:
        root = Path(run["run_dir"]) / "validation"
        try:
            before = completed_rows(root / "step000040000", checkpoint_path(run["run_dir"], 40000))
            after = completed_rows(root / "step000060000", checkpoint_path(run["run_dir"], 60000))
        except (FileNotFoundError,CheckpointIdentityChanged):
            invalidate_gate1(output)
            return
        changes.append(dict(cell=run["cell"], **{"lambda": run["rate_lambda"]},
            seed=run["seed_index"], **paired_change(before, after, run["rate_lambda"])))
    atomic_json(output / "gate1.json", dict(**gate1_decision(changes), runs=changes,
        source="actual holdout128 scene metrics", test_used=False))


def baseline_commands(args):
    commands = []
    native_out = args.root / "gate0/native/validation"
    commands.append((native_out, [sys.executable, str(ROOT / "scripts/run_nfcgs_fresh_rd.py"), "native",
        "--checkpoint", str(args.native_checkpoint), "--run-dir", str(args.root / "baselines/native"),
        "--output", str(native_out), "--preparation", str(args.preparation), "--dataset-root", str(args.dataset_root)]))
    for rate in REFERENCE_JOBS:
        output = args.root / "baselines" / ("legacy_lambda" + rate.replace(".", "p"))
        commands.append((output, [sys.executable, str(ROOT / "scripts/run_nfcgs_fresh_rd.py"), "reference",
            "--rate-lambda", rate, "--checkpoint", str(reference_checkpoint(rate)), "--run-dir", str(output),
            "--output", str(output), "--preparation", str(args.preparation), "--dataset-root", str(args.dataset_root)]))
    return commands


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT / "outputs/nfcgs_fresh_rd")
    p.add_argument("--state", type=Path, default=ROOT / "logs/slurm/fresh_rd/state.json")
    p.add_argument("--phase", choices=("A", "final", "appendix"), default="A")
    p.add_argument("--budget", type=int, choices=(100000, 120000, 150000), default=100000)
    p.add_argument("--wave", type=int, choices=(1, 2), default=1)
    p.add_argument("--preparation", type=Path, default=runner.preparation_for(0))
    p.add_argument("--native-checkpoint", type=Path, default=ROOT / "checkpoints/pretrained/globalsplat-re10k-32k.ckpt")
    p.add_argument("--dataset-root", type=Path, default=Path(os.environ.get("DATASET_ROOT", "/data3/local_datasets/re10k")))
    p.add_argument("--interval", type=float, default=600)
    p.add_argument("--once", action="store_true")
    p.add_argument("--baselines", action="store_true", help="also evaluate optional native and legacy references")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    if a.interval <= 0:
        p.error("interval must be positive")
    convergence = a.root / "convergence"
    steps = list(range(10000, 60001, 10000)) if a.phase == "A" else [a.budget] if a.phase == "final" else list(range(70000, a.budget + 1, 10000))
    if a.dry_run:
        print(f"HOLDOUT_ONLY phase={a.phase} wave={a.wave} steps={steps} NO_TEST_INPUTS")
        for output, command in baseline_commands(a) if a.baselines else []:
            print(shlex.join(command))
        return
    with directory_lock(convergence, f".monitor_{a.wave}_{a.phase}.lock"):
        observations = {}
        while True:
            state = read_json(a.state)
            excluded=set(state.get("wave2_execution",{}).get("optional_cancelled_keys",[])) if a.wave==2 else set()
            optional=set(state.get("wave2_execution",{}).get("optional_keys",[]))
            if excluded and (len(excluded)!=4 or excluded!=optional):
                raise ValueError("only the four registered optional Wave 2 runs may be excluded")
            runs = [r for key,r in state["runs"].items() if r["wave"] == a.wave and key not in excluded]
            expected = 15 if a.wave == 1 else len(wave_runs(2, state.get("msh_lambdas", [".0064", ".0128"]), state.get("omit_seeds", False)))-len(excluded)
            if len(runs) != expected:
                raise ValueError("incomplete submitted run inventory")
            all_done = True
            for output, command in baseline_commands(a) if a.baselines else []:
                if not (output / "complete.json").is_file() and not evaluate(command, output / "monitor.log"):
                    all_done = False
            table = []
            for run in runs:
                for step in steps:
                    path = checkpoint_path(run["run_dir"], step)
                    output = Path(run["run_dir"]) / "validation" / f"step{step:09d}"
                    if not path.is_file():
                        all_done = False
                        continue
                    stat = path.stat()
                    stamp = (stat.st_size, stat.st_mtime_ns)
                    if not (output / "complete.json").exists():
                        if observations.get(str(path)) != stamp:
                            observations[str(path)] = stamp
                            all_done = False
                            continue
                        command = [sys.executable, str(ROOT / "scripts/run_nfcgs_fresh_rd.py"), "validation",
                            "--cell", run["cell"], "--rate-lambda", run["rate_lambda"], "--seed-index", str(run["seed_index"]),
                            "--run-dir", run["run_dir"], "--preparation", run["preparation"],
                            "--step", str(step), "--budget", str(a.budget), "--dataset-root", str(a.dataset_root)]
                        if not evaluate(command, output / "monitor.log"):
                            all_done = False  # partially written checkpoint/evaluation: retry next scan
                            continue
                    try:
                        rows = completed_rows(output, path)
                    except CheckpointIdentityChanged:
                        # Never reuse a result for different checkpoint bytes.
                        # Retain that result separately and re-evaluate only
                        # after the current file passes two stable scans.
                        supersede_evaluation(output,path)
                        observations.pop(str(path),None)
                        invalidate_gate1(convergence)
                        all_done=False
                        continue
                    table.append(summary_row(run, step, rows))
            # One writer for each phase; merge only verified rows with older phases.
            with directory_lock(convergence, ".csv.lock"):
                shard = convergence / f"curve_wave{a.wave}_{a.phase}.json"
                atomic_json(shard, table)
                merged = {}
                for source in sorted(convergence.glob("curve_wave*.json")):
                    for row in read_json(source):
                        merged[(row["cell"], row["rate_lambda"], row["seed"], row["step"])] = row
                atomic_csv(convergence / "holdout_curve.csv", [merged[k] for k in sorted(merged)])
            if a.phase == "A" and a.wave == 1:
                refresh_gate1(convergence, runs)
            if all_done:
                atomic_json(convergence / f"wave{a.wave}_{a.phase}_complete.json", dict(
                    wave=a.wave, phase=a.phase, budget=a.budget, run_count=len(runs), split="validation", test_used=False))
                print(f"MONITOR_COMPLETE phase={a.phase} wave={a.wave}", flush=True)
                return
            print(f"MONITOR_PENDING phase={a.phase} evaluated={len(table)}", flush=True)
            if a.once:
                return
            time.sleep(a.interval)


if __name__ == "__main__":
    main()
