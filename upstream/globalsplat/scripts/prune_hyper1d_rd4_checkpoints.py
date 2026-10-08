#!/usr/bin/env python3
"""Keep verified 50k or recent resume checkpoints in the RD4 sweep only."""

import argparse
from dataclasses import dataclass
import getpass
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import zipfile


REPO = Path(__file__).resolve().parents[1]
STEP_NAME = re.compile(r"step([0-9]+)(?:-v[0-9]+)?\.ckpt\Z")
JOB_NAME = re.compile(r"job_([0-9]+(?:_[0-9]+)?)\Z")


@dataclass(frozen=True)
class File:
    path: Path
    size: int
    mtime_ns: int
    inode: int

    @classmethod
    def capture(cls, path, root):
        path.resolve().relative_to(root.resolve())
        for item in (path, *path.parents):
            if item.is_symlink():
                raise ValueError(f"symlink in checkpoint path: {item}")
            if item == root:
                break
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"not a regular checkpoint file: {path}")
        return cls(path, info.st_size, info.st_mtime_ns, info.st_ino)

    def unchanged(self, root):
        return self == File.capture(self.path, root)


def validate_checkpoint(path, expected_step=None, resume=False):
    # CRC checking also catches a truncated tensor payload after a quota error.
    import torch

    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            bad = archive.testzip()
            if bad is not None:
                raise ValueError(f"invalid checkpoint CRC: {bad}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    else:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", {})
    if not isinstance(state, dict) or not any(
        key.startswith("model.feature_codec.") for key in state
    ):
        raise ValueError("missing integrated feature-codec state_dict")
    step = checkpoint.get("global_step")
    if not isinstance(step, int) or step < 1 or step > 50000:
        raise ValueError(f"unexpected global_step={step}")
    if expected_step is not None and step != expected_step:
        raise ValueError(f"global_step={step}, filename step={expected_step}")
    if resume and not checkpoint.get("optimizer_states"):
        raise ValueError("missing optimizer_states for resume")
    return step


def active_jobs(root=None):
    result = subprocess.run(
        ["squeue", "-h", "-r", "-u", os.environ.get("USER") or getpass.getuser(),
         "-o", "%i"],
        check=True, capture_output=True, text=True,
    )
    active = set(result.stdout.split())
    # Recovery jobs continue writing in the ORIGINAL job's output directory.
    # Map their new Slurm IDs back before allowing any cleanup there.
    root = root or REPO / "outputs/hyper1d_rd4_50k"
    for path in (root / "resumes").glob("*/plan.json"):
        plan = json.loads(path.read_text(encoding="utf-8"))
        for group, job in plan.get("jobs", {}).items():
            for index, task in enumerate(plan["groups"][group]):
                if f"{job}_{index}" in active:
                    active.add(task["original_job"])
    return active


def make_plan(directory, root, keep_recent, validator=validate_checkpoint):
    files = [
        File.capture(path, root)
        for path in directory.iterdir()
        if STEP_NAME.fullmatch(path.name) or path.name == "last.ckpt"
    ]
    named = sorted(
        (file for file in files if STEP_NAME.fullmatch(file.path.name)),
        key=lambda file: (int(STEP_NAME.fullmatch(file.path.name)[1]), file.mtime_ns),
        reverse=True,
    )
    if any(int(STEP_NAME.fullmatch(file.path.name)[1]) > 50000 for file in named):
        raise ValueError("found a checkpoint beyond this sweep's 50k limit")
    verified, invalid = [], []

    def check(file, resume):
        expected = STEP_NAME.fullmatch(file.path.name)
        try:
            step = validator(file.path, int(expected[1]) if expected else None, resume)
            if not file.unchanged(root):
                raise ValueError("checkpoint changed during validation")
            return step
        except Exception as error:
            invalid.append(f"{file.path.name}: {type(error).__name__}: {str(error)[:160]}")
            return None

    # A verified final checkpoint is sufficient for evaluation and later resume.
    for file in named:
        if int(STEP_NAME.fullmatch(file.path.name)[1]) == 50000:
            if check(file, resume=True) == 50000:
                keep = [file]
                return keep, [item for item in files if item not in keep], invalid

    for file in named:
        step = check(file, resume=True)
        if step is not None and step not in {item[0] for item in verified}:
            verified.append((step, file))
        if len(verified) >= keep_recent:
            break
    # Retain last.ckpt only if it contains progress not present in named files,
    # or is the only remaining valid resume checkpoint.
    last = next((file for file in files if file.path.name == "last.ckpt"), None)
    if last is not None:
        step = check(last, resume=True)
        if step is not None and step not in {item[0] for item in verified}:
            verified.append((step, last))
    if not verified:
        raise ValueError("no verified resume checkpoint; preserving every file")
    keep = [file for _, file in sorted(verified, key=lambda item: item[0], reverse=True)[:keep_recent]]
    return keep, [file for file in files if file not in keep], invalid


def apply_plan(keep, remove, root):
    # Check every retained and removed path before deleting any file in this run.
    for file in (*keep, *remove):
        if not file.unchanged(root):
            raise ValueError(f"checkpoint changed after planning: {file.path}")
    for file in remove:
        file.path.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="delete planned surplus checkpoints")
    parser.add_argument("--keep-recent", type=int, default=2, help="valid checkpoints to keep for incomplete runs")
    args = parser.parse_args()
    if args.keep_recent < 1:
        parser.error("--keep-recent must be positive")
    root = REPO / "outputs/hyper1d_rd4_50k"
    root.resolve().relative_to((REPO / "outputs").resolve())
    if not root.is_dir():
        parser.error(f"missing RD4 output directory: {root}")
    try:
        active = active_jobs()
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        if args.apply:
            parser.error(f"cannot check running Slurm jobs; refusing deletion: {error}")
        print("PLAN ONLY: Slurm status unavailable; active jobs cannot be identified.")
        active = set()

    print(f"MODE={'APPLY' if args.apply else 'PLAN'} ROOT={root}", flush=True)
    total_count = total_bytes = 0
    skipped = 0
    for directory in sorted(root.glob("**/checkpoints/hyper1d_12h/version_*")):
        if not directory.is_dir():
            continue
        run = directory.parents[2]
        job = JOB_NAME.fullmatch(run.name)
        if job is None or job[1] in active:
            print(f"SKIP active/unknown job: {run.relative_to(root)}", flush=True)
            skipped += 1
            continue
        try:
            keep, remove, invalid = make_plan(directory, root, args.keep_recent)
            print(f"RUN {run.relative_to(root)}", flush=True)
            for file in keep:
                print(f"  KEEP {file.path.name} ({file.size / 2**20:.1f} MiB)", flush=True)
            for message in invalid:
                print(f"  INVALID {message}", flush=True)
            size = sum(file.size for file in remove)
            print(f"  REMOVE {len(remove)} files / {size / 2**30:.2f} GiB", flush=True)
            if args.apply:
                if job[1] in active_jobs():
                    print("  SKIP: job became active", flush=True)
                    skipped += 1
                    continue
                apply_plan(keep, remove, root)
            total_count += len(remove)
            total_bytes += size
        except (OSError, ValueError) as error:
            print(f"SKIP {run.relative_to(root)}: {error}", flush=True)
            skipped += 1
    action = "DELETED" if args.apply else "WOULD_DELETE"
    print(f"{action}: {total_count} files / {total_bytes / 2**30:.2f} GiB; skipped runs={skipped}")
    if not args.apply:
        print("To delete the listed surplus files: python scripts/prune_hyper1d_rd4_checkpoints.py --apply")


if __name__ == "__main__":
    main()
