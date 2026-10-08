#!/usr/bin/env python3
"""Apply an exact reviewed inventory; default is a read-only preview."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path, PurePosixPath
import subprocess

# The generated self-contained shell script embeds these audit functions.
from audit_nfcgs_checkpoints import live_jobs


def target_path(root, entry):
    relative = PurePosixPath(entry["relative_path"])
    if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "outputs" or relative.suffix != ".ckpt":
        raise ValueError(f"invalid deletion path: {relative}")
    path = root.joinpath(*relative.parts)
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError(f"symlink or path outside repository: {path}")
    return path


def validate_snapshot(root, plan):
    if not (root / "scripts/run_nfcgs.py").is_file():
        raise ValueError(f"repository marker missing: {root}")
    keep = {entry["relative_path"] for entry in plan["keep"]}
    candidates, missing = [], []
    seen = set()
    for entry in plan["keep"]:
        path = root / entry["relative_path"]
        if not path.is_file() or path.stat().st_size < 1:
            raise ValueError(f"required checkpoint missing/empty: {path}")
    for entry in plan["delete"]:
        relative = entry["relative_path"]
        if relative in keep or relative in seen:
            raise ValueError(f"keep/delete overlap or duplicate: {relative}")
        seen.add(relative)
        path = target_path(root, entry)
        if any(path.resolve().is_relative_to(Path(prefix).resolve())
               for prefix in plan.get("protected_output_prefixes", [])):
            raise ValueError(f"protected output tree: {path}")
        if not path.exists():
            missing.append(relative)
            continue
        stat = path.stat()
        expected_time = datetime.fromisoformat(entry["mtime_utc"]).timestamp()
        if not path.is_file() or stat.st_size != entry["bytes"] or abs(stat.st_mtime - expected_time) > .00001:
            raise ValueError(f"checkpoint changed since audit; no deletion: {path}")
        if stat.st_ino != entry["inode"] or stat.st_nlink != 1:
            raise ValueError(f"file identity/link count changed since audit: {path}")
        candidates.append((path, entry))
    return candidates, missing


def validate_live_jobs(plan, jobs, candidates):
    known_ids = {job["job_id"] for job in plan["audit_jobs"]}
    new_ids = {job["job_id"] for job in jobs} - known_ids
    if new_ids:
        raise ValueError(f"new jobs since audit; refresh inventory first: {sorted(new_ids)}")
    root = Path(plan["repo"]).resolve()
    pending_sources = plan.get("confirmed_pending_sources", {})
    for job in jobs:
        workdir = job.get("details", {}).get("WorkDir", "")
        in_repo = bool(workdir) and Path(workdir).resolve().is_relative_to(root)
        if in_repo and job.get("requires_manual_check") and job["job_id"] not in pending_sources:
            raise ValueError(f"unresolved input for repository job {job['job_id']}; confirm its checkpoint before deleting")
        references = {Path(path).resolve() for path in job.get("checkpoint_references", [])}
        references.update(Path(path).resolve() for path in pending_sources.get(job["job_id"], []))
        outputs = [Path(path).resolve() for path in job.get("output_paths", [])]
        for path, _ in candidates:
            resolved = path.resolve()
            if resolved in references or any(resolved.is_relative_to(output) for output in outputs):
                raise ValueError(f"live job {job['job_id']} protects deletion target: {path}")


def main(argv=None, embedded_plan=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, help="reviewed JSON inventory; embedded in generated .sh")
    parser.add_argument("--apply", action="store_true", help="delete exact reviewed .ckpt files using rm")
    args = parser.parse_args(argv)
    plan = json.loads(args.plan.read_text(encoding="utf-8")) if args.plan else embedded_plan
    if plan is None:
        parser.error("--plan is required")
    root = Path(plan["repo"]).resolve(strict=True)
    candidates, missing = validate_snapshot(root, plan)
    total = sum(entry["bytes"] for _, entry in candidates)
    print(f"REPO={root}")
    print(f"KEEP={len(plan['keep'])} DELETE={len(candidates)} ALREADY_MISSING={len(missing)} BYTES={total} GiB={total / 2**30:.3f}")
    for path, _ in candidates:
        print(f"DELETE {path.relative_to(root).as_posix()}")
    if not args.apply:
        print("PREVIEW ONLY: no files deleted. Use --apply to execute this exact list.")
        return
    jobs, errors, complete = live_jobs(plan["user"])
    if errors or not complete:
        raise ValueError(f"cannot refresh live jobs; no files deleted: {errors}")
    validate_live_jobs(plan, jobs, candidates)
    # Validate every file before any rm, then again immediately before each rm.
    validate_snapshot(root, plan)
    for path, entry in candidates:
        stat = path.stat()
        expected_time = datetime.fromisoformat(entry["mtime_utc"]).timestamp()
        if (stat.st_size != entry["bytes"] or abs(stat.st_mtime - expected_time) > .00001
                or stat.st_ino != entry["inode"] or stat.st_nlink != 1
                or path.is_symlink() or not path.resolve().is_relative_to(root)):
            raise ValueError(f"file changed during cleanup; stopped at {path}")
        subprocess.run(["rm", "--", str(path)], check=True)
    print(f"DELETED={len(candidates)} LOGICAL_GiB={total / 2**30:.3f}")


if __name__ == "__main__":
    try:
        main(embedded_plan=globals().get("EMBEDDED_PLAN"))
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        raise SystemExit(f"STOP: {error}")
