#!/usr/bin/env python3
"""Read-only checkpoint and live Slurm inventory. Never deletes or loads weights."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import getpass
import json
import os
from pathlib import Path
import re
import subprocess


# Rendering artifacts cannot contain training checkpoints in the current code.
SKIP_DIRS = {".git", "__pycache__", "images", "videos", "frames", "bitstreams"}
EXTENSIONS = {".ckpt", ".pth", ".pt"}


def run_query(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired) as error:
        return "", str(error)
    if result.returncode:
        return "", result.stderr.strip() or f"exit {result.returncode}"
    return result.stdout, None


def fields_from_scontrol(text):
    keys = ("JobId", "JobName", "JobState", "WorkDir", "Command", "StdOut", "StdErr", "Dependency", "ArrayJobId")
    result = {}
    for key in keys:
        match = re.search(rf"(?:^|\s){key}=(.*?)(?=\s+[A-Za-z][A-Za-z0-9_]*=|$)", text, re.DOTALL)
        if match:
            result[key] = match.group(1).strip()
    return result


def checkpoint_references(text, workdir):
    # Only literal paths: shell templates are not evidence of a resolved input.
    refs = set()
    for value in re.findall(r"[^\s\"'=,()<>]+\.(?:ckpt|pth|pt)\b", text):
        if any(char in value for char in "$}{"):
            continue
        path = Path(value)
        if not path.is_absolute():
            path = Path(workdir) / path
        refs.add(str(path.resolve()))
    return sorted(refs)


def log_paths(fields, job_id):
    for key in ("StdOut", "StdErr"):
        value = fields.get(key)
        if not value or value in ("(null)", "/dev/null"):
            continue
        if "%" in value:
            # scontrol normally resolves these; only expand unambiguous IDs.
            base, _, task = job_id.partition("_")
            value = value.replace("%j", job_id).replace("%A", base)
            if task:
                value = value.replace("%a", task)
            if "%" in value:
                continue
        yield Path(value)


def live_jobs(user):
    queue, error = run_query(["squeue", "-r", "-u", user, "-h", "-o", "%i|%j|%T|%R"])
    if error:
        return [], [f"squeue: {error}"], False
    jobs, errors = [], []
    for row in queue.splitlines():
        columns = row.split("|", 3)
        if len(columns) != 4:
            errors.append(f"unparsed squeue row: {row}")
            continue
        job_id, name, state, reason = columns
        if not re.fullmatch(r"[0-9]+(?:_[0-9]+)?", job_id):
            errors.append(f"unresolved job ID: {job_id}")
            continue
        info, info_error = run_query(["scontrol", "show", "job", "-o", job_id])
        fields = fields_from_scontrol(info)
        if info_error:
            errors.append(f"job {job_id}: {info_error}")
        workdir = fields.get("WorkDir", "/")
        refs = set(checkpoint_references(fields.get("Command", ""), workdir))
        output_paths, log_errors = set(), []
        for path in log_paths(fields, job_id):
            if not path.is_absolute():
                path = Path(workdir) / path
            try:
                # Initial log lines print SOURCE_CHECKPOINT, CHECKPOINT and OUTPUT.
                with path.open("r", encoding="utf-8", errors="replace") as handle:
                    head = handle.read(128 * 1024)
                refs.update(checkpoint_references(head, workdir))
                for value in re.findall(r"(?:TRAIN_OUTPUT|EVAL_OUTPUT|OUTPUT)=([^\s]+)", head):
                    if "$" not in value:
                        output_paths.add(str((Path(workdir) / value).resolve()))
            except OSError as log_error:
                log_errors.append(f"{path}: {log_error}")
        # Pending jobs have no logs yet. Preserve their script references as hints,
        # but exported SOURCE_CHECKPOINT/TOKEN_ORDER values are not in this text.
        batch, batch_error = run_query(["scontrol", "write", "batch_script", job_id, "-"])
        if batch_error:
            errors.append(f"batch script {job_id}: {batch_error}")
        literal_refs = checkpoint_references(batch, workdir)
        jobs.append({"job_id": job_id, "name": name, "state": state, "reason": reason,
                     "details": fields, "checkpoint_references": sorted(refs),
                     "output_paths": sorted(output_paths), "script_literal_references": literal_refs,
                     "log_errors": log_errors,
                     "requires_manual_check": not refs or state == "PENDING"})
    return jobs, errors, not errors


def checkpoint_files(root, scan_paths):
    rows, errors, visited = [], [], set()
    for scan_path in scan_paths:
        path = (root / scan_path).resolve()
        if not path.is_relative_to(root):
            errors.append(f"scan path outside repository skipped: {path}")
            continue
        if not path.exists():
            continue
        def on_error(error):
            errors.append(str(error))
        for folder, dirs, names in os.walk(path, followlinks=False, onerror=on_error):
            dirs[:] = sorted(name for name in dirs if name not in SKIP_DIRS and not (Path(folder) / name).is_symlink())
            for name in sorted(names):
                file = Path(folder) / name
                if file.suffix.lower() not in EXTENSIONS or str(file) in visited:
                    continue
                visited.add(str(file))
                try:
                    stat = file.lstat()
                    rows.append({"path": str(file), "relative_path": file.relative_to(root).as_posix(),
                                 "resolved_path": str(file.resolve()), "bytes": stat.st_size,
                                 "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                                 "is_symlink": file.is_symlink(), "hardlinks": stat.st_nlink,
                                 "device": stat.st_dev, "inode": stat.st_ino})
                except OSError as error:
                    errors.append(f"{file}: {error}")
    return rows, errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--scan", action="append", help="repository-relative scan path; repeatable")
    parser.add_argument("--user", default=getpass.getuser())
    parser.add_argument("--output", type=Path, default=Path("checkpoint_audit.json"))
    args = parser.parse_args(argv)
    root = args.repo.resolve(strict=True)
    scans = args.scan or ["outputs", "checkpoints", "pretrained", "pretrained_models", "weights"]
    jobs, job_errors, queue_complete = live_jobs(args.user)
    files, scan_errors = checkpoint_files(root, scans)
    references = {path for job in jobs for path in job["checkpoint_references"]}
    groups = defaultdict(lambda: {"files": 0, "logical_bytes": 0})
    for row in files:
        row["referenced_by_live_job"] = row["resolved_path"] in references
        row["inside_reported_live_output"] = any(
            Path(row["path"]).is_relative_to(Path(output)) for job in jobs for output in job["output_paths"])
        parts = Path(row["relative_path"]).parts
        group = "/".join(parts[:2])
        groups[group]["files"] += 1
        groups[group]["logical_bytes"] += row["bytes"]
    stamp = datetime.now(timezone.utc).isoformat()
    report = {"created_at_utc": stamp, "repo": str(root), "user": args.user,
              "scan_paths": scans, "slurm_query_complete": queue_complete,
              "live_jobs": jobs, "checkpoints": files, "groups": dict(sorted(groups.items())),
              "errors": job_errors + scan_errors,
              "notes": ["READ ONLY: no files deleted; no checkpoint weights loaded.",
                        "Inventory is a snapshot: rescan live jobs before deletion.",
                        "Pending job exports and future eval inputs require manual confirmation.",
                        "File sizes are logical bytes; hardlinks/symlinks do not imply equivalent reclaimed space.",
                        "Only listed scan paths were scanned; other projects are outside this audit."]}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Audit saved: {args.output.resolve()}")
    print(f"Live jobs: {len(jobs)}; checkpoint-like files: {len(files)}; Slurm query complete: {queue_complete}")
    for name, group in sorted(groups.items(), key=lambda item: item[1]["logical_bytes"], reverse=True):
        print(f"{group['logical_bytes'] / 2**30:9.2f} GiB  {group['files']:5d} files  {name}")
    if report["errors"]:
        print(f"Read/query errors: {len(report['errors'])}; inspect JSON before planning deletion.")


if __name__ == "__main__":
    main()
