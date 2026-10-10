#!/usr/bin/env python3
"""Prune redundant training checkpoints in this repository's outputs only.

Default is a dry run. --apply deletes only the listed numbered/last checkpoints.
Active Slurm runs, initialization/pretrained artifacts, evaluation references,
and recently modified files are protected. No GPU is used.
"""

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import getpass
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time
import zipfile

from prune_hyper1d_rd4_checkpoints import File, STEP_NAME


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "outputs"
JOB = re.compile(r"job_([0-9]+(?:_[0-9]+)?)\Z")
SKIP_DIRS = {".git", ".checkpoint_prune", "__pycache__", "pretrained", "initialization",
             "experiments", "archive", "datasets", "evaluation", "validation", "logs",
             "hydra", "wandb", "images", "bitstreams"}
CKPT_TOKEN = re.compile(r"[^\s\"'<>;()=]+\.ckpt")


@dataclass
class Context:
    jobs: set = field(default_factory=set)
    mapped_jobs: set = field(default_factory=set)
    active_dirs: set = field(default_factory=set)
    pinned: dict = field(default_factory=dict)
    pinned_names: dict = field(default_factory=dict)
    targets: dict = field(default_factory=dict)
    uncertain: list = field(default_factory=list)


def under(path, root):
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def run_directory(directory):
    for parent in (directory, *directory.parents):
        if parent.name == "checkpoints":
            return parent.parent
    return directory.parent if re.fullmatch(r"version_\d+", directory.name) else directory


def query_jobs():
    result = subprocess.run(["squeue", "-h", "-r", "-u", os.environ.get("USER") or getpass.getuser(),
                             "-o", "%i"], check=True, capture_output=True, text=True)
    return set(result.stdout.split())


def log_excerpt(path):
    if not path.is_file():
        return ""
    with path.open("rb") as handle:
        head = handle.read(128 * 1024)
        if path.stat().st_size <= len(head):
            return head.decode("utf-8", errors="replace")
        handle.seek(max(len(head), path.stat().st_size - 16 * 1024))
        return (head + b"\n" + handle.read()).decode("utf-8", errors="replace")


def parse_log(text, root, context, active=False, label="log"):
    observed = set()
    for line in text.splitlines():
        try:
            tokens = shlex.split(line)
        except ValueError:
            continue
        values = dict(token.split("=", 1) for token in tokens if "=" in token)
        testing = values.get("mode") == "test" or "test.actual_bitstream=true" in tokens
        step = values.get("trainer.max_steps", "")
        if "output_dir" in values:
            output = Path(values["output_dir"])
            if output.is_absolute():
                observed.add(run_directory(output).resolve())
            if under(output, root):
                run = run_directory(output)
                if active:
                    context.active_dirs.add(run.resolve())
                if not testing and step.isdigit() and int(step) > 0:
                    context.targets[run.resolve()] = max(context.targets.get(run.resolve(), 0), int(step))
        if active or testing:
            sources = [values[key] for key in ("checkpointing.load", "CHECKPOINT", "SOURCE_CHECKPOINT") if key in values]
            for index, token in enumerate(tokens[:-1]):
                if token in ("--checkpoint", "--vanilla-checkpoint"):
                    sources.append(tokens[index + 1])
            for source in sources:
                path = Path(source)
                if active and path.suffix == ".ckpt" and path.is_absolute():
                    observed.add(run_directory(path.parent).resolve())
                if path.suffix == ".ckpt" and under(path, root):
                    context.pinned[path.resolve()] = f"{label}: checkpoint input"
                    if active:
                        context.active_dirs.add(run_directory(path.parent).resolve())
    return observed


def resume_protection(context, root):
    for path in (root / "hyper1d_rd4_50k/resumes").glob("*/plan.json"):
        plan = json.loads(path.read_text(encoding="utf-8"))
        for group, job in plan.get("jobs", {}).items():
            for index, task in enumerate(plan["groups"][group]):
                if f"{job}_{index}" in context.jobs:
                    context.mapped_jobs.add(f"{job}_{index}")
                    context.active_dirs.add(Path(task["output"]).resolve())
                    source = task.get("source", {}).get("checkpoint")
                    if source:
                        context.pinned[Path(source).resolve()] = "active recovery input"


def script_references(context, repo, root):
    for path in (repo / "scripts").rglob("*"):
        if not path.is_file() or path.suffix not in (".py", ".sh", ".slurm"):
            continue
        if path.name.startswith("prune_"):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for name in ("REPO_DIR", "REPO", "ROOT_DIR"):
            text = text.replace("${" + name + "}", repo.as_posix())
        for token in CKPT_TOKEN.findall(text):
            candidate = Path(token)
            if candidate.is_absolute() and "$" not in token and under(candidate, root):
                context.pinned[candidate.resolve()] = f"script: {path.name}"
            else:
                # Dynamic evaluation templates cannot always be expanded safely.
                # Keep their explicitly named milestones/last aliases wherever found.
                name = candidate.name
                if (STEP_NAME.fullmatch(name) or name == "last.ckpt") and any(
                    key in path.name for key in ("eval", "resolve", "submit", "train", "run_", "common")
                ):
                    context.pinned_names[name] = f"script template: {path.name}"


def collect_context(repo, root, jobs, inspect_slurm=True):
    context = Context(jobs=set(jobs))
    script_references(context, repo, root)
    resume_protection(context, root)
    for path in root.glob("**/manifests/**/*.txt"):
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if text.endswith(".ckpt"):
            candidate = Path(text)
            if candidate.is_absolute() and under(candidate, root):
                context.pinned[candidate.resolve()] = f"manifest: {path.name}"
    log_dirs = {repo / "logs/slurm", repo / "slurm", repo.parents[1] / "slurm"}
    group_dirs = [directory for directory, _ in checkpoint_groups(root)]
    for directory in log_dirs:
        for path in directory.glob("*.out"):
            active = any(path.name.endswith(f"-{job}.out") for job in jobs)
            found = parse_log(log_excerpt(path), root, context, active, path.name)
            if active:
                context.mapped_jobs.update(job for job in jobs if found and path.name.endswith(f"-{job}.out"))
    if inspect_slurm:
        for job in sorted(jobs):
            result = subprocess.run(["scontrol", "-o", "show", "job", job], capture_output=True, text=True)
            if result.returncode:
                if job not in query_jobs():
                    continue
                raise RuntimeError(f"cannot inspect active job {job}: {result.stderr.strip()}")
            values = dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=(.*?)(?=\s[A-Za-z][A-Za-z0-9_]*=|$)", result.stdout.strip()))
            workdir = Path(values.get("WorkDir", "/"))
            stdout = values.get("StdOut", "")
            array, _, index = job.partition("_")
            stdout = stdout.replace("%A", array).replace("%a", index).replace("%j", values.get("JobId", job))
            found = parse_log(log_excerpt(Path(stdout)), root, context, True, f"Slurm {job}") if stdout else set()
            found.update(parse_log(values.get("Command", ""), root, context, True, f"Slurm command {job}"))
            # A directory with the job ID is protected even before stdout exists.
            identifiable = any(JOB.fullmatch(part) and (JOB.fullmatch(part)[1] == job or
                               JOB.fullmatch(part)[1] == array) for directory in group_dirs
                               for part in directory.parts)
            if not found and job not in context.mapped_jobs and not identifiable and (workdir == repo or under(workdir, root)):
                context.uncertain.append(job)
    return context


def checkpoint_groups(root):
    groups = {}
    for directory, subdirs, names in os.walk(root, followlinks=False):
        parent = Path(directory)
        subdirs[:] = [name for name in subdirs if name not in SKIP_DIRS and not (parent / name).is_symlink()]
        files = [parent / name for name in names if STEP_NAME.fullmatch(name) or name == "last.ckpt"]
        if files:
            groups[parent] = files
    return sorted(groups.items())


def is_active(directory, context):
    for part in directory.parts:
        match = JOB.fullmatch(part)
        if match and any(job == match[1] or job.startswith(match[1] + "_") for job in context.jobs):
            return True
    return any(under(directory, parent) for parent in context.active_dirs)


def final_step(directory, context):
    run = run_directory(directory).resolve()
    target = context.targets.get(run)
    paths = (run / "hydra/.hydra/config.yaml", run / "hydra/rank_0/.hydra/config.yaml",
             run / ".hydra/config.yaml")
    for path in paths:
        if path.is_file():
            import yaml
            values = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            step = values.get("trainer", {}).get("max_steps")
            if type(step) is int and step > 0:
                target = max(target or 0, step)
    if target is None and any(part.startswith("hyper1d") for part in directory.parts):
        target = 50000
    return target


def inspect_checkpoint(file, root):
    import torch
    if zipfile.is_zipfile(file.path):
        with zipfile.ZipFile(file.path) as archive:
            bad = archive.testzip()
            if bad:
                raise ValueError(f"corrupt tensor payload: {bad}")
        checkpoint = torch.load(file.path, map_location="cpu", weights_only=False, mmap=True)
    else:
        checkpoint = torch.load(file.path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("not a training checkpoint dictionary")
    state = checkpoint.get("state_dict", {})
    if not isinstance(state, dict) or not any(torch.is_tensor(value) for value in state.values()):
        raise ValueError("missing model state_dict tensors")
    step = checkpoint.get("global_step")
    expected = STEP_NAME.fullmatch(file.path.name)
    if type(step) is not int or step < 1 or (expected and step != int(expected[1])):
        raise ValueError(f"invalid global_step={step} or filename mismatch")
    if not file.unchanged(root):
        raise ValueError("checkpoint changed during inspection")
    return step, bool(checkpoint.get("optimizer_states") and checkpoint.get("lr_schedulers"))


def plan_group(directory, paths, root, context, keep_recent=2, min_age=300):
    files = [File.capture(path, root) for path in paths]
    target = final_step(directory, context)
    named = sorted((file for file in files if STEP_NAME.fullmatch(file.path.name)),
                   key=lambda file: (int(STEP_NAME.fullmatch(file.path.name)[1]), file.mtime_ns), reverse=True)
    cache, invalid, keep = {}, [], {}

    def inspect(file):
        if file not in cache:
            try:
                cache[file] = inspect_checkpoint(file, root)
            except Exception as error:
                cache[file] = None
                invalid.append(f"{file.path.name}: {type(error).__name__}: {str(error)[:160]}")
        return cache[file]

    latest = []
    for file in named:
        info = inspect(file)
        if info and info[1] and info[0] not in {item[0] for item in latest}:
            latest.append((info[0], file))
            if (target and info[0] >= target) or len(latest) >= keep_recent:
                break
        elif info and not info[1]:
            keep[file] = "weights for evaluation; no optimizer/scheduler"
    for file in files:
        if file.path.name == "last.ckpt":
            info = inspect(file)
            if info and info[1] and info[0] not in {item[0] for item in latest}:
                latest.append((info[0], file))
    if not latest:
        raise ValueError("no verified full resume checkpoint; preserve this directory")
    latest.sort(key=lambda item: (item[0], item[1].path.name != "last.ckpt"), reverse=True)
    completed = target is not None and latest[0][0] >= target
    for step, file in latest[:1 if completed else keep_recent]:
        keep[file] = f"final step={step}" if completed else f"resume step={step}"
    for file in files:
        reason = context.pinned.get(file.path.resolve()) or context.pinned_names.get(file.path.name)
        if reason:
            if not inspect(file):
                keep[file] = reason + " (INVALID: repair required)"
            else:
                keep[file] = reason
        if time.time_ns() - file.mtime_ns < min_age * 1_000_000_000:
            keep.setdefault(file, "recently modified")
    return files, keep, [file for file in files if file not in keep], invalid, target


@contextmanager
def run_lock(run):
    import fcntl
    path = run / ".resume.lock"
    if path.is_symlink():
        raise ValueError("symlinked run lock")
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("recovery worker holds the run lock") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def apply_group(directory, files, remove, root):
    present = {path for path in directory.iterdir() if STEP_NAME.fullmatch(path.name) or path.name == "last.ckpt"}
    if present != {file.path for file in files}:
        raise ValueError("checkpoint directory changed after planning")
    for file in files:
        if not file.unchanged(root):
            raise ValueError(f"checkpoint changed after planning: {file.path}")
    for file in remove:
        file.path.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--keep-recent", type=int, default=2)
    parser.add_argument("--min-age-minutes", type=float, default=5)
    args = parser.parse_args(argv)
    if args.keep_recent < 1 or not (0 <= args.min_age_minutes < float("inf")):
        parser.error("keep-recent must be positive; min-age-minutes must be finite and nonnegative")
    root = ROOT
    if not root.is_dir() or root.is_symlink() or root.resolve() != (REPO / "outputs").resolve():
        parser.error(f"missing or unsafe repository outputs: {root}")
    try:
        jobs = query_jobs()
        context = collect_context(REPO, root, jobs)
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.CalledProcessError) as error:
        if args.apply:
            parser.error(f"cannot establish active-job protection; no deletion: {error}")
        print(f"PLAN ONLY: Slurm unavailable ({error}); --apply requires Slurm protection.", flush=True)
        context = collect_context(REPO, root, set(), inspect_slurm=False)
    if context.uncertain:
        parser.error(f"cannot map active project jobs to runs; no deletion: {context.uncertain}")
    report = {"mode": "APPLY" if args.apply else "PLAN", "root": str(root),
              "active_jobs": sorted(context.jobs), "runs": [], "removed_files": 0, "removed_bytes": 0}
    print(f"MODE={report['mode']} ROOT={root}; active Slurm jobs={len(context.jobs)}", flush=True)
    for directory, paths in checkpoint_groups(root):
        record = {"directory": str(directory)}
        report["runs"].append(record)
        if is_active(directory, context):
            record.update(status="SKIP_ACTIVE")
            print(f"SKIP ACTIVE {directory.relative_to(root)}", flush=True)
            continue
        try:
            files, keep, remove, invalid, target = plan_group(directory, paths, root, context,
                                                              args.keep_recent, args.min_age_minutes * 60)
            print(f"RUN {directory.relative_to(root)}; target={target or 'unknown (keep recent)'}", flush=True)
            for file, reason in keep.items():
                print(f"  KEEP {file.path.name}: {reason}", flush=True)
            for message in invalid:
                print(f"  INVALID {message}", flush=True)
            size = sum(file.size for file in remove)
            print(f"  REMOVE {len(remove)} files / {size / 2**30:.2f} GiB", flush=True)
            record.update(keep=[{"path": str(file.path), "reason": reason} for file, reason in keep.items()],
                          remove=[str(file.path) for file in remove], invalid=invalid)
            if args.apply and remove:
                current = query_jobs()
                if current - context.jobs:
                    raise ValueError("new Slurm jobs appeared; rerun to refresh protection")
                with run_lock(run_directory(directory)):
                    apply_group(directory, files, remove, root)
            record["status"] = "DELETED" if args.apply else "WOULD_DELETE"
            report["removed_files"] += len(remove)
            report["removed_bytes"] += size
        except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
            record.update(status="SKIP", reason=str(error))
            print(f"SKIP {directory.relative_to(root)}: {error}", flush=True)
    report_path = root / ".checkpoint_prune" / (datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f_UTC") + ".json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    action = "DELETED" if args.apply else "WOULD_DELETE"
    print(f"{action}: {report['removed_files']} files / {report['removed_bytes'] / 2**30:.2f} GiB", flush=True)
    print(f"REPORT: {report_path}", flush=True)


if __name__ == "__main__":
    main()
