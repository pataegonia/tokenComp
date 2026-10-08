#!/usr/bin/env python3
"""Recover the interrupted RD4 sweep, preserving each original run directory.

Default: inspect retained checkpoints and print a plan. --submit schedules only
unfinished, inactive tasks. --replace-pending replaces the original pending
tasks with the new checkpoint policy; running tasks are always left alone.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime
import getpass
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import zipfile

from prune_hyper1d_rd4_checkpoints import File, STEP_NAME, apply_plan, make_plan


REPO = Path(__file__).resolve().parents[1]
TARGET = 50000
GROUPS = {"one_gpu": (1, "gs-h1d-rd4-1g", "gs-h1d-rd4-r1g"),
          "lowrank": (1, "gs-h1d-rd4-lr", "gs-h1d-rd4-rlr")}


def task_specs(repo, one_id, lowrank_id, lambda4, include_plain4=True):
    tables = {
        "one_gpu": [("single_off", lambda4), ("single_on", "0.0128"),
                    ("single_on", "0.0064"), ("single_on", lambda4),
                    ("dual_off", "0.0128"), ("dual_off", "0.0064"),
                    ("dual_off", lambda4), ("dual_on", "0.0128"),
                    ("dual_on", "0.0064"), ("dual_on", lambda4)],
        "lowrank": [("lowrank_off", "0.0256"), ("lowrank_off", "0.0128"),
                    ("lowrank_off", "0.0064"), ("lowrank_off", lambda4),
                    ("lowrank_on", "0.0128"), ("lowrank_on", "0.0064"),
                    ("lowrank_on", lambda4)],
    }
    if include_plain4:
        tables["one_gpu"].append(("plain4", lambda4))
    root = repo / "outputs/hyper1d_rd4_50k"
    tasks = []
    for group, entries in tables.items():
        array_id = one_id if group == "one_gpu" else lowrank_id
        for index, (variant, rate) in enumerate(entries):
            tag = rate.replace(".", "p")
            tasks.append({
                "group": group, "original_job": f"{array_id}_{index}",
                "variant": variant, "lambda": rate, "gpus": GROUPS[group][0],
                "output": str(root / variant / f"lambda{tag}" / f"job_{array_id}_{index}"),
                "manifest": str(root / "manifests" / group / f"job_{array_id}"
                                / f"{variant}_lambda{tag}.txt"),
            })
    return tasks


def read_queue():
    result = subprocess.run(
        ["squeue", "-h", "-r", "-u", os.environ.get("USER") or getpass.getuser(),
         "-o", "%i|%j|%T"], check=True, capture_output=True, text=True,
    )
    queue = {}
    for line in result.stdout.splitlines():
        if line.strip():
            job, name, state = line.strip().split("|", 2)
            queue[job] = {"name": name, "state": state}
    return queue


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


@contextmanager
def lock_file(path):
    # flock is released automatically if the launcher/worker is killed.
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another recovery process holds {path}") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def check_checkpoint(path, task, root, expected=None, initial=False):
    import torch
    snapshot = File.capture(path, root)
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            bad = archive.testzip()
            if bad is not None:
                raise ValueError(f"invalid checkpoint CRC: {bad}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    else:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must be a dictionary")
    state = checkpoint.get("state_dict", {})
    if not isinstance(state, dict) or not any(key.startswith("model.feature_codec.") for key in state):
        raise ValueError("missing integrated feature-codec weights")
    config = checkpoint.get("feature_codec_config", {})
    variant = task["variant"]
    architecture = "plain4" if variant == "plain4" else "legacy"
    required = {"codec_type": "hyper1d", "architecture": architecture,
                "paths": 2 if variant.startswith(("dual_", "lowrank_")) else 1,
                "base_rank": 56 if variant.startswith("lowrank_") else 0,
                "use_morton": variant.endswith("_on"),
                "n": 256 if architecture == "plain4" else 192,
                "m": 512 if architecture == "plain4" else 320}
    for key, value in required.items():
        # Earlier legacy checkpoints omitted these three default fields.
        fallback = {"architecture": "legacy", "paths": 1, "base_rank": 0}.get(key)
        if config.get(key, fallback) != value:
            raise ValueError(f"codec {key}={config.get(key)}, expected {value}")
    if tuple(config.get("strides", ())) != (2, 2):
        raise ValueError("expected original 4x transform strides")
    step = checkpoint.get("global_step", 0)
    if type(step) is not int or not (0 <= step <= TARGET):
        raise ValueError(f"unexpected global_step={step}")
    if initial:
        if step != 0:
            raise ValueError("initialization checkpoint contains training progress")
    else:
        if step < 1 or (expected is not None and step != expected):
            raise ValueError(f"global_step={step}, filename step={expected}")
        if step < TARGET and (not checkpoint.get("optimizer_states") or not checkpoint.get("lr_schedulers")):
            raise ValueError("missing optimizer/scheduler state for resume")
    if not snapshot.unchanged(root):
        raise ValueError("checkpoint changed during inspection")
    return step


def choose_source(task, repo=REPO):
    output = Path(task["output"])
    root = repo / "outputs"
    output.resolve().relative_to((root / "hyper1d_rd4_50k").resolve())
    manifest = Path(task["manifest"])
    # The original low-rank task 0 may have reused an older completed run.
    if manifest.is_file():
        path = Path(manifest.read_text(encoding="utf-8").strip())
        try:
            if check_checkpoint(path, task, root, expected=TARGET) == TARGET:
                return {"kind": "complete", "checkpoint": str(path), "step": TARGET}
        except Exception as error:
            print(f"  IGNORE manifest {manifest.name}: {error}", flush=True)
    files = [path for path in output.glob("checkpoints/hyper1d_12h/version_*/*.ckpt")
             if STEP_NAME.fullmatch(path.name) or path.name == "last.ckpt"]
    named = sorted((path for path in files if STEP_NAME.fullmatch(path.name)),
                   key=lambda path: (int(STEP_NAME.fullmatch(path.name)[1]), path.stat().st_mtime_ns),
                   reverse=True)
    best = None
    for path in named:
        try:
            step = check_checkpoint(path, task, root, int(STEP_NAME.fullmatch(path.name)[1]))
            best = {"kind": "complete" if step == TARGET else "resume",
                    "checkpoint": str(path), "step": step}
            break
        except Exception as error:
            print(f"  REJECT {path}: {type(error).__name__}: {error}", flush=True)
    # Prefer numbered files. Use last only for genuinely newer/unique progress.
    for path in (path for path in files if path.name == "last.ckpt"):
        if best and best["step"] == TARGET:
            break
        try:
            step = check_checkpoint(path, task, root)
            if best is None or step > best["step"]:
                best = {"kind": "complete" if step == TARGET else "resume",
                        "checkpoint": str(path), "step": step}
        except Exception as error:
            print(f"  REJECT {path}: {type(error).__name__}: {error}", flush=True)
    if best:
        return best
    if files:
        raise ValueError(f"no valid training checkpoint in {output}; refusing to reset progress")
    initial = output / "hyper1d_initial.ckpt"
    if initial.exists():
        check_checkpoint(initial, task, root, initial=True)
        return {"kind": "initial", "checkpoint": str(initial), "step": 0}
    return {"kind": "fresh", "checkpoint": None, "step": 0}


def recovery_jobs(root, queue):
    active = {}
    for path in sorted((root / "resumes").glob("*/plan.json")):
        plan = json.loads(path.read_text(encoding="utf-8"))
        for group, job in plan.get("jobs", {}).items():
            for index, task in enumerate(plan["groups"][group]):
                recovery_job = f"{job}_{index}"
                if recovery_job in queue:
                    active[task["original_job"]] = recovery_job
    return active


def inspect_tasks(tasks, repo, queue, replace_pending):
    active = recovery_jobs(repo / "outputs/hyper1d_rd4_50k", queue)
    groups = {group: [] for group in GROUPS}
    pending = []
    for task in tasks:
        original = task["original_job"]
        if original in active:
            print(f"SKIP {original}: recovery {active[original]} is active", flush=True)
            continue
        if original in queue:
            job = queue[original]
            if job["name"] != GROUPS[task["group"]][1]:
                raise ValueError(f"unexpected Slurm name for {original}: {job['name']}")
            if job["state"] != "PENDING" or not replace_pending:
                print(f"SKIP {original}: original job is {job['state']}", flush=True)
                continue
            pending.append(original)
        source = choose_source(task, repo)
        print(f"{source['kind'].upper():8} {original} {task['variant']} lambda={task['lambda']} "
              f"step={source['step']} gpu={task['gpus']}", flush=True)
        if source["checkpoint"]:
            print(f"  {source['checkpoint']}", flush=True)
        if source["kind"] != "complete":
            groups[task["group"]].append(dict(task, source=source))
    return groups, pending


def train_command(task, source, vanilla, repo=REPO):
    gpus = task["gpus"]
    command = [sys.executable, str(repo / "scripts/run_hyper1d.py"), "train"]
    if source["kind"] in ("resume", "initial"):
        command += ["--checkpoint", source["checkpoint"]]
        if source["kind"] == "resume":
            command += ["--resume"]
    elif source["kind"] == "fresh":
        variant = task["variant"]
        command += ["--vanilla-checkpoint", str(vanilla), "--architecture",
                    "plain4" if variant == "plain4" else "legacy", "--paths",
                    "2" if variant.startswith(("dual_", "lowrank_")) else "1",
                    "--strides", "4x"]
        if variant.startswith("lowrank_"):
            command += ["--base-rank", "56"]
        if variant.endswith("_on"):
            command += ["--morton"]
    else:
        raise ValueError(f"cannot train source {source['kind']}")
    return command + ["--output", task["output"], "--lambda", task["lambda"],
                      "--devices", str(gpus), "--launcher", "srun" if gpus > 1 else "python",
                      "--batch-size", "2", "--accumulate", "1" if gpus > 1 else "4",
                      "--no-time-limit", "--max-steps", str(TARGET), "--warmup-steps", "1000",
                      "--validate-every", "2000", "--checkpoint-every", "5000",
                      "--checkpoint-keep", "2", "--no-save-last"]


def write_manifest(task, source):
    manifest = Path(task["manifest"])
    manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest.with_name(f"{manifest.name}.tmp.{os.getpid()}")
    temporary.write_text(source["checkpoint"] + "\n", encoding="utf-8")
    os.replace(temporary, manifest)


def run_task(plan_path, group, index):
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    task = plan["groups"][group][index]
    output = Path(task["output"])
    output.resolve().relative_to((REPO / "outputs/hyper1d_rd4_50k").resolve())
    with lock_file(output / ".resume.lock"):
        if int(os.environ.get("SLURM_NTASKS", "0")) != task["gpus"]:
            raise RuntimeError("allocated Slurm task count does not match the recovery GPU setting")
        if task["original_job"] in read_queue():
            raise RuntimeError("original task is still active; refusing concurrent writes")
        source = choose_source(task)
        if source["kind"] != "complete":
            command = train_command(task, source, Path(plan["vanilla"]))
            print(f"[RD4 recovery] {task['original_job']} source={source['kind']} "
                  f"step={source['step']} output={output}", flush=True)
            print(shlex.join(command), flush=True)
            subprocess.run(command, cwd=REPO, check=True)
            # Ignore an older manifest until the newly completed run is checked.
            source = choose_source(task)
        if source["kind"] != "complete":
            raise RuntimeError(f"training ended below {TARGET}: {source}")
        write_manifest(task, source)
        print(f"[RD4 ready] {source['checkpoint']}", flush=True)
        # Training has exited and this run is locked. Keep its verified final
        # file; prune old versions independently, retaining their resume fallback.
        root = REPO / "outputs/hyper1d_rd4_50k"
        for directory in output.glob("checkpoints/hyper1d_12h/version_*"):
            try:
                keep, remove, _ = make_plan(directory, root, keep_recent=2)
                apply_plan(keep, remove, root)
                if remove:
                    print(f"[RD4 prune] {directory}: removed {len(remove)} surplus files", flush=True)
            except (OSError, ValueError) as error:
                print(f"[RD4 prune skipped] {directory}: {error}", flush=True)


def submit_groups(groups, plan_path, plan, repo=REPO):
    template = repo / "scripts/slurm/resume_hyper1d_rd4.slurm"
    for group, tasks in groups.items():
        if not tasks:
            continue
        gpus, _, name = GROUPS[group]
        command = ["sbatch", "--parsable", f"--array=0-{len(tasks) - 1}",
                   f"--job-name={name}", f"--gres=gpu:normal:{gpus}",
                   f"--ntasks={gpus}", f"--ntasks-per-node={gpus}",
                   f"--export=ALL,REPO_DIR={repo},RD_RESUME_PLAN={plan_path},RD_RESUME_GROUP={group}", str(template)]
        try:
            result = subprocess.run(command, cwd=repo, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as error:
            print(f"SBATCH FAILED ({group}): {error.stderr or error}", flush=True)
            raise
        job = result.stdout.strip().split(";", 1)[0]
        if not job.isdigit():
            raise ValueError(f"unexpected sbatch result: {result.stdout!r}")
        plan["jobs"][group] = job
        write_json(plan_path, plan)
        print(f"SUBMITTED {group}: {job}; {len(tasks)} tasks x {gpus} GPUs", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--replace-pending", action="store_true")
    parser.add_argument("--one-job-id", type=int, default=442942)
    parser.add_argument("--lowrank-job-id", type=int, default=442943)
    parser.add_argument("--lambda4", choices=("0.0032", "0.0512"), default=os.environ.get("RD_LAMBDA4", "0.0032"))
    parser.add_argument("--no-include-plain4", action="store_true", default=os.environ.get("RD_INCLUDE_PLAIN4") == "0")
    parser.add_argument("--vanilla-checkpoint", type=Path,
                        default=Path(os.environ.get("VANILLA_CHECKPOINT", REPO / "checkpoints/pretrained/globalsplat-re10k-32k.ckpt")))
    parser.add_argument("--run-task", nargs=3, metavar=("PLAN", "GROUP", "INDEX"))
    args = parser.parse_args(argv)
    if args.run_task:
        path, group, index = args.run_task
        run_task(Path(path), group, int(index))
        return
    if min(args.one_job_id, args.lowrank_job_id) <= 0:
        parser.error("original job IDs must be positive")
    if args.lambda4 not in ("0.0032", "0.0512"):
        parser.error("RD_LAMBDA4 must be 0.0032 or 0.0512")
    root = REPO / "outputs/hyper1d_rd4_50k"
    if not root.is_dir():
        parser.error(f"missing original sweep directory: {root}")
    with lock_file(root / "resumes/.submit.lock"):
        queue = read_queue()
        tasks = task_specs(REPO, args.one_job_id, args.lowrank_job_id,
                           args.lambda4, not args.no_include_plain4)
        groups, pending = inspect_tasks(tasks, REPO, queue, args.replace_pending)
        for group, entries in groups.items():
            print(f"PLAN {group}: {len(entries)} tasks; {GROUPS[group][0]} GPUs per task", flush=True)
        print(f"Maximum concurrent total: {sum(len(v) * GROUPS[k][0] for k, v in groups.items())} GPUs "
              "(scheduler availability applies)", flush=True)
        if pending:
            print(f"REPLACE PENDING ONLY: {' '.join(pending)}", flush=True)
        if not args.submit:
            print("PLAN ONLY. Run again with --submit --replace-pending to schedule recovery.")
            return
        for relative in ("scripts/run_hyper1d.py", "scripts/slurm/resume_hyper1d_rd4.slurm"):
            if not (REPO / relative).is_file():
                raise FileNotFoundError(REPO / relative)
        if any(task["source"]["kind"] == "fresh" for entries in groups.values() for task in entries):
            if not args.vanilla_checkpoint.is_file():
                raise FileNotFoundError(args.vanilla_checkpoint)
        if pending:
            # The state filter prevents cancelling a task that started after inspection.
            subprocess.run(["scancel", "--state=PENDING", "--user=" +
                            (os.environ.get("USER") or getpass.getuser()), *pending], check=True)
            queue = read_queue()
            still_pending = [job for job in pending if queue.get(job, {}).get("state") == "PENDING"]
            if still_pending:
                raise RuntimeError(f"pending cancellation not yet visible; retry without duplicating: {still_pending}")
        # Recheck activity after cancellation/inspection, before creating arrays.
        queue = read_queue()
        active = recovery_jobs(root, queue)
        groups = {group: [task for task in entries if task["original_job"] not in queue
                          and task["original_job"] not in active] for group, entries in groups.items()}
        if not any(groups.values()):
            print("No unfinished inactive tasks to submit.")
            return
        (REPO / "logs/slurm").mkdir(parents=True, exist_ok=True)
        plan_path = root / "resumes" / datetime.now().strftime("%Y%m%d_%H%M%S_%f") / "plan.json"
        plan = {"groups": groups, "jobs": {}, "vanilla": str(args.vanilla_checkpoint.resolve())}
        write_json(plan_path, plan)
        print(f"PLAN FILE: {plan_path}", flush=True)
        try:
            submit_groups(groups, plan_path, plan)
        except Exception:
            print(f"Submission stopped. Already submitted: {plan['jobs']}. "
                  "Re-run the same command; active recovery tasks will be skipped.", flush=True)
            raise
        print("Training only; submit full-test evaluation after completion.", flush=True)


if __name__ == "__main__":
    main()
