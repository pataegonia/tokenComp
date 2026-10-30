#!/usr/bin/env python3
"""Recover this ablation sweep and its protected NN-score evals, without scancel."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from globalsplat.compression import CodecConfig  # noqa: E402
from globalsplat.compression.checkpoint import validate_feature_codec_checkpoint  # noqa: E402
from prepare_nfcgs_nn_score_resume import prepare  # noqa: E402

ACTIVE = {"RUNNING", "PENDING", "CONFIGURING", "COMPLETING", "SUSPENDED", "RESIZING",
          "SIGNALING", "STAGE_OUT", "REQUEUED", "REQUEUE_HOLD", "REQUEUE_FED"}
TERMINAL = {"COMPLETED", "CANCELLED", "FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL",
            "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED", "SPECIAL_EXIT"}
PROTECTED = ("450934", "450937", "450939")
PARENT = ("outputs/nfcgs_score_probability10k/20260913_112316/e1_split/rank56/lambda0p0256/"
          "residual_on/checkpoints/nfcgs_score_probability10k_e1_split_rank56_lambda0p0256_"
          "residual_on_morton_on_m1c1s1_split/version_0/step000010000.ckpt")


@dataclass(frozen=True)
class Point:
    key: str
    rate: str
    order: str
    schedule: str
    stages: int
    context: bool
    tag: str
    protected: bool = False

    def config(self):
        return CodecConfig(token_order=self.order, use_centering=False, use_score_norm=False,
            score_mean_condition=False, score_channel_context=False, transform="linear",
            score_spatial_context=self.context, score_context_schedule=self.schedule,
            score_spatial_stages=self.stages, score_spatial_kernel=5, score_context_quantization="ste",
            nn_order_backend="knn" if self.protected else "legacy",
            nn_order_distance_device="input" if self.protected else "cpu")

    def run_root(self, repo):
        schedule = f"{self.schedule}_stages{self.stages}_k5_ste"
        if not self.context:
            schedule += "_context_off"
        if self.protected:
            schedule += "_nn_knn_input_k64"
        return repo / "outputs/nfcgs_order_context" / self.order / schedule / f"lambda{self.rate.replace('.', 'p')}"


def points():
    result = [Point("nn_score_0p0256", "0.0256", "nn_score", "legacy", 3, True, "ns", True),
              Point("nn_score_0p0064", "0.0064", "nn_score", "legacy", 3, True, "ns", True),
              Point("nn_score_0p0128", "0.0128", "nn_score", "legacy", 3, True, "ns", True)]
    for arm, order, schedule, stages, context, tag in (
        ("context_only", "none", "legacy", 3, True, "co"),
        ("hilbert_only", "hilbert", "legacy", 3, False, "ho"),
        ("context_quarter2", "none", "quarter2", 2, True, "cq2"),
        ("context_dyadic4", "none", "dyadic4", 4, True, "cd4"),
    ):
        for rate in ("0.0064", "0.0128", "0.0256"):
            result.append(Point(f"{arm}_{rate.replace('.', 'p')}", rate, order, schedule, stages, context, tag))
    return result


def read_old_state(path):
    if not path.exists():
        return {}
    fields = path.read_text(encoding="utf-8").split()
    if (len(fields) != 4 or fields[0] != "v1" or not re.fullmatch(r"[a-f0-9]{64}", fields[1])
            or not fields[2].isdigit() or (fields[3] != "-" and not fields[3].isdigit())):
        raise ValueError(f"invalid old state: {path}")
    return dict(signature=fields[1], train_job=fields[2], eval_job=None if fields[3] == "-" else fields[3])


def old_record(point, repo, old_dir, parent):
    if point.protected:
        if point.rate == "0.0256":
            path = repo / "logs/slurm/nn_score_resume_retry_450904.state"
            expected = "450934"
            run_dir = point.run_root(repo) / "resume_job_450934"
        else:
            path = repo / "logs/slurm/nn_score_rd_gpu_20261013" / f"nn_score_legacy3_{point.rate.replace('.', 'p')}.state"
            expected = "450937" if point.rate == "0.0064" else "450939"
            run_dir = point.run_root(repo) / f"job_{expected}"
        record = read_old_state(path)
        if record.get("train_job") != expected:
            raise ValueError(f"protected training ID must remain {expected}: {path}")
        return dict(record, run_dir=str(run_dir))
    record = read_old_state(old_dir / f"{point.key}.state")
    if record:
        signature_parts = ["order-context-v1", str(repo), str(parent), point.order, point.schedule,
            str(point.stages), "5", point.rate, "50000", "5000", "", "train-time-limit", "6-00:00:00"]
        if not point.context:
            signature_parts += ["spatial-context", "false"]
        signature = hashlib.sha256(b"".join(value.encode() + b"\0" for value in signature_parts)).hexdigest()
        if signature != record["signature"]:
            raise ValueError(f"old state does not match this experiment: {point.key}")
        record["run_dir"] = str(point.run_root(repo) / f"job_{record['train_job']}")
    return record


def command(args, **kwargs):
    return subprocess.run(args, check=True, capture_output=True, text=True, **kwargs).stdout.strip()


def job_states(ids, user):
    if not ids:
        return {}
    history = command(["sacct", "-X", "-n", "-P", "--starttime=2026-10-13", "-u", user, "-j", ",".join(sorted(ids)),
                       "--format=JobIDRaw,State,User"])
    states = {}
    for line in history.splitlines():
        parts = line.split("|")
        if len(parts) >= 3 and parts[0] in ids:
            if parts[2] != user:
                raise ValueError(f"job ownership mismatch: {parts[0]}")
            states[parts[0]] = parts[1].split()[0].rstrip("+")
    active = command(["squeue", "-h", "-u", user, "-o", "%i|%T"])
    for line in active.splitlines():
        parts = line.strip().split("|")
        if len(parts) == 2 and parts[0] in ids:
            states[parts[0]] = parts[1]
    for job in ids:
        if states.get(job) not in ACTIVE | TERMINAL:
            raise ValueError(f"cannot establish job {job} status; no jobs submitted")
    return states


def latest_checkpoint(run_dir, config):
    if not run_dir:
        return None, 0
    root = Path(run_dir) / "checkpoints/nfcgs_main"
    numbered = [p for p in root.glob("version_*/step?????????.ckpt") if p.is_file()]
    candidates = []
    if numbered:
        candidates.append(max(numbered, key=lambda p: (int(p.stem[4:]), p.stat().st_mtime_ns)))
    candidates.extend(root.glob("version_*/last.ckpt"))
    if not candidates:
        candidates.extend(root.glob("version_*/resume.ckpt"))
    best, best_step = None, -1
    for path in candidates:
        step = prepare(path, 50001, expected_config=config)
        if step > best_step:
            best, best_step = path.resolve(), step
    return best, max(best_step, 0)


def plan(point, record, states, repo, parent):
    train = record.get("train_job")
    evaluation = record.get("eval_job")
    status = states.get(train)
    eval_status = states.get(evaluation)
    if point.protected:
        expected = {"0.0256": "450934", "0.0064": "450937", "0.0128": "450939"}[point.rate]
        if train != expected:
            raise ValueError(f"protected training ID must remain {expected}")
    if status in ACTIVE or status == "COMPLETED":
        if status == "COMPLETED":
            checkpoint, step = latest_checkpoint(record["run_dir"], point.config())
            if step != 50000:
                raise ValueError(f"completed training {train} has no validated step50000 checkpoint")
        return dict(record, key=point.key, mode="reuse", step=None,
                    dependency=train if status in ACTIVE else None,
                    eval_job=evaluation if eval_status in ACTIVE | {"COMPLETED"} else None)
    if point.protected:
        raise ValueError(f"protected NN-score training {train} is {status}; this helper never restarts it")
    if eval_status in ACTIVE and record.get("mode") != "eval_only":
        raise ValueError(f"old eval {evaluation} is still active for terminated training {train}; resolve it before recovery")
    checkpoint, step = latest_checkpoint(record.get("run_dir"), point.config())
    if step == 50000:
        return dict(key=point.key, mode="eval_only", step=step, train_job=None,
                    eval_job=evaluation if eval_status in ACTIVE | {"COMPLETED"} else None,
                    run_dir=record["run_dir"], checkpoint=str(checkpoint))
    return dict(key=point.key, mode="resume" if checkpoint else "restart", step=step,
                checkpoint=str(checkpoint or parent), train_job=None, eval_job=None, run_dir=None)


def save(path, manifest):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, encoding="utf-8", delete=False) as file:
        json.dump(manifest, file, indent=2)
        file.write("\n")
        temporary = file.name
    os.replace(temporary, path)


def environment(point, repo):
    env = os.environ.copy()
    for key in ("EVAL_RUN_DIR", "RESUME_CHECKPOINT", "SOURCE_CHECKPOINT", "SCORE_ORDER_SCALE"):
        env.pop(key, None)
    config = point.config()
    env.update(REPO_DIR=repo.as_posix(), TOKEN_ORDER=point.order, CONTEXT_SCHEDULE=point.schedule,
        STAGES=str(point.stages), KERNEL="5", SPATIAL_CONTEXT=str(point.context).lower(),
        RATE_LAMBDA=point.rate, MAX_STEPS="50000", CHECKPOINT_EVERY="5000", SCORE_ORDER_SCALE="",
        NN_ORDER_BACKEND=config.nn_order_backend, NN_ORDER_DISTANCE_DEVICE=config.nn_order_distance_device,
        NN_ORDER_CANDIDATES="64", PYTHONPATH=str(repo) + os.pathsep + env.get("PYTHONPATH", ""))
    return env


def submit(options, env, repo):
    response = command(["sbatch", "--parsable", "--export=ALL", *options], env=env, cwd=repo)
    job = response.split(";")[0]
    if not job.isdigit():
        raise ValueError(f"invalid sbatch response: {response}")
    return job


def recover(repo, old_dir, state_file, parent, *, dry_run=False, evals_only=False):
    repo, old_dir, parent = repo.resolve(), old_dir.resolve(), parent.resolve()
    for relative in ("scripts/run_nfcgs.py", "scripts/prepare_nfcgs_nn_score_resume.py",
                     "scripts/recover_nfcgs_sort_context_ablation.py",
                     "scripts/slurm/nfcgs_order_context_settings.sh",
                     "scripts/slurm/train_nfcgs_order_context_recovery.slurm",
                     "scripts/slurm/eval_nfcgs_order_context.slurm"):
        path = repo / relative
        if not path.is_file():
            raise ValueError(f"missing server file: {path}")
        if path.suffix in (".sh", ".slurm"):
            data = path.read_bytes()
            if not data.startswith(b"#!/usr/bin/env bash\n") or b"\r" in data:
                raise ValueError(f"shell file must have a BOM-free LF shebang: {path}")
    selected = [p for p in points() if not evals_only or p.protected]
    signature = hashlib.sha256(json.dumps(dict(repo=str(repo), parent=str(parent),
        recipe=[(p.key, p.config().to_dict()) for p in points()]), sort_keys=True).encode()).hexdigest()
    manifest = json.loads(state_file.read_text(encoding="utf-8")) if state_file.exists() else {
        "version": 1, "signature": signature, "points": {}}
    if manifest.get("version") != 1 or manifest.get("signature") != signature:
        raise ValueError("recovery state belongs to a different repository/recipe")
    records = {p.key: manifest["points"].get(p.key) or old_record(p, repo, old_dir, parent) for p in selected}
    ids = {str(r[k]) for r in records.values() for k in ("train_job", "eval_job") if r.get(k)}
    states = job_states(ids, os.environ.get("USER") or getpass.getuser())
    plans = [plan(p, records[p.key], states, repo, parent) for p in selected]
    # Check the common parent against every architecture that needs a warm start.
    if any(p["mode"] == "restart" for p in plans):
        import torch
        checkpoint = torch.load(parent, map_location="cpu", weights_only=False)
        if "model.feature_codec.shared_basis" not in checkpoint.get("state_dict", {}):
            raise ValueError("parent must be an integrated GlobalSplat+codec checkpoint")
        for point, action in zip(selected, plans):
            if action["mode"] == "restart":
                validate_feature_codec_checkpoint(checkpoint, point.config(), allow_score_path_conversion=True)
        del checkpoint
    for action in plans:
        print(f"PLAN {action['key']} MODE={action['mode']} STEP={action['step']} "
              f"TRAIN={action.get('train_job') or 'new'} EVAL={action.get('eval_job') or 'new'}", flush=True)
    if dry_run:
        print("PREVIEW ONLY: no jobs submitted or cancelled; protected training IDs=" + ",".join(PROTECTED))
        return manifest
    state_file.parent.mkdir(parents=True, exist_ok=True)
    lock = Path(str(state_file) + ".lock")
    lock.mkdir()
    try:
        # No concurrent writer can replace the state checked during planning.
        if state_file.exists() and json.loads(state_file.read_text(encoding="utf-8")) != manifest:
            raise ValueError("recovery state changed during preflight; rerun")
        for point, action in zip(selected, plans):
            env = environment(point, repo)
            if action["mode"] in ("resume", "restart"):
                env.update(RECOVERY_MODE=action["mode"], RECOVERY_CHECKPOINT=action["checkpoint"],
                           RECOVERY_CODEC_CONFIG=json.dumps(point.config().to_dict()))
                job = submit([f"--job-name=gs-r-{point.tag}-{point.rate.replace('.', 'p')}",
                              "scripts/slurm/train_nfcgs_order_context_recovery.slurm"], env, repo)
                action["train_job"] = job
                action["dependency"] = job
                action["run_dir"] = str(point.run_root(repo) / f"recovery_job_{job}")
                action["mode"] = "reuse"
                manifest["points"][point.key] = dict(action)
                save(state_file, manifest)
                print(f"TRAIN_SUBMITTED {point.key} ID={job}", flush=True)
            if not action.get("eval_job"):
                env.update(TRAIN_JOB_ID=action.get("train_job") or "completed", EVAL_RUN_DIR=action["run_dir"])
                options = [f"--job-name=gs-r-e-{point.tag}-{point.rate.replace('.', 'p')}"]
                if action.get("dependency"):
                    options += [f"--dependency=afterok:{action['dependency']}"]
                options += ["scripts/slurm/eval_nfcgs_order_context.slurm"]
                action["eval_job"] = submit(options, env, repo)
                print(f"EVAL_SUBMITTED {point.key} ID={action['eval_job']} TRAIN={action.get('train_job')}", flush=True)
            manifest["points"][point.key] = dict(action)
            save(state_file, manifest)
    finally:
        lock.rmdir()
    print(f"RECOVERY_STATE={state_file}; rerun this command to reuse IDs and retry missing submissions")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--old-state-dir", type=Path)
    parser.add_argument("--state-file", type=Path)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--nn-evals-only", action="store_true", help="restore only evals for the three protected trainings")
    parser.add_argument("--copy-checkpoint", type=Path)
    parser.add_argument("--copy-to", type=Path)
    parser.add_argument("--expected-config", help="JSON CodecConfig used when copying a resume checkpoint")
    args = parser.parse_args(argv)
    try:
        if args.copy_checkpoint:
            if not args.copy_to or not args.expected_config:
                parser.error("--copy-checkpoint requires --copy-to and --expected-config")
            step = prepare(args.copy_checkpoint, 50000, args.copy_to,
                           expected_config=CodecConfig.from_mapping(json.loads(args.expected_config)))
            print(f"RESUME_FROM_STEP={step} TARGET_STEP=50000 REMAINING_STEPS={50000-step}")
            return
        repo = args.repo.resolve()
        recover(repo, args.old_state_dir or repo / "logs/slurm/sort_context_ablation_rd_20261013",
                args.state_file or repo / "logs/slurm/sort_context_recovery_20261014.json",
                args.parent or repo / PARENT, dry_run=args.dry_run, evals_only=args.nn_evals_only)
    except subprocess.CalledProcessError as error:
        print(error.stderr or str(error), file=sys.stderr)
        print("Accepted IDs remain in the recovery state; rerun after submission capacity is available.", file=sys.stderr)
        raise SystemExit(1)
    except (ValueError, OSError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
