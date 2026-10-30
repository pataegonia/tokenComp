#!/usr/bin/env python3
"""Resume only train456951/457099 and rebind their existing pending evaluations.

Preview is read-only. --submit accepts up to two resume jobs, preserving the old
run directories and evaluation IDs. There is no restart-from-parent fallback.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import recover_nfcgs_sort_context_ablation as base

FILES = (
    "scripts/recover_nfcgs_paper_controls_20261020.py", "scripts/slurm/resume_nfcgs_paper_controls_20261020.slurm",
    "scripts/recover_nfcgs_sort_context_ablation.py", "scripts/prepare_nfcgs_nn_score_resume.py",
    "scripts/run_nfcgs.py", "scripts/slurm/nfcgs_order_context_settings.sh",
    "scripts/slurm/eval_nfcgs_order_context.slurm", "scripts/check_nfcgs_checkpoint.py",
    "globalsplat/main.py", "globalsplat/checkpointing.py", "globalsplat/model/model_wrapper.py",
    "globalsplat/compression/codec.py", "globalsplat/compression/config.py",
    "globalsplat/compression/checkpoint.py", "globalsplat/compression/score_context.py",
    "globalsplat/dataset/data_module.py", "globalsplat/dataset/re10k.py",
    "config/main.yaml", "config/experiment/re10k_32k_nfcgs.yaml",
    "config/dataset/re10k_nfcgs_paper_train.yaml", "config/model/globalsplat_nfcgs_rank56.yaml",
)


@dataclass(frozen=True)
class Target:
    train: str
    evaluation: str
    point: base.Point


TARGETS = (
    Target("456951", "457098", base.Point("none_l3_off12_0p0512", "0.0512", "none", "legacy", 3, False, "n3off")),
    Target("457099", "457100", base.Point("hilbert_d4_off16_0p0128", "0.0128", "hilbert", "dyadic4", 4, False, "h4off")),
)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            result.update(block)
    return result.hexdigest()


def queue_dependencies(ids, user):
    output = base.command(["squeue", "-h", "-u", user, "-j", ",".join(sorted(ids)), "-o", "%i|%T|%E"])
    result = {}
    for line in output.splitlines():
        fields = line.strip().split("|", 2)
        if len(fields) == 3 and fields[0] in ids:
            result[fields[0]] = dict(state=fields[1], dependency=fields[2])
    return result


def original_records(repo):
    path = repo / "logs/slurm/paper_controls_20261018/submissions.tsv"
    with path.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    records = {}
    for target in TARGETS:
        matched = [row for row in rows if row["train_job"] == target.train]
        if len(matched) != 1:
            raise ValueError(f"original submission must contain exactly one train {target.train}")
        row = matched[0]
        expected_dir = target.point.run_root(repo) / f"job_{target.train}"
        if (row["eval_job"] != target.evaluation or row["order"] != target.point.order
                or row["schedule"] != target.point.schedule or int(row["stages"]) != target.point.stages
                or row["lambda"] != target.point.rate or Path(row["output"]).resolve() != expected_dir.resolve()):
            raise ValueError(f"original submission recipe/path mismatch: {target.train}")
        records[target.point.key] = dict(original_train=target.train, eval_job=target.evaluation,
                                          run_dir=str(expected_dir.resolve()))
    return records


def validated_latest(run_dir, config):
    path, step = base.latest_checkpoint(run_dir, config)
    if path is None:
        raise ValueError(f"no full-state checkpoint under {run_dir}; refusing a fresh warm start")
    expected_parent = (Path(run_dir)/"checkpoints/nfcgs_main/version_0").resolve()
    if path.parent != expected_parent:
        raise ValueError("pending eval reads version_0; a different resume version requires separate review")
    import torch
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not checkpoint.get("loops"):
        raise ValueError("full-state resume requires Lightning loop state")
    del checkpoint
    if step == 50000:
        final = expected_parent / "step000050000.ckpt"
        if not final.is_file() or base.prepare(final, 50001, expected_config=config) != 50000:
            raise ValueError("eval-only recovery requires the named final step000050000.ckpt")
        path = final
    return dict(checkpoint=str(path), step=step, sha256=digest(path))


def copy_for_resume(source, destination, expected_step, expected_sha256, config):
    source, destination = source.resolve(), destination.resolve()
    if source.parent != destination.parent or source.parent.name != "version_0":
        raise ValueError("resume snapshot must be a new file inside the original version_0 directory")
    step = base.prepare(source, 50000, expected_config=config)
    if step != expected_step or digest(source) != expected_sha256:
        raise ValueError("resume checkpoint changed since submission; no training started")
    base.prepare(source, 50000, destination, expected_config=config)
    if digest(destination) != expected_sha256:
        raise ValueError("resume snapshot SHA-256 mismatch")
    print(f"RESUME_SNAPSHOT={destination} RESTORED_STEP={step} TARGET_STEP=50000 REMAINING_STEPS={50000-step}", flush=True)


def recover(repo, *, apply=False, retry_failed=False):
    repo = repo.resolve()
    hashes = {}
    for name in FILES:
        path = repo / name
        if not path.is_file():
            raise FileNotFoundError(f"required server file missing: {path}")
        data = path.read_bytes()
        if path.suffix in (".sh", ".slurm") and (b"\r" in data or not data.startswith(b"#!/usr/bin/env bash\n")):
            raise ValueError(f"shell file must be LF with a valid shebang: {name}")
        hashes[name] = hashlib.sha256(data).hexdigest()
    original = original_records(repo)
    inputs = dict(repo=str(repo), original=original, code_sha256=hashes,
                  protocol="resume_optimizer_scheduler_loops_to_absolute_50000_in_original_version0; fresh-shuffle loader")
    signature = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    state_file = repo / "logs/slurm/paper_controls_recovery_20261020/state.json"
    lock = state_file.with_suffix(".lock")
    if apply:
        state_file.parent.mkdir(parents=True, exist_ok=True)
        lock.mkdir()
    try:
        state = json.loads(state_file.read_text(encoding="utf-8")) if state_file.exists() else {
            "version": 1, "signature": signature, "inputs": inputs, "points": {}}
        if state.get("version") != 1 or state.get("signature") != signature:
            raise ValueError("recovery inputs/code differ from saved state; no duplicate submission")
        user = os.environ.get("USER") or getpass.getuser()
        ids = {target.train for target in TARGETS} | {target.evaluation for target in TARGETS}
        ids |= {record["train_job"] for record in state["points"].values() if record.get("train_job")}
        statuses = base.job_states(ids, user)
        deps = queue_dependencies({target.evaluation for target in TARGETS}, user)
        actions = []
        for target in TARGETS:
            key = target.point.key
            saved = state["points"].get(key, {})
            action = dict(original[key], key=key)
            new_train = saved.get("train_job")
            if statuses[target.train] in base.ACTIVE:
                raise ValueError(f"original train {target.train} is active; no new jobs submitted")
            eval_state = statuses[target.evaluation]
            if eval_state not in {"PENDING", "RUNNING", "COMPLETING", "COMPLETED"}:
                raise ValueError(f"existing eval {target.evaluation} is {eval_state}; this tool preserves its ID")
            if eval_state == "PENDING":
                dep = deps.get(target.evaluation, {}).get("dependency", "")
                owners = set(re.findall(r"after(?:ok|any):(\d+)", dep))
                if owners - {target.train, new_train} or (not owners and new_train is None):
                    raise ValueError(f"eval {target.evaluation} has unexpected dependencies: {dep}")
            if new_train and statuses[new_train] in base.ACTIVE:
                if eval_state != "PENDING":
                    raise ValueError(f"eval {target.evaluation} already started while resume train {new_train} is active")
                action.update(mode="reuse", train_job=new_train, step=saved.get("step"))
            elif new_train and statuses[new_train] != "COMPLETED" and not retry_failed:
                raise ValueError(f"resume train {new_train} is {statuses[new_train]}; inspect it, then use --retry-failed if appropriate")
            else:
                checked = validated_latest(action["run_dir"], target.point.config())
                action.update(checked)
                if checked["step"] == 50000:
                    action.update(mode="eval_only", train_job=new_train)
                elif eval_state != "PENDING":
                    raise ValueError(f"eval {target.evaluation} is {eval_state} but no final50k checkpoint exists")
                elif new_train and statuses[new_train] == "COMPLETED":
                    raise ValueError(f"resume train {new_train} completed without a final50k checkpoint")
                else:
                    action.update(mode="resume", train_job=None)
            action["eval_state"] = eval_state
            actions.append((target, action))
        # Every selected checkpoint and both old evaluations pass before mutation.
        for _, action in actions:
            print(f"PLAN {action['key']} MODE={action['mode']} RESTORED_STEP={action.get('step')} "
                  f"TRAIN={action.get('train_job') or 'new'} EXISTING_EVAL={action['eval_job']} RUN_DIR={action['run_dir']}", flush=True)
        if not apply:
            print("PREVIEW_ONLY=True; use --submit to resume these two controls and update their pending eval dependencies")
            return state
        for target, action in actions:
            key = target.point.key
            if action["mode"] == "resume":
                env = base.environment(target.point, repo)
                env.update(SCORE_SLICE_CHANNELS="16", DATASET_ROOT="/data3/local_datasets/re10k",
                    RESUME_ORIGINAL_TRAIN=target.train,
                    RESUME_RUN_DIR=action["run_dir"], RESUME_CHECKPOINT=action["checkpoint"],
                    RESUME_EXPECTED_STEP=str(action["step"]), RESUME_EXPECTED_SHA256=action["sha256"],
                    RESUME_EXPECTED_CONFIG=json.dumps(target.point.config().to_dict(), separators=(",", ":")))
                job = base.submit([f"--job-name=gs-r20-{target.train}",
                    "scripts/slurm/resume_nfcgs_paper_controls_20261020.slurm"], env, repo)
                previous = state["points"].get(key, {})
                attempts = list(previous.get("attempts", [])) + [dict(job_id=job, step=action["step"],
                    checkpoint=action["checkpoint"], sha256=action["sha256"])]
                action.update(train_job=job, attempts=attempts, mode="reuse", dependency_bound=False)
                state["points"][key] = dict(action)
                base.save(state_file, state)
                print(f"RESUME_SUBMITTED ORIGINAL={target.train} NEW={job} FROM_STEP={action['step']}", flush=True)
            if action["eval_state"] == "PENDING":
                dependency = (f"afterok:{action['train_job']}" if action.get("train_job")
                              else f"afterany:{target.train}")
                observed = deps.get(target.evaluation, {}).get("dependency", "")
                normalized = re.sub(r"\([^)]*\)", "", observed).strip()
                satisfied = (action["mode"] == "eval_only" and state["points"].get(key, {}).get("dependency_bound")
                             and observed in ("", "(null)", "None"))
                if normalized != dependency and not satisfied:
                    base.command(["scontrol", "update", f"JobId={target.evaluation}", f"Dependency={dependency}"])
                action.update(dependency=dependency, dependency_bound=True)
                print(f"EVAL_REBOUND ID={target.evaluation} DEPENDENCY={dependency}", flush=True)
            state["points"][key] = dict(state["points"].get(key, {}), **action)
            base.save(state_file, state)
        print(f"RECOVERY_STATE={state_file}; rerun the same command to reuse accepted training IDs", flush=True)
        return state
    finally:
        if apply:
            lock.rmdir()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--copy-source", type=Path)
    parser.add_argument("--copy-target", type=Path)
    parser.add_argument("--expected-step", type=int)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--expected-config")
    args = parser.parse_args(argv)
    try:
        if args.copy_source:
            if args.copy_target is None or args.expected_step is None or not args.expected_sha256 or not args.expected_config:
                parser.error("copy requires target, expected step, SHA-256 and codec config")
            copy_for_resume(args.copy_source, args.copy_target, args.expected_step,
                            args.expected_sha256, base.CodecConfig.from_mapping(json.loads(args.expected_config)))
        else:
            recover(args.repo, apply=args.submit, retry_failed=args.retry_failed)
    except subprocess.CalledProcessError as error:
        print(error.stderr or str(error), file=sys.stderr)
        print("Accepted training IDs were saved; rerun the same command to retry missing submissions/dependency updates.", file=sys.stderr)
        raise SystemExit(1)
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
