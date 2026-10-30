#!/usr/bin/env python3
"""Run one fresh RD stage or a provenance-checked actual-bitstream evaluation."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import os
from pathlib import Path
import shlex
import stat
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_nfcgs as base
from globalsplat.fresh_rd import (BUDGETS, CELLS, SEEDS, VERSION, atomic_json, canonical_overrides,
    checkpoint_path, converts_initializer, directory_lock, file_hash, milestones, rate_text, read_json, run_key,
    selection_overrides, validate_rows, validate_cohort)

RUNTIME_FILES = ("globalsplat/fresh_rd.py", "globalsplat/fresh_rd_runtime.py", "globalsplat/main.py",
    "globalsplat/model/model_wrapper.py", "globalsplat/dataset/data_module.py", "globalsplat/dataset/holdout.py",
    "globalsplat/compression/codec.py", "globalsplat/compression/checkpoint.py", "globalsplat/compression/score_context.py",
    "globalsplat/compression/residual.py", "globalsplat/compression/architecture.py", "scripts/run_nfcgs.py",
    "scripts/run_nfcgs_fresh_rd.py", "config/main.yaml", "config/experiment/re10k_32k_nfcgs.yaml",
    "globalsplat/model/globalsplat.py", "globalsplat/model/optim.py", "globalsplat/model/rendering.py",
    "globalsplat/compression/initialization.py", "globalsplat/compression/token_order.py", "globalsplat/compression/config.py",
    "globalsplat/compression/entropy.py", "globalsplat/compression/bitstream.py", "globalsplat/dataset/re10k.py",
    "globalsplat/loss/rendering_loss.py", "config/loss/nfcgs_codec_rd.yaml", "config/loss/nfcgs_codec_rd_subset.yaml",
    "config/dataset/re10k_nfcgs_paper_train.yaml", "config/dataset/re10k_eval_all_ctx12.yaml",
    "config/model/globalsplat_nfcgs_rank56.yaml")

RUNTIME_REPAIR_FILES = {"globalsplat/fresh_rd_runtime.py", "globalsplat/model/model_wrapper.py",
                        "scripts/run_nfcgs_fresh_rd.py"}


def protected_stage_a_path(checkpoint):
    return Path(checkpoint).with_name("stage_a_final_step000060000.ckpt")


def freeze_stage_a_checkpoint(checkpoint):
    checkpoint = Path(checkpoint)
    protected = protected_stage_a_path(checkpoint)
    digest = file_hash(checkpoint)
    if not protected.exists():
        os.link(checkpoint, protected)
    if not os.path.samefile(checkpoint, protected) or file_hash(protected) != digest:
        raise ValueError("Stage A final hard link differs from its checkpoint")
    protected.chmod(stat.S_IMODE(protected.stat().st_mode) & ~0o222)
    return dict(protected_checkpoint=str(protected), protected_checkpoint_sha256=digest,
                protection="read_only_hard_link")


def preparation_for(seed, root=ROOT):
    suffix = "initialization_20261019" if seed == 0 else f"initialization_fresh_rd_s{seed}"
    return Path(root) / "outputs/nfcgs_architecture_pilots" / suffix


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("train", "validation", "test", "reference", "native"))
    p.add_argument("--cell", choices=CELLS, default="h_on")
    p.add_argument("--rate-lambda", default="0.0128")
    p.add_argument("--seed-index", type=int, choices=range(3), default=0)
    p.add_argument("--stage", choices=("A", "B", "full"), default="A")
    p.add_argument("--budget", type=int, choices=BUDGETS, default=100000)
    p.add_argument("--preparation", type=Path)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--step", type=int)
    p.add_argument("--split", choices=("validation", "test"), default="validation")
    p.add_argument("--dataset-root", type=Path, default=Path(os.environ.get("DATASET_ROOT", "/data3/local_datasets/re10k")))
    p.add_argument("--launcher", choices=("python", "srun"), default="python")
    p.add_argument("--resume", action="store_true", help="explicitly resume this stage from its last checkpoint")
    p.add_argument("--audit-max-steps", type=int, help="Gate 0 only; at most 2000 optimizer steps")
    p.add_argument("--audit-trace", type=Path)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    a.rate_lambda = rate_text(a.rate_lambda)
    a.preparation = (a.preparation or preparation_for(a.seed_index)).resolve()
    a.run_dir = a.run_dir.resolve()
    if a.stage == "full" and a.budget != 120000:
        p.error("the continuous Wave 2 plan requires --budget 120000")
    if a.mode == "validation" and (a.step is None or a.step < 1):
        p.error("validation requires --step")
    if a.audit_max_steps is not None and (not 1 <= a.audit_max_steps <= 2000 or not a.audit_trace or a.stage != "A"):
        p.error("Gate 0 needs stage A, --audit-trace and 1..2000 steps")
    if a.mode in ("reference", "native") and a.checkpoint is None:
        p.error("baseline evaluation requires --checkpoint")
    return a


def config_for(cell):
    from globalsplat.compression.architecture import pilot_config
    c = CELLS[cell]
    return replace(pilot_config(c.arm), token_order=c.order, use_morton=c.order != "none",
        score_spatial_context=c.context, score_context_schedule=c.schedule, score_spatial_stages=c.stages,
        nn_order_backend=c.backend, nn_order_distance_device="cpu" if c.backend == "legacy" else "input",
        score_order_metric=c.metric)


def identity(args):
    prep = args.preparation
    manifest = read_json(prep / "preparation.json")
    arm = CELLS[args.cell].arm
    result = dict(version=VERSION, key=run_key(args.cell, args.rate_lambda, args.seed_index),
        cell=args.cell, rate_lambda=args.rate_lambda, seed_index=args.seed_index,
        model_seed=SEEDS[args.seed_index][0], data_seed=SEEDS[args.seed_index][1],
        preparation=str(prep), initializer_sha256=manifest["arms"][arm]["sha256"],
        holdout_sha256=manifest["holdout_sha256"], validation_index_sha256=manifest["validation_index_sha256"],
        vanilla_sha256=manifest["vanilla_sha256"], dataset_root=str(args.dataset_root.resolve()),
        codec_config=config_for(args.cell).to_dict(),
        recipe=dict(gpus=1, micro_batch=2, accumulate=4, exact_resume=True, initial_lr=.0001,
            quantile_interval=500, train_deterministic=False, checkpoint_after_quantile=True),
        code_sha256={name: file_hash(ROOT / name) for name in RUNTIME_FILES})
    if CELLS[args.cell].selection:
        result["token_selection"] = selection_overrides(args.cell)
        result["code_sha256"]["globalsplat/compression/token_selection.py"] = file_hash(
            ROOT / "globalsplat/compression/token_selection.py")
    parent_path = args.run_dir / "resume_parent.json"
    if parent_path.exists():
        result["resume_parent"] = read_json(parent_path)
        # The repair may change only the documented runtime files, not the
        # initializer, model, data, seeds or optimization recipe.
        from globalsplat.fresh_rd_runtime import checkpoint_protocol_matches
        parent = result["resume_parent"]
        if not checkpoint_protocol_matches(dict(global_step=60000, fresh_rd_protocol=parent["protocol"]), result):
            raise ValueError("runtime repair changed fields outside the declared parent migration")
    existing = read_json(args.run_dir/"run.json") if (args.run_dir/"run.json").exists() else {}
    if args.stage == "full" or existing.get("training_plan"):
        result["training_plan"] = dict(kind="wave2_continuous_0_to_120k", budget=120000,
            lr_milestones=[84000,108000], stage_boundary_resume=False,
            wave1_difference="Wave 1 resumes at 60k; accepted resaved parents skip one microbatch (two scenes)")
    return result


def chosen_checkpoint(args):
    if args.checkpoint:
        return args.checkpoint.resolve()
    if args.resume:
        return args.run_dir / "checkpoints/nfcgs_main/version_0/last.ckpt"
    if args.mode == "train":
        return (args.preparation / f"{CELLS[args.cell].arm}.ckpt" if args.stage in ("A", "full")
                else checkpoint_path(args.run_dir, 60000))
    return checkpoint_path(args.run_dir, args.step if args.mode == "validation" else args.budget)


def eval_output(args):
    if args.output:
        return args.output.resolve()
    if args.mode == "validation":
        return args.run_dir / "validation" / f"step{args.step:09d}"
    return args.run_dir / ("eval" if args.mode == "test" else f"{args.mode}_{args.split}")


def build_command(args, attempt=None):
    train, c = args.mode == "train", CELLS[args.cell]
    checkpoint = chosen_checkpoint(args)
    output = args.run_dir if train else eval_output(args)
    # The base parser's legacy whitelist is not the fresh sweep's lambda policy.
    argv = ["train" if train else "eval", "--checkpoint", str(checkpoint), "--output", str(output),
        "--dataset-root", str(args.dataset_root), "--rate-lambda", "0",
        "--token-order", c.order, "--score-context-schedule", c.schedule, "--score-spatial-stages", str(c.stages),
        "--score-spatial-kernel", "5", "--score-slice-channels", "16"]
    if not c.context:
        argv.append("--no-spatial-context")
    if c.backend != "legacy":
        argv += ["--nn-order-backend", c.backend, "--nn-order-distance-device", "input"]
    if c.order == "nn_score":
        argv += ["--score-order-metric", c.metric]
    resume = train and (args.stage == "B" or args.resume)
    if train:
        argv += ["--devices", "1", "--launcher", args.launcher, "--scope", "all", "--lr", ".0001",
            "--batch-size", "2", "--accumulate", "4", "--workers", "8", "--max-steps", "60000", "--checkpoint-every", "10000"]
        if resume:
            argv.append("--resume")
        elif converts_initializer(args.cell):
            argv.append("--allow-score-path-conversion")
    command, _, _ = base.build_command(base.parse_args(argv))
    mode = "score_only" if c.arm == "score56_only" else "msh_only" if c.arm == "msh_only" else "two_path"
    changes = [f"loss.rate_lambda={args.rate_lambda}", f"++model.feature_codec.codec_mode={mode}",
        f"model.feature_codec.rank={0 if mode == 'msh_only' else 56}",
        f"model.feature_codec.use_residual={str(mode != 'score_only').lower()}", "test.record_scene_metrics=true",
        *selection_overrides(args.cell)]
    if train:
        maximum = args.audit_max_steps or (60000 if args.stage == "A" else args.budget)
        steps = milestones(args.budget) if args.stage in ("B", "full") else milestones(100000)
        changes += [f"trainer.max_steps={maximum}", f"optimizer.lr_milestones={steps}".replace(" ", ""),
            "checkpointing.exact_resume=true", "checkpointing.record_lr=true", "trainer.log_every_n_steps=1",
            f"checkpointing.override_lr_milestones={str(args.stage == 'B' and not args.resume and args.budget != 100000).lower()}",
            f"seed={SEEDS[args.seed_index][0]}", f"dataset.data_loader_seed={SEEDS[args.seed_index][1]}",
            f"++dataset.train_holdout_manifest={(args.preparation/'holdout.json').as_posix()}",
            "trainer.limit_val_batches=0", "trainer.num_sanity_val_steps=0",
            f"++fresh_rd_manifest={(args.run_dir/'run.json').as_posix()}",
            f"hydra.run.dir={(attempt or args.run_dir/('hydra_'+args.stage)).as_posix()}"]
        if args.audit_max_steps:
            changes += [f"checkpointing.every_n_train_steps={args.audit_max_steps}",
                f"++fresh_rd_audit_trace={args.audit_trace.resolve().as_posix()}"]
    else:
        changes += ["++architecture_entropy_determinism=true", "seed=0", "++dataset.data_loader_seed=0"]
        if args.mode in ("validation", "reference") or args.mode == "native" and args.split == "validation":
            changes += [f"dataset.eval_index_path={(args.preparation/'validation_index.json').as_posix()}",
                "dataset.eval_all_scenes=false",
                f"++dataset.eval_train_holdout_manifest={(args.preparation/'holdout.json').as_posix()}"]
        if args.mode == "native":
            changes += ["model.feature_codec=null", "model.dim_latents=512", "model.freeze_globalsplat=false",
                        "test.actual_bitstream=false", "loss.rate_lambda=0"]
    return canonical_overrides(command, changes), checkpoint


def compose_command(command):
    from hydra import compose, initialize_config_dir
    i = command.index("globalsplat.main")
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "config")):
        return compose(config_name="main", overrides=command[i + 1:])


def validate_training_config(cfg, args):
    from omegaconf import OmegaConf
    expected = {"trainer.devices": 1, "optimizer.batch_size": 2, "trainer.accumulate_grad_batches": 4,
        "optimizer.name": "adam", "optimizer.weight_decay": 0., "optimizer.lr": .0001,
        "optimizer.quantile_training": "deterministic", "loss.quantile_update_interval": 500,
        "loss.subset_consistency": True, "dataset.num_context_views": 24, "dataset.num_target_views": 12,
        "model.freeze_globalsplat": True, "curriculum.final_stage": 3, "checkpointing.exact_resume": True,
        "model.latent_rep_token_amount": 4096, "model.dim_latents": 512,
        "model.feature_codec_train_scope": "all", "curriculum.boundaries": [0,0,0], "curriculum.ramp_iters": 0,
        "loss.gaussian_attributes": 59, "loss.w_mse": 1., "loss.w_lpips": .05, "loss.rate_ramp_steps": 0,
        "dataset.context_gap_warm_up_steps": 0, "dataset.target_gap_warm_up_steps": 0,
        "seed": SEEDS[args.seed_index][0], "dataset.data_loader_seed": SEEDS[args.seed_index][1],
        "loss.rate_lambda": float(args.rate_lambda), "trainer.precision": "bf16-mixed",
        "trainer.gradient_clip_val": .5, "trainer.limit_val_batches": 0}
    for key, value in expected.items():
        if OmegaConf.select(cfg, key) != value:
            raise ValueError(f"fresh training config mismatch: {key}")
    from globalsplat.compression.config import CodecConfig
    actual_codec = CodecConfig.from_mapping(OmegaConf.to_container(cfg.model.feature_codec, resolve=True)).to_dict()
    for key, value in config_for(args.cell).to_dict().items():
        if actual_codec.get(key) != value:
            raise ValueError(f"fresh codec config mismatch: {key}")
    if cfg.get("architecture_entropy_determinism", False):
        raise ValueError("training must retain the non-deterministic numerical policy")


def validate_preparation(args):
    from globalsplat.dataset.holdout import read_holdout
    prep = args.preparation
    manifest = read_json(prep / "preparation.json")
    if manifest.get("kind") != "nfcgs_architecture_preparation_v1" or not manifest.get("shared_initialization_checked"):
        raise ValueError("incomplete preparation")
    if manifest["seed"] != SEEDS[args.seed_index][0]:
        raise ValueError("initializer model seed mismatch")
    for name, key in (("holdout.json", "holdout_sha256"), ("validation_index.json", "validation_index_sha256")):
        if file_hash(prep / name) != manifest[key]:
            raise ValueError("frozen holdout artifact changed")
    if args.seed_index and file_hash(prep / "holdout.json") != file_hash(preparation_for(0) / "holdout.json"):
        raise ValueError("seed preparations must use exactly the same holdout")
    read_holdout(prep / "holdout.json", args.dataset_root)
    return manifest


def validate_checkpoint(args, checkpoint, protocol=None):
    import torch
    from globalsplat.compression.checkpoint import validate_feature_codec_checkpoint
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    initial = args.mode == "train" and args.stage in ("A", "full") and not args.resume
    validate_feature_codec_checkpoint(saved, config_for(args.cell),
        allow_score_path_conversion=initial and converts_initializer(args.cell))
    if initial:
        arm = CELLS[args.cell].arm
        manifest = read_json(args.preparation / "preparation.json")
        if saved.get("global_step", 0) != 0 or any(k in saved for k in ("optimizer_states", "lr_schedulers")):
            raise ValueError("fresh training requires a step-zero initializer without optimizer state")
        if file_hash(checkpoint) != manifest["arms"][arm]["sha256"]:
            raise ValueError("initializer hash differs from preparation")
        signature = saved.get("architecture_pilot_initialization", {}).get("signature", {})
        if any(signature.get(k) != manifest[k] for k in ("seed", "holdout_sha256", "vanilla_sha256")) or signature.get("arm") != arm:
            raise ValueError("initializer provenance mismatch")
    else:
        if saved.get("fresh_rd_protocol") != protocol:
            from globalsplat.fresh_rd_runtime import checkpoint_protocol_matches
            parent = (protocol or {}).get("resume_parent", {})
            inherited_eval = (args.mode == "validation" and saved.get("fresh_rd_protocol") == parent.get("protocol")
                              and args.step in range(10000,60001,10000))
            if not inherited_eval and not checkpoint_protocol_matches(saved, protocol):
                raise ValueError("checkpoint is not from this exact fresh run")
            if parent.get("checkpoints_sha256", {}).get(str(saved.get("global_step"))) != file_hash(checkpoint):
                raise ValueError("runtime migration parent checkpoint bytes differ from the frozen lineage")
        from omegaconf import OmegaConf
        if not saved.get("fresh_rd_training_config"):
            raise ValueError("actual training config is missing from checkpoint")
        training_cfg = OmegaConf.create(saved["fresh_rd_training_config"])
        validate_training_config(training_cfg, args)
        step = saved.get("global_step", -1)
        if args.mode == "train":
            target = args.audit_max_steps or (60000 if args.stage == "A" else args.budget)
            if not args.resume and args.stage == "B" and step != 60000:
                raise ValueError("stage B starts only at the same run's 60k checkpoint")
            if not 0 < step < target or args.stage == "B" and step < 60000:
                raise ValueError("resume checkpoint step outside this stage")
            if not saved.get("optimizer_states") or not saved.get("lr_schedulers"):
                raise ValueError("full optimizer/scheduler state required")
            if not saved.get("loops", {}).get("fit_loop", {}).get("state_dict", {}).get("combined_loader"):
                raise ValueError("exact-resume dataloader state absent")
            expected = milestones(args.budget if args.stage == "full" or args.resume and args.stage == "B" else 100000)
            if sorted(saved["lr_schedulers"][0]["milestones"]) != expected:
                raise ValueError("saved LR milestones differ from this stage")
        elif step != (args.step if args.mode == "validation" else args.budget):
            raise ValueError("evaluation checkpoint is not the requested saved step")
    return saved


def test_cohort(args, rows):
    """Freeze frame identity without requiring any diagnostic to finish first."""
    root = args.run_dir.parents[3]
    path = root / "test_frame_cohort.json"
    if path.exists():
        return read_json(path)["scenes"]
    native = root / "gate0/native/test/nfcgs_main/scene_metrics.json"
    legacy = ROOT / ("outputs/nfcgs_order_context/hilbert/dyadic4_stages4_k5_ste/"
        "lambda0p0128/job_447746/eval_all/nfcgs_main/actual_rate_per_scene.json")
    source = next((p for p in (native, legacy) if p.is_file()), None)
    reference = read_json(source) if source else rows
    if len(reference) != 6991 or len({r["scene"] for r in reference}) != 6991:
        raise ValueError("test frame reference must contain 6991 unique scenes")
    cohort = {r["scene"]:{"context":r["context_frame_ids"], "target":r["target_frame_ids"]} for r in reference}
    # Parallel evals may finish together. The lock protects only this small
    # registry write; a later evaluation always compares with the frozen file.
    import time
    for attempt in range(40):
        try:
            with directory_lock(root, ".test_cohort.lock"):
                if not path.exists():
                    atomic_json(path, dict(scenes=cohort, selection_uses_quality=False,
                        source=str(source) if source else "first completed fresh full-test evaluation",
                        source_sha256=file_hash(source) if source else None))
                return read_json(path)["scenes"]
        except FileExistsError:
            if attempt == 39:
                raise
            time.sleep(.25)


def main(argv=None):
    args = parse_args(argv)
    command, checkpoint = build_command(args)
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    if args.launcher == "srun" and (int(os.environ.get("SLURM_NTASKS", "0")) != 1):
        raise ValueError("one GPU and one task are required")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if args.mode == "native":
        from eval_nfcgs_b0_matched import NATIVE_SHA256
        if file_hash(checkpoint) != NATIVE_SHA256:
            raise ValueError("native baseline must use the released checkpoint")
    validate_preparation(args)
    protocol = None
    if args.mode not in ("native", "reference"):
        protocol = identity(args)
        if (args.run_dir / "run.json").exists():
            if read_json(args.run_dir / "run.json") != protocol:
                raise ValueError("run protocol/code/preparation changed; do not overwrite this run")
        elif args.mode != "train" or args.stage not in ("A", "full") or args.resume:
            raise ValueError("existing fresh run manifest required")
        saved = validate_checkpoint(args, checkpoint, protocol)
        del saved
    elif args.mode == "reference":
        import torch
        from globalsplat.compression.checkpoint import validate_feature_codec_checkpoint
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        validate_feature_codec_checkpoint(saved, config_for(args.cell))
        if saved.get("global_step") != 50000:
            raise ValueError("reference baseline requires the final legacy 50k checkpoint")
        del saved
    destination = args.run_dir if args.mode == "train" else eval_output(args)
    with directory_lock(destination):
        digest = file_hash(checkpoint)
        if args.mode == "train":
            if not (args.run_dir / "run.json").exists():
                if list((args.run_dir / "checkpoints").glob("**/*.ckpt")):
                    raise ValueError("refusing to start over existing checkpoints")
                atomic_json(args.run_dir / "run.json", protocol)
            elif args.stage in ("A", "full") and not args.resume:
                raise ValueError("run already initialized; use explicit --resume for an interrupted stage")
            attempt = args.run_dir / "attempts" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + f"_{os.getpid()}")
            command, _ = build_command(args, attempt / "hydra")
            validate_training_config(compose_command(command), args)
        else:
            attempt = destination
            complete = destination / "complete.json"
            if complete.exists():
                info = read_json(complete)
                expected = 6991 if args.mode == "test" or args.mode == "native" and args.split == "test" else 128
                if (info["checkpoint_sha256"] == digest and info["scene_count"] == expected and info["mode"] == args.mode
                        and file_hash(destination / "nfcgs_main/scene_metrics.json") == info["scene_metrics_sha256"]):
                    print(f"REUSING_COMPLETE={destination}", flush=True)
                    return
                raise ValueError("evaluation output belongs to a different checkpoint")
        atomic_json(attempt / "invocation.json", dict(command=command, checkpoint=str(checkpoint), checkpoint_sha256=digest,
            stage=args.stage, budget=args.budget, resume=args.resume, mode=args.mode))
        subprocess.run(command, cwd=ROOT, check=True)
        if args.mode == "train":
            target = args.audit_max_steps or (60000 if args.stage == "A" else args.budget)
            final = checkpoint_path(args.run_dir, target)
            import torch
            result = torch.load(final, map_location="cpu", weights_only=False)
            if result.get("global_step") != target or result.get("fresh_rd_protocol") != protocol:
                raise ValueError("training did not produce the requested final checkpoint")
            completion = dict(step=target, budget=args.budget, checkpoint=str(final),
                checkpoint_sha256=file_hash(final), invocation=str(attempt / "invocation.json"))
            if args.stage == "A" and target == 60000:
                completion.update(freeze_stage_a_checkpoint(final))
            atomic_json(args.run_dir / f"stage_{args.stage}_complete.json", completion)
        else:
            path = destination / "nfcgs_main/scene_metrics.json"
            rows = read_json(path)
            expected = 6991 if args.mode == "test" or args.mode == "native" and args.split == "test" else 128
            validate_rows(rows, expected, rate_required=args.mode != "native")
            if expected == 128:
                scenes = read_json(args.preparation / "holdout.json")["scenes"]
                cohort = {name:{"context":r["context"], "target":r["target"]} for name,r in scenes.items()}
            elif args.mode == "test":
                cohort = test_cohort(args, rows)
            else:
                cohort = {r["scene"]:{"context":r["context_frame_ids"], "target":r["target_frame_ids"]} for r in rows}
            cohort_sha256 = validate_cohort(rows, cohort)
            if file_hash(checkpoint) != digest:
                raise ValueError("checkpoint changed during evaluation")
            atomic_json(destination / "complete.json", dict(scene_count=expected, checkpoint_sha256=digest,
                scene_metrics_sha256=file_hash(path), frame_cohort_sha256=cohort_sha256,
                mode=args.mode, split="test" if expected == 6991 else "validation"))


if __name__ == "__main__":
    main()
