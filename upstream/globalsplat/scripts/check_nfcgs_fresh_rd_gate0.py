#!/usr/bin/env python3
"""Executable Gate 0: initializer hashes, native boundary, eval/coder parity and real resume."""
from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_nfcgs_fresh_rd as runner
from globalsplat.fresh_rd import CELLS, atomic_json, checkpoint_path, converts_initializer, file_hash, read_json


def tensor_error(a, b):
    import torch
    delta = a.detach().float() - b.detach().float()
    return dict(max_abs=float(delta.abs().max()), mse=float(delta.square().mean()),
        relative_l2=float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(b.detach().float()).clamp_min(1e-12)))


def check_initializers(preparation):
    import torch
    from globalsplat.compression.architecture import tensor_hash
    from globalsplat.compression.codec import ObservableLowRank1DCodec
    from globalsplat.compression.checkpoint import (convert_score_path_state, resize_registered_buffers,
                                                    strip_retired_state, validate_feature_codec_checkpoint)
    manifest = read_json(preparation / "preparation.json")
    result = {}
    common = {}
    for name, cell in CELLS.items():
        path = preparation / f"{cell.arm}.ckpt"
        if file_hash(path) != manifest["arms"][cell.arm]["sha256"]:
            raise ValueError("initializer artifact changed")
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved.get("global_step", 0) != 0 or "optimizer_states" in saved:
            raise ValueError("initializer is not fresh")
        cfg = runner.config_for(name)
        validate_feature_codec_checkpoint(saved, cfg, allow_score_path_conversion=converts_initializer(name))
        torch.manual_seed(manifest["seed"])
        codec = ObservableLowRank1DCodec(cfg)
        source = strip_retired_state(saved["state_dict"])
        converted = convert_score_path_state(source, codec) if cell.arm == "two_path56" else source
        prefix = "model.feature_codec."
        state = {k[len(prefix):]: v for k, v in converted.items() if k.startswith(prefix)}
        resize_registered_buffers(codec, state)
        codec.load_state_dict(state, strict=True)
        copied, created = {}, {}
        for key, value in codec.state_dict().items():
            original = source.get(prefix + key)
            digest = tensor_hash(value)
            if original is not None and original.shape == value.shape:
                if tensor_hash(original) != digest:
                    raise ValueError(f"common initializer tensor was changed: {name}/{key}")
                copied[key] = digest
            else:
                created[key] = dict(sha256=digest, source="seeded target codec constructor", seed=manifest["seed"])
            if name in ("h_on", "h_off", "n_on", "n_off", "m_on", "h_l3"):
                if key in common and common[key] != digest:
                    raise ValueError(f"shared two-path tensors differ: {key}")
                common[key] = digest
        result[name] = dict(copied_tensor_sha256=copied, constructor_tensors=created,
            checkpoint_sha256=file_hash(path), codec_config=cfg.to_dict())
        del codec, saved, state, source, converted
        gc.collect()
    return dict(passed=True, cells=result, shared_two_path_tensors_checked=True)


def load_integrated(path, dataset_root, workers=4):
    import torch
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from globalsplat.main import build_model
    from globalsplat.compression import CodecConfig, resize_registered_buffers
    from globalsplat.compression.checkpoint import strip_retired_state, validate_feature_codec_checkpoint
    saved = torch.load(path, map_location="cpu", weights_only=False)
    config = CodecConfig.from_mapping(saved["feature_codec_config"])
    validate_feature_codec_checkpoint(saved, config)
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "config")):
        cfg = compose(config_name="main", overrides=["+experiment=re10k_32k_nfcgs", "dataset=re10k_eval_all_ctx12"])
    OmegaConf.set_struct(cfg, False)
    cfg.model.feature_codec = OmegaConf.create(config.to_dict())
    cfg.model.dim_latents = config.geometry_channels
    cfg.model.freeze_globalsplat = True
    cfg.dataset.dataset_roots = [str(dataset_root)]
    cfg.dataset.mvsplat_root = str(ROOT / "third_party/ZPressor/mvsplat")
    cfg.dataset.augment = False
    cfg.dataset.data_loader_seed = 0
    cfg.optimizer.num_workers = workers
    cfg.optimizer.batch_size = 1
    model = build_model(cfg.model)
    state = {k[len("model."):]: v for k, v in strip_retired_state(saved["state_dict"]).items() if k.startswith("model.")}
    resize_registered_buffers(model, state)
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False).set_stage(3, mix=1.)
    model.feature_codec.update(force=True, update_quantiles=False)
    return model, cfg


def compare_codec(codec, texture, geometry, positions):
    """Keep both paths under their caller's production autocast policy."""
    import torch
    from globalsplat.compression.bitstream import ResidualBitstream, SceneBitstream
    features, order = codec._ordered_features(codec._pack_features(texture, geometry), positions)
    score = codec._analyze_low_rank(features)
    score = score.transpose(1, 2).unsqueeze(2)
    f_score, _ = codec._score_forward(score, training=False)
    packet = codec.compress(texture, geometry, positions)
    scene = SceneBitstream.unpack(packet.data)
    with torch.autocast(device_type=texture.device.type, enabled=False):
        d_score = codec.score_context.decompress(scene.score, scene.points)
    f_lr = codec._synthesize_low_rank(f_score.squeeze(2).transpose(1, 2))
    d_lr = codec._synthesize_low_rank(d_score.squeeze(2).transpose(1, 2))
    f_residual, d_residual = features - f_lr, features - d_lr
    normalized = (f_residual.transpose(1, 2).unsqueeze(2) - codec.residual_mean) / codec.residual_std.clamp_min(1e-8)
    output = codec.residual_codec(normalized, training=False)
    packed = ResidualBitstream.unpack(scene.residual)
    msh = codec.residual_codec
    z_hat = msh.entropy_bottleneck.decompress(list(packed.z_strings), packed.z_shape)
    scales, means = msh._parameters_from_z(z_hat, packed.y_shape)
    y_hat = msh.gaussian_conditional.decompress(list(packed.y_strings), msh.gaussian_conditional.build_indexes(scales), means=means)
    f_features, _ = codec._analyze_sorted(features, training=False)
    d_texture, d_geometry = codec.decompress(packet.data)
    d_features = torch.cat((d_texture, d_geometry), dim=-1)
    actual_forward = codec(texture, geometry, positions, restore_original_order=False, training=False).reconstruction_sorted
    errors = {"score": tensor_error(f_score, d_score), "low_rank": tensor_error(f_lr, d_lr),
        "residual_input": tensor_error(f_residual, d_residual), "msh_z": tensor_error(output.z_hat, z_hat),
        "msh_y": tensor_error(output.y_hat, y_hat), "msh_reconstruction": tensor_error(output.reconstruction, msh.decompress(scene.residual, output_width=scene.points)),
        "final_feature": tensor_error(f_features, d_features), "forward_reproduction": tensor_error(actual_forward, f_features)}
    width = codec.config.texture_channels
    return errors, (f_features[..., :width], f_features[..., width:]), (d_texture, d_geometry)


def boundary_or_forward(args):
    import torch
    import eval_nfcgs_b0_matched as b0
    import eval_nfcgs_path_diagnostics as diag
    from globalsplat.main import build_datamodule
    torch.manual_seed(0)
    torch.set_num_threads(8)
    selected = args.preparation / "two_path56.ckpt" if args.task == "boundary" else args.reference_checkpoint
    model, cfg = load_integrated(selected, args.dataset_root)
    cfg.dataset.eval_index_path = str(args.preparation / "validation_index.json")
    cfg.dataset.eval_all_scenes = False
    cfg.dataset.eval_train_holdout_manifest = str(args.preparation / "holdout.json")
    data, _ = build_datamodule(cfg)
    model = model.to(args.device)
    native = None
    operators = None
    if args.task == "boundary":
        native, _ = b0.load_native(args)
        operators = b0.validate_model_pair(native, model)
    b0.numerical_policy()
    render, metrics = diag.render_functions()
    rows = []
    with torch.no_grad():
        for batch in data.test_dataloader():
            diag.validate_views(batch)
            inputs = diag.profile.to_device(batch["inputs"], args.device)
            targets = diag.profile.to_device(batch["targets"], args.device)
            gt = targets["images"][0].float().clamp(0, 1)
            with diag.amp(args):
                texture, geometry = model.encode_scene_tokens(inputs)
                if native is not None:
                    a = native.decode_scene_tokens((texture, geometry))
                    b = model.decode_scene_tokens((texture, model.feature_codec.project_geometry(geometry)))
                    errors = {k: tensor_error(getattr(b, k), getattr(a, k)) for k in b0.GAUSSIAN_FIELDS}
                else:
                    xyz = model.gaussian_decoder.decoded_token_centers(model.feature_codec.project_geometry(geometry))
                    errors, f_tokens, d_tokens = compare_codec(model.feature_codec, texture, geometry, xyz)
                    a, b = model.decode_scene_tokens(f_tokens), model.decode_scene_tokens(d_tokens)
                    errors["gaussian"] = {k: tensor_error(getattr(a, k), getattr(b, k)) for k in b0.GAUSSIAN_FIELDS}
            a_metrics, b_metrics = metrics(gt, render(a, targets)), metrics(gt, render(b, targets))
            row = dict(scene=batch["scene_info"]["scene"][0], errors=errors, first=a_metrics, second=b_metrics,
                psnr_delta=b_metrics["psnr"] - a_metrics["psnr"])
            rows.append(row)
            if len(rows) == args.scenes:
                break
    if len(rows) != args.scenes:
        raise ValueError("Gate 0 scene cohort incomplete")
    passed = all(math.isfinite(r["psnr_delta"]) and abs(r["psnr_delta"]) <= args.psnr_tolerance for r in rows)
    if operators:
        passed &= all(v["max_abs"] <= 1e-4 for v in operators["geometry_boundary_effective_readout_errors"].values())
    else:
        passed &= all(e["relative_l2"] <= args.feature_tolerance for r in rows
            for name,e in r["errors"].items() if name != "gaussian")
    return dict(passed=bool(passed), task=args.task, checkpoint=str(selected), checkpoint_sha256=file_hash(selected),
        precision=args.precision, psnr_tolerance=args.psnr_tolerance, feature_relative_tolerance=args.feature_tolerance,
        operators=operators, scenes=rows, code_sha256={n:file_hash(ROOT/n) for n in runner.RUNTIME_FILES})


def compare_traces(continuous, split):
    if len(continuous) != len(split):
        raise ValueError("resume trace lengths differ")
    fields = ("step", "end_step", "scene", "context", "target", "subset_a", "subset_b", "shared_target", "valid", "lr")
    for i, (a, b) in enumerate(zip(continuous, split)):
        if any(a.get(k) != b.get(k) for k in fields):
            raise ValueError(f"resumed sample/frame/subset/LR differs at microbatch {i}")
        if not all(math.isfinite(float(r["loss"])) for r in (a, b)):
            raise ValueError("non-finite resume audit loss")
    return dict(sample_frames_subsets_lr_equal=True, microbatches=len(continuous),
        max_loss_abs_delta=max(abs(a["loss"] - b["loss"]) for a, b in zip(continuous, split)))


def compare_training_states(a, b):
    """CDF buffer lengths can change across non-bit-exact training runs."""
    import torch
    if set(a) != set(b):
        raise ValueError("resumed model state keys differ")
    maximum, exact, buffers = 0., True, []
    for key, value in a.items():
        other = b[key]
        if value.shape != other.shape:
            if not key.endswith(("._quantized_cdf", "._cdf_length", "._offset")):
                raise ValueError(f"resumed model parameter/buffer shape differs: {key}")
            buffers.append(dict(name=key, continuous_shape=list(value.shape), resumed_shape=list(other.shape)))
            exact = False
            continue
        exact &= torch.equal(value, other)
        if value.numel() and torch.is_floating_point(value):
            maximum = max(maximum, tensor_error(value, other)["max_abs"])
    return dict(model_max_abs_delta=maximum, training_bit_exact=bool(exact), entropy_buffer_shape_differences=buffers)


def resume(args):
    import torch
    if args.resume_steps % 2:
        raise ValueError("resume audit step count must be even")
    commands = []
    for label, steps, resume_flag in (("continuous", args.resume_steps, False), ("split", args.resume_steps//2, False), ("split", args.resume_steps, True)):
        run = args.output / label
        trace = run / "trace.jsonl"
        command = [sys.executable, str(ROOT / "scripts/run_nfcgs_fresh_rd.py"), "train", "--cell", "h_on",
            "--run-dir", str(run), "--preparation", str(args.preparation), "--dataset-root", str(args.dataset_root),
            "--audit-max-steps", str(steps), "--audit-trace", str(trace)]
        if resume_flag:
            command += ["--resume"]
        commands.append(command)
        marker = run / f"audit_{steps}_complete.json"
        if not marker.exists():
            subprocess.run(command, cwd=ROOT, check=True)
            atomic_json(marker, dict(step=steps, checkpoint_sha256=file_hash(checkpoint_path(run, steps))))
    import json
    traces = [[json.loads(line) for line in (args.output / label / "trace.jsonl").read_text().splitlines()] for label in ("continuous", "split")]
    comparison = compare_traces(*traces)
    a = torch.load(checkpoint_path(args.output/"continuous", args.resume_steps), map_location="cpu", weights_only=False)
    b = torch.load(checkpoint_path(args.output/"split", args.resume_steps), map_location="cpu", weights_only=False)
    return dict(passed=True, task="resume", optimizer_steps=args.resume_steps, commands=commands,
        **comparison, **compare_training_states(a["state_dict"], b["state_dict"]))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("task", choices=("initialization", "boundary", "forward", "native", "resume", "collect"))
    p.add_argument("--root", type=Path, default=ROOT / "outputs/nfcgs_fresh_rd")
    p.add_argument("--output", type=Path)
    p.add_argument("--preparation", type=Path, default=runner.preparation_for(0))
    p.add_argument("--native-checkpoint", type=Path, default=ROOT / "checkpoints/pretrained/globalsplat-re10k-32k.ckpt")
    p.add_argument("--reference-checkpoint", type=Path)
    p.add_argument("--dataset-root", type=Path, default=Path("/data3/local_datasets/re10k"))
    p.add_argument("--scenes", type=int, default=4)
    p.add_argument("--resume-steps", type=int, default=2000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    p.add_argument("--psnr-tolerance", type=float, default=.1)
    p.add_argument("--feature-tolerance", type=float, default=1e-5)
    a = p.parse_args(argv)
    a.tokens = 4096
    if a.reference_checkpoint is None:
        from monitor_nfcgs_fresh_rd import reference_checkpoint
        a.reference_checkpoint = reference_checkpoint(".0128")
    a.output = a.output or a.root / "gate0" / a.task
    a.output.mkdir(parents=True, exist_ok=True)
    if a.task == "initialization":
        import torch
        torch.set_num_threads(8)
        result = check_initializers(a.preparation)
    elif a.task in ("boundary", "forward"):
        result = boundary_or_forward(a)
    elif a.task == "resume":
        result = resume(a)
    elif a.task == "native":
        for split in ("validation", "test"):
            runner.main(["native", "--checkpoint", str(a.native_checkpoint), "--run-dir", str(a.output),
                "--output", str(a.output / split), "--preparation", str(a.preparation), "--dataset-root", str(a.dataset_root), "--split", split])
        result = dict(passed=True, validation=read_json(a.output/"validation/complete.json"), test=read_json(a.output/"test/complete.json"))
    else:
        reports = {task:read_json(a.root/"gate0"/task/"complete.json") for task in ("initialization", "boundary", "forward", "native", "resume")}
        result = dict(passed=all(r.get("passed") is True for r in reports.values()), reports=reports,
            code_sha256={n:file_hash(ROOT/n) for n in runner.RUNTIME_FILES},
            preparation_sha256=file_hash(a.preparation/"preparation.json"),
            holdout_sha256=file_hash(a.preparation/"holdout.json"),
            native_sha256=file_hash(a.native_checkpoint), dataset_root=str(a.dataset_root.resolve()),
            thresholds=dict(psnr_abs=.1, feature_relative_l2=1e-5))
        a.output = a.root / "gate0"
    atomic_json(a.output / "complete.json", result)
    if not result["passed"]:
        raise RuntimeError(f"Gate 0 {a.task} failed; inspect {a.output}/complete.json")
    print(f"GATE0_PASS={a.task}", flush=True)


if __name__ == "__main__":
    main()
