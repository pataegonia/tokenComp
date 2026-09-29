#!/usr/bin/env python3
"""Evaluate archived Nonlinear32 50k on a reproducible test-minus-validation subset."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys

REPO = Path(__file__).resolve().parents[1]
ARCHIVE = REPO / "archive/codec_experiments_20260915/globalsplat/compression"
CONTEXT_VIEWS = 12
TARGET_VIEWS = 8


def read_scene_ids(path: Path) -> set[str]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(data, dict):
        for key in ("validation_scene_ids", "scene_ids"):
            if key in data:
                data = data[key]
                break
        else:
            data = list(data)
    if not isinstance(data, list):
        raise ValueError("validation JSON must be an ID list, scene-keyed index, or scene records")
    ids = [row.get("scene") if isinstance(row, dict) else row for row in data]
    if not ids or any(not isinstance(scene, str) or not scene for scene in ids):
        raise ValueError("validation JSON must contain nonempty scene IDs")
    return set(ids)


def select_scenes(test_index: dict, validation_ids: set[str], count: int, seed: int, *, rng=None):
    pool = sorted(set(test_index) - validation_ids)
    if count <= 0 or count > len(pool):
        raise ValueError(f"cannot select {count} scenes from a pool of {len(pool)}")
    return sorted((rng or random.Random(seed)).sample(pool, count))


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def activate_archived_codec():
    """Use the historical codec only in this evaluator's Python process."""
    sys.path.insert(0, str(REPO))
    import globalsplat

    if "globalsplat.compression" in sys.modules:
        raise RuntimeError("archive must be activated before importing the current codec")
    spec = importlib.util.spec_from_file_location(
        "globalsplat.compression", ARCHIVE / "__init__.py",
        submodule_search_locations=[str(ARCHIVE)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    globalsplat.compression = module


def build_frame_index(test_root: Path, test_index: dict, scenes: list[str], *, rejected=None):
    """Use the same deterministic 12-context/8-target sampler as full-scene eval."""
    import torch
    from io import BytesIO
    from PIL import Image
    from globalsplat.dataset.data_module import add_repo_to_path
    add_repo_to_path(REPO / "third_party/ZPressor/mvsplat", repo_name="mvsplat")
    from src.dataset.dataset_re10k import DatasetRE10k
    from src.geometry.projection import get_fov
    from src.dataset.view_sampler.view_sampler_deterministic_all import (
        ViewSamplerDeterministicAll, ViewSamplerDeterministicAllCfg,
    )

    sampler = ViewSamplerDeterministicAll(
        ViewSamplerDeterministicAllCfg("deterministic_all", CONTEXT_VIEWS, TARGET_VIEWS),
        "test", False, False, None,
    )
    chunks = defaultdict(set)
    for scene in scenes:
        chunk = test_index[scene]
        if not isinstance(chunk, str):
            raise ValueError("test/index.json must map scene IDs to chunk filenames")
        chunks[test_root / chunk].add(scene)
    frames = {}
    rejected = {} if rejected is None else rejected
    found = set()
    for chunk, wanted in sorted(chunks.items()):
        examples = torch.load(chunk, map_location="cpu", weights_only=False)
        for example in examples:
            scene = example["key"]
            if scene not in wanted:
                continue
            if scene in found:
                raise ValueError(f"duplicate scene in chunks: {scene}")
            found.add(scene)
            num_frames = int(example["cameras"].shape[0])
            if num_frames < CONTEXT_VIEWS + TARGET_VIEWS:
                rejected[scene] = {"scene": scene, "reason": "insufficient_frames",
                                   "num_frames": num_frames,
                                   "required_frames": CONTEXT_VIEWS + TARGET_VIEWS}
                continue
            extrinsics, intrinsics = DatasetRE10k.convert_poses(None, example["cameras"])
            # Match the upstream test loader's max_fov=100 and skip_bad_shape.
            if (get_fov(intrinsics).rad2deg() > 100.0).any():
                rejected[scene] = {"scene": scene, "reason": "field_of_view_above_100_degrees"}
                continue
            context, target = sampler.sample(scene, extrinsics, intrinsics)
            for frame in (*context.tolist(), *target.tolist()):
                if frame >= len(example["images"]):
                    rejected[scene] = {"scene": scene, "reason": "missing_image", "frame": frame}
                    break
                with Image.open(BytesIO(example["images"][frame].numpy().tobytes())) as image:
                    if image.size != (640, 360) or len(image.getbands()) != 3:
                        rejected[scene] = {"scene": scene, "reason": "invalid_image_shape",
                                           "frame": frame, "size": list(image.size), "mode": image.mode}
                        break
            if scene in rejected:
                continue
            frames[scene] = {"context": context.tolist(), "target": target.tolist()}
        del examples
    missing = set(scenes) - found
    if missing:
        raise ValueError(f"sampled scenes missing from their chunks: {sorted(missing)}")
    return {scene: frames[scene] for scene in scenes if scene in frames}


def select_processable_scenes(test_root, test_index, validation_ids, count, seed):
    """Keep the original valid sample and replace rejected scenes deterministically."""
    rng = random.Random(seed)
    pending = select_scenes(test_index, validation_ids, count, seed, rng=rng)
    initial = list(pending)
    pool = sorted(set(test_index) - validation_ids)
    attempted = set()
    frames, rejected = {}, {}
    while len(frames) < count:
        attempted.update(pending)
        before = set(rejected)
        frames.update(build_frame_index(test_root, test_index, pending, rejected=rejected))
        for scene in sorted(set(rejected) - before):
            print(f"Excluded {scene}: {rejected[scene]}", flush=True)
        needed = count - len(frames)
        if needed == 0:
            break
        remaining = [scene for scene in pool if scene not in attempted]
        if len(remaining) < needed:
            raise ValueError(f"cannot fill {count} processable test scenes: "
                             f"{len(frames)} valid, {len(rejected)} rejected, "
                             f"{len(remaining)} untested scenes left")
        pending = sorted(rng.sample(remaining, needed))
    return ({scene: frames[scene] for scene in sorted(frames)},
            {"initial_scene_ids": initial,
             "attempted_scene_count": len(attempted),
             "rejected_scenes": [rejected[scene] for scene in sorted(rejected)]})


def to_device(value, device):
    import torch
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(to_device(item, device) for item in value)
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation-scenes", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("/data3/local_datasets/re10k"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=111123)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--selection-only", action="store_true")
    args = parser.parse_args(argv)
    if args.workers < 0:
        parser.error("--workers must be nonnegative")
    test_root = args.dataset_root.resolve() / "test"
    index_path = test_root / "index.json"
    test_index = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(test_index, dict):
        raise ValueError("test index must be an object keyed by scene ID")
    validation_ids = read_scene_ids(args.validation_scenes)
    # Refuse to overwrite a completed or partial evaluation.
    args.output.mkdir(parents=True, exist_ok=False)
    frames, selection_details = select_processable_scenes(
        test_root, test_index, validation_ids, args.count, args.seed,
    )
    scenes = list(frames)
    frame_path = (args.output / "eval_index.json").resolve()
    write_json(frame_path, frames)
    write_json(args.output / "scene_ids.json", scenes)
    metadata = {
        "checkpoint": str(args.checkpoint.resolve()), "expected_step": 50000,
        "test_index": str(index_path),
        "test_index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
        "validation_source": str(args.validation_scenes.resolve()),
        "validation_sha256": hashlib.sha256(args.validation_scenes.read_bytes()).hexdigest(),
        "validation_scene_ids": sorted(validation_ids),
        "excluded_test_scenes": len(set(test_index) & validation_ids),
        "pool_size": len(set(test_index) - validation_ids),
        "seed": args.seed, "scene_ids": scenes,
        "protocol": "deterministic_all_ctx12_target8", "precision": "bf16-mixed",
        "num_context_views": CONTEXT_VIEWS, "num_target_views": TARGET_VIEWS,
        "selection_version": 2, **selection_details,
    }
    write_json(args.output / "selection.json", metadata)
    print(f"Selected {len(scenes)} scenes, excluded {metadata['excluded_test_scenes']} validation scenes", flush=True)
    if args.selection_only:
        return

    activate_archived_codec()
    import torch
    import pytorch_lightning as pl
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from globalsplat.compression.config import CodecConfig
    from globalsplat.compression.checkpoint import resize_registered_buffers, validate_feature_codec_checkpoint
    from globalsplat.main import build_datamodule, build_model
    from globalsplat.model.rendering import render_static_batched

    pl.seed_everything(args.seed, workers=True)
    torch.backends.cudnn.benchmark = False
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("global_step") != 50000:
        raise ValueError(f"expected step 50000, got {checkpoint.get('global_step')}")
    # Explicitly validate the original transform experiment, not Full or Split.
    codec_config = CodecConfig(transform="nonlinear", transform_hidden=32, rank=56)
    validate_feature_codec_checkpoint(checkpoint, codec_config)
    model_state = {key.removeprefix("model."): value
                   for key, value in checkpoint["state_dict"].items() if key.startswith("model.")}
    with initialize_config_dir(version_base=None, config_dir=str(REPO / "config")):
        cfg = compose(config_name="main", overrides=[
            "+experiment=re10k_32k_nfcgs", "dataset=re10k_eval_all_ctx12",
        ])
    OmegaConf.set_struct(cfg, False)
    cfg.model.feature_codec = OmegaConf.create(codec_config.to_dict())
    cfg.model.latent_rep_token_amount = int(model_state["scene_tokens"].shape[0])
    model = build_model(cfg.model)
    resize_registered_buffers(model, model_state)
    model.load_state_dict(model_state, strict=True)
    del checkpoint, model_state
    model.requires_grad_(False).eval().to("cuda")
    model.set_stage(3, mix=1.0)
    model.feature_codec.update(force=True, update_quantiles=False)

    cfg.dataset.dataset_roots = [str(args.dataset_root.resolve())]
    cfg.dataset.mvsplat_root = str(REPO / "third_party/ZPressor/mvsplat")
    cfg.dataset.eval_index_path = str(frame_path)
    cfg.dataset.eval_all_scenes = False
    cfg.optimizer.batch_size = 1
    cfg.optimizer.num_workers = args.workers
    cfg.seed = args.seed
    data, _ = build_datamodule(cfg)
    from src.evaluation.metrics import compute_psnr, compute_ssim, compute_lpips

    records = {}
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for batch in data.test_dataloader():
            batch = to_device(batch, "cuda")
            scene = batch["scene_info"]["scene"][0]
            if scene not in frames or scene in records:
                raise ValueError(f"unexpected or duplicate scene: {scene}")
            compressed = model.compress_scene(batch["inputs"])
            gaussians = model.decompress_scene(compressed.data)
            targets = batch["targets"]
            gt = targets["images"][0].float().clamp(0, 1)
            rendered = render_static_batched(gaussians, targets, render_depth=False)["img"]
            pred = rendered.reshape_as(gt).float().clamp(0, 1)
            row = {
                "scene": scene, "psnr": float(compute_psnr(gt, pred).mean()),
                "ssim": float(compute_ssim(gt, pred).mean()),
                "lpips": float(compute_lpips(gt, pred).mean()),
                "actual_bytes": len(compressed.data),
                "context_frame_ids": frames[scene]["context"],
                "target_frame_ids": frames[scene]["target"],
            }
            records[scene] = row
            write_json(args.output / "per_scene.json", [records[key] for key in sorted(records)])
            print(f"[{len(records)}/{len(scenes)}] {scene} PSNR={row['psnr']:.4f} bytes={row['actual_bytes']}", flush=True)
            if len(records) == len(scenes):
                break
    missing = set(scenes) - records.keys()
    if missing:
        write_json(args.output / "missing_scenes.json", sorted(missing))
        raise RuntimeError(f"only {len(records)}/{len(scenes)} scenes evaluated; see missing_scenes.json")
    summary = {key: sum(row[key] for row in records.values()) / len(records)
               for key in ("psnr", "ssim", "lpips", "actual_bytes")}
    summary.update(scene_count=len(records), checkpoint=metadata["checkpoint"], global_step=50000)
    write_json(args.output / "scores_all_avg.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
