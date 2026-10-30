#!/usr/bin/env python3
"""N01 boundary metadata, N03 matched full native test, N02 saved T06 rendering.

No existing sender, codec, training recipe or diagnostic artifact is changed.
Heavy model/dataset imports occur only inside the two GPU entry points.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
import os
import tempfile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
POINT = "outputs/nfcgs_order_context/hilbert/dyadic4_stages4_k5_ste/lambda0p0128/job_447746"
METRICS = ("psnr", "ssim", "lpips")
ORDERS = ("slot", "morton", "hilbert")
FINGERPRINT = "069b40a74daef588e7ac01be40231dce159bb304b588e7669c6f4801e4241f29"
READOUTS = ("gaussian_decoder.geo_pos_readout.weight", "gaussian_decoder.geo_param_readout.weight",
            "gaussian_decoder.gate_readout.weight")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    """Write complete JSON atomically; an existing artifact must match exactly."""
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError(f"existing artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, encoding="utf-8", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def report(output, text):
    path = output / "REPORT.md"
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise ValueError("existing REPORT differs")
    else:
        path.write_text(text, encoding="utf-8", newline="\n")


def finite_metrics(row):
    if not all(math.isfinite(row[key]) for key in METRICS):
        raise ValueError("non-finite metrics")


def reference_rows(path, expected=6991):
    rows = read(path)
    if not isinstance(rows, list) or len(rows) != expected or len({r["scene"] for r in rows}) != expected:
        raise ValueError(f"reference must contain exactly {expected} unique scenes")
    for row in rows:
        c, t = row["context_frame_ids"], row["target_frame_ids"]
        if len(c) != 12 or len(set(c)) != 12 or len(t) != 8 or len(set(t)) != 8 or set(c) & set(t):
            raise ValueError("reference is not disjoint C12/T8")
        finite_metrics(row)
        if not math.isfinite(row["actual_bytes"]) or row["actual_bytes"] <= 0:
            raise ValueError("invalid actual packet rate")
    return rows


def boundary_source(args):
    b, p = read(args.b0_run/"metadata.json"), read(args.path_run/"metadata.json")
    bd, pd = read(args.b0_run/"complete.json"), read(args.path_run/"complete.json")
    for meta, done in ((b, bd), (p, pd)):
        protocol = meta["protocol"]
        if (done["scene_count"] != 256 or protocol["scenes"] != 256
                or any(protocol[k] != v for k,v in dict(tokens=4096,seed=0,precision="bf16",
                    context_views=12,target_views=8,image_shape=[256,256],shared_sorted_order=True).items())
                or done["checkpoint_sha256"] != meta["checkpoint"]["sha256"]
                or done["input_fingerprint"] != FINGERPRINT
                or meta["source"]["inputs"]["fingerprint"] != FINGERPRINT):
            raise ValueError("boundary source completion/protocol mismatch")
    # B0 stored path/hash/step; T05 additionally stored checkpoint file size.
    if (any(b["checkpoint"][key] != p["checkpoint"][key] for key in ("path","sha256","global_step"))
            or b["source"]["inputs"] != p["source"]["inputs"]
            or b["manifest"] != p["manifest"] or len(b["source"]["inputs"]["scenes"]) != 256
            or len(set(b["source"]["inputs"]["scenes"])) != 256
            or bd["native_checkpoint_sha256"] != b["native_checkpoint"]["sha256"]):
        raise ValueError("B0/T05 source identities differ")
    if not b["model_pair"]["encoder_tensors_equal"] or not b["model_pair"]["shared_decoder_tensors_equal"]:
        raise ValueError("native/reference shared weights differ")
    operators = b["model_pair"]["geometry_boundary_effective_readout_errors"]
    if set(operators) != set(READOUTS) or not all(
            math.isfinite(v) and v >= 0 for row in operators.values() for v in row.values()):
        raise ValueError("invalid effective readout metadata")
    rows = []
    for i, scene in enumerate(b["source"]["inputs"]["scenes"]):
        br, pr = read(args.b0_run/f"scenes/scene_{i:04d}.json"), read(args.path_run/f"scenes/scene_{i:04d}.json")
        if any(row["scene"] != scene or row["index"] != i for row in (br,pr)):
            raise ValueError("boundary source scene order differs")
        for key in ("input_fingerprint","target_fingerprint","context_frame_ids","target_frame_ids","permutation_sha256"):
            if br[key] != pr[key]:
                raise ValueError(f"boundary paired scene differs: {key}")
        for row in (br["modes"]["b0_native"], br["modes"]["b1_bridge"], *pr["modes"].values()):
            finite_metrics(row)
        rows.append(dict(scene=scene,index=i,b0=br["modes"]["b0_native"],b1=br["modes"]["b1_bridge"],paths=pr["modes"]))
    return b, p, rows


def fixed_source(args):
    meta, done = read(args.fixed_run/"metadata.json"), read(args.fixed_run/"complete.json")
    bundle, manifest, exported = (read(args.bundle/name) for name in ("metadata.json","manifest.json","complete.json"))
    identity = meta["identity"]
    protocol = dict(scenes=256,tokens=4096,seed=0,precision="bf16",context_views=12,
                    target_views=8,image_shape=[256,256],shared_sorted_order=True)
    expected_orders = list(ORDERS)+[f"random_{i}" for i in range(10)]
    if (done["scene_count"] != 256 or done["input_fingerprint"] != FINGERPRINT
            or any(bundle["protocol"].get(k) != v for k,v in protocol.items())
            or any(identity.get(k) != protocol[k] for k in ("tokens","seed","precision"))
            or done["orders"] != expected_orders or identity["orders"] != expected_orders
            or done["summary_sha256"] != digest(args.fixed_run/"summary.json")
            or done["checkpoint_sha256"] != identity["checkpoint"]["sha256"]
            or identity["checkpoint"] != bundle["checkpoint"]
            or identity["source"]["inputs"]["fingerprint"] != FINGERPRINT
            or exported["metadata_sha256"] != digest(args.bundle/"metadata.json")
            or exported["manifest_sha256"] != digest(args.bundle/"manifest.json")
            or exported["scene_count"] != 256 or bundle["protocol"]["scenes"] != 256
            or digest(args.checkpoint) != bundle["checkpoint"]["sha256"]):
        raise ValueError("fixed-order source / target bundle completion or checkpoint differs")
    rows = manifest["rows"]
    names = identity["source"]["inputs"]["scenes"]
    if len(rows) != 256 or [r["scene"] for r in rows] != names or len(set(names)) != 256:
        raise ValueError("fixed-order and target-bundle scene sets/order differ")
    fixed = []
    for i, row in enumerate(rows):
        item = read(args.fixed_run/f"scenes/scene_{i:04d}.json")
        if (row["index"] != i or item["index"] != i or item["scene"] != row["scene"]
                or list(item["orders"]) != expected_orders
                or item["reference_packet_sha256"] != row["packet_sha256"]
                or item["orders"]["hilbert"]["packet_sha256"] != row["packet_sha256"]):
            raise ValueError("fixed-order scene identity / Hilbert packet differs")
        fixed.append(item)
    return meta, bundle, rows, fixed


def preflight(args, tasks):
    """Read existing artifacts on the login node, without model construction."""
    if "boundary" in tasks:
        boundary_source(args)
    if "native" in tasks:
        reference_rows(args.reference)
        if not args.native_checkpoint.is_file():
            raise FileNotFoundError(args.native_checkpoint)
    if "fixed-render" in tasks:
        fixed_source(args)


def runtime_hashes():
    import eval_nfcgs_b0_matched as b0
    import analyze_nfcgs_fixed_order as fixed
    return {**b0.code_hashes(), **fixed.code_hashes(), **{
        name:digest(REPO/name) for name in ("scripts/run_nfcgs_followup_20261020.py",
                                           "scripts/slurm/run_nfcgs_followup_20261020.slurm")}}


@contextmanager
def run_directory(args, identity):
    output = args.output
    if output.exists() and not args.resume:
        raise FileExistsError("use a fresh output or explicit --resume")
    output.mkdir(parents=True, exist_ok=True)
    lock = output/".run-lock"; lock.mkdir()
    try:
        if not (output/"metadata.json").exists() and any(p != lock for p in output.iterdir()):
            raise ValueError("nonempty output without metadata")
        save(output/"metadata.json", identity)
        yield
    finally:
        lock.rmdir()


def boundary(args):
    b,p,rows = boundary_source(args)
    source_files = [folder/name for folder in (args.b0_run,args.path_run)
        for name in ("metadata.json","complete.json","summary.json")]
    source_files += [folder/f"scenes/scene_{i:04d}.json" for folder in (args.b0_run,args.path_run) for i in range(256)]
    identity = dict(task="boundary",version=1,source_sha256={str(path):digest(path) for path in source_files},
                    runner_sha256=digest(__file__), training=False, rendering=False)
    means = {mode:{key:statistics.fmean(row["paths"][mode][key] for row in rows)
             for key in METRICS+("feature_mse","texture_mse","geometry_mse")}
             for mode in rows[0]["paths"]}
    result = dict(scene_count=256,model_pair=b["model_pair"],path_means=means,
        b0={key:statistics.fmean(row["b0"][key] for row in rows) for key in METRICS},
        b1={key:statistics.fmean(row["b1"][key] for row in rows) for key in METRICS},
        b1_bridge_minus_t05_psnr=statistics.fmean(row["b1"]["psnr"]-row["paths"]["b1_direct"]["psnr"] for row in rows),
        max_abs_b1_drift=max(abs(row["b1"]["psnr"]-row["paths"]["b1_direct"]["psnr"]) for row in rows),
        scope="Existing learned-boundary operator metadata and feature errors; no new QR/feature experiment.")
    lines = ["# N01 geometry boundary metadata audit", "", "Matched256; existing B0/T05 artifacts only.", "",
        "| Effective readout W_current P_learned - W_native | RMSE | Max abs | Relative Frobenius |",
        "|---|---:|---:|---:|"]
    for name,row in b["model_pair"]["geometry_boundary_effective_readout_errors"].items():
        lines.append(f"| {name} | {row['rmse']:.8g} | {row['max_abs']:.8g} | {row['relative_frobenius']:.8g} |")
    lines += ["", "| Path | PSNR | Texture MSE to F | Geometry MSE to F |", "|---|---:|---:|---:|"]
    for name,row in means.items():
        lines.append(f"| {name} | {row['psnr']:.5f} | {row['texture_mse']:.8g} | {row['geometry_mse']:.8g} |")
    lines += ["", f"Native / B1 PSNR: {result['b0']['psnr']:.5f} / {result['b1']['psnr']:.5f}.",
        f"Common B1 mean/max absolute PSNR drift: {result['b1_bridge_minus_t05_psnr']:+.8g} / {result['max_abs_b1_drift']:.8g}.", "",
        "Effective readout error tests the learned projection/decoder boundary, not an additive render-loss contribution.",
        "These files do not contain the initial QR operator or a controlled frozen-projection retraining comparison.",
        "No bug, causal attribution or initial-QR preservation conclusion is inferred from B1 PSNR alone.", ""]
    with run_directory(args,identity):
        save(args.output/"summary.json",result)
        report(args.output,"\n".join(lines))
        save(args.output/"complete.json",dict(scene_count=256,summary_sha256=digest(args.output/"summary.json")))
    print(f"BOUNDARY_AUDIT_COMPLETE scenes=256 output={args.output}",flush=True)
    return result


def setup_gpu(args):
    import numpy as np
    import torch
    import eval_nfcgs_b0_matched as b0
    if args.precision == "bf16" and (not torch.cuda.is_available() or torch.device(args.device).type != "cuda"):
        raise ValueError("production BF16 evaluation requires CUDA")
    random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.set_num_threads(args.threads)
    return torch, b0


def scene_name(batch):
    name = batch["scene_info"]["scene"]
    return str(name[0] if isinstance(name,(tuple,list)) else name)


def frame_match(batch, reference, diag):
    diag.validate_views(batch)
    for section,key in (("inputs","context_frame_ids"),("targets","target_frame_ids")):
        if diag.frame_ids(batch[section]) != reference[key]:
            raise ValueError(f"reference frame mismatch: {reference['scene']}/{key}")


def native_scene(model, batch, args, render, metrics, diag):
    inputs = diag.profile.to_device(batch["inputs"],args.device)
    targets = diag.profile.to_device(batch["targets"],args.device)
    with diag.amp(args):
        texture,geometry = model.encode_scene_tokens(inputs)
        if texture.shape[:2] != (1,args.tokens) or geometry.shape[-1] != 512:
            raise ValueError("native unprojected geometry512 / scene-token shape differs")
        gaussians = model.decode_scene_tokens((texture,geometry))
    if gaussians.num_gaussians != args.tokens*8:
        raise ValueError("native decoder did not produce eight Gaussians per token")
    cameras = {key:targets[key] for key in ("intrinsic","extrinsic")}
    cameras["images"] = targets["images"].new_zeros(targets["images"].shape)
    result = metrics(targets["images"][0].float().clamp(0,1),render(gaussians,cameras))
    finite_metrics(result)
    return result


def native(args):
    torch,b0 = setup_gpu(args)
    import eval_nfcgs_independent_receiver as receiver
    diag = b0.diag
    references = reference_rows(args.reference,args.expected_scenes)
    source_hash = digest(args.reference)
    model,checkpoint = b0.load_native(args)
    cfg = receiver.dataset_config(args)
    numerics = b0.numerical_policy()  # after imports that configure backend flags
    hashes = runtime_hashes()
    identity = dict(version=1,task="native",checkpoint=checkpoint,reference=str(args.reference.resolve()),
        reference_sha256=source_hash,code_sha256=hashes,environment=diag.profile.environment(args),numerics=numerics,
        protocol=dict(scenes=args.expected_scenes,tokens=args.tokens,precision=args.precision,seed=0,
            context_views=12,target_views=8,native_geometry_channels=512,order="native original slots",training=False,actual_bitstream=False))
    # GPU UUID may differ on an explicit resume; model/software/numerical policy must match.
    identity["environment"] = {k:v for k,v in identity["environment"].items()
        if k not in ("gpu_uuid","host","temp_directory")}
    selected = {r["scene"]:(i,r) for i,r in enumerate(references)}
    with torch.no_grad(),run_directory(args,identity):
        render,metrics = diag.render_functions()
        results,seen = {},set()
        for batch in diag.load_data(cfg):
            scene = scene_name(batch)
            if scene not in selected:
                continue  # explicit reference cohort; never replace a missing reference scene
            if scene in seen:
                raise ValueError(f"duplicate selected scene: {scene}")
            seen.add(scene)
            index,reference = selected[scene]
            frame_match(batch,reference,diag)
            evidence = dict(scene=scene,index=index,context_frame_ids=diag.frame_ids(batch["inputs"]),
                target_frame_ids=diag.frame_ids(batch["targets"]),
                input_fingerprint=diag.profile.input_fingerprint([batch["inputs"]]),
                target_fingerprint=diag.profile.input_fingerprint([batch["targets"]]),reference=reference)
            path = args.output/f"scenes/scene_{index:05d}.json"
            if path.exists():
                row = read(path)
                if any(row[k] != v for k,v in evidence.items()):
                    raise ValueError("resume scene/input/target identity differs")
                finite_metrics(row["metrics"])
            else:
                measured = native_scene(model,batch,args,render,metrics,diag)
                row = dict(evidence,metrics=measured,
                    codec_minus_native={k:reference[k]-measured[k] for k in METRICS})
                save(path,row)
            results[index] = row
            print(f"NATIVE_FULL_SCENE={len(results)}/{len(references)} scene={scene} psnr={row['metrics']['psnr']:.5f}",flush=True)
            if len(results) == len(references):
                break
        if len(results) != len(references):
            raise RuntimeError(f"missing reference scenes: {sorted(set(selected)-seen)[:10]}; no fallback sampling")
        if (runtime_hashes() != hashes or digest(args.reference) != source_hash
                or digest(args.native_checkpoint) != checkpoint["sha256"]):
            raise RuntimeError("runtime code, reference or native checkpoint changed")
        ordered = [results[i] for i in range(len(references))]
        summary = dict(scene_count=len(ordered),native={k:statistics.fmean(r["metrics"][k] for r in ordered) for k in METRICS},
            codec_reference={k:statistics.fmean(r["reference"][k] for r in ordered) for k in METRICS},
            codec_minus_native={k:statistics.fmean(r["codec_minus_native"][k] for r in ordered) for k in METRICS},
            reference_mean_bytes=statistics.fmean(r["actual_bytes"] for r in references),native_actual_bytes=None,
            matching="Exact scene and C12/T8 frame lists; historical source tensors were not saved for a bit-exact comparison.")
        save(args.output/"summary.json",summary)
        lines = ["# N03 native GlobalSplat on the completed codec full-test cohort", "",
            f"Scenes: {len(ordered)}; C12/T8; 256x256; native geometry512; original slots.", "",
            "| Mode | PSNR | SSIM | LPIPS |", "|---|---:|---:|---:|"]
        for name in ("native","codec_reference","codec_minus_native"):
            row = summary[name]; lines.append(f"| {name} | {row['psnr']:.6f} | {row['ssim']:.6f} | {row['lpips']:.6f} |")
        lines += ["",summary["matching"],"Native has no compressed packet rate. Shared model size is excluded from codec scene rate.",
            "The codec column reuses its historical full-test evaluation; it is not rerendered in this job.",""]
        report(args.output,"\n".join(lines))
        save(args.output/"complete.json",dict(scene_count=len(ordered),reference_sha256=source_hash,
            native_checkpoint_sha256=checkpoint["sha256"],summary_sha256=digest(args.output/"summary.json")))
    print(f"NATIVE_FULL_COMPLETE scenes={len(references)} output={args.output}",flush=True)
    return summary


def fixed_render(args):
    torch,_ = setup_gpu(args)
    import eval_nfcgs_independent_receiver as receiver
    import analyze_nfcgs_fixed_order as fixed
    meta,bundle,rows,source_rows = fixed_source(args)
    diag = receiver.diag
    receiver.shared_code_matches(meta["identity"]["code_sha256"],fixed.code_hashes())
    checked_bundle,checked_rows = receiver.load_bundle(args.bundle)
    if (checked_bundle,checked_rows) != (bundle,rows):
        raise ValueError("bundle changed during preflight")
    env,numerics = receiver.validate_environment(bundle,args)
    model = receiver.load_receiver(args.checkpoint,bundle,args.device)
    hashes = runtime_hashes()
    source_hashes = {str(folder/name):digest(folder/name) for folder in (args.fixed_run,args.bundle)
                     for name in ("metadata.json","complete.json")}
    source_hashes[str(args.bundle/"manifest.json")] = digest(args.bundle/"manifest.json")
    identity = dict(version=1,task="fixed-render",checkpoint=bundle["checkpoint"],source_sha256=source_hashes,
        orders=list(ORDERS),code_sha256=hashes,numerics=numerics,
        environment={k:v for k,v in env.items() if k not in ("gpu_uuid","host","temp_directory")},encoder_constructed=False,
        protocol=bundle["protocol"],feature_error_scope="reused T06 inverse-permutation feature MSE; not recomputed in packet receiver")
    with torch.no_grad(),run_directory(args,identity):
        render,metrics = diag.render_functions()
        results = []
        for reference,source in zip(rows,source_rows):
            index,scene = reference["index"],reference["scene"]
            source_path = args.fixed_run/f"scenes/scene_{index:04d}.json"
            source_sha = digest(source_path)
            target_path = receiver.contained(args.bundle,reference["target_file"])
            if digest(target_path) != reference["target_sha256"]:
                raise ValueError("target bundle SHA-256 mismatch")
            item = torch.load(target_path,map_location="cpu",weights_only=False)
            if set(item) != {"scene","index","targets"} or item["scene"] != scene or item["index"] != index:
                raise ValueError("target artifact identity differs")
            targets = item["targets"]; receiver.validate_targets(targets,reference)
            cameras = {k:diag.profile.to_device(targets[k],args.device) for k in ("intrinsic","extrinsic")}
            cameras["images"] = torch.zeros(targets["images"].shape,device=args.device)
            gt = targets["images"][0].to(args.device).float().clamp(0,1)
            evidence = dict(index=index,scene=scene,source_scene_sha256=source_sha,
                target_sha256=reference["target_sha256"],target_fingerprint=reference["target_fingerprint"],
                target_frame_ids=reference["target_frame_ids"],reference=reference["reference"])
            path = args.output/f"scenes/scene_{index:04d}.json"
            saved = read(path) if path.exists() else None
            if saved and any(saved[k] != value for k,value in evidence.items()):
                raise ValueError("resume source scene/target identity differs")
            measured_orders = {}
            for name in ORDERS:
                cost = source["orders"][name]
                payload = receiver.contained(args.fixed_run,cost["packet_file"]).read_bytes()
                if digest_bytes(payload) != cost["packet_sha256"] or len(payload) != cost["total_bytes"]:
                    raise ValueError("fixed-order packet length/SHA-256 mismatch")
                packed = receiver.SceneBitstream.unpack(payload)
                if packed.points != args.tokens or packed.channels != model.codec.config.observable_channels or packed.rank != model.codec.config.rank:
                    raise ValueError("fixed-order packet dimensions mismatch")
                components = fixed.payload_components(payload)["components"]
                if any(components[k] != cost[k] for k in components):
                    raise ValueError("fixed-order stream accounting differs")
                if saved:
                    value = saved["orders"][name]
                    if value["packet_sha256"] != cost["packet_sha256"] or value["actual_bytes"] != len(payload):
                        raise ValueError("resume packet identity differs")
                    finite_metrics(value["metrics"])
                else:
                    with diag.amp(args):
                        _,gaussians = model.decode(payload)
                    if gaussians.num_gaussians != args.tokens*8:
                        raise ValueError("expected eight Gaussians per scene token")
                    measured = metrics(gt,render(gaussians,cameras)); finite_metrics(measured)
                    value = dict(metrics=measured,actual_bytes=len(payload),packet_sha256=cost["packet_sha256"],
                                 feature_mse_from_t06=cost["feature_mse"])
                measured_orders[name] = value
            delta = {k:measured_orders["hilbert"]["metrics"][k]-reference["reference"][k] for k in METRICS}
            if any(abs(delta[k]) > receiver.TOLERANCES[k] for k in METRICS):
                raise RuntimeError(f"Hilbert render differs from completed T05: {scene}/{delta}")
            row = dict(evidence,orders=measured_orders,hilbert_minus_t05=delta)
            if saved and row != saved:
                raise ValueError("saved scene aggregate differs")
            save(path,row); results.append(row)
            print(f"FIXED_RENDER_SCENE={len(results)}/{len(rows)} scene={scene} "
                  f"slot={measured_orders['slot']['metrics']['psnr']:.5f} hilbert={measured_orders['hilbert']['metrics']['psnr']:.5f}",flush=True)
        # Revalidate historical artifacts and current runtime before writing completion.
        if (runtime_hashes() != hashes or fixed_source(args) != (meta,bundle,rows,source_rows)
                or digest(args.checkpoint) != bundle["checkpoint"]["sha256"]):
            raise RuntimeError("runtime/checkpoint/source changed during rendering")
        summary = dict(scene_count=len(results),orders={name:dict(
            **{k:statistics.fmean(r["orders"][name]["metrics"][k] for r in results) for k in METRICS},
            mean_bytes=statistics.fmean(r["orders"][name]["actual_bytes"] for r in results),
            feature_mse_from_t06=statistics.fmean(r["orders"][name]["feature_mse_from_t06"] for r in results)) for name in ORDERS},
            hilbert_max_abs_t05_delta={k:max(abs(r["hilbert_minus_t05"][k]) for r in results) for k in METRICS},
            scope="Frozen Hilbert-trained packet intervention; not retrained RD curves or equal-quality rate savings.")
        summary["paired_minus_hilbert"] = {name:{k:fixed.interval([
            r["orders"][name]["metrics"][k]-r["orders"]["hilbert"]["metrics"][k] for r in results])
            for k in METRICS} for name in ORDERS if name != "hilbert"}
        save(args.output/"summary.json",summary)
        lines = ["# N02 rendering the saved T06 fixed-order packets", "",f"Scenes: {len(results)}; C12/T8; saved packets only; no re-encoding.","",
            "| Order | PSNR | SSIM | LPIPS | KiB/scene | Feature MSE from T06 |","|---|---:|---:|---:|---:|---:|"]
        for name,value in summary["orders"].items():
            lines.append(f"| {name} | {value['psnr']:.5f} | {value['ssim']:.5f} | {value['lpips']:.5f} | {value['mean_bytes']/1024:.5f} | {value['feature_mse_from_t06']:.8g} |")
        lines += ["",summary["scope"],"Hilbert metrics must match completed T05 within its independent-receiver tolerances.",
            "Only codec and Gaussian decoder are constructed. Target RGB enters metrics only.",
            "Paired scene bootstrap intervals are in summary.json; they are not training-seed intervals.",
            "Feature MSE is reused from the validated T06 scene artifact; inverse-permutation Gaussian errors are not measured here.",""]
        report(args.output,"\n".join(lines))
        save(args.output/"complete.json",dict(scene_count=len(results),orders=list(ORDERS),
            checkpoint_sha256=bundle["checkpoint"]["sha256"],summary_sha256=digest(args.output/"summary.json")))
    print(f"FIXED_RENDER_COMPLETE scenes={len(results)} orders=3 output={args.output}",flush=True)
    return summary


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task",choices=("boundary","native","fixed-render","plan"))
    parser.add_argument("--b0-run",type=Path,default=REPO/"outputs/nfcgs_b0_matched/job_458024")
    parser.add_argument("--path-run",type=Path,default=REPO/"outputs/nfcgs_path_diagnostics/job_458004")
    parser.add_argument("--fixed-run",type=Path,default=REPO/"outputs/nfcgs_fixed_order/job_458818")
    parser.add_argument("--bundle",type=Path,default=REPO/"outputs/nfcgs_independent_receiver/job_458784/bundle")
    parser.add_argument("--checkpoint",type=Path,default=REPO/POINT/"checkpoints/nfcgs_main/version_0/step000050000.ckpt")
    parser.add_argument("--native-checkpoint",type=Path,default=REPO/"checkpoints/pretrained/globalsplat-re10k-32k.ckpt")
    parser.add_argument("--reference",type=Path,default=REPO/POINT/"eval_all/nfcgs_main/actual_rate_per_scene.json")
    parser.add_argument("--dataset-root",type=Path,default=Path("/data3/local_datasets/re10k"))
    parser.add_argument("--output",type=Path)
    parser.add_argument("--num-workers",type=int,default=4)
    parser.add_argument("--threads",type=int,default=8)
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--precision",choices=("bf16",),default="bf16")
    parser.add_argument("--resume",action="store_true")
    args = parser.parse_args(argv)
    args.tokens,args.seed,args.expected_scenes = 4096,0,6991
    if args.num_workers < 0 or args.threads < 1:
        parser.error("invalid workers/threads")
    if args.task != "plan" and args.output is None:
        parser.error("--output is required")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.task == "plan":
        preflight(args,("boundary","native","fixed-render"))
        print("FOLLOWUP_PLAN_PASS boundary=metadata256 native=reference6991 fixed-render=saved256x3 TRAINING=False")
    else:
        {"boundary":boundary,"native":native,"fixed-render":fixed_render}[args.task](args)


if __name__ == "__main__":
    main()
