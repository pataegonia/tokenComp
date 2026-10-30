#!/usr/bin/env python3
"""T06: frozen H+D4 features/Score, actual causal coding under 13 permutations.

This is a fixed-predictor order intervention, not separately trained RD curves.
All non-Hilbert orders are distribution shifts for this Hilbert-trained model.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import random
import statistics
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import eval_nfcgs_independent_receiver as receiver
diag = receiver.diag
from globalsplat.compression.bitstream import SceneBitstream
from globalsplat.compression.mechanism import StageRecorder, channel_scale, observe_stages, payload_components
from globalsplat.compression.morton import MortonOrder, invert_permutation, morton_order_3d
from globalsplat.compression.token_order import hilbert_order_3d

ORDERS = ("slot", "morton", "hilbert", *(f"random_{i}" for i in range(10)))
TRACE_ORDERS = ("slot", "morton", "hilbert", "random_0")
SCALARS = ("total_bytes", "score_raw", "msh_y", "msh_z", "headers", "raw_mae", "raw_mse",
    "normalized_mae", "normalized_mse", "quantization_normalized_mse", "anchor_distance",
    "anchor_score_normalized_mae", "feature_mse", "feature_normalized_mse")


def code_hashes():
    result = receiver.code_hashes()
    for name in ("scripts/analyze_nfcgs_fixed_order.py", "scripts/slurm/analyze_nfcgs_fixed_order.slurm"):
        result[name] = diag.file_hash(REPO/name)
    return result


def state_hash(module):
    # The shared cache fingerprint helper expects tensor dimensions for byte
    # views. Flatten scalar integer buffers, retaining their original shapes.
    return diag.profile.input_fingerprint([{key:(tuple(value.shape),value.reshape(-1))
                                           for key,value in module.state_dict().items()}])


def permutations(positions, *, scene, bits):
    points = positions.shape[1]
    result = dict(slot=torch.arange(points, device=positions.device)[None],
        morton=morton_order_3d(positions, bits).permutation,
        hilbert=hilbert_order_3d(positions, bits).permutation)
    for i in range(10):
        seed = int.from_bytes(hashlib.sha256(f"{i}:{scene}".encode()).digest()[:8], "little") % (2**63)
        generator = torch.Generator().manual_seed(seed)
        result[f"random_{i}"] = torch.randperm(points, generator=generator).to(positions.device)[None]
    return result


def order_object(permutation):
    expected = torch.arange(permutation.shape[1], device=permutation.device)[None]
    if permutation.shape[0] != 1 or not torch.equal(permutation.sort(1).values, expected):
        raise ValueError("expected a single bijective token permutation")
    return MortonOrder(permutation, invert_permutation(permutation), torch.zeros_like(permutation))


def stage_scalars(stages):
    symbols = sum(stage["symbols"] for stage in stages)
    result = {key:sum(stage[key]*stage["symbols"] for stage in stages)/symbols for key in
              ("raw_mae", "raw_mse", "normalized_mae", "normalized_mse", "quantization_normalized_mse")}
    groups = [group for stage in stages for group in stage["groups"] if group["anchor_relation"]["pairs"]]
    weight = sum(group["width"]*group["anchor_relation"]["pairs"] for group in groups)
    for target, source in (("anchor_distance", "xyz_distance"), ("anchor_score_normalized_mae", "normalized_l1")):
        result[target] = sum(group["width"]*group["anchor_relation"]["pairs"]*group["anchor_relation"][source]
                             for group in groups)/weight if weight else None
    return result


@torch.no_grad()
def encode_fixed(codec, features, normalized_score, positions, permutation, *, score_scale, xyz_scale, keep_tensors=False):
    """Encode a fixed F and fixed normalized Score; only their row order changes.

    The computation after analysis matches codec.compress. Score predictors see
    reconstructed causal anchors. MSH residual is recomputed for every order.
    """
    order = order_object(permutation)
    values = order.apply(features)
    score = order.apply(normalized_score).transpose(1,2).unsqueeze(2)
    recorder = StageRecorder(score, order.apply(positions), keep_tensors=keep_tensors)
    # Identical diagnostic scales for all permutations, never coding inputs.
    recorder.scale = score_scale.detach().float().cpu()
    recorder.xyz_scale = xyz_scale.detach().float().cpu()
    with observe_stages(codec.score_context, recorder):
        with torch.autocast(device_type=score.device.type, enabled=False):
            score_payload, score_hat = codec.score_context.compress(score.float())
        shat = score_hat.squeeze(2).transpose(1,2)
        low_rank = codec._synthesize_low_rank(shat)
        residual = (values-low_rank).transpose(1,2).unsqueeze(2)
        residual = (residual-codec.residual_mean)/codec.residual_std.clamp_min(1e-8)
        residual_payload, _ = codec.residual_codec.compress(residual)
        packet = SceneBitstream(points=values.shape[1], channels=values.shape[2], rank=codec.config.rank,
            mean_fp16=b"", score=score_payload, residual=residual_payload, flags=codec.flags).pack()
        # Real receiver, without observing or replacing its causal context.
        with torch.autocast(device_type=score.device.type, enabled=False):
            received = codec.score_context.decompress(score_payload, values.shape[1])
        if not torch.equal(recorder.reconstruction, received[0,:,0].T.cpu()):
            raise RuntimeError("observed causal Score differs from receiver")
        decoded = torch.cat(codec.decompress(packet), -1).float()
    stages = recorder.summary()
    cost = payload_components(packet)
    if sum(stage["raw_bytes"] for stage in stages) != cost["components"]["score_raw"]:
        raise RuntimeError("stage raw bytes differ from Score entropy payload")
    restored = order.restore(decoded)
    std = features.float().std(1, correction=0).clamp_min(1e-6)
    error = restored-features.float()
    if not torch.isfinite(restored).all():
        raise RuntimeError("non-finite packet reconstruction")
    stats = dict(total_bytes=cost["total_bytes"], **cost["components"], stages=stages, **stage_scalars(stages),
        feature_mse=float(error.square().mean()), feature_normalized_mse=float((error/std[:,None]).square().mean()),
        packet_sha256=hashlib.sha256(packet).hexdigest(),
        permutation_sha256=diag.profile.input_fingerprint([permutation]))
    return stats, packet, recorder.tensors


@torch.no_grad()
def analyze_scene(model, inputs, args, scene, reference_sha, *, keep_tensors=False):
    from globalsplat.compression.path_diagnostics import validate_codec
    codec = model.feature_codec
    validate_codec(codec)
    with diag.amp(args):
        texture, geometry = model.encode_scene_tokens(inputs)
        if texture.shape[:2] != (1,args.tokens):
            raise ValueError("unexpected token count")
        xyz = model.gaussian_decoder.decoded_token_centers(codec.project_geometry(geometry))
        features, production_order, _, prepared_score = diag.profile.prepare_score(codec, texture, geometry, xyz)
        # Analyze once in the exact production Hilbert layout, then restore rows.
        fixed_score = production_order.restore(prepared_score.squeeze(2).transpose(1,2))
        orders = permutations(xyz, scene=scene, bits=codec.config.morton_bits)
        if not torch.equal(orders["hilbert"],production_order.permutation):
            raise RuntimeError("production order is not the requested Hilbert permutation")
        production = codec.compress(texture,geometry,xyz).data
        if hashlib.sha256(production).hexdigest() != reference_sha:
            raise RuntimeError(f"Hilbert packet does not reproduce T05: {scene}")
        scale = channel_scale(fixed_score[0])
        xyz_scale = (xyz[0].float()-xyz[0].float().mean(0)).square().sum(-1).mean().sqrt().clamp_min(1e-6)
        result = dict(scene=scene, reference_packet_sha256=reference_sha,
            fixed_features_sha256=diag.profile.input_fingerprint([features]),
            fixed_score_sha256=diag.profile.input_fingerprint([fixed_score]),
            fixed_xyz_sha256=diag.profile.input_fingerprint([xyz]), orders={})
        packets, traces = {}, {}
        # Check the manually reproduced production path before running controls.
        for name in ("hilbert", *(name for name in ORDERS if name!="hilbert")):
            stats, packet, trace = encode_fixed(codec, features, fixed_score, xyz, orders[name],
                score_scale=scale, xyz_scale=xyz_scale, keep_tensors=keep_tensors and name in TRACE_ORDERS)
            if name=="hilbert" and packet!=production:
                raise RuntimeError("fixed-Score Hilbert packet differs from production compress")
            result["orders"][name]=stats
            packets[name]=packet
            if trace:
                traces[name]=dict(permutation=orders[name].cpu(), groups=trace)
    result["orders"]={name:result["orders"][name] for name in ORDERS}
    return result,packets,traces


def scalar_rows(row):
    result = {name:{k:row["orders"][name][k] for k in SCALARS} for name in ORDERS[:3]}
    result["random10_mean"] = {k:statistics.fmean(row["orders"][f"random_{i}"][k] for i in range(10)) for k in SCALARS}
    return result


def interval(values):
    values = np.asarray(values,dtype=float)
    rng = np.random.default_rng(0)
    boot = values[rng.integers(0,len(values),(1000,len(values)))].mean(1)
    return dict(mean=float(values.mean()),ci95=np.quantile(boot,[.025,.975]).tolist())


def summarize(rows):
    per_scene = [scalar_rows(row) for row in rows]
    means = {name:{k:statistics.fmean(row[name][k] for row in per_scene) for k in SCALARS}
             for name in per_scene[0]}
    paired = {name:{k:interval([row[name][k]-row["hilbert"][k] for row in per_scene]) for k in SCALARS}
              for name in means if name!="hilbert"}
    stages = {}
    for name in means:
        stages[name]=[]
        for i in range(4):
            collected = [row["orders"][name]["stages"][i] for row in rows] if name!="random10_mean" else [
                row["orders"][f"random_{j}"]["stages"][i] for row in rows for j in range(10)]
            stages[name].append(dict(stage=i,tokens=collected[0]["tokens"],
                **{k:statistics.fmean(v[k] for v in collected) for k in
                   ("raw_bytes","bits_per_token","raw_mae","normalized_mae","quantization_normalized_mse")}))
    return dict(scene_count=len(rows),means=means,stages=stages,paired_minus_hilbert=paired,
        random_seed_mean_total_bytes=[statistics.fmean(row["orders"][f"random_{i}"]["total_bytes"] for row in rows) for i in range(10)],
        bootstrap="1000 paired scene resamples; random orders averaged within scene first; not training-seed uncertainty",
        interpretation="Frozen Hilbert+D4 predictor and model; non-Hilbert permutations are distribution-shift diagnostics, not retrained RD curves.")


def write_report(output, summary):
    lines=["# T06 frozen H+D4 order intervention","",f"Matched scenes: {summary['scene_count']}; same F, projection, bases, normalized Score, XYZ and weights.",
        "Hilbert packet SHA-256 matches completed T05 for every scene.","",
        "| Order | Total KiB | Score raw KiB | MSH y KiB | MSH z KiB | Header B | Anchor XYZ distance | Score residual normalized MAE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name,row in summary["means"].items():
        lines.append(f"| {name} | {row['total_bytes']/1024:.5f} | {row['score_raw']/1024:.5f} | {row['msh_y']/1024:.5f} | "
            f"{row['msh_z']/1024:.5f} | {row['headers']:.1f} | {row['anchor_distance']:.7f} | {row['normalized_mae']:.7f} |")
    lines += ["", "Paired total bytes minus Hilbert (95% scene bootstrap CI):", ""]
    for name,values in summary["paired_minus_hilbert"].items():
        row=values["total_bytes"]
        lines.append(f"- {name}: {row['mean']:+.3f} B [{row['ci95'][0]:+.3f}, {row['ci95'][1]:+.3f}].")
    lines += ["",summary["interpretation"],summary["bootstrap"],
        "Actual causal reconstructed anchors condition coding; observer statistics never condition the model.",
        "Score raw excludes its wrapper. Headers include both Score wrapper and scene/MSH containers.",
        "MSH residual is recomputed for each decoded Score; all permutations affect both Score and MSH.",
        "No rendering, PSNR or full-test RD claim. Raw traces contain exact anchors/predictions for selected scenes/orders.", ""]
    path=output/"REPORT.md"
    text="\n".join(lines)
    if path.exists():
        if path.read_text(encoding="utf-8")!=text:raise ValueError("existing report differs")
    else:
        with path.open("x",encoding="utf-8") as stream:stream.write(text)


def save_same(path, value):
    if path.exists():
        if diag.read_json(path)!=value:raise ValueError(f"existing artifact differs: {path}")
    else:diag.save_json(path,value)


def preflight(args):
    manifest,spec,meta,reference_rows,_ = receiver.read_source(args)
    source,_ = diag.source_info(args,load_cache=False)
    if meta["source"]["inputs"] != source["inputs"]:
        raise ValueError("T05 and original input cache identities differ")
    return manifest,spec,meta,reference_rows,source


def check_saved(row,item,index,reference,output):
    if (row["scene"]!=item["scene"] or row["index"]!=index or tuple(row["orders"])!=ORDERS
            or row["input_fingerprint"]!=diag.profile.input_fingerprint([item["inputs"]])
            or row["reference_packet_sha256"]!=reference["packet_sha256"]):
        raise ValueError("saved scene identity/protocol differs")
    for value in row["orders"].values():
        payload=receiver.contained(output,value["packet_file"]).read_bytes()
        if hashlib.sha256(payload).hexdigest()!=value["packet_sha256"] or len(payload)!=value["total_bytes"]:
            raise ValueError("saved diagnostic packet changed")
        SceneBitstream.unpack(payload)
        if sum(value[k] for k in ("score_raw","msh_y","msh_z","headers"))!=len(payload):
            raise ValueError("saved byte accounting differs")
    if row.get("trace_file") and diag.file_hash(output/row["trace_file"])!=row["trace_sha256"]:
        raise ValueError("saved trace changed")


@torch.no_grad()
def run(args):
    manifest,spec,meta,references,source=preflight(args)
    if args.output.exists() and not args.resume:
        raise FileExistsError("use a fresh output or explicit --resume")
    if args.precision=="bf16" and (torch.device(args.device).type!="cuda" or not torch.cuda.is_available()):
        raise ValueError("production BF16 requires CUDA; CPU fixtures use fp32")
    torch.manual_seed(0);np.random.seed(0);random.seed(0);torch.set_num_threads(args.threads)
    cache=torch.load(args.input_run/"inputs.pt",map_location="cpu",weights_only=False)
    if ([item["scene"] for item in cache]!=source["inputs"]["scenes"]
            or diag.profile.input_fingerprint(cache)!=source["inputs"]["fingerprint"]):
        raise ValueError("original context cache tensor fingerprint differs")
    model,_,_=diag.profile.load_model(spec,args)
    env,numerics=receiver.validate_environment(dict(source_environment=meta["environment"],source_numerics=meta["numerics"]),args)
    if diag.backbone_hash(model)!=source["backbone_sha256"]:
        raise ValueError("frozen encoder/Gaussian decoder changed")
    model.feature_codec.update(force=True,update_quantiles=False)
    original_state_hash=state_hash(model.feature_codec)
    hashes=code_hashes()
    identity=dict(version=1,manifest=manifest,checkpoint=meta["checkpoint"],source=source,orders=list(ORDERS),
        precision=args.precision,tokens=args.tokens,seed=args.seed,trace_scenes=args.trace_scenes,
        environment={k:env[k] for k in ("torch","compressai","numpy","gpu","cuda_runtime","threads","precision")},
        numerics=numerics,code_sha256=hashes,normalization="common within-scene channel STD floor1e-6; XYZ RMS radius floor1e-6")
    args.output.mkdir(parents=True,exist_ok=True)
    lock=args.output/".analysis-lock";lock.mkdir()
    try:
        metadata_path=args.output/"metadata.json"
        if metadata_path.exists():
            if diag.read_json(metadata_path)["identity"]!=identity:
                raise ValueError("resume identity differs; preserve previous results")
        else:
            if any(path!=lock for path in args.output.iterdir()):
                raise ValueError("nonempty output without metadata")
            diag.save_json(metadata_path,dict(identity=identity,environment=env))
        results=[]
        for index,(item,reference) in enumerate(zip(cache,references)):
            if reference["scene"]!=item["scene"] or reference["index"]!=index:
                raise ValueError("reference scene order differs")
            path=args.output/f"scenes/scene_{index:04d}.json"
            if path.exists():
                row=diag.read_json(path);check_saved(row,item,index,reference,args.output)
            else:
                row,packets,traces=analyze_scene(model,diag.profile.to_device(item["inputs"],args.device),args,
                    item["scene"],reference["packet_sha256"],keep_tensors=index<args.trace_scenes)
                row.update(index=index,input_fingerprint=diag.profile.input_fingerprint([item["inputs"]]))
                for name,packet in packets.items():
                    relative=f"packets/scene_{index:04d}/{name}.bin"
                    packet_path=args.output/relative;packet_path.parent.mkdir(parents=True,exist_ok=True)
                    with packet_path.open("wb") as stream:stream.write(packet)
                    row["orders"][name]["packet_file"]=relative
                if traces:
                    relative=f"traces/scene_{index:04d}.pt"
                    trace_path=args.output/relative;trace_path.parent.mkdir(parents=True,exist_ok=True)
                    torch.save(traces,trace_path)
                    row.update(trace_file=relative,trace_sha256=diag.file_hash(trace_path))
                diag.save_json(path,row)
            results.append(row)
            print(f"FIXED_ORDER_SCENE={len(results)}/{args.scenes} scene={item['scene']} "
                f"hilbert_bytes={row['orders']['hilbert']['total_bytes']} slot_bytes={row['orders']['slot']['total_bytes']}",flush=True)
        if len(results)!=args.scenes or code_hashes()!=hashes:
            raise RuntimeError("incomplete scenes or code changed")
        if state_hash(model.feature_codec)!=original_state_hash:
            raise RuntimeError("codec weights/buffers changed during interventions")
        if diag.file_hash(spec["checkpoint"])!=meta["checkpoint"]["sha256"]:
            raise RuntimeError("checkpoint changed")
        summary=summarize(results)
        save_same(args.output/"summary.json",summary)
        write_report(args.output,summary)
        save_same(args.output/"complete.json",dict(scene_count=len(results),orders=list(ORDERS),
            input_fingerprint=source["inputs"]["fingerprint"],checkpoint_sha256=meta["checkpoint"]["sha256"],
            summary_sha256=diag.file_hash(args.output/"summary.json")))
        print(f"FIXED_ORDER_COMPLETE scenes={len(results)} orders={len(ORDERS)} output={args.output}",flush=True)
        return summary
    finally:lock.rmdir()


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("plan","run"))
    parser.add_argument("--manifest",type=Path,default=REPO/"config/plans/nfcgs_path_diagnostics_20261019.json")
    parser.add_argument("--path-run",type=Path,default=REPO/"outputs/nfcgs_path_diagnostics/job_458004")
    parser.add_argument("--input-run",type=Path,default=REPO/"outputs/nfcgs_legacy3_anchors/job_454995")
    parser.add_argument("--expected-input-fingerprint",default=receiver.FINGERPRINT)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--dataset-root",type=Path,default=Path("/data3/local_datasets/re10k"))
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--precision",choices=("bf16","fp32"),default="bf16")
    parser.add_argument("--scenes",type=int,default=256)
    parser.add_argument("--tokens",type=int,default=4096)
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--threads",type=int,default=8)
    parser.add_argument("--num-workers",type=int,default=0)
    parser.add_argument("--trace-scenes",type=int,default=1)
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args(argv)
    if (args.scenes,args.tokens,args.seed,args.precision)!=(256,4096,0,"bf16"):
        parser.error("production protocol is fixed at matched256/4096/seed0/BF16; CPU is for Python fixtures only")
    if args.threads<1 or args.trace_scenes<0 or args.trace_scenes>args.scenes:
        parser.error("invalid threads/trace-scenes")
    if args.mode=="run" and args.output is None:parser.error("run requires --output")
    return args


def main(argv=None):
    args=parse_args(argv)
    if args.mode=="plan":
        _,spec,_,_,_=preflight(args)
        print(f"FIXED_ORDER_PLAN scenes=256 tokens=4096 orders=13 checkpoint={spec['checkpoint']}")
        print("TRAINING=False RENDERING=False DATASET_LOADER=False CACHE_TENSORS_CHECK=deferred_to_compute_node")
    else:run(args)


if __name__=="__main__":main()
