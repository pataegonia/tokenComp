#!/usr/bin/env python3
"""Freeze a train holdout and initialize comparable codec architectures."""
from __future__ import annotations

import argparse
from io import BytesIO
import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch

from globalsplat.compression.architecture import PILOT_ARMS, pilot_config, shared_fresh_codec_state, apply_shared_fresh_state, tensor_hash
from globalsplat.compression.initialization import initialize_observable_from_vanilla, _checkpoint_state, _READOUTS
from globalsplat.dataset.holdout import file_hash, read_holdout, train_chunk_path

ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path, value):
    data = json.dumps(value, indent=2, sort_keys=True)+"\n"
    if path.exists():
        if path.read_text() != data:
            raise ValueError(f"existing preparation artifact differs: {path}")
        return
    with tempfile.NamedTemporaryFile(mode="w",dir=path.parent,encoding="utf-8",delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data); handle.flush(); os.fsync(handle.fileno())
    try:
        os.replace(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)


def create_holdout(root: Path, output: Path, count=128, seed=0):
    manifest_path = output/"holdout.json"
    if manifest_path.exists():
        result = read_holdout(manifest_path,root)
        if result["scene_count"] != count or result["seed"] != seed:
            raise ValueError("existing holdout has a different size/seed")
    else:
        from PIL import Image
        from globalsplat.dataset.data_module import _import_upstream
        _,ds,_ = _import_upstream(str(ROOT/"third_party/ZPressor/mvsplat"),"ZPressor/MVSplat")
        from src.dataset.view_sampler.view_sampler_deterministic_all import _farthest_camera_indices
        from src.geometry.projection import get_fov
        train=json.loads((root/"train/index.json").read_text())
        test=json.loads((root/"test/index.json").read_text())
        names=sorted(set(train)-set(test),key=lambda key:hashlib.sha256(f"{seed}:{key}".encode()).hexdigest())
        picked={}; current_path=None; chunk={}; skipped=0
        for name in names:
            path=train_chunk_path(root,train[name])
            if path != current_path:
                chunk={item["key"]:item for item in torch.load(path,map_location="cpu",weights_only=False)}
                current_path=path
            item=chunk.get(name)
            if item is None:
                raise ValueError(f"training scene absent from indexed chunk: {name}")
            try:
                cameras=item["cameras"]
                if cameras.shape[0] < 36 or not torch.isfinite(cameras).all():
                    raise ValueError("insufficient/invalid cameras")
                c2w,intrinsics=ds.DatasetRE10k.convert_poses(None,cameras)
                if (get_fov(intrinsics).rad2deg()>100).any():
                    raise ValueError("wide field of view")
                positions=c2w[:,:3,3]
                context=_farthest_camera_indices(positions,12)
                available=torch.tensor([i for i in range(len(positions)) if i not in set(context.tolist())])
                target=available[_farthest_camera_indices(positions[available],8)]
                # Match upstream skip_bad_shape for every fixed validation frame.
                for index in [*context.tolist(),*target.tolist()]:
                    with Image.open(BytesIO(item["images"][index].numpy().tobytes())) as image:
                        if image.size != (640,360) or len(image.getbands()) != 3:
                            raise ValueError("unexpected image shape")
                picked[name]={"chunk":train[name],"context":context.tolist(),"target":target.tolist()}
            except (ValueError,RuntimeError,OSError):
                skipped+=1
                continue
            if len(picked)%16==0:
                print(f"HOLDOUT_SELECTED={len(picked)}/{count} SKIPPED={skipped}",flush=True)
            if len(picked)==count:
                break
        if len(picked)!=count:
            raise ValueError(f"only {len(picked)} valid training scenes available")
        result={"kind":"nfcgs_train_holdout_v1","scene_count":count,"seed":seed,
                "train_index_sha256":file_hash(root/"train/index.json"),
                "test_index_sha256":file_hash(root/"test/index.json"),"scenes":picked}
        atomic_json(manifest_path,result)
        read_holdout(manifest_path,root)
    index={name:{"context":row["context"],"target":row["target"]} for name,row in result["scenes"].items()}
    atomic_json(output/"validation_index.json",index)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vanilla",type=Path,required=True)
    parser.add_argument("--dataset-root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--arms",nargs="+",choices=PILOT_ARMS,default=list(PILOT_ARMS))
    parser.add_argument("--seed",type=int,default=111123)
    parser.add_argument("--holdout-reference",type=Path,
                        help="copy the exact s0 holdout bytes; selection seed stays zero")
    args=parser.parse_args()
    root=args.dataset_root.resolve(); output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    if args.holdout_reference is not None:
        reference = args.holdout_reference.resolve()
        read_holdout(reference, root)
        target = output / "holdout.json"
        if target.exists() and target.read_bytes() != reference.read_bytes():
            raise ValueError("existing holdout differs from the frozen s0 holdout")
        if not target.exists():
            with tempfile.NamedTemporaryFile(dir=output, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(reference.read_bytes()); stream.flush(); os.fsync(stream.fileno())
            os.replace(temporary, target)
    create_holdout(root,output)
    from globalsplat.model.globalsplat import GlobalSplat
    from globalsplat.compression.checkpoint import validate_feature_codec_checkpoint
    vanilla=torch.load(args.vanilla,map_location="cpu",weights_only=False)
    source=_checkpoint_state(vanilla)
    vanilla_hash=file_hash(args.vanilla)
    common=shared_fresh_codec_state(args.seed)
    provenance={"version":1,"seed":args.seed,"vanilla_sha256":vanilla_hash,
                "holdout_sha256":file_hash(output/"holdout.json"),
                "validation_index_sha256":file_hash(output/"validation_index.json"),
                "policy":"fresh_codec_common_compatible_tensors_QR_geometry_trainable"}
    prepared={}
    shared_hashes={}
    for arm in args.arms:
        target=output/f"{arm}.ckpt"
        arm_meta={**provenance,"arm":arm}
        if target.exists():
            saved=torch.load(target,map_location="cpu",weights_only=False)
            if saved.get("architecture_pilot_initialization",{}).get("signature") != arm_meta:
                raise ValueError(f"existing initializer provenance mismatch: {target}")
            validate_feature_codec_checkpoint(saved,pilot_config(arm))
        else:
            torch.manual_seed(args.seed)
            model=GlobalSplat(sh_degree=3,static_only=True,patch_size=8,latent_rep_token_amount=4096,
                             dim_latents=512,dim_rays=256,dim_rgb_feat=512,rounds=4,
                             slot_calib_layers_per_round=2,M_max=16,use_camera_diff_as_input=False,
                             feature_codec=pilot_config(arm).to_dict(),freeze_globalsplat=True)
            missing=[name for name,value in model.state_dict().items() if not name.startswith("feature_codec.")
                     and name not in _READOUTS and (name not in source or source[name].shape != value.shape)]
            if missing:
                raise ValueError(f"vanilla checkpoint does not cover the common backbone: {missing}")
            shared_keys=apply_shared_fresh_state(model.feature_codec,common)
            report=initialize_observable_from_vanilla(model,vanilla)
            if report.max_reparameterization_error>1e-4:
                raise ValueError("observable initialization error exceeds tolerance")
            state=model.feature_codec.state_dict()
            hashes={key:tensor_hash(state[key]) for key in shared_keys}
            hashes.update({key:tensor_hash(model.state_dict()[key]) for key in _READOUTS})
            saved={"state_dict":{f"model.{k}":v.cpu() for k,v in model.state_dict().items()},
                   "feature_codec_config":model.feature_codec.config.to_dict(),"score_mean_offset_enabled":True,
                   "architecture_pilot_initialization":{"signature":arm_meta,"shared_tensor_sha256":hashes,
                                                        "observable_initialization":report.to_dict()}}
            validate_feature_codec_checkpoint(saved,pilot_config(arm))
            with tempfile.NamedTemporaryFile(dir=output,delete=False,suffix=".ckpt") as handle:
                tmp=Path(handle.name)
            try:
                torch.save(saved,tmp)
                os.replace(tmp,target)
            finally:
                tmp.unlink(missing_ok=True)
            del model
        # A resumed preparation on another CPU must not silently mix rounded
        # QR boundaries or differently initialized compatible codec tensors.
        for key,value in saved["architecture_pilot_initialization"]["shared_tensor_sha256"].items():
            if key in shared_hashes and shared_hashes[key]!=value:
                raise ValueError(f"common initialization differs across pilot arms: {key}")
            shared_hashes[key]=value
        prepared[arm]={"path":str(target),"sha256":file_hash(target),"config":pilot_config(arm).to_dict()}
        print(f"INITIALIZED={arm} CHECKPOINT={target}",flush=True)
    atomic_json(output/"preparation.json",{"kind":"nfcgs_architecture_preparation_v1",**provenance,"shared_initialization_checked":True,"arms":prepared})
    print(f"PREPARATION_COMPLETE={output}",flush=True)


if __name__=="__main__":
    main()
