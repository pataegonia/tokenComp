#!/usr/bin/env python3
"""Fail before a 50k GPU run if a full-width codec/receiver cannot roundtrip."""
import argparse
from pathlib import Path
import tempfile
import subprocess
import sys
import torch

from globalsplat.compression.checkpoint import load_feature_codec_checkpoint
from globalsplat.compression.bitstream import SceneBitstream


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    args=parser.parse_args()
    torch.set_num_threads(8)
    if args.device=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("GPU smoke requires CUDA")
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True
    torch.set_float32_matmul_precision("highest")
    codec=load_feature_codec_checkpoint(args.checkpoint).codec.to(args.device).eval()
    codec.update(force=True)
    torch.manual_seed(17)
    inputs=tuple(torch.randn(1,4096,c,device=args.device) for c in (512,512,3))
    with torch.no_grad(),torch.autocast(device_type=args.device,dtype=torch.bfloat16):
        packet=codec.compress(*inputs).data
        expected=torch.cat(codec.decompress(packet),-1).float().cpu()
    scene=SceneBitstream.unpack(packet)
    assert sum(scene.bytes_by_stream.values())==len(packet)
    with tempfile.TemporaryDirectory() as directory:
        folder=Path(directory);(folder/"packet.bin").write_bytes(packet)
        # A fresh receiver process gets no input images, features, XYZ or order.
        code='''import sys, torch
from pathlib import Path
from globalsplat.compression.checkpoint import load_feature_codec_checkpoint
torch.set_num_threads(8)
torch.backends.cudnn.benchmark=False
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
torch.backends.cudnn.deterministic=True
torch.set_float32_matmul_precision("highest")
codec=load_feature_codec_checkpoint(sys.argv[1]).codec.to(sys.argv[3]).eval()
codec.update(force=True)
folder=Path(sys.argv[2])
with torch.no_grad(),torch.autocast(device_type=sys.argv[3],dtype=torch.bfloat16):
    result=torch.cat(codec.decompress((folder/"packet.bin").read_bytes()),-1).float().cpu()
torch.save(result,folder/"decoded.pt")
'''
        subprocess.run([sys.executable,"-c",code,str(args.checkpoint.resolve()),str(folder),args.device],check=True)
        actual=torch.load(folder/"decoded.pt",weights_only=True)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    codec.train();codec.zero_grad(set_to_none=True)
    with torch.autocast(device_type=args.device,dtype=torch.bfloat16):
        output=codec(*inputs)
        loss=output.reconstruction_sorted.float().square().mean()+1e-5*output.estimated_bits+codec.aux_loss()
    if not torch.isfinite(loss):
        raise RuntimeError("non-finite smoke loss")
    loss.backward()
    if codec.geometry_projection.weight.grad is None or not torch.isfinite(codec.geometry_projection.weight.grad).all():
        raise RuntimeError("projection gradient missing or non-finite")
    if any(not torch.isfinite(p.grad).all() for p in codec.parameters() if p.grad is not None):
        raise RuntimeError("non-finite active codec gradient")
    print(f"PILOT_SMOKE_OK mode={codec.config.codec_mode} rank={codec.config.rank} bytes={len(packet)} device={args.device}",flush=True)


if __name__=="__main__":
    main()
