#!/usr/bin/env python3
"""Check that two full-test MSH evaluations used identical scenes and views."""

import argparse
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def load_result(root):
    scores = json.loads((root / "scores_all_avg.json").read_text(encoding="utf-8"))
    rows = json.loads((root / "actual_rate_per_scene.json").read_text(encoding="utf-8"))
    if not rows or scores.get("scene_count") != len(rows):
        raise ValueError(f"Missing or inconsistent scene count in {root}")
    frames = {
        row["scene"]: (row["context_frame_ids"], row["target_frame_ids"])
        for row in rows
    }
    if len(frames) != len(rows):
        raise ValueError(f"Duplicate scene IDs in {root}")
    if any(len(context) != 12 or len(target) != 8
           for context, target in frames.values()):
        raise ValueError(f"Expected 12 context and 8 target views per scene in {root}")
    return scores, frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_id", help="Slurm array job ID, e.g. 440685")
    parser.add_argument("--root", type=Path,
                        default=REPO / "outputs/hyper1d_msh_compare_50k/full_test")
    args = parser.parse_args()

    results = {}
    for task, variant in ((0, "single_msh"), (1, "base_residual_msh")):
        path = args.root / f"job_{args.job_id}_{task}" / variant / "evaluation/hyper1d_12h"
        results[variant] = load_result(path)

    single, single_frames = results["single_msh"]
    dual, dual_frames = results["base_residual_msh"]
    if single_frames != dual_frames:
        only_single = set(single_frames) - set(dual_frames)
        only_dual = set(dual_frames) - set(single_frames)
        changed = {scene for scene in single_frames.keys() & dual_frames.keys()
                   if single_frames[scene] != dual_frames[scene]}
        raise ValueError("Evaluation selections differ: "
                         f"single_only={len(only_single)}, dual_only={len(only_dual)}, "
                         f"frame_mismatches={len(changed)}")

    print(f"Matched scenes and context/target frames: {len(single_frames)}")
    print("model                 PSNR      LPIPS       bytes     KiB      BPGA")
    for name, scores in results.items():
        print(f"{name:21s} {scores['psnr']:7.4f}   {scores['lpips']:7.4f}   "
              f"{scores['actual_bytes']:9.1f}  {scores['actual_bytes']/1024:7.3f}  "
              f"{scores['actual_bpga']:8.5f}")
    print(f"dual - single: PSNR={dual['psnr']-single['psnr']:+.4f}, "
          f"LPIPS={dual['lpips']-single['lpips']:+.4f}, "
          f"bytes={dual['actual_bytes']-single['actual_bytes']:+.1f}, "
          f"BPGA={dual['actual_bpga']-single['actual_bpga']:+.5f}")


if __name__ == "__main__":
    main()
