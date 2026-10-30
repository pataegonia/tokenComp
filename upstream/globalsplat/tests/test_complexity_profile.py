"""Counts and isolated coding paths must reflect the executed codec branches."""
import importlib.util
import json
import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from globalsplat.compression import ObservableLowRank1DCodec
from codec_helpers import config

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("profile_complexity", ROOT / "scripts/profile_nfcgs_complexity.py")
profile = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = profile
SPEC.loader.exec_module(profile)


class TinyDecoder(torch.nn.Module):
    def decoded_token_centers(self, value):
        return value[..., :3]


class TinyModel(torch.nn.Module):
    def __init__(self, codec):
        super().__init__()
        self.feature_codec = codec
        self.gaussian_decoder = TinyDecoder()

    def encode_scene_tokens(self, inputs):
        return inputs["texture"], inputs["geometry"]

    def compress_scene_tokens(self, encoded):
        texture, geometry = encoded
        positions = self.gaussian_decoder.decoded_token_centers(self.feature_codec.project_geometry(geometry))
        return self.feature_codec.compress(texture, geometry, positions)

    def decode_scene_tokens(self, decoded):
        return decoded


class ComplexityProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def make(self, **extra):
        return ObservableLowRank1DCodec(config(score_spatial_kernel=5, **extra)).eval()

    def test_registered_macs_count_functional_matmul_and_transposed_convolution(self):
        x, w = torch.randn(2, 4, 7), torch.randn(7, 3)
        _, count = profile.covered_macs(lambda: x @ w)
        self.assertEqual(count["neural_macs"], 2*4*7*3)
        conv = torch.nn.ConvTranspose2d(2, 3, (1, 5), stride=(1, 2))
        _, count = profile.covered_macs(lambda: conv(torch.randn(1, 2, 1, 8)))
        self.assertEqual(count["neural_macs"], 8*2*3*5)

    def test_compatibility_weights_excluded_even_after_requires_grad_changes(self):
        enabled = self.make(token_order="hilbert", score_spatial_stages=3)
        disabled = self.make(token_order="hilbert", score_spatial_stages=3, score_spatial_context=False)
        a, b = profile.parameter_footprint(enabled), profile.parameter_footprint(disabled)
        self.assertEqual(a["stored_parameters"], b["stored_parameters"])
        self.assertEqual(b["groups"]["context_predictors"], 0)
        self.assertEqual(a["active_architecture_parameters"]-b["active_architecture_parameters"], a["groups"]["context_predictors"])
        enabled.requires_grad_(False)
        self.assertEqual(profile.parameter_footprint(enabled), a)
        anchors = self.make(token_order="none", score_spatial_stages=4, score_context_schedule="dyadic4")
        counts = profile.parameter_footprint(anchors)
        expected = sum(p.numel() for p in anchors.score_context.anchor_predictors.parameters())
        self.assertEqual(counts["groups"]["context_predictors"], expected)

    def test_isolated_streams_match_full_codec_for_all_current_schedules(self):
        cases = [dict(score_spatial_stages=2), dict(score_spatial_stages=3),
                 dict(score_spatial_stages=4),
                 dict(score_spatial_stages=2, score_context_schedule="quarter2"),
                 dict(score_spatial_stages=4, score_context_schedule="dyadic4"),
                 dict(score_spatial_stages=3, score_spatial_context=False)]
        for case in cases:
            with self.subTest(case=case), torch.no_grad():
                codec = self.make(token_order="none", **case)
                codec.update(force=True)
                inputs = tuple(torch.randn(1, 32, n) for n in (8, 8, 3))
                result = profile.neural_profile(codec, inputs)
                self.assertGreater(result["codec_core_encode"]["neural_macs"], result["codec_decode"]["neural_macs"])
                if not case.get("score_spatial_context", True):
                    self.assertEqual(result["score_decode"]["neural_macs"], 0)
                packed = codec.compress(*inputs)
                with patch.object(codec, "_make_order", side_effect=AssertionError("decoder sorted")):
                    codec.decompress(packed.data)

    def test_repeated_scene_encoder_samples_are_not_reinterpreted_as_new_calls(self):
        rows = [dict(stages={"codec": dict(milliseconds=value, cuda_peak_bytes=None)}) for value in (1., 2., 3.)]
        result = profile.summarize_samples(rows)["codec"]
        self.assertEqual(result["milliseconds"]["count"], 3)
        self.assertEqual(result["milliseconds"]["median"], 2.)
        self.assertNotIn("cuda_peak_bytes", result)

    def test_input_fingerprint_detects_view_or_tensor_changes(self):
        x = [dict(scene="scene", inputs={"images": torch.zeros(1, 12, 3, 2, 2)})]
        first = profile.input_fingerprint(x)
        x[0]["inputs"]["images"][0, 0, 0, 0, 0] = 1
        self.assertNotEqual(first, profile.input_fingerprint(x))

    def test_output_never_overwrites_existing_results(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "result.json"
            profile.save_new(path, {"original": True})
            with self.assertRaises(FileExistsError):
                profile.save_new(path, {"original": False})
            self.assertEqual(json.loads(path.read_text()), {"original": True})

    def test_actual_stage_driver_excludes_warmup_and_keeps_real_call_counts(self):
        codec = self.make(token_order="hilbert", score_spatial_stages=3)
        codec.update(force=True)
        cache = [dict(scene=str(i), inputs=dict(texture=torch.randn(1, 32, 8), geometry=torch.randn(1, 32, 8))) for i in range(3)]
        args = SimpleNamespace(device="cpu", precision="fp32", tokens=32, warmup=1, scenes=2, repeats=2)
        with contextlib.redirect_stdout(io.StringIO()):
            result = profile.profile_point(TinyModel(codec), cache, args)
        self.assertEqual(len(result["samples"]), 4)
        self.assertEqual(result["timings"]["scene_encoder"]["milliseconds"]["count"], 2)
        self.assertEqual(result["timings"]["codec_core_encode"]["milliseconds"]["count"], 4)
        self.assertEqual(result["payload_bytes"]["count"], 2)
        self.assertIsNotNone(result["neural"]["pipeline_entropy_encode"])
        self.assertNotIn("cuda_peak_bytes", result["timings"]["codec_decode"])

    def test_partial_run_resumes_missing_points_and_preserves_previous_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / "first.ckpt", root / "second.ckpt"
            first.write_bytes(b"fixture")
            spec = lambda label, path: dict(label=label, checkpoint=str(path), config=self.make(token_order="none", score_spatial_stages=3).config.to_dict())
            manifest = dict(version=1, points=[spec("first", first), spec("second", second)])
            args = SimpleNamespace(device="cpu", precision="fp32", tokens=32, warmup=0, scenes=1,
                repeats=1, threads=1, seed=0, num_workers=0, only=None, skip_missing=True,
                resume=False, output=root / "profile")
            cache = [dict(scene="scene", inputs=dict(texture=torch.randn(1, 32, 8), geometry=torch.randn(1, 32, 8)))]
            def load(*_):
                codec = self.make(token_order="none", score_spatial_stages=3)
                codec.update(force=True)
                return TinyModel(codec), None, profile.parameter_footprint(codec)
            with patch.object(profile, "load_model", side_effect=load), \
                 patch.object(profile, "cache_scenes", return_value=cache), \
                 patch.object(profile, "environment", side_effect=[
                     dict(host="same-host", gpu_uuid="same-gpu", temp_directory="job_first"),
                     dict(host="same-host", gpu_uuid="same-gpu", temp_directory="job_second")]), \
                 contextlib.redirect_stdout(io.StringIO()):
                result = profile.actual_report(manifest, args)
                self.assertEqual(len(result["results"]), 1)
                original = next(args.output.glob("summary_01points_*.json"))
                before = original.read_bytes()
                second.write_bytes(b"fixture")
                args.resume = True
                result = profile.actual_report(manifest, args)
                self.assertEqual(len(result["results"]), 2)
                self.assertEqual(result["results"][0]["environment"]["temp_directory"], "job_first")
                self.assertEqual(result["results"][1]["environment"]["temp_directory"], "job_second")
                self.assertEqual(original.read_bytes(), before)
                self.assertEqual(len(list(args.output.glob("summary_02points_*.json"))), 1)

    def test_slurm_syntax_and_argument_plan(self):
        bash = Path("C:/Program Files/Git/bin/bash.exe")
        if not bash.exists():
            self.skipTest("Git Bash unavailable")
        script = ROOT / "scripts/slurm/profile_nfcgs_complexity.slurm"
        self.assertNotIn(b"\r", script.read_bytes())
        self.assertTrue(script.read_bytes().startswith(b"#!/usr/bin/env bash\n"))
        result = subprocess.run([str(bash), "-n", str(script)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            manifest.write_text("{}")
            env = dict(os.environ, REPO_DIR=ROOT.as_posix(), PROFILE_MANIFEST=manifest.as_posix())
            result = subprocess.run([str(bash), str(script), "plan"], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--mode actual", result.stdout)
            self.assertIn("--device cuda --precision bf16 --tokens 4096", result.stdout)
            self.assertIn("--scenes 32 --repeats 3", result.stdout)


if __name__ == "__main__":
    unittest.main()
