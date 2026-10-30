"""Adjacent-token difference analysis: metric, orders, CPU runner and Slurm wrapper."""
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from globalsplat.compression import CodecConfig, ObservableLowRank1DCodec

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("token_adjacency", ROOT / "scripts/analyze_nfcgs_token_adjacency.py")
adjacency = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = adjacency
spec.loader.exec_module(adjacency)
diag = adjacency.diag

TOKENS = 256


def make_codec():
    torch.manual_seed(11)
    config = CodecConfig(texture_channels=8, geometry_channels=8, geometry_observable_channels=4,
        rank=7, residual_n=6, residual_m=8, adapter_hidden=6, score_slice_channels=3,
        score_spatial_stages=4, score_context_schedule="dyadic4",
        score_spatial_kernel=5, token_order="hilbert")
    codec = ObservableLowRank1DCodec(config).eval()
    with torch.no_grad():  # observable geometry = first four geometry channels
        codec.geometry_projection.weight.zero_()
        codec.geometry_projection.weight[:, :4] = torch.eye(4)
    return codec


class Decoder(torch.nn.Module):
    def decoded_token_centers(self, observable):
        return observable[..., :3]


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_codec = make_codec()
        self.gaussian_decoder = Decoder()

    def encode_scene_tokens(self, inputs):
        return inputs["texture"], inputs["geometry"]


def smooth_scene(seed):
    """Tokens in random slot order whose features vary smoothly with position."""
    generator = torch.Generator().manual_seed(seed)
    xyz = torch.rand(TOKENS, 3, generator=generator)
    weights = torch.randn(3, 12, generator=generator) * 1.5
    field = torch.sin(xyz @ weights) + .01 * torch.randn(TOKENS, 12, generator=generator)
    geometry = torch.cat([xyz, field[:, :5]], -1)
    return dict(texture=field[:, 4:][None].contiguous(), geometry=geometry[None].contiguous())


def arguments(root, **overrides):
    values = dict(input_run=root / "source", output=root / "output", device="cpu", threads=1, tokens=TOKENS,
                  limit=0, random_draws=3, nn_backend="legacy", graph_candidates=8,
                  dataset_root=root / "data", num_workers=0, seed=0)
    values.update(overrides)
    return SimpleNamespace(**values)


class AdjacencyMetricTests(unittest.TestCase):
    def test_mean_step_and_scene_normalization(self):
        values = torch.tensor([[0.], [1.], [3.]])
        result = adjacency.adjacent(values, torch.arange(3), torch.tensor([2.]))
        self.assertAlmostEqual(result["l1"], 1.5)
        self.assertAlmostEqual(result["normalized_l1"], .75)
        self.assertEqual(result, adjacency.adjacent(values, torch.tensor([2, 1, 0]), torch.tensor([2.])))
        xyz = torch.tensor([[0., 0., 0.], [3., 4., 0.]])
        self.assertAlmostEqual(adjacency.adjacent(xyz, torch.arange(2), torch.tensor(5.), euclidean=True)["normalized_l1"], 1.)

    def test_orders_are_deterministic_bijections(self):
        model = Model()
        scene = smooth_scene(1)
        xyz = scene["geometry"][..., :3]
        units = torch.randn(1, TOKENS, 7)
        args = arguments(Path("."))
        first, stats = adjacency.token_orders(model.feature_codec, xyz, units, "scene_a", args)
        second, _ = adjacency.token_orders(model.feature_codec, xyz, units, "scene_a", args)
        self.assertEqual(tuple(first), adjacency.ORDERS)
        for name in adjacency.ORDERS:
            for a, b in zip(first[name], second[name]):
                self.assertTrue(torch.equal(a, b), name)
                self.assertTrue(torch.equal(a.sort().values, torch.arange(TOKENS)), name)
            self.assertGreaterEqual(stats[name]["ms"], 0)
        self.assertEqual(len(first["random"]), 3)
        self.assertFalse(torch.equal(first["random"][0], first["random"][1]))
        self.assertTrue(torch.equal(first["hilbert"][0], adjacency.hilbert_order_3d(xyz, 10).permutation[0]))
        for exact, graph in adjacency.GRAPH_PAIRS:
            self.assertEqual(int(first[exact][0][0]), int(first[graph][0][0]))  # same Morton start
            self.assertTrue(0 <= stats[graph]["shared_edges"] <= 1)
            self.assertGreaterEqual(stats[graph]["jumps"], 0)

    def test_shared_edges_is_direction_free(self):
        order = torch.tensor([0, 1, 2, 3])
        self.assertEqual(adjacency.shared_edges(order, order.flip(0)), 1.0)
        self.assertAlmostEqual(adjacency.shared_edges(order, torch.tensor([0, 2, 1, 3])), 1 / 3)


class AdjacencyRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self, root):
        source = root / "source"
        source.mkdir()
        cache = [dict(scene=f"scene_{i}", inputs=smooth_scene(i)) for i in range(2)]
        fingerprint = diag.profile.input_fingerprint(cache)
        torch.save(cache, source / "inputs.pt")
        diag.save_json(source / "inputs.json", dict(fingerprint=fingerprint, scenes=[c["scene"] for c in cache]))
        diag.save_json(source / "backbone.json", dict(sha256="lineage backbone"))
        checkpoint = root / "step000120000.ckpt"
        checkpoint.write_bytes(b"loader is mocked")
        spec = dict(label="fixture", rate_lambda=.0128, expected_step=120000, checkpoint=str(checkpoint))
        return arguments(root, expected_input_fingerprint=fingerprint), dict(version=1, points=[spec]), spec

    @contextlib.contextmanager
    def patched(self):
        with patch.object(diag.profile, "load_model", return_value=(Model(), None, {})), \
             patch.object(diag, "verify_backbone", return_value=dict(kind="fresh_initializer")), \
             contextlib.redirect_stdout(io.StringIO()):
            yield

    def test_smooth_scene_is_smoother_along_hilbert_and_outputs_are_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            args, manifest, spec = self.fixture(Path(directory))
            with self.patched():
                summary = adjacency.run(args, manifest, spec)
            self.assertEqual(summary["scene_count"], 2)
            orders = summary["orders"]
            for signal in ("xyz", "texture", "geometry", "score"):
                self.assertLess(orders["hilbert"][signal]["vs_random"], .7, signal)
                self.assertLess(orders["morton"][signal]["vs_random"], .7, signal)
                self.assertGreater(orders["slot"][signal]["vs_random"], .8, signal)
            self.assertLess(orders["nn_score"]["score"]["vs_random"], 1)
            self.assertLess(orders["nn_score_graph"]["score"]["vs_random"], 1)
            self.assertLess(orders["nn_xyz"]["xyz"]["vs_random"], .7)
            self.assertLess(orders["nn_xyz_graph"]["xyz"]["vs_random"], .7)
            self.assertAlmostEqual(orders["random"]["features"]["vs_random"], 1.)
            self.assertEqual(set(summary["order_stats"]), set(adjacency.ORDERS))
            self.assertLess(orders["nn_score_gemm"]["score"]["vs_random"], 1)
            self.assertLess(orders["nn_score_l2"]["score"]["vs_random"], 1)
            report = (args.output / "REPORT.md").read_text(encoding="utf-8")
            self.assertIn("| nn_score_gemm |", report)
            self.assertIn("Shared consecutive pairs", report)
            for name in ("metadata.json", "summary.json", "REPORT.md", "complete.json",
                         "scenes/scene_0000.json", "scenes/scene_0001.json"):
                self.assertTrue((args.output / name).is_file(), name)
            self.assertEqual(diag.read_json(args.output / "metadata.json")["backbone"]["kind"], "fresh_initializer")
            with self.assertRaises(FileExistsError), self.patched():
                adjacency.run(args, manifest, spec)

    def test_limit_and_changed_cache_are_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            args, manifest, spec = self.fixture(Path(directory))
            args.limit = 1
            with self.patched():
                self.assertEqual(adjacency.run(args, manifest, spec)["scene_count"], 1)
            args.output, args.expected_input_fingerprint = Path(directory) / "other", "different"
            with self.patched(), self.assertRaisesRegex(ValueError, "context cache differs"):
                adjacency.run(args, manifest, spec)
            self.assertFalse(args.output.exists())


class AdjacencySlurmTests(unittest.TestCase):
    def test_cpu_only_wrapper_passes_manifest_output_and_limit(self):
        bash = Path("C:/Program Files/Git/bin/bash.exe")
        if not bash.is_file():
            bash = shutil.which("bash")
            if not bash:
                self.skipTest("bash unavailable")
        script = ROOT / "scripts/slurm/analyze_nfcgs_token_adjacency.slurm"
        content = script.read_bytes()
        self.assertTrue(content.startswith(b"#!/usr/bin/env bash\n"))
        self.assertNotIn(b"\r", content)
        self.assertNotIn(b"--gres", content)
        syntax = subprocess.run([str(bash), "-n", str(script)], capture_output=True, text=True)
        self.assertEqual(syntax.returncode, 0, syntax.stderr)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scripts/slurm").mkdir(parents=True)
            shutil.copyfile(script, root / "scripts/slurm/analyze_nfcgs_token_adjacency.slurm")
            profile = root / "fake_conda/etc/profile.d"
            profile.mkdir(parents=True)
            (profile / "conda.sh").write_text("conda() { return 0; }\n", newline="\n")
            fake = root / "fake_bin"
            fake.mkdir()
            for name, body in {"conda": 'printf "%s\\n" "$FAKE_CONDA_BASE"', "python": 'printf "%s\\n" "$*" >> "$FAKE_LOG"'}.items():
                (fake / name).write_text(f"#!/usr/bin/env bash\n{body}\n", newline="\n")
                (fake / name).chmod(0o755)
            env = dict(os.environ, REPO_DIR=root.as_posix(), FAKE_CONDA_BASE=(root / "fake_conda").as_posix(),
                       FAKE_LOG=(root / "calls.txt").as_posix(), SLURM_JOB_ID="77", ADJACENCY_LIMIT="16")
            for key in ("ADJACENCY_MANIFEST", "ADJACENCY_OUTPUT", "SLURM_CPUS_PER_TASK"):
                env.pop(key, None)
            result = subprocess.run([str(bash), "-c", 'export PATH="$(pwd)/fake_bin:$PATH"; '
                                     'bash scripts/slurm/analyze_nfcgs_token_adjacency.slurm'],
                                    cwd=root, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            call = (root / "calls.txt").read_text()
            for flag in ("nfcgs_path_diagnostics_fresh120k_h_on_lambda0p0128.json", "outputs/nfcgs_token_adjacency/job_77",
                         "--device cpu", "--threads 16", "--limit 16"):
                self.assertIn(flag, call)


if __name__ == "__main__":
    unittest.main()
