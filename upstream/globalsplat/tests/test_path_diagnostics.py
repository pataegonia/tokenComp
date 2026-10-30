"""Actual causal symbols, recomputed residuals and matched-scene provenance."""
import contextlib
import copy
from dataclasses import replace
import importlib.util
import io
import json
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
from globalsplat.compression.bitstream import SceneBitstream
from globalsplat.compression import path_diagnostics as paths

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("path_diagnostic_runner", ROOT / "scripts/eval_nfcgs_path_diagnostics.py")
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)


def make_codec():
    torch.manual_seed(903)
    config = CodecConfig(texture_channels=8, geometry_channels=8, geometry_observable_channels=4,
        rank=7, residual_n=6, residual_m=8, adapter_hidden=6, score_slice_channels=3,
        score_context_hidden=5, score_spatial_stages=4, score_context_schedule="dyadic4",
        score_spatial_kernel=5, score_context_quantization="ste", token_order="hilbert",
        use_centering=False, use_score_norm=False, transform="linear",
        score_mean_condition=False, score_channel_context=False)
    codec = ObservableLowRank1DCodec(config).eval()
    with torch.no_grad():
        # Nontrivial normalizers ensure bypasses preserve the deployed scaling.
        codec.residual_mean.uniform_(-0.3, 0.3)
        codec.residual_std.uniform_(0.4, 1.8)
        for entropy in codec.score_context.active_entropies(codec.score_entropy):
            entropy.entropy_bottleneck.quantiles[:, 0, 1].fill_(0.31)
    codec.update(force=True, update_quantiles=False)
    return codec


class TinyDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("scalar", torch.tensor(1))

    def decoded_token_centers(self, geometry):
        return geometry[..., :3]


class TinyModel(torch.nn.Module):
    def __init__(self, codec):
        super().__init__()
        self.feature_codec = codec
        self.gaussian_decoder = TinyDecoder()

    def encode_scene_tokens(self, inputs):
        return inputs["texture"], inputs["geometry"]

    def decode_scene_tokens(self, tokens):
        return tokens


class DiagnosticPathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_actual_packet_is_production_coding_and_scores_are_causal_decoded(self):
        for dtype in (None, torch.bfloat16):
            codec = make_codec()
            inputs = tuple(torch.randn(1, 65, width) for width in (8, 8, 3))
            weights = {k: v.clone() for k, v in codec.state_dict().items()}
            with torch.no_grad(), torch.autocast("cpu", dtype=dtype, enabled=dtype is not None), \
                 patch.object(codec, "_score_forward", side_effect=AssertionError("approximate forward is not the oracle")):
                result = paths.build_paths(codec, *inputs)
                packet = codec.compress(*inputs)
                self.assertEqual(result.packet, packet.data)
                self.assertTrue(torch.equal(result.features["actual_packet"], torch.cat(codec.decompress(packet.data), -1)))
                packed = SceneBitstream.unpack(packet.data)
                with torch.autocast("cpu", enabled=False):
                    shat = codec.score_context.decompress(packed.score, torch.zeros(1, 12), 65, codec.score_entropy)
                lr = codec._synthesize_low_rank(shat.squeeze(2).transpose(1, 2))
                self.assertTrue(torch.equal(lr, result.features["lr_decoded"]))
                original = packet.order.apply(codec._pack_features(inputs[0], inputs[1]))
                self.assertTrue(torch.equal(original, result.features["b1_direct"]))
            self.assertEqual(list(result.features), list(paths.MODES))
            self.assertGreater(result.score_mse, 0)
            self.assertEqual(sum(result.stream_bytes.values()), len(result.packet))
            for key, before in weights.items():
                self.assertTrue(torch.equal(before, codec.state_dict()[key]), key)

    def test_changed_score_recomputes_msh_residual_for_each_intervention(self):
        codec = make_codec()
        inputs = tuple(torch.randn(1, 65, width) for width in (8, 8, 3))
        coded_inputs, continuous_inputs = [], []
        real_coded, real_continuous = paths.coded_msh, paths.continuous_msh
        def coded(c, residual):
            coded_inputs.append(residual.clone())
            return real_coded(c, residual)
        def continuous(c, residual):
            continuous_inputs.append(residual.clone())
            return real_continuous(c, residual)
        with patch.object(paths, "coded_msh", side_effect=coded), \
             patch.object(paths, "continuous_msh", side_effect=continuous):
            result = paths.build_paths(codec, *inputs)
        f, ls, lhat = (result.features[name] for name in ("b1_direct", "lr_continuous", "lr_decoded"))
        torch.testing.assert_close(coded_inputs[0], f - ls, rtol=0, atol=0)
        torch.testing.assert_close(continuous_inputs[0], f - lhat, rtol=0, atol=0)
        torch.testing.assert_close(continuous_inputs[1], f - ls, rtol=0, atol=0)
        self.assertFalse(torch.equal(continuous_inputs[0], continuous_inputs[1]))

    def test_continuous_msh_keeps_adapters_scaling_and_crop_but_calls_no_quantizer(self):
        codec = make_codec()
        residual = torch.randn(1, 65, 12)
        with torch.no_grad(), \
             patch.object(codec.residual_codec.entropy_bottleneck, "forward", side_effect=AssertionError("z quantizer")), \
             patch.object(codec.residual_codec.gaussian_conditional, "forward", side_effect=AssertionError("y quantizer")):
            actual = paths.continuous_msh(codec, residual)
            normalized = (residual.transpose(1, 2).unsqueeze(2) - codec.residual_mean) / codec.residual_std
            msh = codec.residual_codec
            expected = msh.synthesis_adapter(msh.g_s(msh.g_a(msh.analysis_adapter(normalized))))[..., :1, :65]
            expected = (expected * codec.residual_std + codec.residual_mean).squeeze(2).transpose(1, 2)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertFalse(torch.equal(actual, residual))

    def test_rejects_nonminimal_or_training_codec(self):
        codec = make_codec()
        codec.train()
        with self.assertRaisesRegex(ValueError, "eval"):
            paths.validate_codec(codec)
        codec.eval()
        codec.config = replace(codec.config, use_centering=True)
        with self.assertRaisesRegex(ValueError, "minimal"):
            paths.validate_codec(codec)


class DiagnosticRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self, directory):
        root = Path(directory)
        source = root / "source"
        source.mkdir()
        model = TinyModel(make_codec()).eval()
        checkpoint = root / "fixture.ckpt"
        checkpoint.write_bytes(b"checkpoint fixture, loader is mocked")
        spec = dict(label="fixture", checkpoint=str(checkpoint), expected_step=50000, tokens=65,
                    rate_lambda=.0128, config=model.feature_codec.config.to_dict())
        manifest = dict(version=1, points=[spec])
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        batches = []
        for i in range(2):
            inputs = dict(texture=torch.randn(1, 65, 8), geometry=torch.randn(1, 65, 8),
                          images=torch.zeros(1, 12, 3, 256, 256), frame_ids=torch.arange(12).reshape(1, 12))
            targets = dict(images=torch.full((1, 8, 3, 256, 256), .4), frame_ids=torch.arange(12, 20).reshape(1, 8))
            batches.append(dict(scene_info=dict(scene=[f"scene_{i}"]), inputs=inputs, targets=targets))
        cache = [dict(scene=f"scene_{i}", inputs=batch["inputs"]) for i, batch in enumerate(batches)]
        fingerprint = runner.profile.input_fingerprint(cache)
        args = SimpleNamespace(manifest=manifest_path, input_run=source, expected_input_fingerprint=fingerprint,
            output=root / "output", device="cpu", precision="fp32", scenes=2, tokens=65,
            seed=0, threads=1, num_workers=0, dataset_root=root / "data")
        identity = {key: getattr(args, key) for key in ("scenes", "tokens", "seed", "precision")}
        torch.save(cache, source / "inputs.pt")
        for name, value in {
            "inputs.json": dict(fingerprint=fingerprint, scenes=[item["scene"] for item in cache]),
            "metadata.json": dict(identity=identity),
            "complete.json": dict(scenes=2, input_fingerprint=fingerprint, points=["fixture"]),
            "backbone.json": dict(sha256=runner.backbone_hash(model)),
        }.items():
            (source / name).write_text(json.dumps(value), encoding="utf-8")
        def render(tokens, targets):
            scalar = torch.sigmoid(torch.cat(tokens, -1).mean())
            return torch.ones_like(targets["images"][0]) * scalar
        def metrics(gt, pred):
            mse = float((gt - pred).square().mean())
            return dict(psnr=-10 * __import__("math").log10(max(mse, 1e-12)), ssim=1-mse, lpips=mse)
        return args, manifest, spec, model, batches, (render, metrics)

    @contextlib.contextmanager
    def patched(self, model, batches, functions):
        with patch.object(runner.profile, "load_model", return_value=(copy.deepcopy(model), None, {})), \
             patch.object(runner, "load_data", return_value=iter(batches)), \
             patch.object(runner, "render_functions", return_value=functions), \
             patch.object(runner, "code_hashes", return_value={"fixture": "unchanged"}), \
             contextlib.redirect_stdout(io.StringIO()):
            yield

    def test_end_to_end_all_modes_rates_fingerprints_reports_and_reordered_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            args, manifest, spec, model, batches, functions = self.fixture(directory)
            with self.patched(model, list(reversed(batches)), functions):
                result = runner.run(args, manifest, spec)
            self.assertEqual(result["scene_count"], 2)
            self.assertEqual(result["renderer_repeat"]["max_pixel_abs"], 0)
            self.assertTrue((args.output / "REPORT.md").is_file())
            self.assertTrue((args.output / "complete.json").is_file())
            self.assertEqual(runner.read_json(args.output / "metadata.json")["backbone"]["kind"], "source_analysis")
            numerics = runner.read_json(args.output / "metadata.json")["numerics"]
            self.assertTrue(numerics["cudnn_deterministic"])
            self.assertFalse(numerics["cudnn_benchmark"])
            self.assertFalse(numerics["cudnn_allow_tf32"])
            self.assertFalse(numerics["matmul_allow_tf32"])
            for i in range(2):
                row = runner.read_json(args.output / f"scenes/scene_{i:04d}.json")
                self.assertEqual(row["scene"], f"scene_{i}")
                self.assertEqual(row["target_frame_ids"], list(range(12, 20)))
                packet = (args.output / row["packet_file"]).read_bytes()
                self.assertEqual(len(packet), row["modes"]["actual_packet"]["actual_bytes"])
                for mode in paths.MODES[:-1]:
                    self.assertIsNone(row["modes"][mode]["actual_bytes"])
                    self.assertIsNone(result["modes"][mode]["mean_actual_bytes"])
            with self.assertRaises(FileExistsError):
                runner.run(args, manifest, spec)

    def test_input_mismatch_is_fatal_without_sampling_replacement_or_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            args, manifest, spec, model, batches, functions = self.fixture(directory)
            batches[0]["inputs"]["texture"][0, 0, 0] += .25
            with self.patched(model, batches, functions), self.assertRaisesRegex(ValueError, "original cache"):
                runner.run(args, manifest, spec)
            self.assertFalse((args.output / "complete.json").exists())

    def test_missing_source_cache_and_changed_cache_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args, *_ = self.fixture(directory)
            torch.save([], args.input_run / "inputs.pt")
            with self.assertRaisesRegex(ValueError, "content differs"):
                runner.source_info(args, load_cache=True)
            (args.input_run / "inputs.pt").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "do not resample"):
                runner.source_info(args, load_cache=False)

    def test_missing_selected_scene_cannot_produce_complete_result(self):
        with tempfile.TemporaryDirectory() as directory:
            args, manifest, spec, model, batches, functions = self.fixture(directory)
            with self.patched(model, batches[:1], functions), self.assertRaisesRegex(RuntimeError, "missing"):
                runner.run(args, manifest, spec)
            self.assertFalse((args.output / "complete.json").exists())

    def test_view_protocol_rejects_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, _, _, batches, _ = self.fixture(directory)
            batches[0]["targets"]["frame_ids"][0, 0] = 0
            with self.assertRaisesRegex(ValueError, "disjoint"):
                runner.validate_views(batches[0])

    def test_plan_does_not_require_login_node_dataset_or_cuda(self):
        with tempfile.TemporaryDirectory() as directory:
            args, manifest, spec, _, _, _ = self.fixture(directory)
            args.plan = True
            with patch.object(runner, "parse_args", return_value=args), \
                 patch.object(runner, "load_spec", return_value=(manifest, spec)), \
                 contextlib.redirect_stdout(io.StringIO()) as output:
                runner.main()
            self.assertIn("DATASET_CHECK=deferred_to_compute_node", output.getvalue())
            self.assertFalse(args.output.exists())

    def test_slurm_plan_and_compute_run_keep_protocol_and_lf(self):
        bash = Path("C:/Program Files/Git/bin/bash.exe")
        if not bash.is_file():
            bash = shutil.which("bash")
            if not bash:
                self.skipTest("bash unavailable")
        script = ROOT / "scripts/slurm/eval_nfcgs_path_diagnostics.slurm"
        self.assertTrue(script.read_bytes().startswith(b"#!/usr/bin/env bash\n"))
        self.assertNotIn(b"\r", script.read_bytes())
        syntax = subprocess.run([str(bash), "-n", str(script)], capture_output=True, text=True)
        self.assertEqual(syntax.returncode, 0, syntax.stderr)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in ("scripts/slurm/eval_nfcgs_path_diagnostics.slurm",
                    "globalsplat/compression/path_diagnostics.py", "scripts/eval_nfcgs_path_diagnostics.py",
                    "config/plans/nfcgs_path_diagnostics_20261019.json", "scripts/profile_nfcgs_complexity.py"):
                dest = root / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / relative, dest)
            profile = root / "fake_conda/etc/profile.d"
            profile.mkdir(parents=True)
            (profile / "conda.sh").write_text("conda() { return 0; }\n", newline="\n")
            fake = root / "fake_bin"
            fake.mkdir()
            stubs = {
                "conda": '#!/usr/bin/env bash\nprintf "%s\\n" "$FAKE_CONDA_BASE"\n',
                "srun": '#!/usr/bin/env bash\nshift\nexec "$@"\n',
                "python": '#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$FAKE_LOG"\n',
            }
            for name, content in stubs.items():
                (fake / name).write_text(content, newline="\n")
                (fake / name).chmod(0o755)
            env = dict(os.environ, REPO_DIR=root.as_posix(),
                       FAKE_CONDA_BASE=(root / "fake_conda").as_posix(),
                       FAKE_LOG=(root / "calls.txt").as_posix(), NFCGS_TMP_ROOT="job_tmp")
            for key in ("SLURM_JOB_ID", "PATH_DIAGNOSTIC_INPUT_RUN", "PATH_DIAGNOSTIC_OUTPUT", "PATH_DIAGNOSTIC_MANIFEST"):
                env.pop(key, None)
            def invoke(option):
                return subprocess.run([str(bash), "-c", 'export PATH="$(pwd)/fake_bin:$PATH"; '
                    'bash scripts/slurm/eval_nfcgs_path_diagnostics.slurm ' + option],
                    cwd=root, env=env, capture_output=True, text=True)
            planned = invoke("plan")
            self.assertEqual(planned.returncode, 0, planned.stdout + planned.stderr)
            self.assertFalse((root / "job_tmp").exists())
            env["SLURM_JOB_ID"] = "12345"
            executed = invoke("")
            self.assertEqual(executed.returncode, 0, executed.stdout + executed.stderr)
            self.assertTrue((root / "job_tmp/path_diagnostics_12345").is_dir())
            calls = (root / "calls.txt").read_text().splitlines()
            self.assertEqual(len(calls), 2)
            for call in calls:
                for flag in ("--scenes 256", "--tokens 4096", "--seed 0", "--precision bf16",
                             "--device cuda", "--input-run", "job_454995", "--expected-input-fingerprint"):
                    self.assertIn(flag, call)
            self.assertIn("--plan", calls[0])
            self.assertNotIn("--plan", calls[1])
            self.assertIn("nfcgs_path_diagnostics_20261019.json", calls[1])
            fresh = root / "config/plans/fresh.json"
            shutil.copyfile(ROOT / "config/plans/nfcgs_path_diagnostics_fresh120k_h_on_lambda0.json", fresh)
            env["PATH_DIAGNOSTIC_MANIFEST"] = fresh.as_posix()
            overridden = invoke("")
            self.assertEqual(overridden.returncode, 0, overridden.stdout + overridden.stderr)
            self.assertIn(f"--manifest {fresh.as_posix()}", (root / "calls.txt").read_text().splitlines()[2])
            env["PATH_DIAGNOSTIC_MANIFEST"] = (root / "config/plans/missing.json").as_posix()
            self.assertNotEqual(invoke("").returncode, 0)


READOUTS = ("gaussian_decoder.geo_pos_readout.weight", "gaussian_decoder.geo_param_readout.weight",
            "gaussian_decoder.gate_readout.weight")


class FrozenFixture(torch.nn.Module):
    """Released encoder and geometry readouts plus a codec-side projection."""

    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(3, 2)
        self.gaussian_decoder = torch.nn.Module()
        for name, rows in (("geo_pos_readout", 1), ("geo_param_readout", 2), ("gate_readout", 1)):
            setattr(self.gaussian_decoder, name, torch.nn.Linear(4, rows))
        self.feature_codec = torch.nn.Module()
        self.feature_codec.geometry_projection = torch.nn.Linear(6, 4, bias=False)


class FreshBackboneTests(unittest.TestCase):
    def fixture(self, root, *, readout_offset=0.0, released_encoder_offset=0.0):
        torch.manual_seed(5)
        model = FrozenFixture()
        released_readouts = torch.randn(4, 6)
        q, r = torch.linalg.qr(released_readouts.double().T, mode="reduced")
        state = model.state_dict()
        state["feature_codec.geometry_projection.weight"] = q.T.float()
        for name, part in zip(READOUTS, (r.T.float() + readout_offset).split((1, 2, 1))):
            state[name] = part
        model.load_state_dict(state)
        released = {k: v.clone() for k, v in state.items() if not k.startswith("feature_codec.")}
        released["encoder.weight"] = released["encoder.weight"] + released_encoder_offset
        for name, part in zip(READOUTS, released_readouts.split((1, 2, 1))):
            released[name] = part
        prep = root / "prep"
        prep.mkdir()
        torch.save({"state_dict": {f"model.{k}": v.clone() for k, v in state.items()}}, prep / "two_path56.ckpt")
        vanilla = root / "vanilla.ckpt"
        torch.save({"state_dict": {f"model.{k}": v for k, v in released.items()}}, vanilla)
        checkpoint = root / "step000120000.ckpt"
        torch.save({"fresh_rd_protocol": dict(cell="h_on", preparation=str(prep),
            initializer_sha256=runner.file_hash(prep / "two_path56.ckpt"),
            vanilla_sha256=runner.file_hash(vanilla))}, checkpoint)
        with torch.no_grad():  # training moves only the codec-side projection
            model.feature_codec.geometry_projection.weight.add_(0.1)
        spec = dict(backbone_reference=dict(kind="fresh_initializer", vanilla_checkpoint=str(vanilla)))
        return model, spec, checkpoint, dict(backbone_sha256="warm-start lineage backbone")

    def test_accepts_frozen_initializer_and_records_factorization(self):
        with tempfile.TemporaryDirectory() as directory:
            model, spec, checkpoint, source = self.fixture(Path(directory))
            result = runner.verify_backbone(model, spec, checkpoint, source)
            self.assertEqual(result["kind"], "fresh_initializer")
            self.assertEqual(result["sha256"], runner.backbone_hash(model))
            self.assertLess(result["max_readout_reparameterization_error"], 1e-5)

    def test_source_analysis_backbone_needs_no_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            model, _, checkpoint, _ = self.fixture(Path(directory))
            result = runner.verify_backbone(model, {}, checkpoint, dict(backbone_sha256=runner.backbone_hash(model)))
            self.assertEqual(result["kind"], "source_analysis")
            with self.assertRaisesRegex(ValueError, "original 256-scene analysis"):
                runner.verify_backbone(model, {}, checkpoint, dict(backbone_sha256="other"))

    def test_rejects_changed_backbone_artifacts_or_factorization(self):
        cases = {
            "changed after initialization": dict(),
            "initializer differs": dict(),
            "released checkpoint differs": dict(),
            "differs from the released checkpoint": dict(released_encoder_offset=0.5),
            "do not factor": dict(readout_offset=1e-2),
        }
        for message, options in cases.items():
            with self.subTest(message), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                model, spec, checkpoint, source = self.fixture(root, **options)
                if message == "changed after initialization":
                    with torch.no_grad():
                        model.encoder.weight.add_(1e-3)
                elif message == "initializer differs":
                    torch.save({"state_dict": {}}, root / "prep/two_path56.ckpt")
                elif message == "released checkpoint differs":
                    torch.save({"state_dict": {}}, root / "vanilla.ckpt")
                with self.assertRaisesRegex(ValueError, message):
                    runner.verify_backbone(model, spec, checkpoint, source)

    def test_fresh_manifests_pass_spec_checks_and_other_steps_are_rejected(self):
        for name in ("nfcgs_path_diagnostics_fresh120k_h_on_lambda0.json",
                     "nfcgs_path_diagnostics_fresh120k_h_on_lambda0p0128.json"):
            with self.subTest(name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest = json.loads((ROOT / "config/plans" / name).read_text(encoding="utf-8"))
                self.assertIn("/lambda0", manifest["points"][0]["checkpoint"])
                manifest["points"][0]["checkpoint"] = str(root / "step000120000.ckpt")
                (root / "step000120000.ckpt").write_bytes(b"server checkpoint placeholder")
                (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
                _, spec = runner.load_spec(root / "manifest.json")
                self.assertEqual(spec["expected_step"], 120000)
                self.assertEqual(Path(spec["backbone_reference"]["vanilla_checkpoint"]),
                                 ROOT / "checkpoints/pretrained/globalsplat-re10k-32k.ckpt")
                manifest["points"][0]["expected_step"] = 60000
                (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "120k"):
                    runner.load_spec(root / "manifest.json")


if __name__ == "__main__":
    unittest.main()
