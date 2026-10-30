"""Native geometry routing, shared weights, eight-Gaussian order and paired joins."""
import contextlib
import copy
import importlib.util
import io
import json
import math
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
from torch import nn

from globalsplat.model.types import Gaussians
from test_path_diagnostics import make_codec

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("b0_matched_runner", ROOT / "scripts/eval_nfcgs_b0_matched.py")
b0 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = b0
spec.loader.exec_module(b0)


class Decoder(nn.Module):
    def __init__(self, geometry):
        super().__init__()
        self.geo_pos_readout = nn.Linear(geometry, 3)
        self.geo_param_readout = nn.Linear(geometry, 3)
        self.gate_readout = nn.Linear(geometry, 1)
        self.texture_readout = nn.Linear(8, 3)

    def decoded_token_centers(self, geometry):
        return self.geo_pos_readout(geometry)

    def forward(self, tokens):
        texture, geometry = tokens
        means = (self.geo_pos_readout(geometry) + self.texture_readout(texture)).repeat_interleave(8, dim=1)
        scales = self.geo_param_readout(geometry).sigmoid().repeat_interleave(8, dim=1)
        opacities = self.gate_readout(geometry).sigmoid().repeat_interleave(8, dim=1)
        rotations = means.new_zeros(*means.shape[:2], 4)
        rotations[..., 0] = 1
        return Gaussians(means, scales, rotations, means.unsqueeze(2), opacities, means.new_zeros(()))


class Model(nn.Module):
    def __init__(self, codec=None):
        super().__init__()
        self.dim_latents = 8
        self.scene_tokens = nn.Parameter(torch.randn(65, 8))
        self.encoder = nn.Linear(8, 8)
        self.feature_codec = codec
        self.gaussian_decoder = Decoder(8 if codec is None else 4)

    def encode_scene_tokens(self, inputs):
        return self.encoder(inputs["texture"]), self.encoder(inputs["geometry"])

    def decode_scene_tokens(self, tokens):
        return self.gaussian_decoder(tokens)


def model_pair():
    native = Model().eval()
    reference = Model(make_codec()).eval()
    state = reference.state_dict()
    for key, value in native.state_dict().items():
        if key not in b0.READOUTS:
            state[key] = value.clone()
    reference.load_state_dict(state, strict=True)
    return native, reference


class B0Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self, directory):
        root = Path(directory)
        source = root / "source"
        source.mkdir()
        native, reference = model_pair()
        checkpoint = root / "reference.ckpt"
        checkpoint.write_bytes(b"integrated current reference fixture")
        args = SimpleNamespace(manifest=root / "manifest.json", native_checkpoint=root / "native.ckpt",
            input_run=source, output=root / "output", scenes=2, tokens=65, seed=0, precision="fp32",
            threads=1, num_workers=0, device="cpu", dataset_root=root / "data")
        spec = dict(label="fixture", checkpoint=str(checkpoint), expected_step=50000,
                    tokens=65, rate_lambda=.0128, config=reference.feature_codec.config.to_dict())
        manifest = dict(version=1, points=[spec])
        args.manifest.write_text(json.dumps(manifest), encoding="utf-8")
        args.native_checkpoint.write_bytes(b"native fixture; loader is mocked")
        batches = [dict(scene_info=dict(scene=[f"scene_{i}"]),
            inputs=dict(texture=torch.randn(1,65,8), geometry=torch.randn(1,65,8),
                        images=torch.zeros(1,12,3,256,256), frame_ids=torch.arange(12).reshape(1,12)),
            targets=dict(images=torch.full((1,8,3,256,256), .4), frame_ids=torch.arange(12,20).reshape(1,8)))
            for i in range(2)]
        cache = [dict(scene=f"scene_{i}", inputs=batch["inputs"]) for i, batch in enumerate(batches)]
        fingerprint = b0.diag.profile.input_fingerprint(cache)
        args.expected_input_fingerprint = fingerprint
        torch.save(cache, source / "inputs.pt")
        identity = {key: getattr(args,key) for key in ("scenes","tokens","seed","precision")}
        for name, value in {
            "inputs.json":dict(fingerprint=fingerprint,scenes=[item["scene"] for item in cache]),
            "metadata.json":dict(identity=identity),
            "complete.json":dict(scenes=2,input_fingerprint=fingerprint,points=["source"]),
            "backbone.json":dict(sha256=b0.diag.backbone_hash(reference)),
        }.items():
            (source/name).write_text(json.dumps(value),encoding="utf-8")
        def render(gaussians,targets):
            return torch.ones_like(targets["images"][0])*gaussians.means.mean().sigmoid()
        def metrics(gt,pred):
            mse=float((gt-pred).square().mean())
            return dict(psnr=-10*math.log10(max(mse,1e-12)),ssim=1-mse,lpips=mse)
        return args,manifest,spec,native,reference,batches,(render,metrics)

    @contextlib.contextmanager
    def patched(self,native,reference,batches,functions):
        with patch.object(b0,"load_native",return_value=(copy.deepcopy(native),dict(sha256="native_fixture"))), \
             patch.object(b0.diag.profile,"load_model",return_value=(copy.deepcopy(reference),None,{})), \
             patch.object(b0.diag,"load_data",return_value=iter(batches)), \
             patch.object(b0.diag,"render_functions",return_value=functions), \
             patch.object(b0,"code_hashes",return_value={"shared":"code"}), \
             contextlib.redirect_stdout(io.StringIO()):
            yield

    def execute(self,directory):
        args,manifest,spec,native,reference,batches,functions=self.fixture(directory)
        with self.patched(native,reference,list(reversed(batches)),functions):
            result=b0.run(args,manifest,spec)
        return args,result

    def test_native_keeps_unprojected_geometry_and_never_uses_codec_branches(self):
        with tempfile.TemporaryDirectory() as directory:
            args,_,_,native,reference,batches,functions=self.fixture(directory)
            with patch.object(native,"decode_scene_tokens",wraps=native.decode_scene_tokens) as native_decode, \
                 patch.object(reference,"decode_scene_tokens",wraps=reference.decode_scene_tokens) as bridge_decode, \
                 patch.object(reference,"encode_scene_tokens",side_effect=AssertionError("encode once")), \
                 patch.object(reference.feature_codec,"_analyze_low_rank",side_effect=AssertionError("no LR")), \
                 patch.object(reference.feature_codec,"compress",side_effect=AssertionError("no entropy coding")):
                result=b0.evaluate_scene(native,reference,batches[0],args,*functions)
            self.assertEqual([call.args[0][1].shape[-1] for call in native_decode.call_args_list],[8,8])
            self.assertEqual(bridge_decode.call_args.args[0][1].shape[-1],4)
            self.assertLess(max(result["native_slot_order_check"]["gaussian_field_max_abs"].values()),1e-5)
            for row in result["modes"].values(): self.assertIsNone(row["actual_bytes"])

    def test_weight_pair_requires_same_encoder_and_shared_decoder(self):
        native,reference=model_pair()
        result=b0.validate_model_pair(native,reference)
        self.assertTrue(result["encoder_tensors_equal"])
        self.assertEqual(set(result["geometry_boundary_effective_readout_errors"]),set(b0.READOUTS))
        with torch.no_grad(): reference.encoder.weight[0,0] += .1
        with self.assertRaisesRegex(ValueError,"weights differ"): b0.validate_model_pair(native,reference)
        native,reference=model_pair()
        with torch.no_grad(): reference.gaussian_decoder.texture_readout.bias[0] += .1
        with self.assertRaisesRegex(ValueError,"weights differ"): b0.validate_model_pair(native,reference)

    def test_wrong_public_checkpoint_hash_is_rejected_before_building(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"wrong.ckpt";path.write_bytes(b"wrong")
            with patch.object(b0,"native_model_config",side_effect=AssertionError("do not construct")), \
                 self.assertRaisesRegex(ValueError,"SHA-256 mismatch"):
                b0.load_native(SimpleNamespace(native_checkpoint=path))

    def test_end_to_end_complete_report_and_source_index_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            args,result=self.execute(directory)
            self.assertEqual(result["scene_count"],2)
            self.assertEqual(list(result["modes"]),list(b0.MODES))
            self.assertTrue((args.output/"REPORT.md").is_file())
            self.assertEqual(b0.diag.read_json(args.output/"complete.json")["native_checkpoint_sha256"],"native_fixture")
            for i in range(2):
                self.assertEqual(b0.diag.read_json(args.output/f"scenes/scene_{i:04d}.json")["scene"],f"scene_{i}")
            with self.assertRaises(FileExistsError): b0.run(args,None,None)

    def test_input_mismatch_and_missing_scenes_cannot_mark_complete(self):
        for mismatch in (True,False):
            with tempfile.TemporaryDirectory() as directory:
                args,manifest,spec,native,reference,batches,functions=self.fixture(directory)
                if mismatch: batches[0]["inputs"]["texture"][0,0,0] += .1
                else: batches=batches[:1]
                error=ValueError if mismatch else RuntimeError
                with self.patched(native,reference,batches,functions), self.assertRaises(error):
                    b0.run(args,manifest,spec)
                self.assertFalse((args.output/"complete.json").exists())

    def path_fixture(self,b0_output):
        output=b0_output.parent/"paths";output.mkdir()
        metadata=b0.diag.read_json(b0_output/"metadata.json")
        complete=b0.diag.read_json(b0_output/"complete.json")
        complete["modes"]=list(b0.diag.MODES)
        b0.diag.save_json(output/"metadata.json",metadata)
        b0.diag.save_json(output/"complete.json",complete)
        b0.diag.save_json(output/"summary.json",dict(renderer_repeat=dict(max_pixel_abs=0.,max_abs_psnr_delta=0.)))
        for i in range(2):
            row=b0.diag.read_json(b0_output/f"scenes/scene_{i:04d}.json")
            metric=dict(row["modes"]["b1_bridge"])
            metric["psnr"]-=.125
            row["modes"]={name:dict(metric) for name in b0.diag.MODES}
            b0.diag.save_json(output/f"scenes/scene_{i:04d}.json",row)
        return output

    def test_comparison_checks_alignment_and_exposes_b1_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            args,_=self.execute(directory)
            paths=self.path_fixture(args.output)
            output=Path(directory)/"comparison"
            with contextlib.redirect_stdout(io.StringIO()):
                result=b0.compare_runs(args.output,paths,output)
            self.assertAlmostEqual(result["mean_b1_bridge_minus_t05_b1_psnr"],.125)
            self.assertAlmostEqual(result["max_abs_b1_bridge_minus_t05_b1_psnr"],.125)
            self.assertTrue((output/"REPORT.md").is_file())
            self.assertEqual(len(b0.diag.read_json(output/"paired_scenes.json")),2)

    def test_comparison_rejects_frame_checkpoint_and_code_mismatch(self):
        for variant in ("frame","checkpoint","code","incomplete"):
            with tempfile.TemporaryDirectory() as directory:
                args,_=self.execute(directory);paths=self.path_fixture(args.output)
                if variant=="frame":
                    path=paths/"scenes/scene_0000.json";row=b0.diag.read_json(path)
                    row["target_frame_ids"][0]=99
                elif variant=="checkpoint":
                    path=paths/"complete.json";row=b0.diag.read_json(path);row["checkpoint_sha256"]="wrong"
                elif variant=="code":
                    path=paths/"metadata.json";row=b0.diag.read_json(path);row["code_sha256"]["shared"]="wrong"
                else:
                    (paths/"complete.json").unlink()
                if variant!="incomplete": path.write_text(json.dumps(row),encoding="utf-8")
                output=Path(directory)/"bad_comparison"
                with self.assertRaises((ValueError,FileNotFoundError)):
                    b0.compare_runs(args.output,paths,output)
                self.assertFalse(output.exists())

    def test_plan_defers_login_node_dataset_model_and_cuda_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            args,manifest,spec,*_=self.fixture(directory);args.plan=True;args.compare_only=False
            with patch.object(b0,"parse_args",return_value=args), \
                 patch.object(b0.diag,"load_spec",return_value=(manifest,spec)), \
                 contextlib.redirect_stdout(io.StringIO()) as text:
                b0.main()
            self.assertIn("deferred_to_compute_node",text.getvalue())
            self.assertFalse(args.output.exists())

    def test_slurm_plan_and_run_are_readonly_one_gpu_with_frozen_protocol(self):
        bash=Path("C:/Program Files/Git/bin/bash.exe")
        if not bash.is_file():
            bash=shutil.which("bash")
            if not bash:self.skipTest("bash unavailable")
        relative="scripts/slurm/eval_nfcgs_b0_matched.slurm"
        raw=(ROOT/relative).read_bytes()
        self.assertTrue(raw.startswith(b"#!/usr/bin/env bash\n"));self.assertNotIn(b"\r",raw)
        syntax=subprocess.run([str(bash),"-n",str(ROOT/relative)],capture_output=True,text=True)
        self.assertEqual(syntax.returncode,0,syntax.stderr)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in (relative,"scripts/eval_nfcgs_b0_matched.py","scripts/eval_nfcgs_path_diagnostics.py",
                    "scripts/profile_nfcgs_complexity.py","globalsplat/compression/path_diagnostics.py",
                    "config/plans/nfcgs_path_diagnostics_20261019.json","scripts/slurm/eval_nfcgs_path_diagnostics.slurm"):
                dest=root/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
            conda=root/"fake_conda/etc/profile.d";conda.mkdir(parents=True)
            (conda/"conda.sh").write_text("conda() { return 0; }\n",newline="\n")
            fake=root/"fake_bin";fake.mkdir()
            for name,content in {
                "conda":'#!/usr/bin/env bash\nprintf "%s\\n" "$FAKE_CONDA_BASE"\n',
                "srun":'#!/usr/bin/env bash\nshift\nexec "$@"\n',
                "python":'#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$FAKE_LOG"\n',
            }.items():
                (fake/name).write_text(content,newline="\n");(fake/name).chmod(0o755)
            env=dict(os.environ,REPO_DIR=root.as_posix(),FAKE_CONDA_BASE=(root/"fake_conda").as_posix(),
                     FAKE_LOG=(root/"calls.txt").as_posix(),NFCGS_TMP_ROOT="tmp_job")
            for key in ("SLURM_JOB_ID","B0_MATCHED_OUTPUT","B0_MATCHED_INPUT_RUN"):env.pop(key,None)
            def invoke(option):
                return subprocess.run([str(bash),"-c",'export PATH="$(pwd)/fake_bin:$PATH"; bash '+relative+' '+option],
                    cwd=root,env=env,capture_output=True,text=True)
            plan=invoke("plan");self.assertEqual(plan.returncode,0,plan.stdout+plan.stderr)
            self.assertFalse((root/"tmp_job").exists())
            env["SLURM_JOB_ID"]="12345"
            run=invoke("");self.assertEqual(run.returncode,0,run.stdout+run.stderr)
            self.assertTrue((root/"tmp_job/b0_matched_12345").is_dir())
            calls=(root/"calls.txt").read_text().splitlines();self.assertEqual(len(calls),2)
            for call in calls:
                for flag in ("--native-checkpoint","--scenes 256","--tokens 4096","--precision bf16","--seed 0","job_454995"):
                    self.assertIn(flag,call)
            self.assertIn("--plan",calls[0]);self.assertNotIn("--plan",calls[1])


if __name__=="__main__":unittest.main()
