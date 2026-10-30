"""Real entropy/decoder fixtures, portable process boundary and failure guards."""
import contextlib
import copy
from dataclasses import replace
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

from test_path_diagnostics import make_codec
from globalsplat.model.globalsplat import GlobalSplat
from globalsplat.model.decoder.gaussian_decoder import TokenCoarseToFine3DGS

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("independent_receiver_runner", ROOT / "scripts/eval_nfcgs_independent_receiver.py")
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)


def fake_functions():
    def render(gaussians, cameras):
        if torch.count_nonzero(cameras["images"]):
            raise AssertionError("GT pixels must not enter rendering")
        return torch.ones_like(cameras["images"][0]) * gaussians.means.float().mean().sigmoid()
    def metrics(gt, pred):
        mse = float((gt-pred).square().mean())
        return dict(psnr=-10*math.log10(max(mse, 1e-12)), ssim=1-mse, lpips=mse)
    return render, metrics


class ReceiverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self, directory, mean_offset=False):
        root = Path(directory)
        codec = make_codec()
        codec.score_context.mean_offset_enabled = mean_offset
        decoder = TokenCoarseToFine3DGS(C=8, geometry_C=4, sh_degree=3, M_max=16).eval()
        decoder.set_stage(3, mix=1.)
        checkpoint = root / "shared.ckpt"
        state = {"model.feature_codec."+k:v.clone() for k,v in codec.state_dict().items()}
        state.update({"model.gaussian_decoder."+k:v.clone() for k,v in decoder.state_dict().items()})
        state["model.scene_tokens"] = torch.randn(65,8)
        state["model.slot_encoder.unused"] = torch.tensor(99.)
        saved = dict(global_step=50000, feature_codec_config=codec.config.to_dict(),
                     score_mean_offset_enabled=mean_offset, state_dict=state)
        torch.save(saved, checkpoint)
        source = root / "source"
        source.mkdir()
        (source / "complete.json").write_text(json.dumps(dict(scene_count=2)), encoding="utf-8")
        args = SimpleNamespace(bundle=root/"bundle", output=root/"receiver", path_run=source,
            manifest=root/"spec.json", checkpoint=checkpoint, dataset_root=root/"dataset",
            device="cpu", precision="fp32", threads=1, num_workers=0)
        (args.dataset_root/"test").mkdir(parents=True)
        (args.dataset_root/"test/index.json").write_text("{}")
        config = codec.config.to_dict()
        spec = dict(checkpoint=str(checkpoint), tokens=65, expected_step=50000, config=config)
        manifest = dict(version=1, points=[spec])
        meta = dict(protocol=dict(scenes=2, tokens=65, precision="fp32", score_mean_offset_enabled=mean_offset),
            checkpoint=dict(sha256=runner.diag.file_hash(checkpoint), global_step=50000),
            environment=runner.diag.profile.environment(args), numerics=runner.numerical_policy())
        rows, batches = [], []
        render, metrics = fake_functions()
        with torch.no_grad():
            for i in range(2):
                packet = codec.compress(torch.randn(1,65,8), torch.randn(1,65,8), torch.randn(1,65,3)).data
                tokens = codec.decompress(packet)
                gaussians = GlobalSplat._decode_scene_tokens(SimpleNamespace(gaussian_decoder=decoder), tokens)
                targets = dict(images=torch.full((1,8,3,256,256), .4),
                    frame_ids=torch.arange(12,20).reshape(1,8),
                    intrinsic=torch.eye(3).expand(1,8,3,3).clone(),
                    extrinsic=torch.eye(4).expand(1,8,4,4).clone())
                cameras = dict(targets, images=torch.zeros_like(targets["images"]))
                row = dict(index=i, scene=f"scene_{i}", packet_file=f"packets/scene_{i:04d}.bin",
                    packet_sha256=runner.hashlib.sha256(packet).hexdigest(), actual_bytes=len(packet),
                    stream_bytes=runner.SceneBitstream.unpack(packet).bytes_by_stream,
                    target_frame_ids=list(range(12,20)), target_fingerprint=runner.diag.profile.input_fingerprint([targets]),
                    reference=metrics(targets["images"][0], render(gaussians,cameras)))
                path = source/row["packet_file"]; path.parent.mkdir(exist_ok=True); path.write_bytes(packet)
                rows.append(row)
                # Preparation can see context on CPU; it must not export it.
                batches.append(dict(scene_info=dict(scene=[row["scene"]]), targets=targets,
                                    inputs=dict(secret_encoder_input=torch.tensor(12345))))
        return args, manifest, spec, meta, rows, batches, codec, decoder

    def export(self, fixture):
        args,manifest,spec,meta,rows,batches,*_ = fixture
        with patch.object(runner,"read_source",return_value=(manifest,spec,meta,rows,runner.code_hashes())), \
             patch.object(runner,"dataset_config",return_value=None), \
             patch.object(runner.diag,"load_data",return_value=list(reversed(batches))), \
             contextlib.redirect_stdout(io.StringIO()):
            runner.prepare(args)

    def test_minimal_receiver_matches_production_decode_fp32_and_bf16(self):
        for offset in (False,True):
            with tempfile.TemporaryDirectory() as directory:
                fixture = self.fixture(directory,offset)
                args,_,spec,meta,rows,_,codec,decoder = fixture
                metadata = dict(meta,config=spec["config"], decoder=dict(C=8,geometry_C=4,sh_degree=3,M_max=16,stage=3,mix=1.))
                with patch.object(GlobalSplat,"__init__",side_effect=AssertionError("no scene encoder construction")):
                    receiver = runner.load_receiver(args.checkpoint,metadata,"cpu")
                self.assertEqual(set(dict(receiver.named_children())),{"codec","gaussian_decoder"})
                self.assertFalse(hasattr(receiver,"scene_tokens"))
                payload = (args.path_run/rows[0]["packet_file"]).read_bytes()
                for mixed in (False,True):
                    with torch.no_grad(),torch.autocast("cpu",dtype=torch.bfloat16,enabled=mixed):
                        expected_tokens = codec.decompress(payload)
                        expected = GlobalSplat._decode_scene_tokens(SimpleNamespace(gaussian_decoder=decoder),expected_tokens)
                        actual_tokens,actual = receiver.decode(payload)
                    for a,b in zip(expected_tokens,actual_tokens): self.assertTrue(torch.equal(a,b))
                    for key in runner.GAUSSIAN_FIELDS: self.assertTrue(torch.equal(getattr(expected,key),getattr(actual,key)),key)

    def test_portable_bundle_receive_without_dataset_source_or_encoder(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(directory); self.export(fixture); args=fixture[0]
            shutil.rmtree(args.path_run); shutil.rmtree(args.dataset_root)
            with patch.object(runner,"dataset_config",side_effect=AssertionError("no dataset")), \
                 patch.object(runner.diag,"load_data",side_effect=AssertionError("no loader")), \
                 patch.object(GlobalSplat,"__init__",side_effect=AssertionError("no encoder")), \
                 patch.object(runner.diag,"render_functions",side_effect=fake_functions), \
                 contextlib.redirect_stdout(io.StringIO()):
                summary=runner.receive(args)
            self.assertTrue(summary["passed"])
            self.assertEqual(summary["max_abs_delta"],dict.fromkeys(runner.METRICS,0.))
            self.assertTrue((args.output/"complete.json").is_file())
            self.assertTrue((args.output/"REPORT.md").is_file())
            meta,rows=runner.load_bundle(args.bundle)
            self.assertEqual([r["scene"] for r in rows],["scene_0","scene_1"])
            for row in rows:
                item=torch.load(args.bundle/row["target_file"],weights_only=False)
                self.assertEqual(set(item),{"scene","index","targets"})
            with self.assertRaises(FileExistsError): runner.receive(args)

    def test_real_fresh_python_process_consumes_only_portable_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture=self.fixture(directory);self.export(fixture);args=fixture[0]
            shutil.rmtree(args.path_run);shutil.rmtree(args.dataset_root)
            code = "\n".join([
                "import sys", f"sys.path.insert(0,{str(ROOT/'tests')!r})",
                "import test_independent_receiver as t",
                "from pathlib import Path", "from types import SimpleNamespace",
                f"args=SimpleNamespace(bundle=Path({str(args.bundle)!r}),output=Path({str(args.output)!r}),checkpoint=Path({str(args.checkpoint)!r}),device='cpu',precision='fp32',threads=1)",
                "t.runner.diag.render_functions=t.fake_functions",
                "t.runner.receive(args)"])
            result=subprocess.run([sys.executable,"-c",code],cwd=ROOT,capture_output=True,text=True,timeout=120)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn("INDEPENDENT_RECEIVER_COMPLETE scenes=2",result.stdout)
            self.assertTrue(runner.diag.read_json(args.output/"summary.json")["passed"])

    def test_corrupt_packet_and_target_never_mark_complete(self):
        for artifact in ("packet_file","target_file"):
            with tempfile.TemporaryDirectory() as directory:
                fixture=self.fixture(directory);self.export(fixture);args=fixture[0]
                _,rows=runner.load_bundle(args.bundle)
                path=args.bundle/rows[0][artifact]
                content=bytearray(path.read_bytes());content[-1]^=1;path.write_bytes(content)
                with patch.object(runner.diag,"render_functions",side_effect=fake_functions), \
                     contextlib.redirect_stdout(io.StringIO()),self.assertRaisesRegex(ValueError,"SHA-256"):
                    runner.receive(args)
                self.assertFalse((args.output/"complete.json").exists())

    def test_bundle_manifest_tampering_and_incomplete_export_are_rejected(self):
        for variant in ("manifest","missing_complete"):
            with tempfile.TemporaryDirectory() as directory:
                fixture=self.fixture(directory);self.export(fixture);args=fixture[0]
                if variant=="manifest":
                    path=args.bundle/"manifest.json";value=runner.diag.read_json(path)
                    value["rows"][0]["reference"]["psnr"]+=1
                    path.write_text(json.dumps(value),encoding="utf-8")
                else:(args.bundle/"complete.json").unlink()
                with self.assertRaises((ValueError,FileNotFoundError)):runner.load_bundle(args.bundle)

    def test_wrong_checkpoint_hash_missing_decoder_and_offset_mismatch_rejected(self):
        for variant in ("hash","decoder","offset"):
            with tempfile.TemporaryDirectory() as directory:
                fixture=self.fixture(directory);self.export(fixture);args=fixture[0]
                meta,_=runner.load_bundle(args.bundle)
                saved=torch.load(args.checkpoint,weights_only=False)
                if variant=="decoder":del saved["state_dict"]["model.gaussian_decoder.geo_pos_readout.weight"]
                elif variant=="offset":saved["score_mean_offset_enabled"]=True
                else:saved["unused"]=1
                torch.save(saved,args.checkpoint)
                if variant!="hash":meta["checkpoint"]["sha256"]=runner.diag.file_hash(args.checkpoint)
                with self.assertRaises((ValueError,RuntimeError)):runner.load_receiver(args.checkpoint,meta,"cpu")

    def test_target_mismatch_and_missing_scene_stop_export(self):
        for variant in ("mismatch","missing"):
            with tempfile.TemporaryDirectory() as directory:
                fixture=list(self.fixture(directory))
                if variant=="mismatch":fixture[5][0]["targets"]["images"][0,0,0,0,0]+=.1
                else:fixture[5]=fixture[5][:1]
                with self.assertRaises((ValueError,RuntimeError)):self.export(fixture)
                self.assertFalse((fixture[0].bundle/"complete.json").exists())

    def test_packet_dimensions_and_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args,_,_,_,rows,_,codec,_=self.fixture(directory)
            payload=(args.path_run/rows[0]["packet_file"]).read_bytes()
            with self.assertRaisesRegex(ValueError,"dimensions"):
                runner.validate_packet(payload,rows[0],codec.config,4096)
            with self.assertRaisesRegex(ValueError,"escapes"):
                runner.contained(args.path_run,"../shared.ckpt")

    def test_metric_failure_writes_report_without_success_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture=self.fixture(directory)
            fixture[4][0]["reference"]["psnr"]+=.1
            self.export(fixture);args=fixture[0]
            with patch.object(runner.diag,"render_functions",side_effect=fake_functions), \
                 contextlib.redirect_stdout(io.StringIO()),self.assertRaisesRegex(RuntimeError,"comparison failed"):
                runner.receive(args)
            self.assertFalse((args.output/"complete.json").exists())
            self.assertFalse(runner.diag.read_json(args.output/"summary.json")["passed"])
            self.assertIn("Status: FAIL",(args.output/"REPORT.md").read_text())

    def test_environment_policy_and_code_mismatch_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture=self.fixture(directory);args=fixture[0]
            meta=dict(source_environment=copy.deepcopy(fixture[3]["environment"]),source_numerics=fixture[3]["numerics"])
            meta["source_environment"]["compressai"]="different"
            with self.assertRaisesRegex(ValueError,"environment mismatch"):runner.validate_environment(meta,args)
            with self.assertRaisesRegex(ValueError,"code differs"):runner.shared_code_matches({"codec":"old"},{"codec":"new"})

    def test_slurm_separate_processes_one_gpu_no_new_sender(self):
        bash=Path("C:/Program Files/Git/bin/bash.exe")
        if not bash.is_file():
            bash=shutil.which("bash")
            if not bash:self.skipTest("bash unavailable")
        relative="scripts/slurm/eval_nfcgs_independent_receiver.slurm"
        raw=(ROOT/relative).read_bytes()
        self.assertTrue(raw.startswith(b"#!/usr/bin/env bash\n"));self.assertNotIn(b"\r",raw)
        syntax=subprocess.run([str(bash),"-n",str(ROOT/relative)],capture_output=True,text=True)
        self.assertEqual(syntax.returncode,0,syntax.stderr)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in (relative,"scripts/eval_nfcgs_independent_receiver.py","scripts/eval_nfcgs_path_diagnostics.py",
                    "scripts/profile_nfcgs_complexity.py","config/plans/nfcgs_path_diagnostics_20261019.json",
                    "globalsplat/compression/path_diagnostics.py","scripts/slurm/eval_nfcgs_path_diagnostics.slurm"):
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
            for key in ("SLURM_JOB_ID","INDEPENDENT_RECEIVER_OUTPUT","INDEPENDENT_RECEIVER_PATH_RUN"):env.pop(key,None)
            def invoke(option):
                return subprocess.run([str(bash),"-c",'export PATH="$(pwd)/fake_bin:$PATH"; bash '+relative+' '+option],
                    cwd=root,env=env,capture_output=True,text=True)
            planned=invoke("plan");self.assertEqual(planned.returncode,0,planned.stdout+planned.stderr)
            self.assertFalse((root/"tmp_job").exists())
            env["SLURM_JOB_ID"]="12345"
            run=invoke("");self.assertEqual(run.returncode,0,run.stdout+run.stderr)
            calls=(root/"calls.txt").read_text().splitlines();self.assertEqual(len(calls),3)
            self.assertIn(".py plan",calls[0]);self.assertIn(".py prepare",calls[1]);self.assertIn(".py receive",calls[2])
            self.assertIn("job_458004",calls[1]);self.assertIn("--dataset-root",calls[1])
            self.assertNotIn("--dataset-root",calls[2]);self.assertNotIn("--path-run",calls[2])
            self.assertIn("--precision bf16",calls[2]);self.assertIn("step000050000.ckpt",calls[2])
            self.assertTrue((root/"tmp_job/independent_receiver_12345").is_dir())


if __name__=="__main__":unittest.main()
