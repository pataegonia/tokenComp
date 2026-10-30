"""Matched cohort guards, real packet rendering, and partial Slurm submission."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import run_nfcgs_followup_20261020 as runner
import submit_nfcgs_followup_20261020 as submit
import eval_nfcgs_b0_matched as b0
import eval_nfcgs_independent_receiver as receiver
import analyze_nfcgs_fixed_order as fixed
import test_independent_receiver as receiver_fixtures
from test_independent_receiver import fake_functions


class FollowupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def reference(self,i):
        return dict(scene=f"scene_{i}",context_frame_ids=list(range(12)),target_frame_ids=list(range(12,20)),
                    psnr=20.+i,ssim=.8,lpips=.2,actual_bytes=100+i)

    def boundary_fixture(self,directory):
        root=Path(directory)
        args=SimpleNamespace(b0_run=root/"b0",path_run=root/"path",output=root/"audit",resume=False)
        protocol=dict(scenes=256,tokens=4096,seed=0,precision="bf16",context_views=12,target_views=8,
                      image_shape=[256,256],shared_sorted_order=True)
        checkpoint=dict(path="checkpoint",sha256="shared",global_step=50000)
        source=dict(inputs=dict(fingerprint=runner.FINGERPRINT,scenes=[f"scene_{i}" for i in range(256)]))
        pair=dict(encoder_tensors_equal=True,shared_decoder_tensors_equal=True,
            geometry_boundary_effective_readout_errors={key:dict(rmse=.1,max_abs=.2,relative_frobenius=.3) for key in runner.READOUTS})
        for folder,meta in ((args.b0_run,dict(protocol=protocol,checkpoint=checkpoint,source=source,manifest={},
                model_pair=pair,native_checkpoint=dict(sha256="native"))),
                (args.path_run,dict(protocol=protocol,checkpoint=dict(checkpoint,size=1234),source=source,manifest={}))):
            runner.save(folder/"metadata.json",meta)
            runner.save(folder/"complete.json",dict(scene_count=256,checkpoint_sha256="shared",
                input_fingerprint=runner.FINGERPRINT,native_checkpoint_sha256="native"))
            runner.save(folder/"summary.json",{})
        for i in range(256):
            common=dict(scene=f"scene_{i}",index=i,input_fingerprint="input",target_fingerprint="target",
                context_frame_ids=list(range(12)),target_frame_ids=list(range(12,20)),permutation_sha256="permutation")
            metric=dict(psnr=24.,ssim=.7,lpips=.2)
            runner.save(args.b0_run/f"scenes/scene_{i:04d}.json",dict(common,modes=dict(b0_native=metric,b1_bridge=metric)))
            row=dict(metric,feature_mse=.1,texture_mse=.2,geometry_mse=.3)
            runner.save(args.path_run/f"scenes/scene_{i:04d}.json",dict(common,modes={name:row for name in b0.diag.MODES}))
        return args

    def test_existing_boundary_schema_audit_and_paired_frame_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            args=self.boundary_fixture(directory)
            with contextlib.redirect_stdout(io.StringIO()):result=runner.boundary(args)
            self.assertEqual(result["scene_count"],256)
            self.assertEqual(result["max_abs_b1_drift"],0)
            self.assertTrue((args.output/"complete.json").exists())
            path=args.b0_run/"scenes/scene_0000.json";row=runner.read(path)
            row["target_frame_ids"]=list(range(20,28));path.write_text(json.dumps(row))
            with self.assertRaisesRegex(ValueError,"paired scene"):
                runner.boundary_source(args)

    def test_fixed_source_protocol_checkpoint_and_scene_pairing_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/"fixed";bundle=root/"bundle";checkpoint=root/"shared.ckpt"
            checkpoint.write_bytes(b"shared checkpoint identity")
            args=SimpleNamespace(fixed_run=source,bundle=bundle,checkpoint=checkpoint)
            checkpoint_meta=dict(path=str(checkpoint),sha256=runner.digest(checkpoint),global_step=50000,size=26)
            names=[f"scene_{i}" for i in range(256)]
            orders=list(runner.ORDERS)+[f"random_{i}" for i in range(10)]
            identity=dict(checkpoint=checkpoint_meta,source=dict(inputs=dict(fingerprint=runner.FINGERPRINT,scenes=names)),
                          orders=orders,tokens=4096,seed=0,precision="bf16")
            runner.save(source/"metadata.json",dict(identity=identity))
            runner.save(source/"summary.json",{})
            runner.save(source/"complete.json",dict(scene_count=256,input_fingerprint=runner.FINGERPRINT,orders=orders,
                checkpoint_sha256=checkpoint_meta["sha256"],summary_sha256=runner.digest(source/"summary.json")))
            runner.save(bundle/"metadata.json",dict(checkpoint=checkpoint_meta,protocol=dict(scenes=256,tokens=4096,seed=0,
                precision="bf16",context_views=12,target_views=8,image_shape=[256,256],shared_sorted_order=True)))
            rows=[dict(index=i,scene=name,packet_sha256=f"packet_{i}") for i,name in enumerate(names)]
            runner.save(bundle/"manifest.json",dict(rows=rows))
            runner.save(bundle/"complete.json",dict(scene_count=256,metadata_sha256=runner.digest(bundle/"metadata.json"),
                manifest_sha256=runner.digest(bundle/"manifest.json")))
            for i,name in enumerate(names):
                runner.save(source/f"scenes/scene_{i:04d}.json",dict(scene=name,index=i,
                    reference_packet_sha256=f"packet_{i}",orders={order:dict(packet_sha256=f"packet_{i}") for order in orders}))
            self.assertEqual(len(runner.fixed_source(args)[2]),256)
            checkpoint.write_bytes(b"changed checkpoint")
            with self.assertRaisesRegex(ValueError,"checkpoint"):
                runner.fixed_source(args)

    def test_reference_count_duplicate_and_frame_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"reference.json"
            good = [self.reference(i) for i in range(2)]
            runner.save(path,good)
            self.assertEqual(runner.reference_rows(path,2),good)
            with self.assertRaisesRegex(ValueError,"6991"):
                runner.reference_rows(path)
            for bad in ([good[0],good[0]], [dict(good[0],target_frame_ids=list(range(8))),good[1]],
                        [dict(good[0],psnr=float("nan")),good[1]]):
                path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):runner.reference_rows(path,2)

    def native_fixture(self,directory):
        root=Path(directory)
        args=SimpleNamespace(reference=root/"reference.json",native_checkpoint=root/"native.ckpt",
            output=root/"native",resume=False,precision="fp32",device="cpu",threads=1,num_workers=0,
            tokens=65,seed=0,expected_scenes=2,dataset_root=root/"data")
        refs=[self.reference(i) for i in range(2)]; runner.save(args.reference,refs)
        args.native_checkpoint.write_bytes(b"native fixture checkpoint")
        batches=[]
        for i in range(2):
            batches.append(dict(scene_info=dict(scene=[f"scene_{i}"]),
                inputs=dict(images=torch.zeros(1,12,3,256,256),frame_ids=torch.arange(12)[None],
                    texture=torch.randn(1,65,8),geometry=torch.randn(1,65,512)),
                targets=dict(images=torch.full((1,8,3,256,256),.4),frame_ids=torch.arange(12,20)[None],
                    intrinsic=torch.eye(3).expand(1,8,3,3),extrinsic=torch.eye(4).expand(1,8,4,4))))
        class Native:
            feature_codec=None
            def encode_scene_tokens(self,inputs):return inputs["texture"],inputs["geometry"]
            def decode_scene_tokens(self,tokens):
                means=tokens[1][...,:3].repeat_interleave(8,1)
                return SimpleNamespace(means=means,num_gaussians=means.shape[1])
        return args,batches,Native(),dict(sha256=runner.digest(args.native_checkpoint))

    def run_native(self,fixture,batches=None):
        args,data,model,identity=fixture
        render,metrics=fake_functions()
        with patch.object(runner,"setup_gpu",return_value=(torch,b0)), \
             patch.object(b0,"load_native",return_value=(model,identity)), \
             patch.object(receiver,"dataset_config",return_value=None), \
             patch.object(b0.diag,"load_data",return_value=batches if batches is not None else list(reversed(data))), \
             patch.object(b0.diag,"render_functions",return_value=(render,metrics)), \
             patch.object(runner,"runtime_hashes",return_value={}), \
             contextlib.redirect_stdout(io.StringIO()):
            return runner.native(args)

    def test_native_real_tensor_route_matches_reference_order_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture=self.native_fixture(directory)
            summary=self.run_native(fixture)
            args=fixture[0]
            self.assertEqual(summary["scene_count"],2)
            self.assertIsNone(summary["native_actual_bytes"])
            self.assertEqual(runner.read(args.output/"scenes/scene_00000.json")["scene"],"scene_0")
            self.assertTrue((args.output/"complete.json").exists())
            with self.assertRaises(FileExistsError):self.run_native(fixture)
            args.resume=True
            with patch.object(runner,"native_scene",side_effect=AssertionError("do not rerender completed scenes")):
                self.assertEqual(self.run_native(fixture),summary)

    def test_native_missing_duplicate_and_changed_frames_never_complete(self):
        for kind in ("missing","duplicate","frames"):
            with tempfile.TemporaryDirectory() as directory:
                fixture=self.native_fixture(directory);args,batches,*_=fixture
                if kind=="missing":batches=batches[:1]
                if kind=="duplicate":batches=[batches[0],batches[0]]
                if kind=="frames":batches[0]["targets"]["frame_ids"]=torch.arange(20,28)[None]
                with self.assertRaises((ValueError,RuntimeError)):self.run_native(fixture,batches)
                self.assertFalse((args.output/"complete.json").exists())

    def fixed_fixture(self,directory):
        helper=receiver_fixtures.ReceiverTests()
        fixture=helper.fixture(directory);helper.export(fixture)
        original,_,_,_,rows,_,codec,_=fixture
        root=Path(directory);source=root/"fixed";source.mkdir()
        runner.save(source/"metadata.json",{});runner.save(source/"complete.json",{})
        bundle,rows=receiver.load_bundle(original.bundle)
        all_rows=[]
        for i,reference in enumerate(rows):
            record=dict(index=i,scene=reference["scene"],orders={})
            for name in runner.ORDERS:
                payload=(original.bundle/reference["packet_file"]).read_bytes() if name=="hilbert" else codec.compress(
                    torch.randn(1,65,8),torch.randn(1,65,8),torch.randn(1,65,3)).data
                cost=fixed.payload_components(payload)
                path=source/f"packets/{i}_{name}.bin";path.parent.mkdir(exist_ok=True);path.write_bytes(payload)
                record["orders"][name]=dict(**cost["components"],total_bytes=len(payload),
                    packet_sha256=runner.digest_bytes(payload),packet_file=path.relative_to(source).as_posix(),feature_mse=.1)
            runner.save(source/f"scenes/scene_{i:04d}.json",record)
            all_rows.append(record)
        args=SimpleNamespace(fixed_run=source,bundle=original.bundle,checkpoint=original.checkpoint,
            output=root/"render",resume=False,tokens=65,precision="fp32",device="cpu",threads=1)
        return args,(dict(identity=dict(code_sha256={})),bundle,rows,all_rows)

    def run_fixed(self,fixture):
        args,source=fixture
        with patch.object(runner,"setup_gpu",return_value=(torch,b0)), \
             patch.object(runner,"fixed_source",return_value=source), \
             patch.object(receiver,"validate_environment",return_value=({},{})), \
             patch.object(runner,"runtime_hashes",return_value={}), \
             patch.object(receiver.diag,"render_functions",side_effect=fake_functions), \
             contextlib.redirect_stdout(io.StringIO()):
            return runner.fixed_render(args)

    def test_saved_packets_render_with_real_minimal_receiver_and_resume(self):
        from globalsplat.model.globalsplat import GlobalSplat
        with tempfile.TemporaryDirectory() as directory:
            fixture=self.fixed_fixture(directory);args,_=fixture
            with patch.object(GlobalSplat,"__init__",side_effect=AssertionError("no scene encoder")):
                summary=self.run_fixed(fixture)
            self.assertEqual(summary["scene_count"],2)
            self.assertEqual(set(summary["orders"]),set(runner.ORDERS))
            self.assertTrue(all(v==0 for v in summary["hilbert_max_abs_t05_delta"].values()))
            args.resume=True
            with patch.object(receiver.PacketReceiver,"decode",side_effect=AssertionError("reuse completed scene")):
                self.assertEqual(self.run_fixed(fixture),summary)

    def test_corrupt_packet_and_hilbert_reference_never_complete(self):
        for kind in ("packet","reference"):
            with tempfile.TemporaryDirectory() as directory:
                fixture=self.fixed_fixture(directory);args,(_,_,rows,source_rows)=fixture
                if kind=="packet":
                    cost=source_rows[0]["orders"]["slot"]
                    path=args.fixed_run/cost["packet_file"];data=bytearray(path.read_bytes());data[-1]^=1;path.write_bytes(data)
                else:
                    rows[0]["reference"]["psnr"]+=1
                    # The real bundle loader also needs the altered reference for this tolerance test.
                    bundle=runner.read(args.bundle/"manifest.json");bundle["rows"]=rows
                    (args.bundle/"manifest.json").write_text(json.dumps(bundle))
                    complete=runner.read(args.bundle/"complete.json")
                    complete["manifest_sha256"]=runner.digest(args.bundle/"manifest.json")
                    (args.bundle/"complete.json").write_text(json.dumps(complete))
                with self.assertRaises((ValueError,RuntimeError)):self.run_fixed(fixture)
                self.assertFalse((args.output/"complete.json").exists())

    def test_cli_and_output_identity_cannot_silently_change(self):
        with self.assertRaises(SystemExit):runner.parse_args(["native","--precision","fp32","--output","unused"])
        with tempfile.TemporaryDirectory() as directory:
            args=SimpleNamespace(output=Path(directory)/"out",resume=False)
            with runner.run_directory(args,dict(protocol=1)):pass
            args.resume=True
            with self.assertRaisesRegex(ValueError,"differs"):
                with runner.run_directory(args,dict(protocol=2)):pass


class SubmissionTests(unittest.TestCase):
    def test_partial_eval_rejection_reuses_every_accepted_job(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/"state.json";state=dict(jobs={})
            calls=[]
            def sbatch(command,**kwargs):
                calls.append((command,kwargs["env"]))
                if len(calls)==5:
                    raise subprocess.CalledProcessError(1,command,stderr="QOS rejection")
                return SimpleNamespace(stdout=f"{500+len(calls)};cluster\n")
            with patch.object(submit.subprocess,"run",side_effect=sbatch),contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(subprocess.CalledProcessError):
                    submit.submit_missing(root,{},state,path)
            saved=runner.read(path)
            self.assertEqual(len(saved["jobs"]),4)
            self.assertEqual(saved["jobs"]["none-off-train"],"504")
            with patch.object(submit.subprocess,"run",return_value=SimpleNamespace(stdout="506\n")) as call, \
                 contextlib.redirect_stdout(io.StringIO()):
                submit.submit_missing(root,{},saved,path)
            self.assertEqual(call.call_count,1)
            self.assertIn("--dependency=afterok:504",call.call_args.args[0])
            self.assertEqual(call.call_args.kwargs["env"]["TRAIN_JOB_ID"],"504")

    def test_task_resources_and_none_off_recipe(self):
        self.assertFalse(any("gres" in value for value in submit.task_command("boundary")))
        self.assertIn("--gres=gpu:normal:1",submit.task_command("native"))
        env=submit.none_environment(ROOT,ROOT/"parent.ckpt",ROOT/"logs/state")
        self.assertEqual((env["TOKEN_ORDER"],env["SPATIAL_CONTEXT"],env["STAGES"],env["SCORE_SLICE_CHANNELS"]),
                         ("none","false","4","16"))
        command=submit.validate_recipe(ROOT/"parent.ckpt")
        for value in ("model.feature_codec.token_order=none","model.feature_codec.score_spatial_context=false",
            "model.feature_codec.score_context_schedule=dyadic4","trainer.max_steps=50000",
            "trainer.devices=2","optimizer.batch_size=2","trainer.accumulate_grad_batches=2"):
            self.assertIn(value,command)
        for name in ("scripts/slurm/run_nfcgs_followup_20261020.slurm",):
            raw=(ROOT/name).read_bytes()
            self.assertNotIn(b"\r",raw)
            self.assertTrue(raw.startswith(b"#!/usr/bin/env bash\n"))

    def test_state_signature_and_dependency_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"state.json"
            runner.save(path,dict(version=1,signature="other",inputs={},jobs={}))
            with self.assertRaisesRegex(ValueError,"differs"):
                submit.load_state(path,"expected",{})
            path.write_text(json.dumps(dict(version=1,signature="ok",inputs={},jobs={"none-off-eval":"100"})))
            with self.assertRaisesRegex(ValueError,"dependency"):
                submit.load_state(path,"ok",{})


if __name__ == "__main__":unittest.main()
