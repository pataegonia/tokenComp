"""Frozen-score causal coding, production packet reproduction and safe resume."""
import contextlib
import copy
import hashlib
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

from test_path_diagnostics import make_codec, TinyModel

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("fixed_order_runner",ROOT/"scripts/analyze_nfcgs_fixed_order.py")
runner=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=runner
spec.loader.exec_module(runner)


class FixedOrderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def inputs(self):
        return dict(texture=torch.randn(1,65,8),geometry=torch.randn(1,65,8))

    def reference(self,model,inputs,args):
        with torch.no_grad(),runner.diag.amp(args):
            texture,geometry=model.encode_scene_tokens(inputs)
            xyz=model.gaussian_decoder.decoded_token_centers(model.feature_codec.project_geometry(geometry))
            packet=model.feature_codec.compress(texture,geometry,xyz).data
        return hashlib.sha256(packet).hexdigest()

    def test_frozen_score_packet_matches_production_fp32_and_bf16(self):
        for precision in ("fp32","bf16"):
            model=TinyModel(make_codec()).eval();inputs=self.inputs()
            args=SimpleNamespace(tokens=65,precision=precision,device="cpu")
            reference=self.reference(model,inputs,args)
            state=runner.state_hash(model)
            with patch.object(model,"encode_scene_tokens",wraps=model.encode_scene_tokens) as encode, \
                 patch.object(model.feature_codec,"_analyze_low_rank",wraps=model.feature_codec._analyze_low_rank) as analysis:
                row,packets,traces=runner.analyze_scene(model,inputs,args,"same_scene",reference,keep_tensors=True)
            self.assertEqual(encode.call_count,1)
            self.assertEqual(analysis.call_count,2) # one fixed Score extraction, one production oracle
            self.assertEqual(tuple(row["orders"]),runner.ORDERS)
            self.assertEqual(len(packets),13)
            self.assertEqual(row["orders"]["hilbert"]["packet_sha256"],reference)
            self.assertEqual(set(traces),set(runner.TRACE_ORDERS))
            self.assertEqual(state,runner.state_hash(model))
            for name,stats in row["orders"].items():
                self.assertEqual([s["tokens"] for s in stats["stages"]],[9,8,16,32])
                self.assertEqual(sum(stats[k] for k in ("score_raw","msh_y","msh_z","headers")),len(packets[name]))
                self.assertEqual(sum(s["raw_bytes"] for s in stats["stages"]),stats["score_raw"])
                self.assertTrue(torch.isfinite(torch.tensor(stats["feature_mse"])))

    def test_random_permutations_are_scene_local_and_do_not_consume_global_rng(self):
        xyz=torch.randn(1,65,3)
        state=torch.get_rng_state().clone()
        first=runner.permutations(xyz,scene="a",bits=10)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        repeated=runner.permutations(xyz,scene="a",bits=10)
        other=runner.permutations(xyz,scene="b",bits=10)
        self.assertEqual(tuple(first),runner.ORDERS)
        for name in runner.ORDERS:
            self.assertTrue(torch.equal(first[name],repeated[name]))
            self.assertTrue(torch.equal(first[name].sort(1).values,torch.arange(65)[None]))
        self.assertFalse(torch.equal(first["random_0"],other["random_0"]))
        with self.assertRaisesRegex(ValueError,"bijective"):
            runner.order_object(torch.zeros(1,65,dtype=torch.long))

    def test_historical_packet_mismatch_stops_before_order_interventions(self):
        model=TinyModel(make_codec()).eval();args=SimpleNamespace(tokens=65,precision="fp32",device="cpu")
        with patch.object(runner,"encode_fixed",side_effect=AssertionError("must stop first")), \
             self.assertRaisesRegex(RuntimeError,"does not reproduce T05"):
            runner.analyze_scene(model,self.inputs(),args,"scene","wrong")

    def test_traces_use_decoded_anchors_and_shared_diagnostic_scale(self):
        model=TinyModel(make_codec()).eval();inputs=self.inputs()
        args=SimpleNamespace(tokens=65,precision="fp32",device="cpu")
        reference=self.reference(model,inputs,args)
        row,packets,traces=runner.analyze_scene(model,inputs,args,"scene",reference,keep_tensors=True)
        for name,trace in traces.items():
            packet=runner.SceneBitstream.unpack(packets[name])
            with torch.no_grad():
                received=model.feature_codec.score_context.decompress(packet.score,65)[0,:,0].T
            for group in trace["groups"]:
                ref=group["references"];valid=group["reference_valid"]
                start=group["start"];width=group["target"].shape[1]
                if valid.any():
                    self.assertTrue(torch.equal(group["reference_inputs"][valid],received[ref[valid],start:start+width]))
        # The all-symbol residual is weighted by stage sizes, never equal stage means.
        stages=row["orders"]["hilbert"]["stages"]
        expected=sum(s["normalized_mae"]*s["symbols"] for s in stages)/(65*7)
        self.assertAlmostEqual(row["orders"]["hilbert"]["normalized_mae"],expected)

    def test_random_replicates_are_averaged_within_scene_before_bootstrap(self):
        rows=[]
        for offset in (0,100):
            orders={}
            for name in runner.ORDERS:
                value=10+int(name.split("_")[-1])+offset if name.startswith("random_") else 30+offset
                record={k:0. for k in runner.SCALARS};record["total_bytes"]=value
                record["stages"]=[dict(tokens=n,raw_bytes=0.,bits_per_token=0.,raw_mae=0.,normalized_mae=0.,
                    quantization_normalized_mse=0.) for n in (9,8,16,32)]
                orders[name]=record
            rows.append(dict(orders=orders))
        summary=runner.summarize(rows)
        self.assertEqual(summary["means"]["random10_mean"]["total_bytes"],64.5)
        self.assertEqual(summary["paired_minus_hilbert"]["random10_mean"]["total_bytes"],dict(mean=-15.5,ci95=[-15.5,-15.5]))
        self.assertEqual(summary["random_seed_mean_total_bytes"],list(range(60,70)))

    def fixture(self,directory):
        root=Path(directory);source=root/"source";source.mkdir()
        args=SimpleNamespace(input_run=source,output=root/"output",resume=False,scenes=2,tokens=65,seed=0,
            precision="fp32",device="cpu",threads=1,num_workers=0,trace_scenes=1,dataset_root=root/"unavailable")
        model=TinyModel(make_codec()).eval()
        cache=[dict(scene=f"scene_{i}",inputs=self.inputs()) for i in range(2)]
        torch.save(cache,source/"inputs.pt")
        checkpoint=root/"reference.ckpt";checkpoint.write_bytes(b"fixture checkpoint; model loader mocked")
        spec=dict(label="fixture",checkpoint=str(checkpoint),tokens=65,expected_step=50000,config=model.feature_codec.config.to_dict())
        meta=dict(checkpoint=dict(sha256=runner.diag.file_hash(checkpoint),global_step=50000),
            environment=runner.diag.profile.environment(args),numerics=runner.receiver.numerical_policy())
        info=dict(inputs=dict(scenes=[r["scene"] for r in cache],fingerprint=runner.diag.profile.input_fingerprint(cache)),
                  backbone_sha256=runner.diag.backbone_hash(model))
        references=[dict(index=i,scene=r["scene"],packet_sha256=self.reference(model,r["inputs"],args)) for i,r in enumerate(cache)]
        return args,model,cache,(dict(version=1,points=[spec]),spec,meta,references,info)

    @contextlib.contextmanager
    def patched(self,fixture):
        args,model,cache,preflight=fixture
        with patch.object(runner,"preflight",return_value=preflight), \
             patch.object(runner.diag.profile,"load_model",side_effect=lambda *a:(copy.deepcopy(model),None,None)), \
             contextlib.redirect_stdout(io.StringIO()):yield

    def test_end_to_end_partial_resume_and_completed_packet_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture=self.fixture(directory);args=fixture[0]
            source_bytes=(args.input_run/"inputs.pt").read_bytes()
            original=runner.analyze_scene;calls=0
            def stop(*a,**kw):
                nonlocal calls
                calls+=1
                if calls==2:raise RuntimeError("interruption")
                return original(*a,**kw)
            with self.patched(fixture),patch.object(runner,"analyze_scene",side_effect=stop), \
                 self.assertRaisesRegex(RuntimeError,"interruption"):runner.run(args)
            first=args.output/"scenes/scene_0000.json";before=first.read_bytes()
            self.assertFalse((args.output/".analysis-lock").exists())
            args.resume=True
            with self.patched(fixture),patch.object(runner,"analyze_scene",wraps=original) as execute:
                result=runner.run(args)
            self.assertEqual(execute.call_count,1);self.assertEqual(result["scene_count"],2)
            self.assertEqual(first.read_bytes(),before)
            self.assertEqual((args.input_run/"inputs.pt").read_bytes(),source_bytes)
            with self.patched(fixture),patch.object(runner,"analyze_scene",side_effect=AssertionError("already done")):
                self.assertEqual(runner.run(args),result)
            packet=args.output/"packets/scene_0000/slot.bin"
            raw=bytearray(packet.read_bytes());raw[-1]^=1;packet.write_bytes(raw)
            with self.patched(fixture),self.assertRaisesRegex(ValueError,"packet changed"):runner.run(args)

    def test_report_written_before_interruption_can_be_finalized_on_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture=self.fixture(directory);args=fixture[0]
            save=runner.diag.save_json
            def interrupted(path,value):
                if Path(path).name=="complete.json":raise RuntimeError("before completion marker")
                return save(path,value)
            with self.patched(fixture),patch.object(runner.diag,"save_json",side_effect=interrupted), \
                 self.assertRaisesRegex(RuntimeError,"before completion"):runner.run(args)
            self.assertTrue((args.output/"REPORT.md").exists())
            self.assertFalse((args.output/"complete.json").exists())
            args.resume=True
            with self.patched(fixture):runner.run(args)
            self.assertTrue((args.output/"complete.json").exists())

    def test_cache_mismatch_and_changed_resume_protocol_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture=self.fixture(directory);args=fixture[0]
            with self.patched(fixture):runner.run(args)
            args.resume=True;args.trace_scenes=0
            with self.patched(fixture),self.assertRaisesRegex(ValueError,"identity differs"):runner.run(args)
            cache=fixture[2];cache[0]["inputs"]["texture"][0,0,0]+=.1
            torch.save(cache,args.input_run/"inputs.pt")
            with self.patched(fixture),self.assertRaisesRegex(ValueError,"fingerprint differs"):runner.run(args)

    def test_cli_plan_is_frozen_and_never_loads_cache_or_gpu(self):
        with patch.object(runner,"preflight",return_value=(None,dict(checkpoint="current"),None,None,None)), \
             patch.object(torch,"load",side_effect=AssertionError("no cache tensors")), \
             patch.object(torch.cuda,"is_available",side_effect=AssertionError("no CUDA")), \
             contextlib.redirect_stdout(io.StringIO()) as output:runner.main(["plan"])
        self.assertIn("orders=13",output.getvalue())
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):runner.parse_args(["plan","--seed","1"])

    def test_slurm_plan_run_and_explicit_resume(self):
        bash=Path("C:/Program Files/Git/bin/bash.exe")
        if not bash.is_file():
            bash=shutil.which("bash")
            if not bash:self.skipTest("bash unavailable")
        relative="scripts/slurm/analyze_nfcgs_fixed_order.slurm"
        raw=(ROOT/relative).read_bytes();self.assertNotIn(b"\r",raw)
        self.assertTrue(raw.startswith(b"#!/usr/bin/env bash\n"))
        syntax=subprocess.run([str(bash),"-n",str(ROOT/relative)],capture_output=True,text=True)
        self.assertEqual(syntax.returncode,0,syntax.stderr)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in (relative,"scripts/analyze_nfcgs_fixed_order.py","scripts/eval_nfcgs_independent_receiver.py",
                    "scripts/slurm/eval_nfcgs_independent_receiver.slurm","scripts/eval_nfcgs_path_diagnostics.py",
                    "scripts/profile_nfcgs_complexity.py","globalsplat/compression/mechanism.py"):
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
            for key in ("SLURM_JOB_ID","FIXED_ORDER_OUTPUT","FIXED_ORDER_RESUME","FIXED_ORDER_TRACE_SCENES"):env.pop(key,None)
            def invoke(option):
                return subprocess.run([str(bash),"-c",'export PATH="$(pwd)/fake_bin:$PATH"; bash '+relative+' '+option],
                    cwd=root,env=env,capture_output=True,text=True)
            planned=invoke("plan");self.assertEqual(planned.returncode,0,planned.stdout+planned.stderr)
            self.assertFalse((root/"tmp_job").exists())
            env["SLURM_JOB_ID"]="12345"
            launched=invoke("");self.assertEqual(launched.returncode,0,launched.stdout+launched.stderr)
            env["FIXED_ORDER_RESUME"]="1"
            resumed=invoke("");self.assertEqual(resumed.returncode,0,resumed.stdout+resumed.stderr)
            calls=(root/"calls.txt").read_text().splitlines();self.assertEqual(len(calls),3)
            self.assertIn(".py plan",calls[0]);self.assertIn(".py run",calls[1]);self.assertNotIn("--resume",calls[1])
            self.assertIn("--resume",calls[2]);self.assertIn("--precision bf16",calls[1])
            self.assertIn("job_454995",calls[1]);self.assertIn("job_458004",calls[1])
            self.assertTrue((root/"tmp_job/fixed_order_12345").is_dir())


if __name__=="__main__":unittest.main()
