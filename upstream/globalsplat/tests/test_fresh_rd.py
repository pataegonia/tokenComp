"""CPU validation of the new protocol, real entropy coding and resumable training."""
from collections import Counter
from dataclasses import replace
import importlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_nfcgs_fresh_rd as runner
import monitor_nfcgs_fresh_rd as monitor
import submit_nfcgs_fresh_rd as submit
import check_nfcgs_fresh_rd_gate0 as gate0
from globalsplat.fresh_rd import *
from globalsplat.fresh_rd_runtime import FreshRDState, FreshRDCheckpoint, override_scheduler_milestones

torch.set_num_threads(1)


def test_inventory():
    a, b = wave_runs(1), wave_runs(2)
    assert len(a) == 15 and len(b) == 20 and len(set(a+b)) == 35
    assert len(wave_runs(2, omit_seeds=True)) + len(a) == 27
    assert all(sum(c == cell and r == rate and s == 0 for c,r,s in a+b) == 1 for cell in CORE for rate in LAMBDAS)
    assert milestones(120000) == [84000,108000]
    with pytest.raises(ValueError): wave_runs(2, (".0032", ".0064"))


@pytest.mark.parametrize("cell", CELLS)
@pytest.mark.parametrize("stage,budget", [("A",100000),("B",100000),("B",120000),("B",150000)])
def test_hydra_training(cell,stage,budget,tmp_path):
    a=runner.parse_args(["train","--cell",cell,"--run-dir",str(tmp_path),"--stage",stage,"--budget",str(budget)])
    command, checkpoint=runner.build_command(a)
    keys=[s.split("=",1)[0].lstrip("+") for s in command if "=" in s]
    assert len(keys)==len(set(keys))
    cfg=runner.compose_command(command)
    runner.validate_training_config(cfg,a)
    assert cfg.trainer.max_steps==(60000 if stage=="A" else budget)
    assert list(cfg.optimizer.lr_milestones)==milestones(100000 if stage=="A" else budget)
    assert cfg.checkpointing.resume==(stage=="B")
    assert cfg.checkpointing.override_lr_milestones==(stage=="B" and budget!=100000)
    assert not cfg.checkpointing.allow_score_path_conversion if stage=="B" else True


@pytest.mark.parametrize("mode", ["validation","test","reference","native"])
def test_evaluation_commands(mode,tmp_path):
    args=[mode,"--run-dir",str(tmp_path),"--checkpoint",str(tmp_path/"model.ckpt")]
    if mode=="validation": args += ["--step","40000"]
    a=runner.parse_args(args); cfg=runner.compose_command(runner.build_command(a)[0])
    assert cfg.architecture_entropy_determinism and cfg.test.record_scene_metrics
    assert cfg.test.actual_bitstream==(mode!="native")
    assert cfg.seed==0 and (cfg.dataset.num_context_views,cfg.dataset.num_target_views)==(12,8)
    assert bool(cfg.dataset.get("eval_train_holdout_manifest"))==(mode!="test")
    if mode == "native":
        assert cfg.loss.rate_lambda == 0
        from globalsplat.model.model_wrapper import GlobalSplatModule
        from omegaconf import OmegaConf
        GlobalSplatModule(torch.nn.Linear(1,1), eval_mode=True,
            loss_w_dict=OmegaConf.to_container(cfg.loss, resolve=True))


def test_additional_lambda(tmp_path):
    a=runner.parse_args(["train","--cell","msh","--rate-lambda",".0016","--run-dir",str(tmp_path)])
    cfg=runner.compose_command(runner.build_command(a)[0])
    assert cfg.loss.rate_lambda==.0016


def test_scheduler_override_restores_actual_saved_schedule():
    p=torch.nn.Parameter(torch.tensor(1.))
    opt=torch.optim.Adam([p],lr=.0001)
    scheduler=torch.optim.lr_scheduler.MultiStepLR(opt,[7,9],gamma=.1)
    for _ in range(6): opt.step(); scheduler.step()
    saved=dict(global_step=6,optimizer_states=[opt.state_dict()],lr_schedulers=[scheduler.state_dict()])
    override_scheduler_milestones(saved,[8,11])
    opt2=torch.optim.Adam([torch.nn.Parameter(torch.tensor(1.))],lr=.0001)
    s2=torch.optim.lr_scheduler.MultiStepLR(opt2,[7,9],gamma=.1)
    opt2.load_state_dict(saved["optimizer_states"][0]);s2.load_state_dict(saved["lr_schedulers"][0])
    values=[]
    for _ in range(6): opt2.step();s2.step();values.append(opt2.param_groups[0]["lr"])
    assert values==pytest.approx([.0001,.00001,.00001,.00001,.000001,.000001])
    assert s2.last_epoch==12
    saved["global_step"]=8
    with pytest.raises(ValueError): override_scheduler_milestones(saved,[8,11])


def sample_row(scene="a",mse=.01):
    return dict(scene=scene,mse=mse,psnr=20.,ssim=.7,lpips=.2,actual_bytes=1000,
        actual_score_bytes=200,actual_residual_y_bytes=600,actual_residual_z_bytes=100,
        actual_mean_bytes=0,actual_container_bytes=100,context_frame_ids=list(range(12)),target_frame_ids=list(range(12,20)))


def test_direct_mse_and_paired_gate():
    before=[sample_row(str(i),.01+i*.00001) for i in range(128)]
    after=[dict(r,mse=r["mse"]-.0001) for r in before]
    change=paired_change(before,after,.0128)
    assert change["delta_db_j"]>0 and change["improvement_above_2se"]
    assert objective(before[0],.0128)==pytest.approx(.01+.05*.2+.0128*8000/(32768*59))
    bad=[dict(r) for r in after];bad[0]["target_frame_ids"]=list(range(13,21))
    with pytest.raises(ValueError): paired_change(before,bad,.0128)
    changes=[dict(cell=c,**{"lambda":r},**change) for c,r,s in wave_runs(1) if c in CORE]
    assert gate1_decision(changes)["recommended_budget"]==100000
    changes[0].update(delta_db_j=.21,paired_se_db_j=.01,improvement_above_2se=True)
    assert gate1_decision(changes)["consider_150k"]
    expected={r["scene"]:dict(context=r["context_frame_ids"],target=r["target_frame_ids"]) for r in before}
    assert validate_cohort(before,expected)
    with pytest.raises(ValueError): validate_cohort(bad,expected)


@pytest.mark.parametrize("cell", CELLS)
def test_real_entropy_roundtrip_and_forward(cell):
    from globalsplat.compression.codec import ObservableLowRank1DCodec
    cfg=replace(runner.config_for(cell),texture_channels=8,geometry_channels=8,geometry_observable_channels=4,
        rank=0 if CELLS[cell].arm=="msh_only" else 4,residual_n=8,residual_m=12,adapter_hidden=6,score_slice_channels=2)
    torch.manual_seed(123)
    codec=ObservableLowRank1DCodec(cfg).eval()
    codec.update(force=True)
    t,g,xyz=torch.randn(1,32,8),torch.randn(1,32,8),torch.randn(1,32,3)
    with torch.no_grad():
        f=codec(t,g,xyz,restore_original_order=False,training=False).reconstruction_sorted
        packet=codec.compress(t,g,xyz)
        d=torch.cat(codec.decompress(packet.data),dim=-1)
    torch.testing.assert_close(f,d,atol=1e-6,rtol=1e-5)
    if cell=="h_on":
        with torch.no_grad(): errors,_,_=gate0.compare_codec(codec,t,g,xyz)
        assert all(r["max_abs"]<1e-5 for r in errors.values())


def test_initializer_load_and_hashes_all_cells(tmp_path):
    from globalsplat.compression.codec import ObservableLowRank1DCodec
    from globalsplat.compression.architecture import tensor_hash
    original=runner.config_for
    def small(cell):
        return replace(original(cell),texture_channels=8,geometry_channels=8,geometry_observable_channels=4,
            rank=0 if CELLS[cell].arm=="msh_only" else 4,residual_n=8,residual_m=12,adapter_hidden=6,score_slice_channels=2)
    common=None;arms={}
    for cell in ("h_on","s56","msh"):
        torch.manual_seed(111123)
        codec=ObservableLowRank1DCodec(small(cell))
        state=codec.state_dict()
        if common is None: common={k:v.clone() for k,v in state.items()}
        else:
            for k,v in state.items():
                if k in common and v.shape==common[k].shape: state[k]=common[k].clone()
        arm=CELLS[cell].arm;path=tmp_path/(arm+".ckpt")
        torch.save(dict(state_dict={"model.feature_codec."+k:v for k,v in state.items()},feature_codec_config=small(cell).to_dict()),path)
        arms[arm]={"sha256":file_hash(path)}
    atomic_json(tmp_path/"preparation.json",dict(seed=111123,arms=arms))
    with patch.object(runner,"config_for",side_effect=small): result=gate0.check_initializers(tmp_path)
    assert result["passed"] and len(result["cells"])==len(CELLS)


def test_new_cells_convert_only_switches_of_their_initializer():
    from globalsplat.compression.checkpoint import validate_feature_codec_checkpoint
    from globalsplat.compression.codec import ObservableLowRank1DCodec
    assert not any(converts_initializer(c) for c in ("s56","msh"))
    assert all(converts_initializer(c) for c in ("h_on","nx_on","ns_on","s56_off","s56_n_on","s56_n_off","msh_n"))
    assert runner.config_for("ns_on").score_order_metric=="l2" and runner.config_for("ns1_on").nn_order_backend=="graph_gemm"
    assert runner.config_for("nx_on").nn_order_distance_device=="input" and runner.config_for("h_on").nn_order_backend=="legacy"
    def small(cell):
        return replace(runner.config_for(cell),texture_channels=8,geometry_channels=8,geometry_observable_channels=4,
            rank=0 if CELLS[cell].arm=="msh_only" else 4,residual_n=8,residual_m=12,adapter_hidden=6,
            score_slice_channels=2)
    for base,variants in (("s56",("s56_off","s56_n_on","s56_n_off")),("msh",("msh_n",))):
        codec=ObservableLowRank1DCodec(small(base))
        saved=dict(state_dict={"model.feature_codec."+k:v for k,v in codec.state_dict().items()},
                   feature_codec_config=small(base).to_dict())
        for cell in variants:
            validate_feature_codec_checkpoint(saved,small(cell),allow_score_path_conversion=True)
            with pytest.raises(ValueError): validate_feature_codec_checkpoint(saved,small(cell))
            # Tensors load unchanged into the variant's codec.
            ObservableLowRank1DCodec(small(cell)).load_state_dict(codec.state_dict(),strict=True)
        for other in (replace(small(base),residual_n=16),replace(small(variants[0]),score_slice_channels=1)
                      if base=="s56" else replace(small(variants[0]),adapter_hidden=9)):
            with pytest.raises(ValueError): validate_feature_codec_checkpoint(saved,other,allow_score_path_conversion=True)
    with pytest.raises(ValueError):
        codec=ObservableLowRank1DCodec(small("s56"))
        saved=dict(state_dict={"model.feature_codec."+k:v for k,v in codec.state_dict().items()},
                   feature_codec_config=small("s56").to_dict())
        validate_feature_codec_checkpoint(saved,small("h_on"),allow_score_path_conversion=True)


def test_gate0_integrated_loader_keeps_encoder_width(tmp_path):
    from globalsplat.compression.codec import ObservableLowRank1DCodec
    from globalsplat.compression.config import CodecConfig
    import globalsplat.main as main
    cfg=replace(runner.config_for("h_on"),texture_channels=8,geometry_channels=8,geometry_observable_channels=4,
        rank=4,residual_n=8,residual_m=12,adapter_hidden=6,score_slice_channels=2)
    codec=ObservableLowRank1DCodec(cfg)
    path=tmp_path/"initial.ckpt"
    torch.save(dict(state_dict={"model.feature_codec."+k:v for k,v in codec.state_dict().items()},feature_codec_config=cfg.to_dict()),path)
    class Integrated(torch.nn.Module):
        def __init__(self,model_cfg):
            super().__init__()
            assert model_cfg.dim_latents==8
            self.feature_codec=ObservableLowRank1DCodec(cfg)
        def set_stage(self,*args,**kwargs): pass
    with patch.object(main,"build_model",side_effect=Integrated):
        model,resolved=gate0.load_integrated(path,tmp_path)
    assert resolved.model.dim_latents==8 and model.feature_codec.config.geometry_observable_channels==4


def fake_args(tmp_path):
    import argparse
    return argparse.Namespace(dry_run=False,state=tmp_path/"state.json",root=tmp_path/"out")


def test_submission_survives_partial_failure(tmp_path):
    import subprocess
    args=fake_args(tmp_path)
    state=dict(jobs={},runs={},lanes={})
    s=submit.Submission(args,state)
    ok=subprocess.CompletedProcess([],0,"12345;cluster\n","")
    with patch.object(submit.subprocess,"run",return_value=ok) as call:
        s.add("one","job.slurm",{},lane=0)
        s.add("one","job.slurm",{},lane=0)
        assert call.call_count==1
    assert read_json(args.state)["jobs"]["one"]["id"]=="12345"
    with patch.object(submit.subprocess,"run",side_effect=KeyboardInterrupt):
        with pytest.raises(KeyboardInterrupt): s.add("two","job.slurm",{},lane=0)
    assert read_json(args.state)["pending_submission"]["key"]=="two"
    with patch.object(submit.subprocess,"run") as call:
        with pytest.raises(ValueError): s.add("two","job.slurm",{},lane=0)
        call.assert_not_called()


def test_monitor_completed_fake_checkpoint(tmp_path):
    checkpoint=tmp_path/"checkpoint.ckpt";checkpoint.write_bytes(b"frozen fixture")
    output=tmp_path/"validation";path=output/"nfcgs_main/scene_metrics.json"
    rows=[sample_row(str(i)) for i in range(128)]
    atomic_json(path,rows)
    atomic_json(output/"complete.json",dict(split="validation",mode="validation",scene_count=128,
        checkpoint_sha256=file_hash(checkpoint),scene_metrics_sha256=file_hash(path)))
    assert len(monitor.completed_rows(output,checkpoint))==128
    row=monitor.summary_row(dict(cell="h_on",rate_lambda=".0128",seed_index=0),40000,rows)
    assert row["MSE"]==.01 and row["scene_count"]==128
    checkpoint.write_bytes(b"different")
    with pytest.raises(ValueError): monitor.completed_rows(output,checkpoint)


def test_dependency_repair_is_durable(tmp_path):
    import argparse,subprocess
    args=fake_args(tmp_path)
    args.resume_run="h_on/lambda0p0128/s0";args.stage="A";args.budget=None
    args.dataset_root=tmp_path/"data";args.preparation=tmp_path/"prep";args.native_checkpoint=tmp_path/"native.ckpt"
    key=args.resume_run
    run=dict(key=key,cell="h_on",rate_lambda="0.0128",seed_index=0,wave=1,preparation=str(args.preparation),run_dir=str(tmp_path/"run"))
    state=dict(runs={key:run},jobs={key+"/A":dict(id="100",dependencies=[],history=[]),key+"/B":dict(id="101",dependencies=["100"])},lanes={})
    s=submit.Submission(args,state)
    responses=[subprocess.CompletedProcess([],0,"102\n",""),subprocess.CalledProcessError(1,["scontrol"])]
    with patch.object(submit,"job_state",side_effect=lambda j:"FAILED" if j=="100" else "PENDING"),patch.object(submit.subprocess,"run",side_effect=responses):
        with pytest.raises(subprocess.CalledProcessError): submit.resume_run(s,args)
    assert read_json(args.state)["jobs"][key+"/A"]["id"]=="102"
    assert read_json(args.state)["dependency_repairs"][0]["complete"] is False
    with patch.object(submit,"job_state",return_value="PENDING"),patch.object(submit.subprocess,"run",return_value=subprocess.CompletedProcess([],0)):
        submit.repair_dependencies(s)
    assert read_json(args.state)["jobs"][key+"/B"]["dependencies"]==["102"]


def test_complete_sweep_gpu_chains_and_idempotent_dry_run(tmp_path,capsys):
    state=tmp_path/"state.json"
    submit.main(["--dry-run","--state",str(state),"--root",str(tmp_path/"out")])
    log=capsys.readouterr().out
    assert not state.exists()
    assert log.count("ACCEPTED ")==113
    assert "--dependency=afterok:" in log
    assert "PLAN: 15 wave-1 + 20 wave-2" in log


def test_optional_gate_and_preserved_diagnostic_submission(tmp_path,capsys):
    import argparse,copy
    inputs=dict(root="/server/fresh",dataset="/data/re10k",preparation="/server/prep",native_checkpoint="/server/native",code={"runner.py":"new"})
    state=dict(version=VERSION,inputs=dict(inputs,code={"runner.py":"old"}),runs={},
        jobs={"gate0/forward":dict(id="459889",dependencies=[],lane=1)},lanes={"1":"459889"})
    submit.adopt_inputs(state, inputs)
    args=argparse.Namespace(dry_run=True,state=tmp_path/"state.json",root=tmp_path/"fresh",dataset_root=tmp_path/"data",
        preparation=tmp_path/"prep",native_checkpoint=tmp_path/"native",budget=None,require_gate0=False)
    sub=submit.Submission(args,state)
    submit.submit_wave1(sub,args)
    assert state["jobs"]["gate0/forward"]["id"]=="459889"
    assert len(state["runs"])==15
    assert state["diagnostic_lanes"]=={"1":"459889"}
    assert all("459889" not in v["dependencies"] for k,v in state["jobs"].items() if not k.startswith("gate0/"))
    assert not (tmp_path/"state.json").exists()
    changed=copy.deepcopy(inputs);changed["code"]["runner.py"]="third"
    with pytest.raises(ValueError): submit.adopt_inputs(state,changed)
    # No diagnostic result exists; default submission must not read one.
    args.dry_run=False
    submit.require_gate0(args)
    args.require_gate0=True
    with pytest.raises(FileNotFoundError): submit.require_gate0(args)


def test_variable_entropy_buffers_do_not_crash_resume_report():
    a={"model.w":torch.ones(2),"model.entropy._quantized_cdf":torch.zeros(3,185,dtype=torch.int32)}
    b={"model.w":torch.ones(2),"model.entropy._quantized_cdf":torch.zeros(3,193,dtype=torch.int32)}
    result=gate0.compare_training_states(a,b)
    assert result["model_max_abs_delta"]==0
    assert not result["training_bit_exact"]
    assert len(result["entropy_buffer_shape_differences"])==1
    b["model.w"]=torch.ones(3)
    with pytest.raises(ValueError): gate0.compare_training_states(a,b)


def test_full_test_needs_no_native_gate(tmp_path):
    import argparse
    args=argparse.Namespace(run_dir=tmp_path/"sweep/h_on/lambda0p0128/s0/job_1")
    rows=[dict(scene=str(i),context_frame_ids=list(range(12)),target_frame_ids=list(range(12,20))) for i in range(6991)]
    with patch.object(runner,"ROOT",tmp_path):
        cohort=runner.test_cohort(args,rows)
        assert validate_cohort(rows,cohort)
        frozen=runner.test_cohort(args,[])
        assert frozen==cohort
    meta=read_json(tmp_path/"sweep/test_frame_cohort.json")
    assert meta["source"]=="first completed fresh full-test evaluation"
    assert meta["selection_uses_quality"] is False


@pytest.mark.parametrize("dataset_size,cut,total,shift_schedule", [(128,4,8,False),(32,6,12,False),(128,6,12,True)])
def test_lightning_full_state_resume_sample_rng_lr(tmp_path,dataset_size,cut,total,shift_schedule):
    import pytorch_lightning as pl
    from torch.utils.data import TensorDataset
    from globalsplat.dataset.data_module import ResumableDataLoader
    class Data(pl.LightningDataModule):
        def train_dataloader(self):
            gen=torch.Generator().manual_seed(403+0x9E3779B1*int(self.trainer.current_epoch))
            return ResumableDataLoader(TensorDataset(torch.arange(dataset_size)),batch_size=2,shuffle=True,generator=gen,num_workers=0)
    class Model(pl.LightningModule):
        def __init__(self,new_schedule=False,override=False):
            super().__init__();self.w=torch.nn.Parameter(torch.tensor(1.));self.lr_milestones=(8,11) if new_schedule else (7,9)
            self.override_lr_milestones=override;self.model=None
        def on_load_checkpoint(self,checkpoint):
            from globalsplat.model.model_wrapper import GlobalSplatModule
            GlobalSplatModule.on_load_checkpoint(self,checkpoint)
        def training_step(self,batch,idx):
            return (self.w*(batch[0].float()+torch.rand_like(batch[0].float()))).square().mean()
        def on_train_batch_end(self,outputs,batch,batch_idx): pass
        def configure_optimizers(self):
            opt=torch.optim.Adam([self.w],lr=.0001)
            return dict(optimizer=opt,lr_scheduler=dict(scheduler=torch.optim.lr_scheduler.MultiStepLR(opt,self.lr_milestones),interval="step"))
    def train(label,maximum,load=None):
        pl.seed_everything(7,workers=True)
        model=Model(new_schedule=shift_schedule and (label=="cont" or load is not None),override=shift_schedule and load is not None)
        callback=FreshRDCheckpoint(dirpath=tmp_path/label,filename="step{step}",auto_insert_metric_name=False,
            every_n_train_steps=maximum,save_top_k=-1,save_last=True)
        trainer=pl.Trainer(accelerator="cpu",devices=1,max_steps=maximum,accumulate_grad_batches=4,
            logger=False,enable_progress_bar=False,enable_model_summary=False,num_sanity_val_steps=0,
            reload_dataloaders_every_n_epochs=1,callbacks=[FreshRDState(tmp_path/(label+".jsonl")),callback])
        trainer.fit(model,datamodule=Data(),ckpt_path=str(load) if load else None)
        return model
    cont=train("cont",total)
    train("split",cut)
    split=train("split",total,tmp_path/"split/last.ckpt")
    a=[json.loads(s) for s in (tmp_path/"cont.jsonl").read_text().splitlines()]
    b=[json.loads(s) for s in (tmp_path/"split.jsonl").read_text().splitlines()]
    assert [r["sample"] for r in a]==[r["sample"] for r in b]
    assert [r["lr"] for r in a]==[r["lr"] for r in b]
    torch.testing.assert_close(cont.w,split.w,atol=0,rtol=0)
