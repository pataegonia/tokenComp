"""Regression tests for the 120k repair, its lineage, and continuous Wave 2."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_nfcgs_fresh_rd as runner
import submit_nfcgs_fresh_rd as submit
import test_fresh_rd_provisional as legacy
from globalsplat.fresh_rd import atomic_json,checkpoint_path,file_hash,read_json,paired_change,gate1_decision
from globalsplat.fresh_rd_runtime import FreshRDCheckpoint,FreshRDState,checkpoint_protocol_matches

torch.set_num_threads(1)


def set_evaluation(folder,step,mse):
    out=folder/'validation'/f'step{step:09d}'
    path=out/'nfcgs_main/scene_metrics.json'
    rows=read_json(path)
    for row in rows:row['mse']=mse
    atomic_json(path,rows)
    done=read_json(out/'complete.json')
    done.update(checkpoint_sha256=file_hash(checkpoint_path(folder,step)),scene_metrics_sha256=file_hash(path))
    atomic_json(out/'complete.json',done)
    return rows


def repair_sweep(tmp_path):
    args,state=legacy.setup(tmp_path)
    legacy.provision(args,state);legacy.gate(args,state,120000)
    args.provisional=False;args.budget=120000;args.restart_stage_b_from_60k=True;args.accept_resaved_60k=True
    changes=[]
    for run in state['runs'].values():
        folder=Path(run['run_dir']);protocol=read_json(folder/'run.json')
        saved=dict(global_step=60000,epoch=7,fresh_rd_protocol=protocol,fresh_rd_quantile_step=60000,
            state_dict={'weight':torch.tensor([1.,2.])},optimizer_states=[{'state':{},'param_groups':[{'lr':.0001}]}],
            lr_schedulers=[dict(last_epoch=60000,milestones={70000:1,90000:1},base_lrs=[.0001])],
            loops={'fit_loop':{'epoch_loop.batch_progress':{'total':{'processed':239979},'current':{'processed':8860}},
                               'state_dict':{'combined_loader':{'consumed':8860}}}})
        torch.save(saved,checkpoint_path(folder,60000))
        atomic_json(folder/'stage_A_complete.json',dict(step=60000,checkpoint_sha256=file_hash(checkpoint_path(folder,60000))))
        before=set_evaluation(folder,40000,.01);after=set_evaluation(folder,60000,.009)
        changes.append(dict(cell=run['cell'],**{'lambda':run['rate_lambda']},seed=run['seed_index'],
                            **paired_change(before,after,run['rate_lambda'])))
    atomic_json(args.root/'convergence/gate1.json',dict(**gate1_decision(changes),runs=changes,test_used=False))
    run=next(iter(state['runs'].values()));folder=Path(run['run_dir'])
    original=file_hash(checkpoint_path(folder,60000))
    shutil.copytree(folder/'validation/step000060000',folder/'validation/step000060000.superseded_1')
    saved=torch.load(checkpoint_path(folder,60000),weights_only=False)
    saved['loops']['fit_loop']['epoch_loop.batch_progress']['total']['processed']=239980
    saved['loops']['fit_loop']['epoch_loop.batch_progress']['current']['processed']=8861
    torch.save(saved,checkpoint_path(folder,60000));set_evaluation(folder,60000,.009)
    return args,state,folder,original


def test_accept_resaved_parent_records_exact_checks_and_known_difference(tmp_path):
    args,state,folder,original=repair_sweep(tmp_path)
    source=submit.stage_a_branch_sources(submit.Submission(args,state),args)[0]
    assert source['source_60k_mode']=='resaved_after_first_microbatch'
    assert source['original_stage_a_sha256']==original
    assert source['checkpoint_60k_sha256']==file_hash(checkpoint_path(folder,60000))!=original
    assert source['known_differences']==dict(skipped_microbatches=1,skipped_scenes=2,
        first_resumed_optimizer_microbatches=3,original_processed=239979,resaved_processed=239980)
    assert source['superseded_comparisons'][0]['max_mse_difference']==0
    assert 'gate1_recomputed_from_current_scene_metrics' in source['passed_checks']
    decision=submit.gate1_evidence(args)
    assert decision['recommended_budget']==120000 and decision['consider_150k']
    assert 'schedule' in decision['reason']


@pytest.mark.parametrize('case',['flag','lr','milestone','milestone_count','step','epoch','last_epoch','processed_original',
    'processed_plus_two','evaluation_sha','archive_mse','archive_lpips','archive_bytes','quantile_step','loader'])
def test_resaved_rejections_happen_before_any_cancellation(tmp_path,case):
    args,state,folder,_=repair_sweep(tmp_path)
    path=checkpoint_path(folder,60000);saved=torch.load(path,weights_only=False)
    if case=='flag':args.accept_resaved_60k=False
    elif case=='lr':saved['optimizer_states'][0]['param_groups'][0]['lr']=.00001
    elif case=='milestone':saved['lr_schedulers'][0]['milestones']={84000:1,108000:1}
    elif case=='milestone_count':saved['lr_schedulers'][0]['milestones']={70000:2,90000:1}
    elif case=='step':saved['global_step']=60001
    elif case=='epoch':saved['epoch']=8
    elif case=='last_epoch':saved['lr_schedulers'][0]['last_epoch']=60001
    elif case=='processed_original':saved['loops']['fit_loop']['epoch_loop.batch_progress']['total']['processed']=239979
    elif case=='processed_plus_two':saved['loops']['fit_loop']['epoch_loop.batch_progress']['total']['processed']=239981
    elif case=='quantile_step':saved['fresh_rd_quantile_step']=59500
    elif case=='loader':saved['loops']['fit_loop']['state_dict']={}
    if case not in ('flag','evaluation_sha','archive_mse','archive_lpips','archive_bytes'):
        torch.save(saved,path);set_evaluation(folder,60000,.009)
    if case=='evaluation_sha':
        complete=folder/'validation/step000060000/complete.json';body=read_json(complete)
        body['checkpoint_sha256']='wrong';atomic_json(complete,body)
    if case.startswith('archive_'):
        archive=folder/'validation/step000060000.superseded_1'
        metrics=archive/'nfcgs_main/scene_metrics.json';rows=read_json(metrics)
        field={'archive_mse':'mse','archive_lpips':'lpips','archive_bytes':'actual_bytes'}[case]
        rows[0][field]+=1
        if field=='actual_bytes':rows[0]['actual_container_bytes']+=1
        atomic_json(metrics,rows);body=read_json(archive/'complete.json')
        body['scene_metrics_sha256']=file_hash(metrics);atomic_json(archive/'complete.json',body)
    with patch.object(submit.subprocess,'run') as command:
        with pytest.raises(ValueError,match='50k checkpoint'):
            submit.submit_stage_b(submit.Submission(args,state),args)
        command.assert_not_called()
    assert not (args.root/'branches/B120000').exists()


def test_real_state_dry_run_inspects_sources_ids_and_file_plan_without_mutation(tmp_path,capsys):
    args,state,folder,_=repair_sweep(tmp_path)
    args.dry_run=True;atomic_json(args.state,state)
    before=args.state.read_bytes();parent=file_hash(checkpoint_path(folder,60000))
    with patch.object(submit,'job_state',return_value='RUNNING'),patch.object(submit,'verify_provisional_job') as verify, \
            patch.object(submit.subprocess,'run') as command:
        submit.main(['--stage-b','--wave','1','--budget','120000','--restart-stage-b-from-60k',
            '--accept-resaved-60k','--dry-run','--state',str(args.state),'--root',str(args.root),
            '--preparation',str(args.preparation),'--dataset-root',str(args.dataset_root),
            '--native-checkpoint',str(args.native_checkpoint)])
        command.assert_not_called();assert verify.call_count==15
    output=capsys.readouterr().out
    assert 'resaved_after_first_microbatch' in output
    assert 'WOULD_STOP_PROVISIONAL_ONLY=8000' in output
    assert 'WOULD_LINK=' in output and 'step000060000.ckpt' in output
    assert 'step000070000.ckpt ->' not in output and 'last.ckpt ->' not in output
    assert args.state.read_bytes()==before and file_hash(checkpoint_path(folder,60000))==parent
    assert not (args.root/'branches').exists()
    if os.environ.get('NFCGS_DRY_RUN_REPORT'):
        Path(os.environ['NFCGS_DRY_RUN_REPORT']).write_text(
            'Synthetic CPU fixture dry-run; these are not server checkpoints or live Slurm states.\n'+output,
            encoding='utf-8')


def test_repaired_fork_preserves_source_files_and_full_state_120k_schedule(tmp_path):
    args,state,folder,_=repair_sweep(tmp_path)
    parent_hash=file_hash(checkpoint_path(folder,60000));old_state=copy.deepcopy(state['inputs'])
    with patch.object(submit,'job_state',return_value='CANCELLED'),patch.object(submit.subprocess,'run',side_effect=legacy.responses()):
        submit.submit_stage_b(submit.Submission(args,state),args)
    new=args.root/'branches/B120000'/next(iter(state['runs']))/folder.name
    assert file_hash(checkpoint_path(folder,60000))==parent_hash
    assert file_hash(checkpoint_path(new,60000))==parent_hash
    assert os.path.samefile(checkpoint_path(new,60000),runner.protected_stage_a_path(checkpoint_path(new,60000)))
    assert not (checkpoint_path(new,60000).stat().st_mode & stat.S_IWUSR)
    assert not checkpoint_path(new,70000).exists()
    assert not (checkpoint_path(new,60000).parent/'last.ckpt').exists()
    assert state['inputs']==old_state
    args2=runner.parse_args(['train','--stage','B','--budget','120000','--run-dir',str(new)])
    command,_=runner.build_command(args2)
    assert 'optimizer.lr_milestones=[84000,108000]' in command
    assert 'checkpointing.override_lr_milestones=true' in command


def test_runtime_migration_is_limited_to_declared_repair_files(tmp_path):
    code={name:'old' for name in runner.RUNTIME_FILES}
    parent=dict(version='test',key='run',code_sha256=code,recipe={'accumulate':4})
    current=copy.deepcopy(parent)
    current['code_sha256']['globalsplat/fresh_rd_runtime.py']='new'
    current['resume_parent']=dict(protocol=parent,changed_runtime_files=['globalsplat/fresh_rd_runtime.py'],
        source_60k_mode='resaved_after_first_microbatch',checkpoints_sha256={'60000':'sha'})
    saved=dict(global_step=60000,fresh_rd_protocol=parent)
    assert checkpoint_protocol_matches(saved,current)
    current['code_sha256']['globalsplat/compression/codec.py']='changed codec'
    assert not checkpoint_protocol_matches(saved,current)
    current['code_sha256']['globalsplat/compression/codec.py']='old'
    current['recipe']['accumulate']=3
    assert not checkpoint_protocol_matches(saved,current)


def test_branch_runtime_identity_and_model_restore_accept_only_the_frozen_parent(tmp_path):
    args,state,folder,_=repair_sweep(tmp_path)
    manifest=dict(arms={arm:{'sha256':'initializer'} for arm in ('two_path56','score56_only','msh_only')},
        holdout_sha256='holdout',validation_index_sha256='index',vanilla_sha256='vanilla')
    atomic_json(args.preparation/'preparation.json',manifest)
    for run in state['runs'].values():
        source=Path(run['run_dir'])
        a=runner.parse_args(['train','--cell',run['cell'],'--rate-lambda',run['rate_lambda'],
            '--run-dir',str(source),'--preparation',str(args.preparation),'--dataset-root',str(args.dataset_root)])
        protocol=runner.identity(a)
        for name in runner.RUNTIME_REPAIR_FILES:protocol['code_sha256'][name]='legacy-runtime'
        atomic_json(source/'run.json',protocol)
        checkpoint=checkpoint_path(source,60000);saved=torch.load(checkpoint,weights_only=False)
        saved['fresh_rd_protocol']=protocol
        from omegaconf import OmegaConf
        saved['fresh_rd_training_config']=OmegaConf.to_container(runner.compose_command(runner.build_command(a)[0]),resolve=True)
        resaved=run['key']==next(iter(state['runs']))
        saved['loops']['fit_loop']['epoch_loop.batch_progress']['total']['processed']=239979
        torch.save(saved,checkpoint)
        original=file_hash(checkpoint)
        atomic_json(source/'stage_A_complete.json',dict(step=60000,checkpoint_sha256=original))
        if resaved:
            archive=source/'validation/step000060000.superseded_1/complete.json'
            old=read_json(archive);old['checkpoint_sha256']=original;atomic_json(archive,old)
            saved['loops']['fit_loop']['epoch_loop.batch_progress']['total']['processed']=239980
            torch.save(saved,checkpoint)
        set_evaluation(source,60000,.009)
    for name in runner.RUNTIME_REPAIR_FILES:state['inputs']['code'][name]='legacy-runtime'
    old_inputs=copy.deepcopy(state['inputs'])
    with patch.object(submit,'job_state',return_value='CANCELLED'),patch.object(submit.subprocess,'run',side_effect=legacy.responses()):
        submit.submit_stage_b(submit.Submission(args,state),args)
    target=args.root/'branches/B120000'/next(iter(state['runs']))/folder.name
    a=runner.parse_args(['train','--stage','B','--budget','120000','--cell','h_on','--rate-lambda','.0032',
        '--run-dir',str(target),'--preparation',str(args.preparation),'--dataset-root',str(args.dataset_root)])
    current=runner.identity(a)
    assert current==read_json(target/'run.json')
    assert read_json(folder/'run.json')==read_json(target/'source_run.json')
    assert state['inputs']==old_inputs
    with patch('globalsplat.compression.checkpoint.validate_feature_codec_checkpoint'):
        loaded=runner.validate_checkpoint(a,checkpoint_path(target,60000),current)
    from globalsplat.model.model_wrapper import GlobalSplatModule
    fake=SimpleNamespace(fresh_rd_protocol=current,override_lr_milestones=True,lr_milestones=[84000,108000],model=None)
    GlobalSplatModule.on_load_checkpoint(fake,loaded)
    assert sorted(loaded['lr_schedulers'][0]['milestones'])==[84000,108000]
    assert loaded['optimizer_states'][0]['param_groups'][0]['lr']==.0001
    assert fake._last_quantile_update_step==60000


def test_quantile_step_is_restored_and_not_updated_twice():
    from globalsplat.model.model_wrapper import GlobalSplatModule
    fake=Mock();fake.model=Mock(feature_codec=None);fake.override_lr_milestones=False
    fake.fresh_rd_protocol={'same':'protocol'}
    GlobalSplatModule.on_load_checkpoint(fake,dict(fresh_rd_protocol=fake.fresh_rd_protocol,fresh_rd_quantile_step=60000))
    assert fake._last_quantile_update_step==60000
    fake.global_step=60000;fake.quantile_training='deterministic';fake.quantile_update_interval=500
    fake.model.feature_codec=Mock()
    GlobalSplatModule.on_train_batch_end(fake,None,None,0)
    fake.model.feature_codec.update.assert_not_called()


def test_lightning_230_resume_does_not_resave_boundary_or_change_hardlinked_parent(tmp_path):
    import pytorch_lightning as pl
    from torch.utils.data import TensorDataset
    from globalsplat.dataset.data_module import ResumableDataLoader
    class Data(pl.LightningDataModule):
        def train_dataloader(self):return ResumableDataLoader(TensorDataset(torch.arange(128)),batch_size=2,num_workers=0)
    class Model(pl.LightningModule):
        def __init__(self):super().__init__();self.w=torch.nn.Parameter(torch.tensor(1.))
        def training_step(self,batch,idx):return (self.w*batch[0].float()).square().mean()
        def configure_optimizers(self):return torch.optim.Adam([self.w],lr=.0001)
        def on_train_batch_end(self,*args):pass
    def train(folder,maximum,load=None):
        callback=FreshRDCheckpoint(dirpath=folder,filename='step{step:09d}',auto_insert_metric_name=False,
            every_n_train_steps=4,save_top_k=-1,save_last=True)
        trainer=pl.Trainer(accelerator='cpu',devices=1,max_steps=maximum,accumulate_grad_batches=4,
            logger=False,enable_progress_bar=False,enable_model_summary=False,num_sanity_val_steps=0,
            callbacks=[FreshRDState(),callback])
        with patch.object(callback,'_save_checkpoint',wraps=callback._save_checkpoint) as saves:
            trainer.fit(Model(),datamodule=Data(),ckpt_path=str(load) if load else None)
        return [Path(c.args[1]).name for c in saves.call_args_list]
    source=tmp_path/'source';branch=tmp_path/'branch';branch.mkdir()
    train(source,4);original=source/'step000000004.ckpt';parent=branch/original.name
    os.link(original,parent);runner.freeze_stage_a_checkpoint(parent)
    digest=file_hash(original);mtime=original.stat().st_mtime_ns
    saved=train(branch,8,parent)
    assert 'step000000004.ckpt' not in saved
    assert (branch/'step000000008.ckpt').exists()
    assert file_hash(original)==file_hash(parent)==digest and original.stat().st_mtime_ns==mtime
    # Keep local pytest cleanup possible on Windows; production retains read-only links.
    parent.chmod(stat.S_IMODE(parent.stat().st_mode)|stat.S_IWUSR)


def test_continuous_wave2_submits_twenty_ordered_jobs_and_one_monitor(tmp_path,capsys):
    args,state=legacy.setup(tmp_path)
    args.budget=120000;args.dry_run=True;args.msh_lambdas=['.0064','.0128'];args.omit_seeds=False
    submit.submit_wave2(submit.Submission(args,state),args)
    jobs=state['jobs'];plan=state['wave2_execution']
    assert len(plan['main_keys'])==16 and len(plan['optional_keys'])==4
    assert [k.split('/')[0] for k in plan['optional_keys']]==['msh','msh','m_on','h_l3']
    front={jobs[k+'/full']['id'] for k in plan['main_keys']}
    for key in plan['main_keys']+plan['optional_keys']:
        record=jobs[key+'/full']
        assert record['env']['FRESH_STAGE']=='full' and record['env']['FRESH_TEST_AFTER_TRAIN']=='true'
        assert not any(key+s in jobs for s in ('/A','/B','/test'))
        if key in plan['optional_keys']:assert front<=set(record['dependencies'])
    assert len(jobs['monitor/2/final']['dependencies'])==20
    output=capsys.readouterr().out
    assert output.count('ACCEPTED ')==21
    a=runner.parse_args(['train','--stage','full','--budget','120000','--run-dir',str(tmp_path/'continuous')])
    command,initial=runner.build_command(a);cfg=runner.compose_command(command)
    runner.validate_training_config(cfg,a)
    assert cfg.trainer.max_steps==120000 and list(cfg.optimizer.lr_milestones)==[84000,108000]
    assert not cfg.checkpointing.resume and cfg.checkpointing.exact_resume


def test_optional_cancel_changes_only_four_jobs_and_unblocks_sixteen_run_monitor(tmp_path):
    import monitor_nfcgs_fresh_rd as monitor
    args,state=legacy.setup(tmp_path)
    args.budget=120000;args.dry_run=True;args.msh_lambdas=['.0064','.0128'];args.omit_seeds=False;args.reason='deadline'
    submit.submit_wave2(submit.Submission(args,state),args)
    # Replace dry IDs with numeric IDs, as a successful real submission would.
    front_keys=state['wave2_execution']['main_keys'];optional_keys=state['wave2_execution']['optional_keys']
    keys=[k+'/full' for k in front_keys+optional_keys]+['monitor/2/final']
    for i,key in enumerate(keys):state['jobs'][key]['id']=str(9000+i)
    before={key:copy.deepcopy(state['jobs'][key+'/full']) for key in front_keys}
    args.dry_run=False
    front={state['jobs'][k+'/full']['id'] for k in front_keys}
    extra={state['jobs'][k+'/full']['id'] for k in optional_keys}
    stopped=set()
    def status(job):
        if job in front:return 'COMPLETED'
        if job in stopped:return 'CANCELLED'
        return 'PENDING'
    def execute(command,**kwargs):
        if command[0]=='scancel':stopped.add(command[1])
        else:assert command[:2]==['scontrol','update']
        return subprocess.CompletedProcess(command,0,'','')
    with patch.object(submit,'job_state',side_effect=status),patch.object(submit,'verify_registered_job'), \
            patch.object(submit.subprocess,'run',side_effect=execute):
        submit.cancel_wave2_optional(submit.Submission(args,state),args)
    assert stopped==extra
    assert {key:state['jobs'][key+'/full'] for key in front_keys}==before
    assert state['jobs']['monitor/2/final']['dependencies']==[]
    assert set(state['wave2_execution']['optional_cancellation']['core_completed_ids'])==front
    assert set(state['wave2_execution']['optional_cancelled_keys'])==set(optional_keys)
    # With no 120k files present this performs a pending scan, exercising the
    # selected inventory without running evaluation or requesting a GPU.
    atomic_json(args.state,state)
    monitor.main(['--root',str(args.root),'--state',str(args.state),'--wave','2','--phase','final','--budget','120000','--once'])


@pytest.mark.parametrize('argv',[
    ['--accept-resaved-60k'],
    ['--stage-b','--wave','1','--budget','100000','--accept-resaved-60k'],
    ['--stage-b','--wave','1','--budget','150000','--restart-stage-b-from-60k','--accept-resaved-60k']])
def test_accept_resaved_flag_requires_explicit_120k_restart(argv):
    with pytest.raises(SystemExit):submit.main(argv)
