#!/usr/bin/env python3
"""Read-only checks of the frozen 60k parents, 70k checkpoint LR and TensorBoard LR."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from globalsplat.fresh_rd import atomic_json,checkpoint_path,file_hash,read_json


def read_lr_events(paths):
    from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
    from tensorboard.util.tensor_util import make_ndarray
    count=0;last_step=None;violations=[];tags=set()
    for path in paths:
        for event in EventFileLoader(str(path)).Load():
            for value in event.summary.value:
                tag=value.tag.lower();head=tag.split('/')[0]
                # optim.py names the scheduler "lr", so Lightning logs lr/pg1, lr/pg2.
                if not (head=='lr' or head.startswith('lr-') or tag.endswith('/lr')):
                    continue
                scalar=float(make_ndarray(value.tensor)) if value.HasField('tensor') else float(value.simple_value)
                count+=1;tags.add(value.tag);last_step=max(event.step,last_step or 0)
                if event.step<84000 and not math.isclose(scalar,.0001,rel_tol=1e-6,abs_tol=1e-10):
                    violations.append(dict(file=str(path),step=event.step,tag=value.tag,lr=scalar))
    return dict(lr_events=count,last_logged_step=last_step,tags=sorted(tags),pre_84k_violations=violations)


def verify(state_path):
    state=read_json(state_path)
    origin=state.get('stage_b_origin',{})
    if state.get('budget')!=120000 or origin.get('budget')!=120000:
        raise ValueError('use the Wave 1 120k branch state')
    import torch
    rows=[];errors=[]
    for source in origin['source_runs']:
        key=source['run']['key'];expected=source['checkpoint_60k_sha256']
        original=checkpoint_path(source['run']['run_dir'],60000)
        folder=Path(state['runs'][key]['run_dir']);branch=checkpoint_path(folder,60000)
        protected=branch.with_name('stage_a_final_step000060000.ckpt')
        row=dict(key=key,expected_60k_sha256=expected,source_60k_mode=source['source_60k_mode'])
        for label,path in (('original',original),('branch',branch),('protected',protected)):
            actual=file_hash(path) if path.exists() else None
            row[label+'_sha256']=actual
            if actual!=expected:errors.append(f'{key}: {label} 60k SHA changed or file missing')
        path70=checkpoint_path(folder,70000)
        row['checkpoint_70k_exists']=path70.exists()
        if path70.exists():
            saved=torch.load(path70,map_location='cpu',weights_only=False)
            schedulers=saved.get('lr_schedulers',[])
            groups=[g for opt in saved.get('optimizer_states',[]) for g in opt.get('param_groups',[])]
            row['checkpoint_70k_lrs']=[g['lr'] for g in groups]
            if (saved.get('global_step')!=70000 or not groups
                    or any(abs(g['lr']-.0001)>1e-12 for g in groups)
                    or len(schedulers)!=1 or sorted(schedulers[0].get('milestones',{}))!=[84000,108000]
                    or saved.get('fresh_rd_protocol')!=read_json(folder/'run.json')):
                errors.append(f'{key}: invalid 70k protocol, LR or milestones')
            del saved
        lr=read_lr_events(sorted((folder/'tensorboard').glob('**/events.out.tfevents.*')))
        row['tensorboard']=lr
        if lr['pre_84k_violations']:errors.append(f'{key}: TensorBoard LR differs before 84k')
        rows.append(row)
        print(f"VERIFY={key} sha60_ok={all(row[k+'_sha256']==expected for k in ('original','branch','protected'))} checkpoint70={path70.exists()} lr_events={lr['lr_events']}",flush=True)
    return dict(utc=datetime.now(timezone.utc).isoformat(),state=str(state_path),runs=rows,errors=errors,
        checkpoints_modified=False,all_70k_checkpoints_observed=all(r['checkpoint_70k_exists'] for r in rows),
        all_tensorboard_lr_observed=all(r['tensorboard']['lr_events']>0 for r in rows))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state',type=Path,default=ROOT/'logs/slurm/fresh_rd/state_B120000.json')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args(argv);args.state=args.state.resolve()
    result=verify(args.state)
    output=args.output or args.state.parent/('verify_B120000_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')
    atomic_json(output,result)
    print(f"VERIFICATION_REPORT={output} errors={len(result['errors'])} all_70k={result['all_70k_checkpoints_observed']} all_lr_logs={result['all_tensorboard_lr_observed']}",flush=True)
    return 1 if result['errors'] else 0


if __name__=='__main__':raise SystemExit(main())
