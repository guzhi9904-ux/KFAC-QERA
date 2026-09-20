#!/usr/bin/env python3
"""Explicit resource-gated stages. No scheduler, lease renewal, or remote-login code."""
import argparse
from pathlib import Path
import sys
import traceback
from common import HERE,PLAN,read,save_json,atomic_bytes,source_identity,require,sha_file
from records import load_record
from runtime_io import Resources,BudgetReached,hardware,lock


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('config',type=Path);p.add_argument('output',type=Path)
    p.add_argument('stage',choices=['prepare','pilot','run','report'])
    args=p.parse_args();config=read(args.config);root=args.output.resolve()
    require(sys.flags.optimize==0,'Python -O is forbidden for numerical acceptance')
    require(config['profile']=='dual4090' and config['modules']==[PLAN['module']],'Only registered L10 dual4090 allowed')
    parent=Path(config['output_parent']).resolve();require(root!=parent and root.is_relative_to(parent),'Run must be a child of output parent')
    for field in ('assets','model','wikitext','geometry_history','paired_source','quantized_source'):
        protected=Path(config[field]).resolve();require(not root.is_relative_to(protected) and not protected.is_relative_to(root),'Output overlaps protected data')
    with lock(root):
        manifest=source_identity(config);identity=manifest['identity'];mp=root/'manifest.json'
        if mp.exists():require(read(mp)==manifest,'Frozen code/config changed; use an explicitly versioned new run')
        else:
            require(not any(p.name!='.run.lock' for p in root.iterdir()),'New run directory is not empty')
            save_json(mp,manifest);atomic_bytes(root/'protocol.md',(HERE/'protocol.md').read_bytes())
        if args.stage=='report':
            from report import report
            report(root);return
        resources=Resources(root,identity,config['budget_hours']);status='FAILED';experiment=None
        try:
            from assets import inventory
            from data import prepare
            hp=root/'hardware.json';hw=hardware(root)
            if not hp.exists():
                require(hw['free_disk_GiB']>=PLAN['minimum_free_GiB'],'Need at least 144 GiB free disk before pilot')
                require(hw['memory_limit_bytes'] is not None and hw['memory_limit_bytes']>=64*2**30,'Require confirmed >=64 GiB host limit')
                save_json(hp,hw)
            with resources.timed('asset_hash_inventory'):
                inv=inventory(config);ap=root/'asset_identity.json'
                if ap.exists():require(read(ap)==inv,'Source asset inventory changed')
                else:save_json(ap,inv)
            with resources.timed('freeze_data_and_sample_IDs'):prepare(root,config,identity,inv)
            atomic_bytes(root/'teacher_identity.json',(Path(config['assets'])/'exp03/teacher_identity.json').read_bytes())
            if args.stage=='prepare':status='PREPARED';print('CPU PREPARE COMPLETE',flush=True);return
            import torch
            require(torch.cuda.is_available() and torch.cuda.device_count()==2,'Exactly two visible CUDA GPUs required')
            require(all('4090' in torch.cuda.get_device_name(i) for i in range(2)),'RTX4090 devices required')
            torch.set_num_threads(config['cpu_threads']);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            from experiment import Experiment
            experiment=Experiment(root,config,identity,resources)
            if args.stage=='pilot':
                experiment.pilot();status='PILOT_COMPLETE';print('FIT-ONLY PILOT COMPLETE; read budget_freeze.json before run',flush=True);return
            pilot=load_record(root/'pilot.json',identity);budget=load_record(root/'budget_freeze.json',identity)
            require(pilot['passed'],'Pilot not accepted')
            if not budget['fits_budget']:raise BudgetReached('Pilot upper estimate exceeds frozen budget; formal sampling not started')
            remaining=config['budget_hours']*3600-resources.data['active_seconds']
            require(remaining>0,'No remaining frozen budget')
            experiment.collect_fit();experiment.fit();experiment.evaluate()
            from diagnostics import run
            run(experiment)
            from report import report
            require(report(root,verify=True),'Final report verification did not complete')
            for path,value in inv['files'].items():require(sha_file(path)==value,'Original input changed during run')
            status='COMPLETE';print('EXPERIMENT_COMPLETE',flush=True)
        except BudgetReached as exc:
            status='INCOMPLETE_BUDGET';save_json(root/'incomplete.json',dict(identity=identity,reason=str(exc)));print(str(exc),flush=True)
        except Exception as exc:
            save_json(root/'failure.json',dict(identity=identity,error=repr(exc),traceback=traceback.format_exc()));raise
        finally:
            if experiment is not None:experiment.offline()
            resources.close(status)
            save_json(root/'status.json',dict(identity=identity,status=status))
            if status=='COMPLETE':
                from report import report
                report(root,verify=False)
                save_json(root/'final_hardware.json',hardware(root))


if __name__=='__main__':main()
