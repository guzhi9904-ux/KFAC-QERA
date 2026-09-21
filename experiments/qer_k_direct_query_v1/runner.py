import argparse
import os
import signal
import torch
from d_common import (Path,read,save_json,identity,PLAN,require,lock,load_record,checked_files,
                      Resources,ReplayTeacher,Source,audit_assets)
from d_statistics import collect
from d_solve import solve
from d_evaluate import evaluate
from d_report import report


def main():
    p=argparse.ArgumentParser();p.add_argument('--ko-run',type=Path,required=True)
    p.add_argument('--audit-run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--check-only',action='store_true');args=p.parse_args()
    require(not __import__('sys').flags.optimize,'Python -O forbidden')
    for k in ('HF_HUB_OFFLINE','HF_DATASETS_OFFLINE','TRANSFORMERS_OFFLINE'):os.environ[k]='1'
    torch.set_num_threads(8);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    kroot=args.ko_run.resolve();aroot=args.audit_run.resolve();root=args.output.resolve()
    manifest=identity(kroot,aroot);ident=manifest['identity'];config=read(kroot/'manifest.json')['config']
    for src in (kroot,aroot,Path(config['parent_run'])):
        require(not root.is_relative_to(src) and not src.is_relative_to(root),'Output overlaps frozen input')
    config=dict(config,budget_hours=PLAN['budget_hours'],disk_limit_GiB=PLAN['disk_limit_GiB'],checkpoint_every=1)
    root.mkdir(parents=True,exist_ok=True)
    with lock(root):
        if (root/'manifest.json').exists():require(read(root/'manifest.json')==manifest,'Frozen run changed')
        else:save_json(root/'manifest.json',manifest)
        if (root/'complete.json').exists():checked_files(root,load_record(root/'complete.json',ident)['files']);print('DIRECT_QUERY_ALREADY_COMPLETE');return
        if args.check_only:print('SOURCE_IDENTITIES_VERIFIED; GPU work not started');return
        resources=Resources(root,ident,config);teacher=ReplayTeacher(config,root,ident,resources.timed)
        for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:setattr(resources,'stop',True))
        try:
            source=Source(kroot);audit_assets(root,ident,source,resources)
            collect(root,config,ident,resources,teacher,source)
            solve(root,config,ident,resources,source)
            evaluate(root,config,ident,resources,teacher,source)
            report(root,config,ident,resources,source)
        except BaseException as exc:
            save_json(root/'failure.json',dict(identity=ident,error=repr(exc),no_relaxed_tolerances=True));raise
        finally:
            teacher.unload();resources.flush();save_json(root/'resource_usage.json',resources.data)


if __name__=='__main__':main()
