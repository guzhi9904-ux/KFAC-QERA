import argparse
import os
import signal
import torch
from j_common import (Path,read,save_json,identity,PLAN,require,lock,load_record,checked_files,
                      Resources,ReplayTeacher,Source)
from j_assets import prepare_assets
from j_statistics import collect
from j_solve import solve
from j_evaluate import evaluate
from j_report import report


def main():
    p=argparse.ArgumentParser();p.add_argument('--ko-run',type=Path,required=True)
    p.add_argument('--sens-run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);mode=p.add_mutually_exclusive_group()
    mode.add_argument('--check-only',action='store_true');mode.add_argument('--prepare-only',action='store_true');args=p.parse_args()
    require(not __import__('sys').flags.optimize,'Python -O forbidden')
    for k in ('HF_HUB_OFFLINE','HF_DATASETS_OFFLINE','TRANSFORMERS_OFFLINE'):os.environ[k]='1'
    torch.set_num_threads(8);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    kroot=args.ko_run.resolve();sroot=args.sens_run.resolve();root=args.output.resolve();manifest=identity(kroot,sroot);ident=manifest['identity']
    config=read(kroot/'manifest.json')['config']
    for src in (kroot,sroot,Path(config['parent_run'])):
        require(not root.is_relative_to(src) and not src.is_relative_to(root),'Output overlaps frozen parent')
    config=dict(config,budget_hours=PLAN['budget_hours'],disk_limit_GiB=PLAN['disk_limit_GiB'])
    root.mkdir(parents=True,exist_ok=True)
    with lock(root):
        if (root/'manifest.json').exists():require(read(root/'manifest.json')==manifest,'Frozen run changed')
        else:save_json(root/'manifest.json',manifest)
        if (root/'complete.json').exists():checked_files(root,load_record(root/'complete.json',ident)['files']);print('TOKEN_ONE_ALREADY_COMPLETE');return
        source=Source(kroot,sroot)
        if args.check_only:print('SOURCE_AND_DATA_IDENTITIES_VERIFIED; GPU work not started');return
        resources=Resources(root,ident,config);teacher=None
        for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:setattr(resources,'stop',True))
        try:
            with resources.timed('small_parent_asset_audit'):assets=prepare_assets(root,ident,source)
            if args.prepare_only:
                require(all(m['status']=='pending_stream_validation' for m in assets['modules'].values()),'Small asset audit failed; see parent_assets_preflight.json')
                print('SMALL_ASSETS_VERIFIED; GPU work not started',flush=True);return
            collect(root,ident,resources,kroot,sroot,assets)
            solve(root,ident,resources,source)
            teacher=ReplayTeacher(config,root,ident,resources.timed)
            evaluate(root,config,ident,resources,teacher,source)
            report(root,config,ident,resources,source)
        except BaseException as exc:
            save_json(root/'failure.json',dict(identity=ident,error=repr(exc),no_relaxed_tolerances=True));raise
        finally:
            if teacher is not None:teacher.unload()
            resources.flush();save_json(root/'resource_usage.json',resources.data)


if __name__=='__main__':main()
