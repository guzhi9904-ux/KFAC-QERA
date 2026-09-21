import argparse
import os
import shutil
import signal
import torch
from v_common import Path,HERE,read,save_json,require,identity,lock,load_record,checked_files,Resources,ReplayTeacher
from configure import validate
from v_assets import audit_assets
from v_pilot import pilot
from v_statistics import collect_statistics
from v_solve import solve
from v_evaluate import evaluate
from v_report import report


def main():
    p=argparse.ArgumentParser();p.add_argument('config',type=Path);p.add_argument('output',type=Path);p.add_argument('stage',choices=['pilot','run','report']);a=p.parse_args()
    c=read(a.config);validate(c);root=a.output.resolve();parent=Path(c['output_parent']).resolve()
    require(root!=parent and root.is_relative_to(parent),'Output must be beneath configured parent')
    require(not __import__('sys').flags.optimize,'Python -O forbidden')
    for k in ('HF_HUB_OFFLINE','HF_DATASETS_OFFLINE','TRANSFORMERS_OFFLINE'):os.environ[k]='1'
    torch.set_num_threads(c['cpu_threads']);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    root.mkdir(parents=True,exist_ok=True);manifest=identity(c);ident=manifest['identity']
    with lock(root):
        if (root/'manifest.json').exists():require(read(root/'manifest.json')==manifest,'Frozen prototype source/config changed; use new run')
        else:
            require(shutil.disk_usage(root).free>=c['minimum_free_GiB']*2**30,'Insufficient free disk')
            save_json(root/'manifest.json',manifest);shutil.copyfile(HERE/'protocol.md',root/'protocol.md')
        if (root/'complete.json').exists():checked_files(root,load_record(root/'complete.json',ident)['files']);print('PROTOTYPE_ALREADY_COMPLETE');return
        resources=Resources(root,ident,c);teacher=ReplayTeacher(c,root,ident,resources.timed)
        for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:setattr(resources,'stop',True))
        try:
            source=audit_assets(root,c,ident,resources)
            if a.stage=='report':report(root,c,ident,resources,source);return
            pilot(root,c,ident,resources,teacher,source)
            if a.stage=='pilot':return
            collect_statistics(root,c,ident,resources,teacher,source)
            solve(root,c,ident,resources,source)
            evaluate(root,c,ident,resources,teacher,source)
            report(root,c,ident,resources,source)
        except BaseException as exc:
            save_json(root/'failure.json',dict(identity=ident,error=repr(exc),no_relaxed_tolerances=True));raise
        finally:
            teacher.unload();resources.flush();save_json(root/'resource_usage.json',resources.data)


if __name__=='__main__':main()
