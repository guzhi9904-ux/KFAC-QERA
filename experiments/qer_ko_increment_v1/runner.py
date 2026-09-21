import argparse
import os
from pathlib import Path
import shutil
import signal
import torch
from ko_common import read,save_json,require,identity,lock,load_record,checked_files
from configure import validate
from ko_resources import Resources
from ko_assets import Source,prepare,collect
from ko_replay import ReplayTeacher
from ko_acceptance import accept
from ko_fit import fit
from ko_evaluate import evaluate,report


def main():
    p=argparse.ArgumentParser();p.add_argument('config',type=Path);p.add_argument('output',type=Path)
    p.add_argument('stage',choices=['prepare','pilot','run','report']);a=p.parse_args()
    require(not __import__('sys').flags.optimize,'Python -O forbidden')
    c=read(a.config);validate(c);root=a.output.resolve();parent=Path(c['output_parent']).resolve()
    require(root!=parent and root.is_relative_to(parent),'Run must be beneath output parent')
    for key in ('HF_HUB_OFFLINE','HF_DATASETS_OFFLINE','TRANSFORMERS_OFFLINE'):os.environ[key]='1'
    torch.set_num_threads(c['cpu_threads']);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    root.mkdir(parents=True,exist_ok=True);manifest=identity(c);ident=manifest['identity']
    with lock(root):
        if (root/'manifest.json').exists():require(read(root/'manifest.json')==manifest,'Frozen incremental source/config changed')
        else:
            require(shutil.disk_usage(root).free>c['minimum_free_GiB']*2**30,'Insufficient free disk')
            save_json(root/'manifest.json',manifest)
        if (root/'complete.json').exists():
            done=load_record(root/'complete.json',ident);checked_files(root,done['files']);print('INCREMENT_ALREADY_COMPLETE');return
        resources=Resources(root,ident,c);teacher=ReplayTeacher(c,root,ident,resources.timed)
        for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:setattr(resources,'stop',True))
        try:
            source=Source(c);prepare(root,c,ident,source)
            if a.stage=='prepare':print('INCREMENT_PREPARED');return
            if a.stage=='report':report(root,c,ident,source);return
            acceptance=accept(root,c,ident,resources,teacher,source)
            if a.stage=='pilot':return
            collect(root,c,ident,resources,teacher,source,acceptance)
            fit(root,c,ident,resources)
            evaluate(root,c,ident,resources,teacher,source,acceptance)
        finally:teacher.unload();resources.flush()


if __name__=='__main__':main()
