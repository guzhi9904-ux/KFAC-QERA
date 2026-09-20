#!/usr/bin/env python3
"""Explicit stages; no background launcher, no automatic long experiment."""
import argparse
import os
from pathlib import Path
import shutil
import sys
from bridge import identity,read,save_json,require,lock,load_record,checked_files
from configure import validate
from planning import make_plan
from resources import Resources


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('config',type=Path);p.add_argument('output',type=Path)
    p.add_argument('stage',choices=('prepare','pilot','collect','fit','evaluate','report','run'));args=p.parse_args()
    require(not sys.flags.optimize,'Python -O is forbidden')
    for key in ('HF_HUB_OFFLINE','HF_DATASETS_OFFLINE','TRANSFORMERS_OFFLINE'):os.environ[key]='1'
    config=read(args.config);validate(config);root=args.output.resolve();parent=Path(config['output_parent']).resolve()
    import torch
    torch.set_num_threads(config['cpu_threads']);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    require(root!=parent and root.is_relative_to(parent),'Run output must be a child of configured output_parent')
    root.mkdir(parents=True,exist_ok=True);manifest=identity(config);ident=manifest['identity']
    with lock(root):
        if (root/'manifest.json').exists():require(read(root/'manifest.json')==manifest,'Frozen source/config mismatch; use a new run')
        else:save_json(root/'manifest.json',manifest)
        if args.stage=='prepare':
            from dataset import prepare
            from assets_local import prepare_quantized
            index=prepare(root,config,ident);shapes=prepare_quantized(root,config,ident)
            plan=make_plan(shapes,config['budgets'],config['labels_per_window'],config['sequence_length'],config['validation_windows'],index['windows']['test'])
            save_json(root/'work_plan.json',plan)
            require(shutil.disk_usage(root).free/2**30>=config['minimum_free_GiB'],'Insufficient filesystem free space; also check account quota')
            print(plan);return
        resources=Resources(root,ident,config)
        try:
            if args.stage=='pilot':
                from acceptance import acceptance
                acceptance(root,config,manifest,resources);return
            pilot=load_record(root/'acceptance.json',ident);require(pilot['passed'],'Pilot must pass first')
            for frozen in ('data/freeze.json','quantized/freeze.json'):checked_files(root,load_record(root/frozen,ident)['files'])
            from assets_local import prepare_quantized
            prepare_quantized(root,config,ident)
            from shared_teacher import SharedTeacher
            teacher=SharedTeacher(config,root,ident,resources.timed)
            if args.stage in ('collect','run'):
                from collection import collect
                collect(root,config,ident,resources,teacher)
            if args.stage in ('fit','run'):
                from fitting import fit
                fit(root,config,ident,resources)
            if args.stage in ('evaluate','run'):
                from evaluation import evaluate
                evaluate(root,config,ident,resources,teacher)
            if args.stage=='report':
                from evaluation import report
                report(root,config,ident)
        finally:resources.flush()


if __name__=='__main__':main()
