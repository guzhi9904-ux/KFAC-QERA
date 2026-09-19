#!/usr/bin/env python3
"""Explicit CPU preparation, bounded GPU run, or CPU report; no remote actions."""
import argparse
import os
import sys
from pathlib import Path
import traceback
from common import PLAN, HERE, read, save_json, sha_file, atomic_bytes, require, source_identity
from resources import Resources, BudgetReached, single_worker


def setup(root, config):
    manifest=source_identity(config)
    if (root/'manifest.json').exists(): require(read(root/'manifest.json')==manifest,'Source/configuration changed: use a new run directory')
    else:
        require(not any(p.name!='.run.lock' for p in root.iterdir()),'New run directory is not empty')
        save_json(root/'manifest.json',manifest)
        atomic_bytes(root/'protocol.md',(HERE/'protocol.md').read_bytes())
    return manifest['identity']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('config',type=Path); p.add_argument('output',type=Path)
    p.add_argument('stage',choices=['prepare','run','report'])
    args=p.parse_args(); config=read(args.config); root=args.output.resolve()
    require(config['schema']==1,'Configuration schema differs')
    require(config['modules'] in (PLAN['modules'][:1],PLAN['modules']),'Only preregistered L10 then optional L31 allowed')
    require(config['profile'] in ('a6000','dual4090') and config['budget_hours']>0,'Invalid profile or budget')
    require(root.is_relative_to(Path(config['output_parent']).resolve()) and root!=Path(config['output_parent']).resolve(),'Run must be a child of configured output parent')
    for field in ('assets','model','wikitext'):
        protected=Path(config[field]).resolve()
        require(not root.is_relative_to(protected) and not protected.is_relative_to(root),'Output overlaps protected asset')
    require(not root.is_relative_to(HERE),'Output cannot be inside versioned experiment source')
    with single_worker(root):
        identity=setup(root,config)
        if args.stage=='report':
            from report import report
            report(root); print('CPU report:',root/'RESULTS.md',flush=True); return
        resources=Resources(root,identity,config['budget_hours'],PLAN['disk_budget_GiB']); status='FAILED'
        try:
            import assets
            import data_split
            with resources.timed('asset_inventory_and_hashes'):
                inventory=assets.inventory(config)
                ap=root/'asset_identity.json'
                if ap.exists(): require(read(ap)==inventory,'Parent/data asset set changed')
                else: save_json(ap,inventory)
            with resources.timed('article_selection_and_freeze'):
                data_split.build(root,config,identity,inventory)
            if args.stage=='prepare':
                status='PREPARED'; print('CPU preparation complete. No GPU experiment started.',flush=True); return
            require(not os.environ.get('PYTHONOPTIMIZE') and sys.flags.optimize==0,'Do not disable scientific assertions with Python -O')
            import torch
            require(torch.cuda.is_available(),'CUDA unavailable; prepared inputs retained, no samples imputed')
            from engine import Experiment
            experiment=Experiment(root,config,identity,inventory,resources)
            for name in config['modules']:
                resources.boundary(); experiment.run_module(name)
                print('MODULE_COMPLETE',name,flush=True)
            for path,value in inventory['files'].items():
                require(sha_file(path)==value,'Input asset changed during execution: '+path)
            status='COMPLETE'
        except BudgetReached as e:
            status='INCOMPLETE_BUDGET'; print(str(e),flush=True)
        except Exception as e:
            save_json(root/'failure.json',dict(identity=identity,error=repr(e),traceback=traceback.format_exc()))
            raise
        finally:
            resources.close(status)
            save_json(root/'status.json',dict(identity=identity,status=status))
            if args.stage=='run':
                from report import report
                try: report(root)
                except Exception as e:
                    print('Report validation incomplete:',repr(e),flush=True)
                    if status=='COMPLETE': raise


if __name__=='__main__': main()
