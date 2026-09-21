import argparse
from pathlib import Path
from ko_common import read,save_json,require,NEW,OLD,METHODS


def validate(c):
    require(c['modules']==NEW+OLD and c['methods']==METHODS,'Exactly8 new modules plus12 A-only baselines')
    require(c['budgets']==[256] and c['labels_per_window']==1 and c['rank']==64 and c['sequence_length']==2048,'Frozen fit design changed')
    require(c['validation_windows']==16 and c['test_windows']==0,'Single-module validation only')
    require(c['offline_workers']==2 and c['profile']=='dual4090','Require two4090 workers')
    require(c['gradient_tolerance']==1e-5 and c['forward_tolerance']==1e-7,'Acceptance thresholds changed')


def main():
    p=argparse.ArgumentParser();p.add_argument('--parent-run',type=Path,required=True)
    p.add_argument('--output-parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();parent=args.parent_run.resolve();old=read(parent/'manifest.json')['config']
    c=dict(old,parent_run=str(parent),output_parent=str(args.output_parent.resolve()),modules=NEW+OLD,methods=METHODS,
           budgets=[256],test_windows=0,budget_hours=8.,disk_limit_GiB=180.,minimum_free_GiB=190.,
           gradient_tolerance=1e-5,forward_tolerance=1e-7,offline_workers=2,cache_GiB=2.)
    validate(c);require(not args.output.exists(),'Configuration already exists')
    require(not args.output_parent.resolve().is_relative_to(parent) and not parent.is_relative_to(args.output_parent.resolve()),'Output overlaps parent')
    save_json(args.output,c);print('CONFIGURED',args.output)


if __name__=='__main__':main()
