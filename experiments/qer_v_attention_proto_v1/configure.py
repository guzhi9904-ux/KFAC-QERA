import argparse
from v_common import Path,read,save_json,require,MODULES


def validate(c):
    require(c['modules']==MODULES and c['budgets']==[256] and c['labels_per_window']==1,'Wrong module/sample scope')
    require(c['rank']==64 and c['sequence_length']==2048 and c['validation_windows']==16 and c['test_windows']==0,'Wrong rank/eval design')
    require(c['profile']=='dual4090' and c['checkpoint_every']==4,'Wrong profile/checkpoints')


def main():
    p=argparse.ArgumentParser();p.add_argument('--parent-run',type=Path,required=True);p.add_argument('--output-parent',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();parent=a.parent_run.resolve()
    c=dict(read(parent/'manifest.json')['config'],parent_run=str(parent),output_parent=str(a.output_parent.resolve()),
           modules=MODULES,methods=['Attention-aware'],budgets=[256],test_windows=0,checkpoint_every=4,
           budget_hours=8.,disk_limit_GiB=16.,minimum_free_GiB=48.,cache_GiB=0.,offline_workers=2)
    validate(c);require(not a.output.exists(),'Config already exists')
    out=a.output_parent.resolve();require(not out.is_relative_to(parent) and not parent.is_relative_to(out),'Output overlaps parent')
    save_json(a.output,c);print('CONFIGURED',a.output)


if __name__=='__main__':main()
