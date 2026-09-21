#!/usr/bin/env python3
"""Resume frozen evaluation after rental renewal with an explicit total runtime allowance."""
import argparse
import math
import os
from pathlib import Path
import sys
import time
import uuid

HERE=Path(__file__).resolve().parent
ENTRY=HERE.parents[1]/'experiments/qer_multimodule_three_fp64_v1'
sys.path.insert(0,str(ENTRY))
import runner
from bridge import read,require,identity,load_record,save_json,sha_file


def runtime_config(config,total_hours,active_seconds):
    require(math.isfinite(total_hours) and total_hours>=config['budget_hours'],'Total hours cannot reduce the original allowance')
    require(total_hours*3600>active_seconds,'Requested total allowance is already exhausted')
    return dict(config,budget_hours=total_hours)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--total-hours',type=float,required=True,help='Cumulative allowance INCLUDING historical active time, not extra hours')
    parser.add_argument('--test-windows',type=int,choices=(16,32),help='Evaluate a fixed prefix of test windows; write a separate subset report, never full completion')
    args=parser.parse_args();root=args.output.resolve();config=read(args.config)
    frozen=read(root/'manifest.json');require(identity(config)==frozen,'Frozen science/config changed; do not edit original config')
    if (root/'complete.json').exists():
        require(load_record(root/'complete.json',frozen['identity'])['passed'],'Completion record not passed')
        print('EXPERIMENT_ALREADY_COMPLETE',flush=True);return
    require((root/'candidate_freeze.json').exists(),'All72 candidates must be frozen before evaluation-only resume')
    source_class=runner.Resources
    class ExtendedResources(source_class):
        def __init__(self,root,ident,config):
            super().__init__(root,ident,config)
            # The runner already holds the OS file lock. Preserve all elapsed
            # time and scientific identities; only this instance's allowance changes.
            self.config=runtime_config(config,args.total_hours,self.base)
            receipt=root/'runtime_extensions'/('resume_'+uuid.uuid4().hex+'.json')
            save_json(receipt,dict(identity=ident,stage='evaluate',started_utc=time.strftime('%FT%TZ',time.gmtime()),
                pid=os.getpid(),active_seconds_before=self.base,original_config_hours=config['budget_hours'],
                requested_total_hours=args.total_hours,manifest_sha256=sha_file(root/'manifest.json'),
                launcher_sha256=sha_file(__file__),test_windows=args.test_windows,
                changes='Runtime allowance and optional explicitly labelled test subset; no time reset, no candidate/config/threshold changes'))
            self.data.update(runtime_budget_hours=args.total_hours,runtime_budget_receipt=str(receipt))
            self.flush();print('EVALUATION_RESUME',dict(used_hours=self.base/3600,total_hours=args.total_hours),flush=True)
    runner.Resources=ExtendedResources
    sys.argv=[str(ENTRY/'runner.py'),str(args.config.resolve()),str(root),'evaluate']
    if args.test_windows is None:
        runner.main()
    else:
        from subset_eval import limited_test
        with limited_test(args.test_windows):runner.main()


if __name__=='__main__':main()
