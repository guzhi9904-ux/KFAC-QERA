#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import signal
import time
import traceback
from bridge import PLAN,read,torch,save_json,request_pause,Paused
from bridge import start_memory_monitor
from runtime import Runtime,BudgetReached
from construction import Construction
from evaluation import Evaluation


class ModuleExperiment(Construction,Evaluation,Runtime):pass


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--phase',choices=['pilot','construct','evaluate'],required=True)
    parser.add_argument('--modules',nargs='+',required=True);parser.add_argument('--deadline',type=float,required=True)
    parser.add_argument('--worker-id',required=True);args=parser.parse_args()
    assert os.name=='posix' and torch.cuda.device_count()==2
    monitor=start_memory_monitor()
    assert set(args.modules)<=set(PLAN['modules']) and len(set(args.modules))==len(args.modules)
    signal.signal(signal.SIGTERM,request_pause);signal.signal(signal.SIGINT,request_pause)
    path=Path(args.output)/'workers'/f'{args.phase}_{args.worker_id}.json'
    record=dict(phase=args.phase,worker=args.worker_id,modules=args.modules,started=time.time(),completed_modules=[])
    save_json(path,record)
    try:
        for name in args.modules:
            e=ModuleExperiment(args.output,name,args.deadline)
            try:
                if args.phase=='pilot' and e.is_parent:
                    # Fail on a different teacher before spending time on dense decompositions.
                    e.reference(0)
                if args.phase in ('pilot','construct'):
                    e.construct();e.clean_temporary()
                if args.phase=='pilot':e.evaluate(pilot=True);e.clean_temporary()
                elif args.phase=='evaluate':e.evaluate();e.clean_temporary()
                record['completed_modules'].append(name);save_json(path,record)
            except BaseException as error:
                e.status('BUDGET_REACHED' if isinstance(error,BudgetReached) else 'PAUSED' if isinstance(error,Paused) else 'FAILED',
                         module=name,error=repr(error),traceback=traceback.format_exc())
                raise
            finally:e.unload()
    finally:
        monitor.set()
        record['finished']=time.time();record['all_completed']=record['completed_modules']==args.modules;save_json(path,record)


if __name__=='__main__':main()
