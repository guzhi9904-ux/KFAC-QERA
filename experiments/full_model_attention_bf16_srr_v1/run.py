#!/usr/bin/env python3
import argparse
import traceback
from common import *

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','pilot','evaluate','report','resume'])
    parser.add_argument('--config',default=str(HERE/'config.json'));args=parser.parse_args()
    cfg=read(args.config);cache=Path(cfg['base'])/'huggingface_cache'
    for key,value in dict(HF_HOME=cache,HF_DATASETS_CACHE=cache/'datasets',HF_HUB_CACHE=cache/'hub').items():os.environ.setdefault(key,str(value))
    os.environ['TOKENIZERS_PARALLELISM']='false'
    torch.set_num_threads(cfg['cpu_threads']);torch.manual_seed(1234)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    from benchmarks import prepare
    from evaluation import pilot,evaluate,report
    import fcntl
    root=Path(cfg['run']);root.mkdir(parents=True,exist_ok=True)
    with (root/'run.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        ctx=Context(args.config)
        require(torch.cuda.device_count()==2,'Requires two GPUs')
        write(root/'launch.json',dict(identity=ctx.identity,pid=os.getpid(),argv=sys.argv,time=time.time()))
        stages=dict(prepare=prepare,pilot=pilot,evaluate=evaluate,report=report)
        try:
            for stage in stages if args.stage=='resume' else [args.stage]:
                with timed(ctx,stage):stages[stage](ctx)
        except BaseException as error:
            write(root/'failure.json',dict(identity=ctx.identity,stage=stage,error=repr(error),traceback=traceback.format_exc(),time=time.time()))
            raise

if __name__=='__main__':main()
