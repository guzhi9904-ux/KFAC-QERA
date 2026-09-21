#!/usr/bin/env python3
"""Ordered, fail-closed, identity-bound stages; resume never starts a second writer."""
import argparse
import os
from pathlib import Path
import sys
import traceback
from fm_common import *

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['audit','prepare-data','pilot','collect-a','collect-functional','solve','freeze','eval-kl-ppl','eval-downstream','report','resume'])
    parser.add_argument('--config',default=str(HERE/'config.yaml'))
    args=parser.parse_args();cfg=read(args.config)
    cache=Path(cfg['base'])/'huggingface_cache'
    os.environ.setdefault('HF_HOME',str(cache));os.environ.setdefault('HF_DATASETS_CACHE',str(cache/'datasets'))
    os.environ.setdefault('HF_HUB_CACHE',str(cache/'hub'));os.environ['TOKENIZERS_PARALLELISM']='false'
    torch.set_num_threads(cfg['cpu_threads']);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    from audit import audit
    from data import prepare
    from pilot import pilot
    from collect import collect_a,collect_functional
    from solve import solve,freeze
    from evaluate import eval_kl_ppl,eval_downstream
    from report import report
    stages={'audit':audit,'prepare-data':prepare,'pilot':pilot,'collect-a':collect_a,'collect-functional':collect_functional,
            'solve':solve,'freeze':freeze,'eval-kl-ppl':eval_kl_ppl,'eval-downstream':eval_downstream,'report':report}
    ctx=Context(args.config)
    import fcntl
    with (ctx.root/'run.lock').open('a+') as handle:
        fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        write(ctx.root/'launch.json',dict(pid=os.getpid(),argv=sys.argv,identity=ctx.identity,time=time.time()))
        try:
            for stage in stages if args.stage=='resume' else [args.stage]:
                with ctx.timed(stage):stages[stage](ctx)
        except Exception as error:
            traceback.print_exc()
            write(ctx.root/'failure.json',dict(identity=ctx.identity,stage=stage,time=time.time(),error=repr(error),traceback=traceback.format_exc()))
            try:report(ctx)
            except Exception:traceback.print_exc()
            raise
        finally:ctx.teacher.unload()

if __name__=='__main__':main()
