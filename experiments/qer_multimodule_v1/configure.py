"""Write an explicit12-module configuration without starting GPU work."""
import argparse
from pathlib import Path
from bridge import read,save_json,require,MODULES,METHODS
from planning import defaults


def validate(config):
    require(config['modules']==MODULES and config['methods']==METHODS,'This version registers exactly the chosen12 modules and four methods')
    require(config['sequence_length']==2048 and config['rank']==64,'Frozen numerical primitives require length2048/rank64')
    require(config['budgets']==[128,256] and config['labels_per_window']==1,'This first protocol uses nested128/256 windows and one shared label set per window')
    require(config['validation_windows']==16 and not config['diagnostics'],'Main protocol requires16 validation windows and excludes heavy diagnostics')
    require(config['offline_workers'] in (1,2) and 0<=config['cache_GiB']<=8 and config['budget_hours']>0,'Invalid resources')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--from-config',type=Path,required=True)
    p.add_argument('--wikitext',type=Path,required=True,help='Local DatasetDict or train/validation/test dataset directories')
    p.add_argument('--quantized-run',type=Path,required=True);p.add_argument('--output-parent',type=Path,required=True)
    p.add_argument('--hours',type=float,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();base=read(args.from_config);config=defaults()
    config.update({k:base[k] for k in ('assets','model','cpu_threads','vocab_chunk')})
    config.update(profile='dual4090',wikitext=str(args.wikitext.resolve()),quantized_run=str(args.quantized_run.resolve()),
        output_parent=str(args.output_parent.resolve()),budget_hours=args.hours)
    validate(config)
    out=args.output_parent.resolve()
    for k in ('assets','model','wikitext','quantized_run'):
        source=Path(config[k]).resolve();require(not out.is_relative_to(source) and not source.is_relative_to(out),'Output overlaps an input')
    require(not args.output.exists(),'Configuration already exists');save_json(args.output,config)
    print('CONFIGURED',args.output.resolve(),'No GPU work started. Next: prepare, then pilot.')


if __name__=='__main__':main()
