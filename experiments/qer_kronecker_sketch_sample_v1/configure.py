#!/usr/bin/env python3
"""Host-local, explicit dual4090 registration. Does not start model work."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--from-geometry-config',type=Path,required=True)
    p.add_argument('--geometry-history',type=Path,required=True)
    p.add_argument('--output-parent',type=Path,required=True)
    p.add_argument('--hours',type=float,required=True)
    p.add_argument('--cache-gib',type=float,default=8.)
    p.add_argument('--gram-block',type=int,default=8)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();base=json.loads(args.from_geometry_config.read_text())
    if args.hours<=0 or not 0<=args.cache_gib<=16 or not 1<=args.gram_block<=8:p.error('Invalid time/memory/block budget')
    config={k:base[k] for k in ('assets','model','wikitext','calibration','cpu_threads','vocab_chunk')}
    config.update(schema=1,profile='dual4090',modules=['model.layers.31.self_attn.q_proj'],
        geometry_history=str(args.geometry_history.resolve()),output_parent=str(args.output_parent.resolve()),
        budget_hours=args.hours,cache_GiB=args.cache_gib,gram_block=args.gram_block,
        extra_history_windows=base.get('extra_history_windows',[]),extra_article_manifests=base.get('extra_article_manifests',[]))
    out=args.output_parent.resolve()
    for value in [config[k] for k in ('assets','model','wikitext','geometry_history')]+[str(Path(__file__).parent)]:
        path=Path(value).resolve()
        if out.is_relative_to(path) or path.is_relative_to(out):p.error('Output parent overlaps source or immutable asset')
    with args.output.open('x',encoding='utf8') as f:json.dump(config,f,indent=2);f.write('\n')
    print('Saved',args.output.resolve(),'No experiment started. Next stage: prepare, then pilot.')


if __name__=='__main__':main()
