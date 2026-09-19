#!/usr/bin/env python3
"""Create a host-local preregistration. This command never loads a model or starts CUDA."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--from-portable-config', type=Path, help='Existing qer_4090_v1.json: reuse assets/model only')
    for name in ('assets','model','wikitext','calibration','output-parent'):
        p.add_argument('--'+name, type=Path)
    p.add_argument('--profile', choices=['a6000','dual4090'], default='dual4090')
    p.add_argument('--modules', type=int, choices=[1,2], default=2, help='Preregister L10 only, or L10 then L31')
    p.add_argument('--hours', type=float, required=True, help='Cumulative active wall budget over all attempts')
    p.add_argument('--cpu-threads', type=int, default=8)
    p.add_argument('--vocab-chunk', type=int, default=64)
    p.add_argument('--extra-history-windows', type=Path, action='append', default=[])
    p.add_argument('--extra-article-manifests', type=Path, action='append', default=[])
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    base = json.loads(args.from_portable_config.read_text()) if args.from_portable_config else {}
    for name in ('assets','model'):
        if getattr(args,name) is None and name in base: setattr(args,name,Path(base[name]))
    if args.assets:
        inputs = args.assets.resolve().parent/'geometry_inputs'
        args.wikitext = args.wikitext or inputs/'wikitext2'
        args.calibration = args.calibration or inputs/'calibration.safetensors'
    if args.output_parent is None: args.output_parent = args.output.resolve().parent/'qera_runs/geometry_search_v1'
    if not args.assets or not args.model: p.error('Pass --assets and --model, or --from-portable-config')
    if args.hours <= 0 or args.cpu_threads < 1 or not 1 <= args.vocab_chunk <= 128: p.error('Invalid resource settings')
    config = {k:str(getattr(args,k).resolve()) for k in ('assets','model','wikitext','calibration','output_parent')}
    out = Path(config['output_parent'])
    protected = [Path(config[k]) for k in ('assets','model','wikitext')]+[Path(__file__).resolve().parent]
    if any(out == v or out.is_relative_to(v) or v.is_relative_to(out) for v in protected):
        p.error('Output parent must be separate from source/model/parent assets')
    plan = json.loads((Path(__file__).parent/'plan.json').read_text())
    config.update(schema=1, profile=args.profile, modules=plan['modules'][:args.modules],
        budget_hours=args.hours, cpu_threads=args.cpu_threads, vocab_chunk=args.vocab_chunk,
        extra_history_windows=[str(v.resolve()) for v in args.extra_history_windows],
        extra_article_manifests=[str(v.resolve()) for v in args.extra_article_manifests])
    with args.output.open('x', encoding='utf8') as f: json.dump(config,f,indent=2); f.write('\n')
    print('Configuration saved:', args.output.resolve())
    for name in ('assets','model','wikitext','calibration'):
        print(name+':', config[name], '(exists)' if Path(config[name]).exists() else '(MISSING: required before prepare)')
    print('Preregistered modules:', ', '.join(config['modules']))
    print('No experiment started. Next: run.sh CONFIG NEW_RUN_DIRECTORY prepare')


if __name__ == '__main__': main()
