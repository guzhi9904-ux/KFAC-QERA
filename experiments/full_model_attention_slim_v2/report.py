import csv
import math
from fm_common import *

def csv_file(path,rows):
    if not rows:return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def report(ctx):
    rows=[];per_window=[];missing=[]
    for state in STATES:
        row={'state':state}
        for corpus,prefix in [('validation','kl'),('wikitext2','ppl'),('c4','ppl')]:
            expected=16 if corpus=='validation' else (138 if corpus=='wikitext2' else 128)
            if corpus=='wikitext2' and (ctx.root/'data_manifest.json').exists():
                expected=read(ctx.root/'data_manifest.json')['datasets']['wikitext2']['windows']
            paths=sorted((ctx.root/prefix/state/corpus).glob('w*.json'))
            records=[read(p) for p in paths];per_window.extend(records)
            if len(records)!=expected:
                missing.append(f'{state}/{corpus}: {len(records)}/{expected}')
                row[corpus+'_NLL']=None;row[corpus+'_PPL']=None
                if corpus=='validation':row['KL']=None
                continue
            tokens=sum(r['tokens'] for r in records);nll=sum(r['NLL_sum'] for r in records)/tokens
            row[corpus+'_NLL']=nll;row[corpus+'_PPL']=math.exp(nll)
            if corpus=='validation':row['KL']=sum(r['KL_sum'] for r in records)/tokens
        scores=[]
        for task,metric in TASKS.items():
            path=ctx.root/'downstream'/state/task/'complete.json'
            value=read(path)['value'] if path.exists() else None;row[task+'_'+metric]=value
            if value is not None:scores.append(value)
            elif state!='C1':missing.append(f'{state}/{task}')
        row['task_macro_mean']=sum(scores)/5 if len(scores)==5 else None;rows.append(row)
    baseline=next(r['KL'] for r in rows if r['state']=='C0')
    for row in rows:row['recovery_percent']=100*(1-row['KL']/baseline) if baseline and row['KL'] is not None else None
    deltas=[]
    for left,right in [('C0','C1'),('C1','C2'),('C2','C3'),('C3','C4'),('C2','C4')]:
        a=next(r for r in rows if r['state']==left);b=next(r for r in rows if r['state']==right)
        for metric in a:
            if metric!='state':
                deltas.append(dict(comparison=left+'->'+right,metric=metric,
                    right_minus_left=None if a[metric] is None or b[metric] is None else b[metric]-a[metric],
                    better_direction='positive' if metric.endswith(('acc','acc_norm','percent','mean')) else 'negative'))
    csv_file(ctx.root/'summary/results.csv',rows);csv_file(ctx.root/'summary/differences.csv',deltas)
    if per_window:
        keys=sorted(set().union(*(r.keys() for r in per_window)))
        csv_file(ctx.root/'summary/per_window.csv',[{k:r.get(k) for k in keys} for r in per_window])
    complete=not missing and bool(ctx.done('downstream/complete.json',model_tasks=25)) and bool(ctx.done('ppl/complete.json',states=6,corpora=2,KL_windows=16))
    status='EXPERIMENT_COMPLETE' if complete else 'PARTIAL/BLOCKED'
    write(ctx.root/'summary/status.json',dict(status=status,missing=missing,identity=ctx.identity))
    def fmt(x):return 'NA' if x is None else f'{x:.7g}'
    lines=['# Full-model Structured Attention / SlimPajama','',status,'',
      '| State | KL16 | Recovery % | WT2 token-PPL | C4 fixed128 token-PPL | Five-task mean |',
      '|---|---:|---:|---:|---:|---:|']
    lines += ['| '+' | '.join([r['state'],fmt(r['KL']),fmt(r['recovery_percent']),fmt(r['wikitext2_PPL']),fmt(r['c4_PPL']),fmt(r['task_macro_mean'])])+' |' for r in rows]
    lines += ['', 'Individual tasks and signed absolute differences are in summary/results.csv and differences.csv. '
              'C1 has no downstream tasks by design. Missing results are NA and never filled from historical scores.', '',
              'C0→C2: total compensation; C1→C2: attention increment above fixed MLP-A; '
              'C2→C3: predictive Marginal increment; C3→C4: frozen combined K/V structure rule; C2→C4: total geometry increment.', '',
              'Statistics: 128 shared ordinary A, 128 attention G, 32 K A/G pairs, 32 V A/G pairs; '
              '416 unique rank64 factors when complete. Old WT2 geometry is not reused. '
              'Resource evidence is in resources/timings.json and audit/estimated_cost.md.', '',
              'The WT2 validation windows were development data; some WT2 test results were seen historically. '
              'This run uses SlimPajama calibration and C4 fixed128 as an additional corpus. '
              'No per-layer winners, new seeds, rank tuning, Hessian-fit claims, or packed-inference speed claims. '
              'C3→C4 jointly changes K and V and cannot identify their separate effects.', '',
              'Missing items: '+('; '.join(missing) if missing else 'none')]
    for comparison in ('C2->C3','C3->C4'):
        selected=[d for d in deltas if d['comparison']==comparison and d['metric'] in ('KL','wikitext2_PPL','c4_PPL','task_macro_mean')]
        lines += ['',comparison+': '+', '.join(d['metric']+' right-minus-left='+fmt(d['right_minus_left']) for d in selected)]
    (ctx.root/'READOUT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if complete:
        ctx.commit('complete.json',[ctx.root/'READOUT.md',ctx.root/'summary/results.csv',ctx.root/'summary/differences.csv',
                   ctx.root/'downstream/complete.json',ctx.root/'ppl/complete.json',ctx.root/'candidates/complete.json'],status=status)
    log(status,missing=len(missing))
