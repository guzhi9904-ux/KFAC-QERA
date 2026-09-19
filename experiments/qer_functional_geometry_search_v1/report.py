"""CPU-only independent reaggregation and portable SVG/Markdown reporting."""
import html
import math
from pathlib import Path
import numpy as np
from common import PLAN, read, save_json, save_csv, sha_file, atomic_bytes, require, slug
from records import load_record, checked_files
from geometry import choose, candidate_key
from analysis_math import ratio_summary


def fmt(x): return 'unresolved' if x is None else f'{x:.6g}'


def heatmap(path, scores, selected):
    # No external plotting dependencies on either server.
    values = [scores[candidate_key(a,b)] for a in PLAN['exponents'] for b in PLAN['exponents']]
    lo,hi = min(values),max(values)
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="600" height="410" viewBox="0 0 600 410">',
        '<rect width="600" height="410" fill="white"/>',
        '<g font-family="sans-serif" fill="#172b4d"><text x="30" y="30" font-size="20">Search q_H / None (selection only)</text>']
    for i,a in enumerate(PLAN['exponents']):
        for j,b in enumerate(PLAN['exponents']):
            key = candidate_key(a,b); value = scores[key]
            f = (value-lo)/(hi-lo) if hi>lo else .5; color=f'rgb({int(222+25*f)},{int(245-70*f)},{int(242-80*f)})'
            x,y = 85+155*j, 65+100*i
            parts.append(f'<rect x="{x}" y="{y}" width="145" height="90" fill="{color}" stroke="{("#152f71" if key==selected else "white")}" stroke-width="3"/>')
            ratio=value/scores['None'] if scores['None']>PLAN['denominator_floor'] else None
            parts.append(f'<text x="{x+12}" y="{y+28}" font-size="14">a={a:g}, b={b:g}</text><text x="{x+12}" y="{y+57}" font-size="17">{html.escape(fmt(ratio))}</text>')
    parts.append('<text x="85" y="389" font-size="13">Dark outline: frozen selection. Test data never used here.</text></g></svg>')
    atomic_bytes(path, '\n'.join(parts).encode())


def module_report(root, name, identity):
    folder=root/'modules'/slug(name)
    if not (folder/'completion.json').exists():
        nsearch=len(list((folder/'atomic/search').glob('w*_k*.json')))
        ntest=len(list((folder/'atomic/test').glob('w*_k*.json')))
        return dict(module=name,status='INCOMPLETE',search_gradients=nsearch,test_gradients=ntest)
    done=load_record(folder/'completion.json',identity)
    freeze=load_record(folder/'selection_freeze.json',identity); checked_files(root,freeze['files'])
    require(freeze['source_manifest_sha256']==sha_file(root/'manifest.json'),'Source freeze changed')
    cm=load_record(folder/'candidate_manifest.json',identity); checked_files(folder,cm['files'])
    grid={candidate_key(a,b):[a,b] for a in PLAN['exponents'] for b in PLAN['exponents']}
    search=[load_record(folder/'atomic/search'/f'w{c:02d}_k{k:03d}.json',identity)
        for c in range(PLAN['search_windows']) for k in range(PLAN['search_labels'])]
    scores={key:math.fsum(r['physical'][key]['score'] for r in search)/len(search) for key in ['None']+list(grid)}
    require(scores==freeze['search_scores'],'Search scores do not recompute exactly')
    selected=choose({k:scores[k] for k in grid},grid,scores['None'],PLAN['denominator_floor'])['selected']
    require(selected==freeze['selection']['selected'],'Selection does not match search argmin')
    heatmap(folder/'search_heatmap.svg',scores,selected)
    if selected is None:
        require(done['status']=='DENOMINATOR_UNRESOLVED','Incorrect degenerate completion state')
        return dict(module=name,status='DENOMINATOR_UNRESOLVED',selection=freeze['selection'],search_scores=scores)
    require(done['selection_freeze_sha256']==sha_file(folder/'selection_freeze.json'),'Completion freeze binding differs')
    mapping=done['role_mapping']; roles=['None','A-only','Marginal-AG','Selected']; unique=list(dict.fromkeys(mapping.values()))
    expected=dict(zip(roles,['None',candidate_key(1,0),candidate_key(1,1),selected]))
    require(mapping==expected,'Test role alias differs')
    for phase,phase_rows in [('search',search),('test',None)]:
        if phase_rows is None:
            phase_rows=[load_record(folder/'atomic/test'/f'w{c:02d}_k{k:03d}.json',identity)
                for c in range(PLAN['test_windows']) for k in range(PLAN['test_labels'])]
            test=phase_rows
        for row in phase_rows:
            lp=root/'data/labels'/phase/f'w{row["window"]:02d}_k{row["replicate"]:03d}.safetensors'
            require(sha_file(lp)==row['label_file_sha256'],'Label file differs during independent report')
            require(row['candidate_manifest_sha256']==sha_file(folder/'candidate_manifest.json'),'Projection source binding differs')
            for key,item in row['physical'].items():
                require(item['weight_hash']==cm['candidates'][key]['weight_hash'],'Projected weight differs')
                square=item['d']*item['d']
                require(item['squared']==square and item['score']==square/(2*PLAN['T']),'Projection score normalization differs')
        checks=load_record(folder/(phase+'_numerical_checks.json'),identity)
        require(checks['passed'],'Failed deployment numerical acceptance')
    values=np.asarray([[r['physical'][mapping[role]]['score'] for role in roles] for r in test]).reshape(PLAN['test_windows'],PLAN['test_labels'],4)
    qsummary=ratio_summary(values,roles,PLAN['bootstrap_seed'],PLAN['bootstrap_count'],PLAN['denominator_floor'])
    kl=[]; repeat_guard=0.
    for c in range(PLAN['test_windows']):
        table={key:load_record(folder/'atomic/kl'/f'w{c:02d}_{key}.json',identity) for key in unique}
        for key,row in table.items():
            require(row['candidate_file_sha256']==cm['candidates'][key]['sha256'],'KL candidate changed')
            require(abs(row['self_KL'])<=PLAN['tolerances']['self_KL'],'Self-KL acceptance missing')
            if c==0:
                require(row['repeat'] is not None and row['repeat']['difference']<=row['repeat']['allowed'],'KL repeat acceptance missing')
                repeat_guard=max(repeat_guard,row['repeat']['difference'],row['repeat']['allowed'])
        kl.append([[table[mapping[r]]['value'] for r in roles]])
    ksummary=ratio_summary(kl,roles,PLAN['bootstrap_seed']+100,PLAN['bootstrap_count'],PLAN['denominator_floor'],kl=True,repeat_guard=repeat_guard)
    baseline=freeze['selection']['baseline_selected']
    qstatus=qsummary['comparisons'][0]['article']['status']; kstatus=ksummary['comparisons'][0]['article']['status']
    if baseline=='Marginal-AG': conclusion='Baseline-selected: this search found no extra improvement over Marginal-AG.'
    elif qstatus=='IMPROVED' and kstatus=='IMPROVED': conclusion='Pilot support on this module: independent q_H and actual KL both improved versus Marginal-AG.'
    elif qstatus=='IMPROVED': conclusion='Independent quadratic improvement has not established an actual-KL improvement.'
    else: conclusion='Independent pilot has not established improvement on both metrics; unresolved is not proof of equivalence.'
    result=dict(module=name,status='COMPLETE',selection=freeze['selection'],search_scores=scores,
        search_recovery={role:1-scores[key]/scores['None'] for role,key in mapping.items()},
        functional=qsummary,KL=ksummary,conclusion=conclusion,role_mapping=mapping,
        counts=dict(search_gradients=len(search),test_gradients=len(test),search_logical_scores=len(search)*10,
            test_logical_scores=len(test)*4,KL_main=PLAN['test_windows']*len(unique),KL_replays=len(unique)),
        search_interpretation='selected/training performance only, not an unbiased generalization estimate')
    save_json(folder/'comparisons.json',result)
    save_json(folder/'numerical_checks.json',dict(identity=identity,passed=True,
        parent_replay=load_record(folder/'parent_replay.json',identity),
        search=load_record(folder/'search_numerical_checks.json',identity),
        test=load_record(folder/'test_numerical_checks.json',identity),KL_repeat_max_absolute=repeat_guard,
        exact_aliases=baseline,full_new_S_saved=False))
    return result


def report(root):
    root=Path(root); manifest=read(root/'manifest.json'); identity=manifest['identity']
    results=[module_report(root,n,identity) for n in manifest['config']['modules']]
    resources=read(root/'resource_usage.json') if (root/'resource_usage.json').exists() else {}
    complete=all(r['status'] in ('COMPLETE','DENOMINATOR_UNRESOLVED') for r in results)
    save_json(root/'summary.json',dict(identity=identity,execution_complete=complete,modules=results))
    lines=['# Functional geometry search pilot','',f'Execution complete: **{complete}**. Registered modules: {len(results)}.',
        '', 'Fixed teacher/MXINT3 residual, ideal rank 64, nine fixed (a,b) residual-SVD geometries. No H(E) SVD or amplitude fitting.',
        '', 'Each reported comparison uses the search-frozen selection. New test articles did not select the geometry.']
    tables=[]
    for r in results:
        lines += ['', '## '+r['module'], '', 'Status: '+r['status']]
        if r['status']!='COMPLETE':
            lines += ['', 'No missing test/module result is imputed as zero or as failure. '+str(r.get('selection',{}))]; continue
        lines += ['',f"Selected: **{r['selection']['parameters']}**; baseline alias: {r['selection']['baseline_selected'] or 'none'}.",
            '',r['conclusion'],'',f"![Search grid](modules/{slug(r['module'])}/search_heatmap.svg)",
            '', '| Role | Search q recovery | Test q recovery | Test KL recovery |','|---|---:|---:|---:|']
        for role in r['role_mapping']:
            lines.append(f"| {role} | {fmt(r['search_recovery'][role])} | {fmt(r['functional']['candidates'][role]['recovery'])} | {fmt(r['KL']['candidates'][role]['recovery'])} |")
        lines += ['', 'Recovery is 1 - ratio of mean damages. Values are not clipped.',
            '', '| Metric | Baseline | Relative gain d | Article 95% CI | State | Conditional-label 95% CI |',
            '|---|---|---:|---|---|---|']
        for metric in ('functional','KL'):
            for row in r[metric]['comparisons']:
                ci=row['article']; label=row.get('conditional_labels')
                labeltext=f"[{fmt(label['ci_low'])}, {fmt(label['ci_high'])}]" if label else 'not applicable'
                lines.append(f"| {metric} | {row['baseline']} | {fmt(row['d'])} | [{fmt(ci['ci_low'])}, {fmt(ci['ci_high'])}] | {ci['status']} | {labeltext} |")
                tables.append(dict(module=r['module'],metric=metric,**row))
        lines += ['', 'Counts: '+str(r['counts'])]
    lines += ['', '## Resources and limits','',f"Cumulative active wall time: {resources.get('active_seconds',0)/3600:.3f} h; output: {resources.get('output_GiB',0):.3f} GiB.",
        '',f"GPU allocated peaks (GiB): {resources.get('max_GPU_allocated_GiB',{})}. Host peak RSS (GiB): {resources.get('host_peak_RSS_GiB','unavailable')}.",
        '', 'See resource_usage.json for model loading, historical rebuild/replay, sampling, gradient/projection, eigendecomposition/SVD, KL and I/O costs.',
        '', 'Intervals use 2000 paired article resamples. Conditional-label resamples are reported separately, never nested/added; KL has article intervals only.',
        '', '32 search articles and 16 test articles are a pilot budget, not a power guarantee. Results apply only to these registered modules and this same-corpus spectral family. No all-model PPL, global oracle, cross-corpus or novelty claim.',
        '', 'No follow-up experiment is started automatically.']
    save_csv(root/'comparisons.csv',tables); atomic_bytes(root/'RESULTS.md', ('\n'.join(lines)+'\n').encode())
    return complete
