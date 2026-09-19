"""Fixed-window paired inference; old fit labels are descriptive only."""
import hashlib
import json
from pathlib import Path
import numpy as np

CANDIDATES = ('None','SVD64','A64','Marginal-AG64','Full-fit-AG64')


def decision(lo, hi, tau=.02):
    if lo > tau: return 'IMPROVED'
    if hi < -tau: return 'DEGRADED'
    if lo >= -tau and hi <= tau: return 'SMALL_WITHIN_BUDGET'
    return 'UNRESOLVED_AT_K16'


def summarize(rows, modules, k, indices=None, floor=1e-12):
    expected = {(m,c,j,d) for m in modules for c in range(8) for j in range(k) for d in CANDIDATES}
    grid = {(r['module'],r['window'],r['replicate'],r['candidate']):r for r in rows}
    assert len(rows) == len(grid) and set(grid) == expected
    quality, paired, windows, draws = [], [], [], {}
    for m in modules:
        v = np.array([[[grid[m,c,j,d]['b'] for d in CANDIDATES] for j in range(k)] for c in range(8)])
        assert np.isfinite(v).all()
        means = v.mean(axis=(0,1))
        boot = None if indices is None else sum(v[c][indices[:,c]].mean(axis=1)/8 for c in range(8))
        valid = bool(means[0] > floor and (boot is None or np.all(boot[:,0]>floor)))
        gamma = 1-means/means[0] if valid else [None]*5
        bgamma = None if boot is None or not valid else 1-boot/boot[:,0,None]
        draws[m] = dict(q=boot,gamma=bgamma)
        for i,d in enumerate(CANDIDATES):
            row = dict(module=m,candidate=d,q=float(means[i]),gamma=None if not valid else float(gamma[i]),
                       ratio_status='VALID' if valid else 'RATIO_UNRESOLVED')
            if boot is not None:
                row.update(q_ci_low=float(np.quantile(boot[:,i],.025)),q_ci_high=float(np.quantile(boot[:,i],.975)))
                if valid:
                    row.update(gamma_ci_low=float(np.quantile(bgamma[:,i],.025)),gamma_ci_high=float(np.quantile(bgamma[:,i],.975)))
            quality.append(row)
            for c in range(8):
                wm=v[c].mean(axis=0)
                windows.append(dict(module=m,window=c,candidate=d,q=float(wm[i]),
                                    gamma=float(1-wm[i]/wm[0]) if wm[0]>floor else None))
        delta=float(means[3]-means[4]);d=delta/means[0] if valid else None
        row=dict(module=m,Delta=delta,d=None if d is None else float(d),tau=.02)
        if boot is None:
            row.update(status='EMPIRICAL_IMPROVEMENT' if d is not None and d>.02 else
                       'EMPIRICAL_DEGRADATION' if d is not None and d<-.02 else
                       'EMPIRICAL_WITHIN_BUDGET' if d is not None else 'RATIO_UNRESOLVED',
                       uncertainty='Fixed fitting objective; no generalization CI')
        else:
            row['SE']=float(np.sqrt((v[:,:,3]-v[:,:,4]).var(axis=1,ddof=1).sum()/(64*k)))
            if valid:
                lo,hi=np.quantile((boot[:,3]-boot[:,4])/boot[:,0],[.025,.975])
                row.update(ci_low=float(lo),ci_high=float(hi),status=decision(lo,hi),
                           direction='POSITIVE' if lo>0 else 'NEGATIVE' if hi<0 else 'UNRESOLVED')
            else: row.update(status='RATIO_UNRESOLVED',direction='UNRESOLVED')
        paired.append(row)
    return dict(quality=quality,paired=paired,windows=windows),draws


def build(root, modules, oldrows, newrows, metrics, historical):
    from storage import save_json,save_csv,sha_file
    indices=np.random.Generator(np.random.PCG64(2026091805)).integers(0,16,size=(2000,8,16),dtype=np.int64)
    old,_=summarize(oldrows,modules,4)
    new,draws=summarize(newrows,modules,16,indices)
    np.savez_compressed(root/'bootstrap_indices.npz',indices=indices.astype(np.uint8))
    save_json(root/'bootstrap.json',dict(seed=2026091805,rng='numpy.PCG64',count=2000,quantile='linear',
        indices_sha256=sha_file(root/'bootstrap_indices.npz'),scope='Within each fixed window, jointly resample 16 labels for every candidate; no refitting; no old-label bootstrap'))
    save_csv(root/'stage_a/by_window.csv',old['windows']);save_csv(root/'stage_b/by_window.csv',new['windows'])
    comparison=[]
    for domain,data in [('old_fit',old),('new_labels',new)]:
        comparison.extend(dict(domain=domain,**r) for r in data['quality'])
    for r in historical['correction_quality']:
        comparison.append(dict(domain='historical_validation',module=r['module'],candidate=r['candidate'],
            q=r['q_H'],q_ci_low=r['q_H_ci_low'],q_ci_high=r['q_H_ci_high'],gamma=r['gamma_H'],
            gamma_ci_low=r['gamma_H_ci_low'],gamma_ci_high=r['gamma_H_ci_high'],interval_source='Unchanged Exp-3: q uses conditional normal SE; gamma paired bootstrap'))
    paired=[dict(domain=domain,**r) for domain,data in [('old_fit',old),('new_labels',new)] for r in data['paired']]
    for r in historical['paired_comparisons']:
        if r['quantity']=='correction_d_H':
            paired.append(dict(domain='historical_validation',module=r['module'],Delta=r['raw_difference'],
                d=r['mean'],SE=r['raw_SE'],ci_low=r['ci_low'],ci_high=r['ci_high'],status=r['status'],tau=.02))
    gaps=[]
    for r in metrics:
        for domain in ('old_fit','new_labels','historical_validation'):
            q=next(s for s in comparison if (s['domain'],s['module'],s['candidate'])==(domain,r['module'],r['candidate']))
            row=dict(domain=domain,**r,gamma_D=q['gamma'],prediction_gap=r['gamma_K']-q['gamma'] if q['gamma'] is not None else None)
            if 'gamma_ci_low' in q:
                row.update(gap_ci_low=r['gamma_K']-q['gamma_ci_high'],gap_ci_high=r['gamma_K']-q['gamma_ci_low'])
            gaps.append(row)
    save_csv(root/'summary/three_domain_comparison.csv',comparison)
    save_csv(root/'summary/paired_comparisons.csv',paired)
    save_csv(root/'summary/metric_prediction_gaps.csv',gaps)
    result=dict(comparison=comparison,paired=paired,gaps=gaps)
    save_json(root/'analysis.json',result)
    return result
