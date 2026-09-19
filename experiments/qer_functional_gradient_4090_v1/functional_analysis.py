"""Paired, fixed-window summaries. A fitted sample is never bootstrapped as a test."""
import math
import numpy as np

RANKS=(16,32,64,128)
METHODS=('Residual-SVD','Residual-SVD-Cal','Gradient-SVD','Gradient-SVD-Cal')
CANDIDATES=('None',)+tuple(f'{m}-r{r}' for r in RANKS for m in METHODS)


def classify(lo,hi,tau=.02):
    if lo>tau:return 'IMPROVED'
    if hi<-tau:return 'DEGRADED'
    if lo>=-tau and hi<=tau:return 'SMALL_WITHIN_BUDGET'
    return 'UNRESOLVED_AT_K16'


def summarize(module,records,k,bootstrap=False):
    grid={(r['window'],r['replicate']):r for r in records}
    assert len(records)==len(grid)==8*k and set(grid)=={(c,j) for c in range(8) for j in range(k)}
    v=np.array([[[grid[c,j]['scores'][name]['b'] for name in CANDIDATES] for j in range(k)] for c in range(8)])
    assert np.isfinite(v).all() and np.all(v>=0)
    q=v.mean(axis=(0,1));boot=None
    if bootstrap:
        assert k==16
        indices=np.random.Generator(np.random.PCG64(2026091902)).integers(0,16,size=(2000,8,16),dtype=np.int64)
        boot=sum(v[c][indices[:,c]].mean(axis=1)/8 for c in range(8))
    valid=bool(q[0]>1e-12 and (boot is None or np.all(boot[:,0]>1e-12)))
    gamma=1-q/q[0] if valid else [None]*len(CANDIDATES)
    bgamma=None if boot is None or not valid else 1-boot/boot[:,0,None]
    curves=[];paired=[];windows=[]
    for i,name in enumerate(CANDIDATES):
        rank=0 if name=='None' else int(name.rsplit('-r',1)[1])
        method=name if name=='None' else name.rsplit('-r',1)[0]
        row=dict(module=module,candidate=name,rank=rank,method=method,q=float(q[i]),
            gamma=float(gamma[i]) if valid else None,ratio_valid=valid)
        if bootstrap:
            row.update(q_ci_low=float(np.quantile(boot[:,i],.025)),q_ci_high=float(np.quantile(boot[:,i],.975)))
            if valid:row.update(gamma_ci_low=float(np.quantile(bgamma[:,i],.025)),gamma_ci_high=float(np.quantile(bgamma[:,i],.975)))
        curves.append(row)
        for c in range(8):
            wm=v[c].mean(axis=0)
            windows.append(dict(module=module,window=c,candidate=name,q=float(wm[i]),
                                gamma=float(1-wm[i]/wm[0]) if wm[0]>1e-12 else None))
    for rank in RANKS:
        for suffix,comparison in [('', 'raw'),('-Cal','calibrated')]:
            i=CANDIDATES.index(f'Residual-SVD{suffix}-r{rank}');j=CANDIDATES.index(f'Gradient-SVD{suffix}-r{rank}')
            delta=float(q[i]-q[j]);d=delta/q[0] if valid else None
            row=dict(module=module,rank=rank,comparison=comparison,Delta=delta,d=d,tau=.02,
                     primary=rank==64,reference=f'Residual-SVD{suffix}-r{rank}',candidate=f'Gradient-SVD{suffix}-r{rank}')
            if bootstrap:
                row['SE']=float(np.sqrt((v[:,:,i]-v[:,:,j]).var(axis=1,ddof=1).sum()/(64*k)))
                if valid:
                    lo,hi=np.quantile((boot[:,i]-boot[:,j])/boot[:,0],[.025,.975])
                    row.update(ci_low=float(lo),ci_high=float(hi),status=classify(lo,hi),
                               direction='POSITIVE' if lo>0 else 'NEGATIVE' if hi<0 else 'UNRESOLVED')
                else:row.update(status='RATIO_UNRESOLVED',direction='UNRESOLVED')
            else:
                row['status']='EMPIRICAL_IMPROVEMENT' if d is not None and d>.02 else 'EMPIRICAL_DEGRADATION' if d is not None and d<-.02 else 'EMPIRICAL_SMALL' if d is not None else 'RATIO_UNRESOLVED'
            paired.append(row)
    return dict(curves=curves,paired=paired,by_window=windows)


def summarize_kl(module,rows):
    names=('None',)+tuple(f'{m}-r64' for m in METHODS)
    grid={(r['window'],r['candidate']):r for r in rows}
    assert len(rows)==len(grid)==40 and set(grid)=={(c,n) for c in range(8) for n in names}
    means={n:math.fsum(grid[c,n]['KL'] for c in range(8))/8 for n in names}
    valid=means['None']>1e-12
    maximum_repeat=max(grid[0,n]['repeat_difference'] for n in names)
    quality=[dict(module=module,candidate=n,KL=q,gamma_KL=1-q/means['None'] if valid else None) for n,q in means.items()]
    paired=[]
    for suffix,comparison in [('', 'raw'),('-Cal','calibrated')]:
        b,f=f'Residual-SVD{suffix}-r64',f'Gradient-SVD{suffix}-r64'
        difference=means[b]-means[f];d=difference/means['None'] if valid else None
        guard=2*max(maximum_repeat,1e-12)/means['None'] if valid else None
        status=classify(d-guard,d+guard) if valid else 'RATIO_UNRESOLVED'
        if status=='UNRESOLVED_AT_K16':status='NUMERICALLY_UNRESOLVED'
        paired.append(dict(module=module,rank=64,comparison=comparison,Delta=difference,d=d,
            relative_to_residual_percent=100*difference/means[b] if means[b]>1e-12 else None,
            numerical_guard=guard,status=status,tau=.02,maximum_repeat_difference=maximum_repeat,
            uncertainty='deterministic fixed texts and weights; no label bootstrap CI'))
    return dict(quality=quality,paired=paired)


def test_statistics():
    records=[]
    for c in range(8):
        for k in range(16):
            scores={n:dict(b=(c+1)*(k+1)*(1-.02*i)) for i,n in enumerate(CANDIDATES)}
            records.append(dict(window=c,replicate=k,scores=scores))
    a=summarize('test',records,16,True)
    for r in a['paired']:
        assert abs(r['d']-.04)<1e-12 and abs(r['ci_low']-.04)<1e-12 and r['status']=='IMPROVED'
    old=summarize('test',[r for r in records if r['replicate']<4],4)
    assert all('ci_low' not in r and 'SE' not in r for r in old['paired'])
    assert classify(-.02,.02)=='SMALL_WITHIN_BUDGET' and classify(-.03,.01)=='UNRESOLVED_AT_K16'
    try:summarize('test',records+records[:1],16,True)
    except AssertionError:pass
    else:raise AssertionError('Duplicate accepted')
