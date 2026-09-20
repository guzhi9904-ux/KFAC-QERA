"""All preregistered paired contrasts, common bootstrap draws and 2pp decision rule."""
import numpy as np
from common import PLAN,seed,require


def state(lo,hi,guard=0.):
    t=PLAN['practical_threshold']
    if lo>t+guard:return 'IMPROVED_OVER_2PP'
    if hi<-t-guard:return 'DEGRADED_OVER_2PP'
    if lo>=-t+guard and hi<=t-guard:return 'DIFFERENCE_WITHIN_2PP'
    return 'UNRESOLVED'


def contrasts():
    rows=[]
    for budget in PLAN['budgets']:
        for method in PLAN['methods'][1:]:
            rows.append(dict(kind='primary',label=budget+': '+method+' vs Marginal',baseline=budget+'__Marginal',candidate=budget+'__'+method))
        for method in PLAN['methods']:
            rows.append(dict(kind='A_only',label=budget+': '+method+' vs A-only',baseline='A32' if budget=='S2' else 'A8',candidate=budget+'__'+method))
    for method in PLAN['methods']:
        for old,new in [('S0','S1'),('S0','S2'),('S1','S2')]:
            rows.append(dict(kind='sample_budget',label=method+': '+new+' minus '+old,
                baseline=old+'__'+method,candidate=new+'__'+method,interpretation='positive = reference damage minus new damage'))
    rows.append(dict(kind='sample_budget',label='A32 vs A8',baseline='A8',candidate='A32',interpretation='input-statistics-only text coverage control'))
    return rows


def summarize(q,kl,keys,repeat_guard=0.):
    q=np.asarray(q,dtype=np.float64);kl=np.asarray(kl,dtype=np.float64)
    require(q.ndim==3 and kl.shape==(q.shape[0],q.shape[2]),'Evaluation array dimensions differ')
    require(np.isfinite(q).all() and np.isfinite(kl).all() and (q>=0).all() and (kl>=0).all(),'Nonfinite or negative loss')
    n,m,_=q.shape;B=PLAN['bootstrap_count'];none=keys.index('None')
    rng=np.random.Generator(np.random.PCG64(seed('bootstrap','',0)))
    article_indices=rng.integers(n,size=(B,n))
    labels=np.random.Generator(np.random.PCG64(seed('bootstrap','',1))).integers(m,size=(B,n,m))
    qwindow=q.mean(axis=1)
    samples={'q_article':qwindow[article_indices].mean(axis=1),'KL_article':kl[article_indices].mean(axis=1),
        'q_conditional_labels':q[np.arange(n)[None,:,None],labels].mean(axis=(1,2))}
    means={'q':qwindow.mean(axis=0),'KL':kl.mean(axis=0)};candidate_rows=[];comparisons=[]
    for metric in ('q','KL'):
        mu=means[metric];sample=samples[metric+'_article'];valid=bool(mu[none]>PLAN['denominator_floor'] and np.all(sample[:,none]>PLAN['denominator_floor']))
        for i,key in enumerate(keys):
            ci=np.quantile(1-sample[:,i]/sample[:,none],[.025,.975]) if valid else (None,None)
            candidate_rows.append(dict(metric=metric,candidate=key,damage=float(mu[i]),
                recovery=float(1-mu[i]/mu[none]) if mu[none]>PLAN['denominator_floor'] else None,
                ci_low=float(ci[0]) if valid else None,ci_high=float(ci[1]) if valid else None))
        for pair in contrasts():
            old=keys.index(pair['baseline']);new=keys.index(pair['candidate']);absolute=float(mu[old]-mu[new])
            guard=max(2*repeat_guard if metric=='KL' else 0.,1e-10*max(float(mu[old]),float(mu[new])))
            row=dict(pair,metric=metric,Delta=absolute,d=absolute/float(mu[none]) if mu[none]>PLAN['denominator_floor'] else None,absolute_numerical_guard=guard)
            for scope in (['article','conditional_labels'] if metric=='q' else ['article']):
                sample=samples[metric+'_'+scope];delta=sample[:,old]-sample[:,new]
                valid=bool(mu[none]>PLAN['denominator_floor'] and np.all(sample[:,none]>PLAN['denominator_floor']))
                ci=np.quantile(delta/sample[:,none],[.025,.975]) if valid else (None,None)
                absolute_ci=np.quantile(delta,[.025,.975])
                row[scope]=dict(ci_low=float(ci[0]) if valid else None,ci_high=float(ci[1]) if valid else None,
                    absolute_ci=[float(v) for v in absolute_ci],status=state(*ci,guard/float(mu[none])) if valid else 'DENOMINATOR_UNRESOLVED')
            comparisons.append(row)
    return dict(candidates=candidate_rows,comparisons=comparisons,bootstrap_count=B,
        bootstrap_seed=seed('bootstrap','',0),label_bootstrap_seed=seed('bootstrap','',1),
        same_article_draws_for_all_candidates_contrasts_and_metrics=True,
        caveats='Exploratory pointwise intervals, no family-wise control; no refitting uncertainty. Article and conditional-label intervals are separate, not nested or added.')
