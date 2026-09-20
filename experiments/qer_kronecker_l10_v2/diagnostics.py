"""Frozen-candidate fit/eval curvature and ideal rank-constrained bound analysis."""
import math
import torch
from common import PLAN,save_csv,save_json,require
from records import commit,load_record
import sketch_math as sm


def run(e):
    e.offline();keys=e.check_candidates();root=e.root
    collections={'fit':[r['id'] for r in e.index['fit']],'eval':[r['id'] for r in e.index['eval']]}
    grams={}
    for role,ids in collections.items():
        path=root/'diagnostics/grams'/(role+'.safetensors')
        if path.exists():grams[role]=e.store.get(path)[0]['B'];continue
        with e.resources.timed('reference_Gram',role=role,samples=len(ids)):
            b,hnorm2=sm.gram_blocked(ids,lambda key:e.s(key,role),PLAN['T'],e.device,
                block=e.config['gram_block'],boundary=e.boundary)
            e.store.put(path,dict(B=b),samples=ids,H_norm2=hnorm2,N=len(ids),T=PLAN['T'])
            grams[role]=b
    fit_locations={key:i for i,key in enumerate(collections['fit'])};hnorms={}
    for budget,ids in e.index['budgets'].items():
        ii=torch.tensor([fit_locations[key] for key in ids]);block=grams['fit'][ii[:,None],ii[None,:]]
        hnorms[budget]=float(block.square().sum()/(len(ids)**2*PLAN['T']**2))
    hnorms['eval']=float(grams['eval'].square().sum()/(len(collections['eval'])**2*PLAN['T']**2))
    for row in e.index['fit']:
        p=root/'diagnostics/fit_q'/(row['id']+'.json')
        if p.exists():load_record(p,e.identity);continue
        applicable=[budget for budget,ids in e.index['budgets'].items() if row['id'] in ids]
        candidates=[k for k in keys if k=='None' or any(k.startswith(b+'__') for b in applicable)
            or (k=='A8' and any(b in applicable for b in ('S0','S1'))) or (k=='A32' and 'S2' in applicable)]
        with e.resources.timed('fit_diagnostic_projections',sample=row['id']):
            s=e.s(row['id']).to(e.device);scores={k:e.project(s,k) for k in candidates}
            commit(p,e.identity,sample=row,scores=scores);del s
    q_means={}
    references=list(PLAN['budgets'])+['eval']
    for reference in references:
        ids=e.index['budgets'][reference] if reference!='eval' else collections['eval']
        records=[load_record(root/('diagnostics/fit_q' if reference!='eval' else 'scores/atomic_q')/(key+'.json'),e.identity) for key in ids]
        candidates=[k for k in keys if reference=='eval' or k=='None' or k.startswith(reference+'__') or k==('A32' if reference=='S2' else 'A8')]
        q_means[reference]={key:{field:math.fsum(r['scores'][key][field] for r in records)/len(records) for field in ('q','ideal_q')} for key in candidates}
    qualities=[];bounds=[]
    for reference in references:
        ids=e.index['budgets'][reference] if reference!='eval' else collections['eval'];role='fit' if reference!='eval' else 'eval'
        moment=e.moments(ids,reference,role);ma=moment['StS'].to(e.device);mg=moment['SSt'].to(e.device)
        candidates=[k for k in q_means[reference] if k!='None']
        def quality_task(worker,key):
            e=worker
            ma=moment['StS'].to(e.device);mg=moment['SSt'].to(e.device)
            p=root/'diagnostics/quality'/reference/(key+'.json')
            if p.exists():record=load_record(p,e.identity)
            else:
                with e.resources.timed('metric_curvature',reference=reference,candidate=key,samples=len(ids)):
                    ft,fm=e.store.get(root/'factors'/key/'solve.safetensors')
                    a,g=ft['A_raw'].to(e.device),ft['G_raw'].to(e.device)
                    if fm['method']=='A-only':
                        # G is a scalar identity after gauge; this is an exact contraction.
                        scalar=float(g.diag().mean());sm.scalar_check(float((g-scalar*torch.eye(len(g),device=g.device,dtype=g.dtype)).norm()),0.,float(g.norm()))
                        inner=scalar*float((a*ma).sum())
                    else:inner=sm.raw_metric_inner(e.stream(ids,role),a,g,PLAN['T'])
                    audit=fm['audit'];inner_solve=sm.solve_metric_inner(inner,a,g,ma,mg,audit['A_damping']['lambda'],audit['G_damping']['lambda'])
                    t,_=e.store.get(root/'corrections'/(key+'.safetensors'));rr=t['R64'].to(e.device);rd=t['R_deploy'].to(e.device)
                    rows=[]
                    for representation,hk in [('raw',inner),('solve',inner_solve)]:
                        aa=ft['A_'+representation].to(e.device);gg=ft['G_'+representation].to(e.device)
                        quality=sm.curvature_quality(hnorms[reference],hk,aa,gg)
                        qe=sm.qmetric(e.error.to(e.device),aa,gg);qr=sm.qmetric(rr,aa,gg);qd=sm.qmetric(rd,aa,gg)
                        rows.append(dict(reference=reference,candidate=key,budget=fm['budget'],method=fm['method'],representation=representation,
                            qK_E=qe,qK_R64=qr,qK_Rdeploy=qd,proxy_recovery=1-qd/qe if qe>0 else None,
                            norm_C=audit['norm_C'],condition_A=audit['A_damping']['condition'],condition_G=audit['G_damping']['condition'],
                            spectral_tail_half=audit['tail_energy_half'],**quality))
                    record=commit(p,e.identity,rows=rows)
                    del ft,a,g,aa,gg,t,rr,rd
            return record['rows']
        from parallel import offline_map
        for rows in offline_map(e,candidates,quality_task):qualities.extend(rows)
        del ma,mg
        for budget in PLAN['budgets']:
            if reference!='eval' and reference!=budget:continue
            for method in PLAN['methods']:
                key=budget+'__'+method
                quality=next(r for r in qualities if r['reference']==reference and r['candidate']==key and r['representation']=='solve')
                ft,_=e.store.get(root/'factors'/key/'solve.safetensors');a=ft['A_solve'].to(e.device);g=ft['G_solve'].to(e.device)
                rk=e.store.get(root/'corrections'/(key+'.safetensors'))[0]['R64'].to(e.device)
                for baseline in [budget+'__Marginal','A32' if budget=='S2' else 'A8']:
                    r0=e.store.get(root/'corrections'/(baseline+'.safetensors'))[0]['R64'].to(e.device)
                    bound=sm.excess_bound(quality,rk,r0,a,g,q_means[reference][key]['ideal_q'],q_means[reference][baseline]['ideal_q'],q_means[reference]['None']['ideal_q'])
                    require(bound['scaled_proxy_improvement']<=1e-8*max(abs(quality['s_star']*quality['qK_E']),PLAN['denominator_floor']),'SVD candidate is not proxy-optimal against feasible reference')
                    bounds.append(dict(reference=reference,budget=budget,method=method,candidate=key,baseline=baseline,**bound))
                del ft,a,g,rk,r0
    save_csv(root/'summary/curvature_quality.csv',qualities);save_csv(root/'summary/bound_diagnostics.csv',bounds)
    save_json(root/'summary/diagnostics.json',dict(identity=e.identity,H_norms_squared=hnorms,q_means=q_means,quality=qualities,bounds=bounds))
    return qualities,bounds
