"""Exact FP64 empirical Kronecker operations; no residual-specific fitting."""
import math
import time
import sys
import torch
from common import HERE, PLAN, require
sys.path.insert(0,str(HERE.parent/'qer_teacher_kl_exp03'))
import ag_math as parent
sys.path.insert(0,str(HERE))

sym=parent.sym
gauge=parent.gauge
qmetric=parent.qmetric
relative=parent.rel


def scalar_check(a,b,scale,tol=1e-10):
    absolute=64*torch.finfo(torch.float64).eps*abs(scale)
    allowed=tol*max(abs(a),abs(b))+absolute
    require(math.isfinite(a+b+scale) and abs(a-b)<=allowed, f'Scalar mismatch: {a}, {b}, allowed {allowed}')
    return dict(absolute_error=abs(a-b),allowed=allowed,scale=scale)


def token_step(samples,a,g,T):
    # Both contractions use the OLD pair. x,g are per-position full sum-NLL values.
    aa=torch.zeros_like(a); gg=torch.zeros_like(g); count=0
    na=float(a.square().sum()); ng=float(g.square().sum())
    require(na>0 and ng>0,'Zero Token-joint factor')
    for x,z in samples():
        x=x.to(a.device,dtype=torch.float64); z=z.to(a.device,dtype=torch.float64)
        wx=((z@g)*z).sum(-1); wz=((x@a)*x).sum(-1)
        aa.add_(x.T@(x*wx[:,None])); gg.add_(z.T@(z*wz[:,None]));count+=1
    require(count>0,'Empty token stream')
    return sym(aa/(count*T*ng)),sym(gg/(count*T*na))


def sequence_moments(samples,T,device):
    aa=gg=None;count=0
    for s in samples():
        s=s.to(device,dtype=torch.float64)
        if aa is None:
            aa=torch.zeros((s.shape[1],s.shape[1]),dtype=torch.float64,device=device)
            gg=torch.zeros((s.shape[0],s.shape[0]),dtype=torch.float64,device=device)
        aa.add_(s.T@s);gg.add_(s@s.T);count+=1
    require(count>0,'Empty sequence stream')
    return sym(aa/(count*T)),sym(gg/(count*T)),count


def sequence_one_step(moments_a,moments_g):
    # Identity denominators are m and n, not ||newly updated A||^2.
    return moments_a/len(moments_g),moments_g/len(moments_a)


def full_step(samples,a,g,T):
    def clock():
        if a.is_cuda:torch.cuda.synchronize(a.device)
        return time.perf_counter()
    started=clock()
    a,g=gauge(sym(a),sym(g)); old_a,old_g=a,g
    before_g=clock()
    from exact_ops import contraction
    ug,count=contraction(samples,a,'G',T)
    g_seconds=clock()-before_g
    norm_a=a.square().sum();j0=float(norm_a*g.square().sum()-2*(g*ug).sum())
    g=ug/norm_a;jg=float(norm_a*g.square().sum()-2*(g*ug).sum())
    before_a=clock()
    ua,count2=contraction(samples,g,'A',T);require(count==count2,'Unequal ALS streams')
    a_seconds=clock()-before_a
    norm_g=g.square().sum();require(float(norm_g)>0,'Zero ALS block')
    check=float(norm_a*norm_g-2*(a*ua).sum())
    a=ua/norm_g;ja=float(a.square().sum()*norm_g-2*(a*ua).sum())
    scale=max(abs(j0),abs(jg),abs(ja));require(scale>0,'Zero ALS objective')
    scalar_check(jg,check,scale)
    require(jg<=j0+1e-10*scale and ja<=jg+1e-10*scale,'ALS monotonicity failed')
    a,g=gauge(sym(a),sym(g));change,guard=parent.product_change(a,g,old_a,old_g)
    row=dict(J_before=j0,J_after_G=jg,J_after_A=ja,relative_J_improvement=(j0-ja)/scale,
        product_relative_change=change,distance_roundoff=guard,samples=count,
        A_spectrum=parent.spectrum(a),G_spectrum=parent.spectrum(g),synchronous=False,
        J_note='J excludes ||H||^2; a negative J is valid and is not an error norm')
    row.update(contraction_seconds=g_seconds+a_seconds,fixed_seconds=clock()-started-g_seconds-a_seconds)
    return a,g,row


def solve(error,a,g,rank,wq,w0):
    parent.spectrum(a);parent.spectrum(g)
    t,audit=parent.weighted_svd(error,a,g,rank,PLAN['eta_A'],PLAN['eta_G'])
    weight=wq.to(error.device)+t['C64'].float()
    ideal=error-t['C64']; deployed=w0.to(error.device).double()-weight.double()
    q64=qmetric(ideal,t['A_solve'],t['G_solve']);qd=qmetric(deployed,t['A_solve'],t['G_solve'])
    scale=max(abs(q64),abs(qd),PLAN['denominator_floor']);drift=abs(qd-q64)/scale
    require(drift<=PLAN['tolerances']['deployment'],'FP32 deployment proxy drift exceeds 1e-4')
    audit.update(proxy_ideal=q64,proxy_deployed=qd,deployment_relative_drift=drift,
        deployment_absolute_drift=abs(qd-q64),deployment_floor=PLAN['denominator_floor'],
        norm_C=float(t['C64'].norm()),norm_R64=float(ideal.norm()),norm_Rdeploy=float(deployed.norm()),
        residual_relative_drift=relative(deployed,ideal),deployment=PLAN['deployment'])
    return t,dict(P64=t['P64'],Q64=t['Q64'],R64=ideal,R_deploy=deployed,W_deploy=weight),audit


def raw_metric_inner(samples,a,g,T):
    if hasattr(samples,'groups'):
        from exact_ops import metric_inner
        return metric_inner(samples,a,g,T)
    total=0.;count=0
    for s in samples():
        s=s.to(a.device,dtype=torch.float64)
        total+=float(((g@s@a)*s).sum());count+=1
    require(count>0,'Empty metric stream')
    return total/(count*T)


def solve_metric_inner(raw_inner,a,g,moment_a,moment_g,lambda_a,lambda_g):
    # Exact damping expansion saves a second full stream; verified directly in pilot.
    return (raw_inner+lambda_a*float((g*moment_g).sum())+
        lambda_g*float((a*moment_a).sum())+lambda_a*lambda_g*float(moment_a.trace()))


def curvature_quality(hnorm2,inner,a,g):
    knorm2=float(a.square().sum()*g.square().sum())
    require(hnorm2>0 and knorm2>0 and inner>0,'Degenerate empirical H/K')
    c=inner/math.sqrt(hnorm2*knorm2);raw_square=1-c*c
    require(raw_square>=-1e-10,'Curvature cosine exceeds one beyond roundoff')
    bounded_square=max(0.,raw_square);delta2=hnorm2*bounded_square
    return dict(H_norm=math.sqrt(hnorm2),K_norm=math.sqrt(knorm2),inner_H_K=inner,
        cosine=c,s_star=inner/knorm2,relative_error=math.sqrt(bounded_square),delta_F=math.sqrt(delta2),
        relative_error_square_before_guard=raw_square,roundoff_guard_used=raw_square<0,
        scope='exact finite-sample reference, not a population-unbiased curvature estimate')


def excess_bound(quality,rk,r0,a,g,qh_k,qh_0,qh_none):
    proxy=quality['s_star']*(qmetric(rk,a,g)-qmetric(r0,a,g))
    norms=float(rk.square().sum()+r0.square().sum())
    error=.5*quality['delta_F']*norms; upper=proxy+error;actual=qh_k-qh_0
    protection=1e-10*max(abs(upper),abs(actual),abs(proxy),abs(error),PLAN['denominator_floor'])
    require(actual<=upper+protection,'Empirical excess-loss bound violated')
    return dict(empirical_excess=actual,scaled_proxy_improvement=proxy,error_term=error,U=upper,
        U_over_None=upper/qh_none if qh_none>PLAN['denominator_floor'] else None,
        norm_RK=float(rk.norm()),norm_R0=float(r0.norm()),passed=True,absolute_protection=protection,
        residual='ideal FP64 rank-constrained R64 only')


def gram_blocked(paths,read_s,T,device,block=8,boundary=lambda:None):
    """Bounded device blocks; result is N by N, NEVER (mn) by (mn)."""
    from exact_ops import gram_features
    return gram_features(paths,read_s,T,device,boundary=boundary)


def gram_original(paths,read_s,T,device,block=8,boundary=lambda:None):
    n=len(paths);result=torch.empty((n,n),dtype=torch.float64)
    for i in range(0,n,block):
        boundary();x=torch.stack([read_s(p).reshape(-1) for p in paths[i:i+block]]).to(device)
        for j in range(0,i+1,block):
            boundary();y=x if i==j else torch.stack([read_s(p).reshape(-1) for p in paths[j:j+block]]).to(device)
            term=(x@y.T).cpu();result[i:i+len(x),j:j+len(y)]=term
            if i!=j:result[j:j+len(y),i:i+len(x)]=term.T
            del y,term
        del x
    require(bool(torch.isfinite(result).all()),'Nonfinite sample Gram')
    return result,float(result.square().sum()/(n*n*T*T))
