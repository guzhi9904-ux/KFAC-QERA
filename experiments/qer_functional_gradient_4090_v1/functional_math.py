"""FP64 fixed-geometry functional-gradient construction; never tunes factors."""
import math
import torch


def relative(a,b):
    n=float(b.norm());e=float((a-b).norm())
    return e/n if n else (0. if e==0 else math.inf)


def check_scalar(a,b,tolerance=1e-8,scale=0.):
    # A roundoff-sized absolute allowance protects cancellation, not scientific effect sizes.
    denominator=max(abs(a),abs(b),abs(scale)*32*torch.finfo(torch.float64).eps/tolerance)
    error=abs(a-b)/denominator if denominator else (0. if a==b else math.inf)
    if not math.isfinite(error) or error>tolerance:raise ArithmeticError((a,b,error,tolerance))
    return error


def roots(solve,max_condition=1e8,tolerance=1e-8):
    assert solve.dtype==torch.float64 and bool(torch.isfinite(solve).all())
    if relative(solve,solve.T)>1e-10:raise ArithmeticError('Factor symmetry')
    eigen,u=torch.linalg.eigh(solve)
    lo,hi=float(eigen[0]),float(eigen[-1])
    if lo<=0 or hi/lo>max_condition:raise ArithmeticError(('Solve factor condition',lo,hi))
    root=(u*eigen.sqrt())@u.T
    inverse=(u*eigen.rsqrt())@u.T
    reconstruction=relative(root@root,solve)
    inverse_error=relative(inverse@root,torch.eye(len(solve),dtype=solve.dtype,device=solve.device))
    if max(reconstruction,inverse_error)>tolerance:raise ArithmeticError('Square-root identity')
    return root,inverse,dict(min_eigenvalue=lo,max_eigenvalue=hi,condition=hi/lo,
                             reconstruction_relative_error=reconstruction,inverse_relative_error=inverse_error)


def metric(r,a,g):
    return float((r*(g@r@a)).sum()/2)


def inner_factors(matrix,p,q):
    return float(((p.T@matrix)*q).sum())


def metric_factors(p,q,a,g):
    return float(((p.T@g@p)*(q@a@q.T).T).sum())


def decompose(error,m,a,g,max_rank=128):
    ar,ai,ac=roots(a);gr,gi,gc=roots(g)
    results={};audits={'A':ac,'G':gc}
    for method,target in [('Residual',gr@error@ar),('Gradient',gi@m@ai)]:
        args={'driver':'gesvd'} if target.is_cuda else {}
        u,s,vh=torch.linalg.svd(target,full_matrices=False,**args)
        reconstruction=relative((u*s)@vh,target)
        if reconstruction>1e-8:raise ArithmeticError('SVD reconstruction')
        r=min(max_rank,len(s))
        p=gi@(u[:,:r]*s[:r].sqrt());q=(s[:r].sqrt()[:,None]*vh[:r])@ai
        results[method]=dict(P=p,Q=q,singular_values=s)
        audits[method]=dict(SVD_relative_error=reconstruction,input_norm_squared=float(target.square().sum()),
                            algorithm='torch.linalg.svd full_matrices=False; CUDA gesvd; no approximation')
        for rank in [x for x in (16,32,64,128) if x<=r]:
            pp,qq=p[:,:rank],q[:rank]
            transformed=(gr@pp)@(qq@ar)
            prefix=(u[:,:rank]*s[:rank])@vh[:rank]
            transform_error=relative(transformed,prefix)
            tail=float(s[rank:].square().sum())
            tail_actual=float((target-transformed).square().sum())
            tail_error=check_scalar(tail_actual,tail,scale=audits[method]['input_norm_squared'])
            curvature=metric_factors(pp,qq,a,g);sigma2=float(s[:rank].square().sum())
            curvature_error=check_scalar(curvature,sigma2,scale=audits[method]['input_norm_squared'])
            if transform_error>1e-8:raise ArithmeticError('Inverse transform')
            linear=inner_factors(m,pp,qq)
            audit=dict(back_transform_relative_error=transform_error,tail_squared=tail,tail_check_error=tail_error,
                proxy_curvature=curvature,singular_prefix_squared=sigma2,proxy_curvature_identity_error=curvature_error,
                a_from_M=linear)
            if method=='Gradient':audit['gradient_linear_identity_error']=check_scalar(linear,sigma2,scale=float(m.norm())*float((pp@qq).norm()))
            else:
                q_residual=metric(error-pp@qq,a,g)
                audit['residual_metric_tail_error']=check_scalar(q_residual,tail/2,scale=float(target.square().sum()))
            audits[method][str(rank)]=audit
        del u,s,vh,target
    return results,audits


def calibrated_ray(d_error,d_direction,norm,T):
    """Normalize the ray before fitting its signed amplitude; no epsilon clipping."""
    assert d_error.dtype==d_direction.dtype==torch.float64 and norm>=0
    if norm==0:
        if bool(torch.any(d_direction!=0)):raise ArithmeticError('Zero correction has nonzero projection')
        return dict(a=0.,b=0.,beta=0.,unit_amplitude=0.,ideal_gain=0.)
    unit=d_direction/norm
    au=float((d_error*unit).mean()/T);bu=float(unit.square().mean()/T)
    if bu==0:
        if au!=0:raise ArithmeticError('PSD null direction has nonzero linear term')
        amplitude=0.
    else:amplitude=au/bu
    beta=amplitude/norm
    values=dict(a=au*norm,b=bu*norm*norm,beta=beta,unit_amplitude=amplitude,
                ideal_gain=au*au/(2*bu) if bu else 0.)
    if not all(math.isfinite(v) for v in values.values()):raise ArithmeticError('Nonfinite amplitude')
    return values


def small_checks():
    rng=torch.Generator().manual_seed(2026091901)
    rand=lambda *shape:torch.randn(*shape,dtype=torch.float64,generator=rng)
    s=rand(9,5,4);e=rand(5,4);a0=rand(4,4);g0=rand(5,5)
    a=a0@a0.T+torch.eye(4);g=g0@g0.T+torch.eye(5);T=7
    vec=lambda x:x.T.contiguous().flatten()
    h=sum(torch.outer(vec(x),vec(x)) for x in s)/(len(s)*T)
    d=(s*e).sum((1,2));m=(s*d[:,None,None]).sum(0)/(len(s)*T)
    assert relative(vec(m),h@vec(e))<1e-12
    c=rand(5,4)
    gain=float((vec(e)@h@vec(e)-vec(e-c)@h@vec(e-c))/2)
    assert check_scalar(gain,float((m*c).sum()-vec(c)@h@vec(c)/2))<1e-12
    assert relative(torch.kron(a,g)@vec(c),vec(g@c@a))<1e-12
    def construct(aa,gg,mm,kind):
        ar,ai,_=roots(aa);gr,gi,_=roots(gg)
        target=gr@e@ar if kind=='B' else gi@mm@ai
        u,sv,vh=torch.linalg.svd(target,full_matrices=False)
        correction=gi@((u[:,:2]*sv[:2])@vh[:2])@ai
        return correction,target,sv[:2]
    cb,b,sv=construct(a,g,g@e@a,'B');cf,f,sf=construct(a,g,g@e@a,'F')
    assert relative(b,f)<1e-12 and relative(cb,cf)<1e-12
    assert check_scalar(float((cb*(g@e@a)).sum()),float((cb*(g@cb@a)).sum()))<1e-12
    checks={}
    for method in ('B','F'):
        c,target,sv=construct(a,g,m,method)
        proj=(s*c).sum((1,2));cal=calibrated_ray(d,proj,float(c.norm()),T)
        q0=float(d.square().mean()/(2*T));q1=float((d-proj).square().mean()/(2*T));qc=float((d-cal['beta']*proj).square().mean()/(2*T))
        check_scalar(q0-qc,cal['ideal_gain'],scale=q0)
        assert qc<=q0+1e-12 and qc<=q1+1e-12
        gauge,_,_=construct(a*3,g/3,m,method);assert relative(c,gauge)<1e-11
        scaled,_,_=construct(a,g*7,m,method)
        expected=c if method=='B' else c/7
        assert relative(expected,scaled)<1e-11
        newcal=calibrated_ray(d,(s*scaled).sum((1,2)),float(scaled.norm()),T)
        assert relative(c*cal['beta'],scaled*newcal['beta'])<1e-10
        if method=='F':
            linear=float((m*c).sum());proxy=float((c*(g@c@a)).sum());sigma2=float(sv.square().sum())
            check_scalar(linear,sigma2);check_scalar(proxy,sigma2)
            check_scalar(newcal['beta'],7*cal['beta'])
            target_tail=float((target-roots(g)[0]@c@roots(a)[0]).square().sum())
            check_scalar(q0-linear+proxy/2,q0-float(target.square().sum())/2+target_tail/2,scale=q0+sigma2)
        checks[method]=dict(beta=cal['beta'],q0=q0,q1=q1,q_cal=qc,gauge_and_scale_passed=True)
    negative=calibrated_ray(d,-d,float(e.norm()),T);assert negative['beta']<0
    zero=calibrated_ray(d,torch.zeros_like(d),0,T);assert zero['beta']==0
    return dict(passed=True,explicit_H_and_column_vec=True,HK_equal_inputs_and_beta1=True,
                surrogate_identity=True,gauge_and_scaling=True,signed_and_zero_rays=True,checks=checks)
