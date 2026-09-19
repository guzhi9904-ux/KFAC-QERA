"""Canonical full-curvature contractions and exact dense weighted SVD.

No absolute epsilon, eigenvalue floor, random trace or low-rank statistics.
"""
import math
import time
import torch


def rel(a, b):
    n=float(torch.linalg.vector_norm(b))
    e=float(torch.linalg.vector_norm(a-b))
    return e/n if n else (0. if e==0 else math.inf)


def sym(a, tolerance=1e-10):
    assert a.dtype==torch.float64 and torch.isfinite(a).all()
    e=rel(a,a.T)
    if e>tolerance:raise RuntimeError(f'Symmetry failure: {e}')
    return (a+a.T)/2


def gauge(a,g):
    n=float(a.norm())
    if not n>0 or not math.isfinite(n):raise RuntimeError('Zero/nonfinite factor')
    return a/n,g*n


def qmetric(r,a,g):
    return float(((g@r@a)*r).sum()/2)


def product_change(a,g,b,h):
    n=float(a.square().sum()*g.square().sum())
    m=float(b.square().sum()*h.square().sum())
    cross=float((a*b).sum()*(g*h).sum())
    z=n+m-2*cross
    # Cancellation in a squared distance; not a PSD factor/score repair.
    if z < -1e-12*max(n,m):raise RuntimeError('Product-distance cancellation invalid')
    return math.sqrt(max(z,0)/m),{'distance_square_before_roundoff_guard':z,'scale':m}


def contraction(samples,factor,side,T):
    result=None;count=0
    for s in samples():
        s=s.to(factor.device,dtype=torch.float64)
        term=s@factor@s.T if side=='G' else s.T@factor@s
        if result is None:result=torch.zeros_like(term)
        result.add_(term);count+=1
    if not count:raise RuntimeError('Empty empirical curvature')
    return sym(result/(count*T)),count


def spectrum(a,negative_tolerance=1e-10):
    v=torch.linalg.eigvalsh(sym(a));lo,hi=float(v[0]),float(v[-1])
    if hi<=0 or lo < -negative_tolerance*hi:raise RuntimeError(f'Invalid PSD spectrum: {lo}, {hi}')
    return {'min':lo,'max':hi,'trace':float(v.sum()),'frobenius':float(a.norm()),
            'negative_eigenvalues':int((v<0).sum()),'symmetry_error':rel(a,a.T)}


def als(samples,a,g,T,max_iterations,objective_tolerance,product_tolerance,callback=None):
    a,g=gauge(sym(a),sym(g));history=[];stable=0
    stop='MAX_ITERATIONS_REACHED'
    for iteration in range(1,max_iterations+1):
        tick=time.perf_counter();old_a,old_g=a,g
        ug,count=contraction(samples,a,'G',T)
        aa=a.square().sum();j0=float(aa*g.square().sum()-2*(g*ug).sum())
        g=ug/aa
        jg=float(aa*g.square().sum()-2*(g*ug).sum())
        ua,count2=contraction(samples,g,'A',T);assert count==count2
        gg=g.square().sum()
        if not float(gg)>0:raise RuntimeError('Zero ALS block')
        jg_check=float(aa*gg-2*(a*ua).sum())
        a=ua/gg
        ja=float(a.square().sum()*gg-2*(a*ua).sum())
        scale=max(abs(j0),abs(jg),abs(ja))
        if scale==0:raise RuntimeError('Zero ALS objective scale')
        if abs(jg-jg_check)>1e-10*scale or jg>j0+1e-10*scale or ja>jg+1e-10*scale:
            raise RuntimeError('Exact-block objective monotonicity failed')
        a,g=gauge(a,g)
        change,roundoff=product_change(a,g,old_a,old_g)
        improvement=(j0-ja)/scale
        row={'iteration':iteration,'J_before':j0,'J_after_G':jg,'J_after_A':ja,
             'G_block_crosscheck':jg_check,'relative_J_improvement':improvement,
             'product_relative_change':change,'product_distance_audit':roundoff,
             'sample_count':count,'A_spectrum':spectrum(a),'G_spectrum':spectrum(g),
             'seconds':time.perf_counter()-tick}
        history.append(row)
        stable=stable+1 if improvement<=objective_tolerance and change<=product_tolerance else 0
        if callback:callback(row,a,g)
        if stable>=2:stop='TWO_CONSECUTIVE_CONVERGED_CYCLES';break
    return a,g,history,stop


def damp(a,eta,max_condition=1e8):
    a=sym(a);v,u=torch.linalg.eigh(a)
    lo,hi=float(v[0]),float(v[-1])
    if hi<=0 or lo < -1e-10*hi:raise RuntimeError('Raw factor not PSD')
    scale=float(a.trace())/len(a)
    if scale<=0:raise RuntimeError('Nonpositive trace')
    lam=eta*scale;vv=v+lam
    condition=float(vv[-1]/vv[0]) if float(vv[0])>0 else math.inf
    if not condition<=max_condition:raise RuntimeError('Solve condition limit failed')
    solve=a+lam*torch.eye(len(a),dtype=a.dtype,device=a.device)
    root=(u*vv.sqrt().unsqueeze(0))@u.T
    error=rel(root@root,solve)
    if error>1e-10:raise RuntimeError('Square-root reconstruction failed')
    return solve,root,{'eta':eta,'lambda':lam,'trace_scale':scale,'condition':condition,
                       'raw_min_eigenvalue':lo,'raw_max_eigenvalue':hi,'root_error':error,
                       'eigenvalue_clipping':'none','floor':'none'}


def weighted_svd(error,a,g,rank,eta_A,eta_G):
    a,g=gauge(a,g)
    aa,ar,am=damp(a,eta_A);gg,gr,gm=damp(g,eta_G)
    b=gr@error@ar
    args={'driver':'gesvd'} if b.is_cuda else {}
    tick=time.perf_counter();u,s,vh=torch.linalg.svd(b,full_matrices=False,**args)
    p=torch.linalg.solve(gr,u[:,:rank]*s[:rank].sqrt())
    q=torch.linalg.solve(ar,(s[:rank].sqrt()[:,None]*vh[:rank]).T).T
    c=p@q;br=(u[:,:rank]*s[:rank])@vh[:rank]
    tail=float(s[rank:].square().sum()/2)
    objective=qmetric(error-c,aa,gg)
    tail_error=abs(objective-tail)/tail if tail else abs(objective-tail)
    transform_error=rel(gr@c@ar,br)
    if tail_error>1e-8 or transform_error>1e-8:raise RuntimeError('Weighted-SVD acceptance failed')
    audit={'rank':rank,'algorithm':'dense torch.linalg.svd gesvd on CUDA; no approximation',
           'SVD_and_recovery_seconds':time.perf_counter()-tick,'tail_energy_half':tail,
           'solve_objective':objective,'tail_relative_error':tail_error,'back_transform_error':transform_error,
           'singular_at_rank':float(s[rank-1]),'singular_after_rank':float(s[rank]),
           'relative_rank_gap':float((s[rank-1]-s[rank])/s[rank-1]),'A_damping':am,'G_damping':gm}
    return {'A_raw':a,'G_raw':g,'A_solve':aa,'G_solve':gg,'P64':p,'Q64':q,'C64':c,'singular_values':s},audit
