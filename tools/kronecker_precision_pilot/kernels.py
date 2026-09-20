"""FP32 GEMMs / FP64 sample accumulation, with an identical FP64 comparator."""
import math
import torch
from bootstrap import PLAN,bridge
require=bridge.require


def sym(a):
    require(a.dtype==torch.float64 and bool(torch.isfinite(a).all()),'Invalid FP64 accumulator')
    return (a+a.T)*.5


class Ops:
    def __init__(self,dtype,device):
        require(dtype in (torch.float32,torch.float64),'Precision must be explicit')
        self.dtype=dtype;self.device=device
    def cast(self,x):return x.to(self.device,dtype=self.dtype)
    def mm(self,a,b):return self.cast(a)@self.cast(b)
    def marginal(self,samples,L,T):
        aa=gg=None;n=0
        for x,z in samples():
            x=self.cast(x);z=self.cast(z);at=(x.T@x).double();gt=(z.T@z).double()
            aa=at if aa is None else aa+at;gg=gt if gg is None else gg+gt;n+=1
        require(n>0,'No samples');return sym(aa/(n*L)),sym(gg/(n*T))
    def contraction(self,samples,factor,side,T):
        require(side in ('A','G'),'Invalid contraction side');total=None;n=0;factor=self.cast(factor)
        for x,z in samples():
            x=self.cast(x);z=self.cast(z)
            term=z.T@((x@factor)@x.T)@z if side=='G' else x.T@((z@factor)@z.T)@x
            term=term.double();total=term if total is None else total+term;n+=1
        require(n>0,'No samples');return sym(total/(n*T))
    def token(self,samples,a,g,T):
        aa=torch.zeros_like(a);gg=torch.zeros_like(g);n=0
        af=self.cast(a);gf=self.cast(g)
        for x,z in samples():
            x=self.cast(x);z=self.cast(z)
            # Dot products / reductions stay FP64; the large GEMMs are the
            # only precision change. Per-sample GEMM errors are not erased by
            # accumulating their results in FP64.
            wx=((z@gf).double()*z.double()).sum(-1)
            wz=((x@af).double()*x.double()).sum(-1)
            aa.add_((x.T@(x*wx.to(self.dtype)[:,None])).double())
            gg.add_((z.T@(z*wz.to(self.dtype)[:,None])).double());n+=1
        require(n>0,'No samples')
        return sym(aa/(n*T*g.square().sum())),sym(gg/(n*T*a.square().sum()))
    def moments(self,samples,T):
        aa=gg=None;n=0
        for x,z in samples():
            x=self.cast(x);z=self.cast(z)
            at=(x.T@(z@z.T)@x).double();gt=(z.T@(x@x.T)@z).double()
            aa=at if aa is None else aa+at;gg=gt if gg is None else gg+gt;n+=1
        require(n>0,'No samples');return sym(aa/(n*T)),sym(gg/(n*T))
    def full(self,samples,a,g,T):
        a,g=bridge.sm.gauge(sym(a),sym(g));old_a,old_g=a,g
        ug=self.contraction(samples,a,'G',T);na=a.square().sum()
        j0=float(na*g.square().sum()-2*(g*ug).sum());g=ug/na
        jg=float(na*g.square().sum()-2*(g*ug).sum())
        ua=self.contraction(samples,g,'A',T);ng=g.square().sum()
        require(float(ng)>0,'Zero ALS factor')
        cross=float(na*ng-2*(a*ua).sum());a=ua/ng
        ja=float(a.square().sum()*ng-2*(a*ua).sum());scale=max(abs(j0),abs(jg),abs(ja))
        require(scale>0 and math.isfinite(scale),'Invalid ALS objective')
        tol=1e-10 if self.dtype==torch.float64 else PLAN['tolerances']['round_objective_relative_fp32']
        a,g=bridge.sm.gauge(sym(a),sym(g))
        change,_=bridge.sm.parent.product_change(a,g,old_a,old_g)
        row=dict(J_before=j0,J_after_G=jg,J_after_A=ja,cross_relative=abs(jg-cross)/scale,
            relative_J_improvement=(j0-ja)/scale,product_relative_change=change,
            numeric_round_passed=abs(jg-cross)<=tol*scale and jg<=j0+tol*scale and ja<=jg+tol*scale,
            objective_tolerance=tol)
        # Pilot keeps failed checks visible and does not silently promote them.
        return a,g,row


def factor_comparison(a,g,refa,refg):
    a,g=bridge.sm.gauge(a,g);refa,refg=bridge.sm.gauge(refa,refg)
    da=a-refa;dg=g-refg
    square=float(da.square().sum()*refg.square().sum()+a.square().sum()*dg.square().sum()+2*(da*a).sum()*(refg*dg).sum())
    require(square>=-1e-12*float(refa.square().sum()*refg.square().sum()),'Kronecker distance cancellation')
    relative=math.sqrt(max(square,0)/float(refa.square().sum()*refg.square().sum()))
    row=dict(A_relative=bridge.sm.relative(a,refa),G_relative=bridge.sm.relative(g,refg),product_relative=relative)
    row['passed']=max(row['A_relative'],row['G_relative'])<=PLAN['tolerances']['factor_relative'] and relative<=PLAN['tolerances']['product_relative']
    return row


def damp(a,mixed):
    a=sym(a);v,u=torch.linalg.eigh(a);lo,hi=float(v[0]),float(v[-1]);lam=PLAN['eta']*float(a.trace())/len(a)
    tol=PLAN['tolerances']['raw_negative_relative_fp32'] if mixed else 1e-10
    require(hi>0 and lam>0 and lo>=-tol*hi,'Raw PSD numerical tolerance failed')
    require(not mixed or lo>=-PLAN['tolerances']['negative_over_damping']*lam,'Negative raw mode is material relative to damping')
    shifted=v+lam;condition=float(shifted[-1]/shifted[0]) if float(shifted[0])>0 else math.inf
    require(condition<=PLAN['tolerances']['condition'],'Damped condition failed')
    solve=a+lam*torch.eye(len(a),dtype=a.dtype,device=a.device);root=(u*shifted.sqrt()[None,:])@u.T
    error=bridge.sm.relative(root@root,solve);require(error<=PLAN['tolerances']['root'],'FP64 root reconstruction failed')
    return solve,root,dict(raw_min=lo,raw_max=hi,negative_relative=max(0.,-lo/hi),lambda_=lam,condition=condition,root_relative=error,clipping=False)


def solve(error,a,g,w0,wq,rank,mixed):
    # Only the raw-factor PSD allowance is precision-specific and preregistered.
    # Damping, dense gesvd, solve/tail checks and deployment stay FP64/unchanged.
    a,g=bridge.sm.gauge(a,g);aa,ar,am=damp(a,mixed);gg,gr,gm=damp(g,mixed)
    b=gr@error@ar;kw={'driver':'gesvd'} if b.is_cuda else {}
    u,s,vh=torch.linalg.svd(b,full_matrices=False,**kw)
    p=torch.linalg.solve(gr,u[:,:rank]*s[:rank].sqrt());q=torch.linalg.solve(ar,(s[:rank].sqrt()[:,None]*vh[:rank]).T).T
    c=p@q;tail=float(s[rank:].square().sum()/2);q64=bridge.sm.qmetric(error-c,aa,gg)
    transformed=bridge.sm.relative(gr@c@ar,(u[:,:rank]*s[:rank])@vh[:rank])
    tail_error=abs(q64-tail)/tail if tail else abs(q64-tail)
    require(max(transformed,tail_error)<=PLAN['tolerances']['svd'],'FP64 weighted SVD acceptance failed')
    weight=wq.to(error.device)+c.float();residual=w0.to(error.device).double()-weight.double()
    qd=bridge.sm.qmetric(residual,aa,gg);drift=abs(qd-q64)/max(abs(qd),abs(q64),1e-12)
    require(drift<=PLAN['tolerances']['deployment'],'FP32 deployment drift failed')
    audit=dict(A=am,G=gm,tail_relative=tail_error,transform_relative=transformed,deployment_relative=drift,
        singular_at_rank=float(s[rank-1]),singular_after_rank=float(s[rank]),mixed_raw_factors=mixed,solver_dtype='float64')
    return dict(P64=p,Q64=q,W_deploy=weight),audit
