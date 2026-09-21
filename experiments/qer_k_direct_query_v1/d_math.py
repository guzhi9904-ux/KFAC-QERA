import math
import torch
from d_common import require
from k_math import compare,transpose_rotate,softmax_row_audit,centered_formula_audit


def contract_grams(ds,qs,x,mapping,hkv):
    """Exact FP64 contraction; raw G sum includes1/d and every query row."""
    x=x.double();d=qs.shape[-1]
    b=torch.zeros(x.shape[0],x.shape[0],dtype=torch.float64,device=x.device)
    g=torch.zeros(hkv,d,d,dtype=torch.float64,device=x.device)
    for a,kv in enumerate(mapping):
        e=ds[a].double();q=qs[a].double();b.add_(e.T@e);g[kv].add_(q.T@q/d)
    return x.T@(b@x),g


def normalize(a,g,n,length,hq):
    require(n>0,'Empty statistics')
    abar=a/(n*length*hq);gbar=torch.block_diag(*list(g/(n*length*hq)))
    actual=torch.block_diag(*list(g/(n*(length-1))))
    return abar,actual,gbar


def effective(state,delta,device,pilot=False):
    x=state['x'].reshape(2048,4096).to(device).double();length,n=x.shape;d=128
    dv=delta.reshape(length,32,d).to(device);v=state['v'].reshape(length,8,d).to(device)
    z=state['z'].reshape(length,32,d).to(device)
    b=torch.zeros(length,length,dtype=torch.float64,device=device)
    gblocks=torch.zeros(8,d,d,dtype=torch.float64,device=device)
    gtrue=torch.zeros(8,length,d,dtype=torch.float64,device=device) if pilot else None
    proxy_g=torch.zeros_like(gtrue) if pilot else None
    proxy_score=torch.zeros(8,d,n,dtype=torch.float64,device=device) if pilot else None
    checks=[];explicit_gram=torch.zeros(n,n,dtype=torch.float64,device=device) if pilot else None
    gram_b=torch.zeros_like(b) if pilot else None
    c=state['cos'].to(device).double();s=state['sin'].to(device).double()
    for a in range(32):
        kv=a//4;p=state['prob'][a].to(device);da=dv[:,a,:];va=v[:,kv,:];dp=da@va.T
        e=torch._softmax_backward_data(dp.contiguous(),p.contiguous(),-1,torch.float32)
        # Reuse accepted audit-v3 native/FP64 checks; never treat surrogate S as true S.
        formula=centered_formula_audit(p,dp,e,da,va,z[:,a,:]);rowcheck=softmax_row_audit(p,dp,e)
        require(torch.count_nonzero(p.triu(1))==0 and torch.count_nonzero(e.triu(1))==0,'Masked edge contributes')
        require(torch.count_nonzero(e[-1])==0,'Final loss-free query gradient nonzero')
        ee=e.double();q=state['q'][a].to(device).double()
        b.add_(ee.T@ee);gblocks[kv].add_(q.T@q/d)
        checks.append(dict(head=a,formula=formula,softmax=rowcheck))
        if pilot:
            qt=state['qt'][a].to(device).double()
            gtrue[kv].add_(transpose_rotate(ee.T@qt/math.sqrt(d),c,s))
            proxy_g[kv].add_(ee.T@q/math.sqrt(d))
            bar=ee@x;proxy_score[kv].add_(q.T@bar/math.sqrt(d))
            if a in (0,4):
                sub=ee[:16];bx=sub@x;explicit_gram.add_(bx.T@bx);gram_b.add_(sub.T@sub)
    result=dict(A_sum=x.T@(b@x),G_blocks_sum=gblocks,checks=checks)
    require(torch.isfinite(result['A_sum']).all() and torch.isfinite(gblocks).all(),'Nonfinite Gram')
    if pilot:
        pg=proxy_g.permute(1,0,2).reshape(length,-1)
        result.update(gram_identity=compare(explicit_gram,x.T@(gram_b@x),1e-10),
                      proxy_score_identity=compare(proxy_score.reshape(1024,n),pg.T@x,1e-10),
                      true_g=gtrue.permute(1,0,2).reshape(length,-1),
                      surrogate_is_not_true_score=True)
    return result
