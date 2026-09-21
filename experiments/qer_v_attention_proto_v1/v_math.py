import torch
from v_common import require


def compare(actual,reference,tolerance):
    a=actual.detach().double();b=reference.detach().to(a.device).double();error=a-b
    norm=float(b.norm());absolute=float(error.norm());relative=absolute/max(norm,1e-12)
    degenerate=norm<=1e-12
    # Degenerate references are judged on the implied absolute bound, reported explicitly.
    passed=absolute<=tolerance*1e-12 if degenerate else relative<=tolerance
    row=dict(relative_frobenius=relative,absolute_frobenius=absolute,max_abs=float(error.abs().max()),
             reference_norm=norm,denominator_floor=1e-12,degenerate=degenerate,tolerance=tolerance,passed=passed)
    require(passed,'Numerical comparison failed: '+str(row))
    return row


def effective(prob,x,delta,mapping,hkv,device,explicit=False):
    """Original FP32 variables -> exact FP64 contractions; no head averaging."""
    x=x.reshape(x.shape[-2],-1).to(device,dtype=torch.float64)
    length,n=x.shape;hq=len(mapping);d=delta.shape[-1]//hq
    delta=delta.reshape(length,hq,d).to(device,dtype=torch.float64)
    b=torch.zeros((length,length),dtype=torch.float64,device=device)
    blocks=torch.zeros((hkv,d,d),dtype=torch.float64,device=device)
    gv=torch.zeros((hkv,length,d),dtype=torch.float64,device=device)
    explicit_a=torch.zeros((n,n),dtype=torch.float64,device=device) if explicit else None
    effective_s=torch.zeros((hkv*d,n),dtype=torch.float64,device=device) if explicit else None
    for a,kv in enumerate(mapping):
        p=prob[a].to(device,dtype=torch.float64);z=delta[:,a,:]
        b.add_(p.T@p);blocks[kv].add_(z.T@z);gv[kv].add_(p.T@z)
        if explicit:
            bar=p@x;explicit_a.add_(bar.T@bar);effective_s[kv*d:(kv+1)*d].add_(z.T@bar)
    a_sum=x.T@(b@x)
    gv=gv.permute(1,0,2).reshape(length,hkv*d)
    out=dict(A_sum=a_sum,G_blocks_sum=blocks,g_v=gv)
    if explicit:out.update(explicit_A_sum=explicit_a,S_effective=effective_s,S_source=gv.T@x)
    return out


def normalize(a_sum,g_blocks,count,length,hq):
    require(count>0,'No statistics');a=a_sum/(count*length*hq)
    g=torch.block_diag(*list(g_blocks/(count*(length-1))))
    return a,g
