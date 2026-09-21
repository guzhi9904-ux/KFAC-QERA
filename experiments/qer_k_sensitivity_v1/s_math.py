import torch
from s_common import require,sm


def compare(actual,reference,tol=1e-10,scale=None):
    a=torch.as_tensor(actual,dtype=torch.float64).detach();b=torch.as_tensor(reference,dtype=torch.float64,device=a.device).detach()
    require(a.shape==b.shape and torch.isfinite(a).all() and torch.isfinite(b).all(),'Invalid comparison')
    norm=float(b.norm());error=float((a-b).norm());scale=norm if scale is None else abs(float(scale))
    bound=tol*norm+64*torch.finfo(torch.float64).eps*scale
    row=dict(reference_norm=norm,absolute_error=error,relative_error=error/max(norm,1e-300),absolute_bound=bound,passed=error<=bound)
    require(row['passed'],'FP64 identity failed: '+str(row));return row


def weighted_sum(x,g):
    x=x.double();g=g.double();w=g.square().sum(-1)
    require(torch.isfinite(w).all(),'Nonfinite sensitivity weights')
    return x.T@(x*w[:,None]),w.sum(),w


def pilot(x,g):
    # Fixed first16 positions/features, full output-channel norm retained.
    x=x.reshape(-1,x.shape[-1])[:16,:16].double();g=g.reshape(-1,g.shape[-1])[:16].double()
    u,d,w=weighted_sum(x,g)
    explicit=sum(w[i]*torch.outer(x[i],x[i]) for i in range(len(w)))
    weighted=x*w.sqrt()[:,None]
    first,_=sm.token_step(lambda:iter([(x,g)]),torch.eye(x.shape[1],device=x.device,dtype=torch.float64),
                         torch.eye(g.shape[1],device=g.device,dtype=torch.float64),len(x)-1)
    const=torch.full_like(w,2.)
    return dict(explicit=compare(explicit,u),sqrt_implementation=compare(weighted.T@weighted,u),
        first_Token_A=compare(first,u/((len(x)-1)*g.shape[1])),
        constant_weight=compare(x.T@(x*const[:,None])/const.sum(),x.T@x/len(x)),
        weight_sum=float(d),input_positions=len(x),input_features=x.shape[1],gradient_channels=g.shape[1])


def scale_relation(raw,a,g):
    factor=float(a.norm());require(factor>0,'Degenerate parent A')
    checks=dict(A_gauge=compare(raw['A_raw'],a/factor),G_gauge=compare(raw['G_raw'],g*factor))
    return dict(raw_G_over_canonical=factor,checks=checks)
