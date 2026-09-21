import torch
from j_common import require,compare,sm


def contract(x,g,a):
    x=x.double();g=g.double();a=a.double()
    u=((x@a)*x).sum(-1);scale=float(u.abs().max())
    require(torch.isfinite(u).all() and float(u.min())>=-1e-10*scale,'Significantly negative/nonfinite x A_M x; no clamping allowed')
    v=g.T@(g*u[:,None])
    require(torch.isfinite(v).all(),'Nonfinite weighted G')
    row=dict(sum_u=float(u.sum()),sum_u2=float(u.square().sum()),sum_u_g2=float((u*g.square().sum(-1)).sum()),
             min_u=float(u.min()),max_u=float(u.max()),negative_u_count=int((u<0).sum()),clamped=False)
    return v,u,row


def pilot(samples,a,contributions,T):
    require(len(samples)==len(contributions)==2,'Pilot requires original windows0 and1')
    aa,gg=sm.token_step(lambda:iter(samples),a,torch.eye(samples[0][1].shape[1],device=a.device,dtype=torch.float64),T)
    local_a=sum(x.T@(x*g.square().sum(-1)[:,None]) for x,g in samples)/(2*T*samples[0][1].shape[1])
    local_g=sum(v for v,u in contributions)/(2*T*a.square().sum())
    checks=dict(parent_first_A=compare(local_a,aa),parent_first_G=compare(local_g,gg),windows=[])
    for (x,g),(v,u) in zip(samples,contributions):
        xs=x[:16];gs=g[:16];us=u[:16]
        direct=torch.stack([torch.dot(z,a@z) for z in xs])
        diag=torch.diagonal((xs@a)@xs.T)
        outer=sum(us[i]*torch.outer(gs[i],gs[i]) for i in range(len(xs)))
        checks['windows'].append(dict(row_quadratic=compare(us,direct),diagonal_identity=compare(us,diag),
                                       weighted_outer=compare(gs.T@(gs*us[:,None]),outer)))
    return checks


def recover_a(parent,n=256,T=2047,m=1024):
    u=parent['U'];d=parent['D'].reshape(());a=parent['A_raw'];g=parent['G_raw']
    require(torch.isfinite(d) and float(d)>0,'Invalid inherited gradient mass')
    require(torch.isfinite(u).all() and torch.isfinite(a).all() and torch.isfinite(g).all(),'Nonfinite inherited statistics')
    a1=sm.sym(u/(n*T*m));ratio=d/(n*T*m)
    checks=dict(U_over_D=compare(sm.sym(u/d),a),trace_G=compare(d/(n*T),g.trace()),
                A1_scale=compare(a1,a*ratio),A1_over_sensitivity_A=float(ratio))
    return a1,checks
