"""Exact reassociation of S=g.T@x; all cross-position terms remain present."""
import torch
from common import require


class GroupedStream:
    def __init__(self,dense,groups):self.dense=dense;self.groups=groups
    def __call__(self):return self.dense()


def contraction(samples,factor,side,T):
    from sketch_math import parent,sym
    if not hasattr(samples,'groups'):return parent.contraction(samples,factor,side,T)
    total=None;count=0
    require(side in ('A','G'),'Invalid contraction side')
    for x,gradients in samples.groups():
        x=x.to(factor.device,dtype=torch.float64)
        if side=='G':
            shared=x@factor@x.T
            for z in gradients:
                z=z.to(factor.device,dtype=torch.float64);term=z.T@shared@z
                total=term if total is None else total+term;count+=1
        else:
            shared=None
            for z in gradients:
                z=z.to(factor.device,dtype=torch.float64);term=z@factor@z.T
                shared=term if shared is None else shared+term;count+=1
            require(shared is not None,'Empty article gradient group')
            term=x.T@shared@x;total=term if total is None else total+term
    require(count>0,'Empty grouped stream')
    return sym(total/(count*T)),count


def metric_inner(samples,a,g,T):
    total=torch.zeros((),dtype=torch.float64,device=a.device);count=0
    for x,gradients in samples.groups():
        x=x.to(a.device,dtype=torch.float64);xx=x@a@x.T
        for z in gradients:
            z=z.to(a.device,dtype=torch.float64)
            total.add_(((z@g@z.T)*xx).sum());count+=1
    require(count>0,'Empty grouped metric stream')
    return float(total)/(count*T)


def gram_features(paths,read_s,T,device,boundary=lambda:None,features=262144):
    """CPU mmap references; only N*features numbers reside on CUDA at a time."""
    require(bool(paths) and features>0,'Empty Gram or invalid panel size')
    rows=[]
    for p in paths:
        boundary();s=read_s(p)
        require(s.dtype==torch.float64 and s.device.type=='cpu','Gram requires CPU FP64 cache')
        rows.append(s.reshape(-1))
    width=rows[0].numel();require(all(r.numel()==width for r in rows),'Gram shapes differ')
    result=torch.zeros((len(rows),len(rows)),device=device,dtype=torch.float64)
    for start in range(0,width,features):
        boundary();panel=torch.stack([r[start:start+features] for r in rows]).to(device)
        result.add_(panel@panel.T);del panel
    result=result.cpu();require(bool(torch.isfinite(result).all()),'Nonfinite Gram')
    return result,float(result.square().sum()/(len(rows)**2*T*T))


def validate_reassociation(samples,a,g,T):
    """Pilot acceptance against independent dense-S operations on actual fit data."""
    from sketch_math import parent,relative,raw_metric_inner,scalar_check,full_step
    checks={}
    for side,factor in [('A',g),('G',a)]:
        fast,n=contraction(samples,factor,side,T)
        slow,m=parent.contraction(samples,factor,side,T)
        error=relative(fast,slow);require(n==m and error<=1e-10,'Reassociation differs: '+side)
        checks[side+'_relative']=error
    fast=metric_inner(samples,a,g,T);slow=raw_metric_inner(lambda:samples(),a,g,T)
    checks['inner']=scalar_check(fast,slow,abs(slow))
    af,gf,hf=full_step(samples,a,g,T)
    ad,gd,hd=full_step(lambda:samples(),a,g,T)
    for key,fast,slow in [('ALS_A',af,ad),('ALS_G',gf,gd)]:
        error=relative(fast,slow);require(error<=1e-10,'ALS reassociation differs')
        checks[key+'_relative']=error
    checks['ALS_J']=scalar_check(hf['J_after_A'],hd['J_after_A'],abs(hd['J_after_A']))
    checks['passed']=True
    return checks
