"""Exact shared symmetric roots and dense weighted SVD, with parent gates."""
from concurrent.futures import ThreadPoolExecutor
import gc
import math
import time
import torch
from fm_common import *
from audit import original_weight

def roots(a):
    scale=float(a.norm());require(scale>0,'Zero A')
    aa,ar,audit=sm.parent.damp(a/scale,.001)
    return aa,ar,scale,audit

def solve_one(error,a,g,w0,wq,rank=64,prepared=None):
    """G=None is the exact scalar-I path, including the parent's gauge/damping."""
    aa,ar,scale,am=prepared if prepared is not None else roots(a)
    if g is None:
        scalar=scale*(1+.001);gr=math.sqrt(scalar);gg=None
        gm=dict(eta=.001,lambda_=scale*.001,trace_scale=scale,condition=1.,root_error=0.,scalar_identity=True)
    else:
        gg,gr,gm=sm.parent.damp(g*scale,.001)
    b=(error@ar)*gr if g is None else gr@error@ar
    start=time.monotonic()
    u,s,vh=torch.linalg.svd(b,full_matrices=False,**({'driver':'gesvd'} if b.is_cuda else {}))
    left=u[:,:rank]*s[:rank].sqrt()
    p=left/gr if g is None else torch.linalg.solve(gr,left)
    q=torch.linalg.solve(ar,(s[:rank].sqrt()[:,None]*vh[:rank]).T).T
    c=p@q;target=(u[:,:rank]*s[:rank])@vh[:rank]
    back=(c@ar)*gr if g is None else gr@c@ar
    transform=sm.relative(back,target)
    def objective(r):
        return float(((r@aa)*r).sum()*scalar/2) if g is None else sm.qmetric(r,aa,gg)
    residual=error-c;tail=float(s[rank:].square().sum()/2);obj=objective(residual)
    tail_error=abs(obj-tail)/tail if tail else abs(obj-tail)
    require(transform<=1e-8 and tail_error<=1e-8,'Weighted-SVD/back-transform gate failed')
    deployed=wq.to(error.device)+c.float()
    actual=w0.to(error.device).double()-deployed.double();qd=objective(actual)
    drift=abs(qd-obj)/max(abs(qd),abs(obj),1e-30)
    require(drift<=1e-4,'FP32 deployment proxy drift failed')
    audit=dict(rank=rank,A_gauge_divisor=scale,G_gauge_multiplier=scale,A_damping=am,G_damping=gm,
        dense_SVD=True,driver='gesvd' if b.is_cuda else 'default_cpu',seconds=time.monotonic()-start,
        tail_energy_half=tail,solve_objective=obj,tail_relative_error=tail_error,back_transform_error=transform,
        singular_at_rank=float(s[rank-1]),singular_after_rank=float(s[rank]) if rank<len(s) else None,
        deployment_relative_drift=drift,deployed_objective=qd,W_deploy_hash=mo.digest_tensor(deployed),
        deployment='Wq + float32(P64 @ Q64); FP64 product; FP32 addition',scalar_G_path=g is None)
    return {'P64':p,'Q64':q},audit

def statistics(ctx,i,kind,family):
    if family=='Token-joint-one':
        row=load_file(str(ctx.stat(i,'K_one')));return row['A'],row['G']
    if family=='Attention-aware':
        row=load_file(str(ctx.stat(i,'V_attention')));return row['A'],row['G']
    a=load_file(str(ctx.stat(i,'A_'+a_group(kind))))['A']
    g=None if family=='A-only' else load_file(str(ctx.stat(i,'G_'+kind)))['G']
    return a,g

def solve(ctx):
    require(ctx.done('statistics/functional_complete.json',windows=N,groups=8),'Functional statistics incomplete')
    if ctx.done('factors/complete.json',factors=416):
        return
    ctx.teacher.unload();warm_and_probe(['cuda:0','cuda:1'])
    quant=read(ctx.root/'quantization_manifest.json')['modules']
    def worker(device,layers):
        torch.cuda.set_device(device)
        for i in layers:
            for family_key,kinds in [('qkv',('q','k','v')),('o',('o',)),('gate_up',('gate','up')),('down',('down',))]:
                a=load_file(str(ctx.stat(i,'A_'+family_key)))['A'].to(device)
                with ctx.timed('shared_A_root',layer=i,family=family_key,device=device):prepared=roots(a)
                for kind in kinds:
                    families=['A-only']+(['Marginal'] if kind in ('q','k','v','o') else [])
                    for family in families:
                        path=ctx.factor(i,kind,family);cp=path.with_suffix('.complete.json')
                        if ctx.done(cp.relative_to(ctx.root),rank=64):continue
                        with ctx.timed('factor_solve',layer=i,kind=kind,family=family,device=device):
                            key=name(i,kind);row=quant[key]
                            require(sha(row['path'])==row['file_sha256'],'Wq changed before solve')
                            wq=load_file(row['path'])['Wq'].to(device);w0=original_weight(ctx,key,device)
                            g=None if family=='A-only' else load_file(str(ctx.stat(i,'G_'+kind)))['G'].to(device)
                            factors,audit=solve_one(w0.double()-wq.double(),a,g,w0,wq,prepared=prepared)
                            tensors(path,factors);write(path.with_suffix('.json'),audit)
                            ctx.commit(cp.relative_to(ctx.root),[path,path.with_suffix('.json')],rank=64,module=key,family=family,
                                       Wq_hash=row['Wq_hash'],statistics_a_sha256=sha(ctx.stat(i,'A_'+family_key)),
                                       statistics_g_sha256=None if g is None else sha(ctx.stat(i,'G_'+kind)))
                            del g,wq,w0,factors
                del a,prepared;gc.collect();torch.cuda.empty_cache()
            for kind,family,stat in [('k','Token-joint-one','K_one'),('v','Attention-aware','V_attention')]:
                path=ctx.factor(i,kind,family);cp=path.with_suffix('.complete.json')
                if ctx.done(cp.relative_to(ctx.root),rank=64):continue
                with ctx.timed('factor_solve',layer=i,kind=kind,family=family,device=device):
                    raw=load_file(str(ctx.stat(i,stat)));a=raw['A'].to(device);g=raw['G'].to(device)
                    key=name(i,kind);row=quant[key];require(sha(row['path'])==row['file_sha256'],'Wq changed')
                    wq=load_file(row['path'])['Wq'].to(device);w0=original_weight(ctx,key,device)
                    factors,audit=solve_one(w0.double()-wq.double(),a,g,w0,wq)
                    tensors(path,factors);write(path.with_suffix('.json'),audit)
                    ctx.commit(cp.relative_to(ctx.root),[path,path.with_suffix('.json')],rank=64,module=key,family=family,
                               Wq_hash=row['Wq_hash'],statistics_sha256=sha(ctx.stat(i,stat)))
                    del raw,a,g,wq,w0,factors
                gc.collect();torch.cuda.empty_cache()
            log('SOLVE_LAYER_COMPLETE',layer=i,unique_factors=13)
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(worker,f'cuda:{d}',range(d*16,(d+1)*16)) for d in range(2)]
        for job in jobs:job.result()
    records=list((ctx.root/'factors').rglob('*.complete.json'))
    require(len(records)==416,'Missing/extra independent factors')
    ctx.commit('factors/complete.json',records,factors=416)

def freeze(ctx):
    require(ctx.done('factors/complete.json',factors=416),'Factor solve incomplete')
    if ctx.done('candidates/complete.json',states=6):return
    quant=read(ctx.root/'quantization_manifest.json');files=[]
    expected={'Teacher':0,'C0':0,'C1':96,'C2':224,'C3':224,'C4':224}
    for state in STATES:
        modules={};count=0
        for i in range(32):
            for kind in KINDS:
                key=name(i,kind);family=method(state,kind);row=dict(quant['modules'][key],family=family)
                if family:
                    path=ctx.factor(i,kind,family);require(ctx.done(path.with_suffix('.complete.json').relative_to(ctx.root),rank=64),'Missing factor')
                    row.update(factor=str(path),factor_sha256=sha(path),W_deploy_hash=read(path.with_suffix('.json'))['W_deploy_hash']);count+=1
                else:row['W_deploy_hash']=row['W0_hash'] if state=='Teacher' else row['Wq_hash']
                modules[key]=row
        require(count==expected[state],'Candidate budget mismatch')
        material=dict(run_identity=ctx.identity,state=state,modules=modules,compensated_modules=count,
                      teacher_sha256=sha(ctx.root/'teacher_identity.json'),eval_sha256=sha(ctx.root/'eval_manifest.json'),rank=64,
                      non_target='unchanged teacher FP32',deployment='Wq + float32(P64 @ Q64)')
        path=ctx.root/'candidates'/f'{state}.json';write(path,dict(identity=digest(material),**material));files.append(path)
    ctx.commit('candidates/complete.json',files,states=6,before_model_scores=True)
