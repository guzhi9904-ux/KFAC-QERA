#!/usr/bin/env python3
"""Read-only reuse of L10 S2 checkpoints; extend Token to5, compare1/2/3/5 on validation."""
import argparse
import gc
import math
from pathlib import Path
import sys
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'experiments/qer_multimodule_v1'))
import bridge as b
from shared_teacher import SharedTeacher
from resources import Resources
from common import source_identity,PLAN as OLD
from dataset import windows

PLAN=dict(module='model.layers.10.self_attn.q_proj',budget='S2',rounds=[1,2,3,5],
    fitting_dtype='float64',validation_windows=16,rank=64,bootstrap=2000,seed=20260920,
    scope='One module,32 old train windows x4 teacher-label draws; official validation16; no official test, no new gradients',
    interpretation='Fixed-round ablation, not proof of convergence; directional factors reported separately from alternating global scale')


class ReadOnlyStore(b.TensorStore):
    def get(self,path):
        b.require(Path(path).with_suffix('.json').exists(),'Source receipt absent; do not create one')
        return super().get(path)
    def put(self,*args,**kwargs):raise RuntimeError('Source is read-only')


def directions(a,g,ra,rg):
    na,ng,nra,nrg=[float(x.norm()) for x in (a,g,ra,rg)]
    b.require(min(na,ng,nra,nrg)>0,'Zero factor')
    a,g,ra,rg=a/na,g/ng,ra/nra,rg/nrg
    ca=float((a*ra).sum());cg=float((g*rg).sum())
    return dict(A_direction_relative=float((a-ra).norm()),G_direction_relative=float((g-rg).norm()),
        Kronecker_direction_cosine=max(-1.,min(1.,ca*cg)),product_norm=na*ng,reference_product_norm=nra*nrg)


def paired_summary(values):
    baseline=values['None'];ref=values['round3'];n=len(ref)
    b.require(n>0 and float(baseline.mean())>0 and bool((ref>0).all()),'Degenerate KL reference')
    generator=torch.Generator().manual_seed(PLAN['seed'])
    draws=torch.randint(n,(PLAN['bootstrap'],n),generator=generator);rows=[]
    for key,v in values.items():
        mean=float(v.mean());refmean=float(ref.mean());base=float(baseline.mean())
        boot=100*(ref[draws].mean(1)-v[draws].mean(1))/ref[draws].mean(1)
        bounds=torch.quantile(boot,torch.tensor([.025,.975],dtype=torch.float64)).tolist()
        rows.append(dict(candidate=key,KL=mean,recovery_percent=100*(1-mean/base),
            KL_reduction_vs_round3_percent=100*(refmean-mean)/refmean,
            paired_bootstrap95_KL_reduction_percent=bounds,
            recovery_change_vs_round3_pp=100*(refmean-mean)/base))
    return rows


def run(source,validation,root,ident,config,resources):
    oldmanifest=b.read(source/'manifest.json');oldid=oldmanifest['identity'];old=ReadOnlyStore(oldid,10)
    store=b.TensorStore(ident,0);source_files={}
    def get(path):
        value=old.get(path);source_files[str(path)]=old.verified[str(path)][2];return value
    index=b.read(source/'data/sample_index.json');b.require(index['identity']==oldid,'Index identity differs')
    keys=index['budgets']['S2'];entries={r['id']:r for r in index['fit']}
    b.require(len(keys)==128 and len({entries[k]['window'] for k in keys})==32,'Expected32 windows x4 labels')
    quantpath=Path(config['quantized_source']);quant,_=b.read_tensors(quantpath)
    source_files[str(quantpath)]=b.sha_file(quantpath)
    expected=b.read(Path(config['assets'])/'exp03/teacher_identity.json')['tensor_hashes'][PLAN['module']+'.weight']
    b.require(b.mo.digest_tensor(quant['W0'])==expected['hash'],'Original weight not frozen teacher weight')
    base,_=get(source/'corrections/None.safetensors')
    b.require(torch.equal(base['W_deploy'],quant['Wq']),'Quantized baseline changed');del base
    error=(quant['W0'].double()-quant['Wq'].double()).to('cuda:0')
    b.warm_and_probe(['cuda:0','cuda:1'])
    factors={}
    for iteration in (1,2,3):
        path=source/'factors/S2/Token-joint'/f'iteration_{iteration:02d}.safetensors'
        t,m=get(path);b.require(m['samples']==keys,'Checkpoint samples differ');factors[iteration]=t
        store.put(root/'factors'/f'round{iteration}.safetensors',t,source_sha256=source_files[str(path)],source_identity=oldid)
    a,g=(factors[3][k].to('cuda:0') for k in ('A','G'))
    def samples():
        for key in keys:
            resources.boundary();row=entries[key]
            xt,xm=get(source/'cache/fit_x'/f'w{row["window"]:02d}.safetensors');gt,gm=get(source/'cache/fit_g'/(key+'.safetensors'))
            b.require(gm['x_hash']==xm['tensor_hashes']['x'] and gm['sample']==row,'Gradient/input binding differs')
            yield xt['x'].reshape(OLD['L'],-1),gt['g']
    for iteration in (4,5):
        path=root/'factors'/f'round{iteration}.safetensors'
        if path.exists():t,_=store.get(path);a,g=(t[k].to('cuda:0') for k in ('A','G'))
        else:
            with resources.timed('token_joint_round',iteration=iteration,samples=len(keys),device='cuda:0'):
                a,g=b.sm.token_step(samples,a,g,OLD['T']);a,g=b.sm.gauge(a,g)
                audit=dict(A=b.sm.parent.spectrum(a),G=b.sm.parent.spectrum(g))
                store.put(path,dict(A=a,G=g),audit=audit,samples=keys)
        factors[iteration]={'A':a.cpu(),'G':g.cpu()}
    reference=factors[3];direction_rows=[]
    for iteration,t in factors.items():
        direction_rows.append(dict(iteration=iteration,vs_round3=directions(t['A'],t['G'],reference['A'],reference['G']),
            vs_previous=None if iteration==1 else directions(t['A'],t['G'],factors[iteration-1]['A'],factors[iteration-1]['G'])))
    b.commit(root/'factor_directions.json',ident,rows=direction_rows)
    for iteration in PLAN['rounds']:
        path=root/'candidates'/f'round{iteration}.safetensors'
        if path.exists():store.get(path);continue
        if iteration==3:
            t,m=get(source/'corrections/S2__Token-joint.safetensors')
            store.put(path,{k:t[k] for k in ('P64','Q64','W_deploy')},source_identity=oldid,audit=m['audit']);del t
        else:
            t=factors[iteration]
            with resources.timed('weighted_SVD',iteration=iteration,device='cuda:0'):
                solved,c,audit=b.sm.solve(error,t['A'].to('cuda:0'),t['G'].to('cuda:0'),OLD['rank'],quant['Wq'],quant['W0'])
                store.put(path,{k:c[k] for k in ('P64','Q64','W_deploy')},audit=audit);del c,solved
    store.put(root/'candidates/None.safetensors',dict(W_deploy=quant['Wq']))
    b.commit(root/'source_assets.json',ident,files=source_files,source_read_only=True)
    b.commit(root/'candidate_freeze.json',ident,files=b.file_table(root,[p for d in ('candidates','factors') for p in (root/d).rglob('*') if p.is_file()]),validation_seen=False)
    del factors,reference,t,a,g,error,quant;old.clear();gc.collect();torch.cuda.empty_cache()
    vid=b.read(validation/'manifest.json')['identity'];ids_all=windows(validation,'validation',vid)
    b.require(len(ids_all)==16 and b.read(validation/'data/validation.json')['split']=='validation','Expected official16 validation windows')
    fit_ids,_=b.read_tensors(source/'data/fit_windows.safetensors')
    fit_hashes={b.mo.digest_tensor(row) for row in fit_ids['input_ids']}
    b.require(not fit_hashes&{b.mo.digest_tensor(row) for row in ids_all},'Fit/validation full-window overlap')
    teacher=SharedTeacher(config,root,ident,resources.timed);candidates=['None']+[f'round{i}' for i in PLAN['rounds']]
    try:
        for c,ids in enumerate(ids_all):
            paths={key:root/'scores'/f'w{c:02d}'/(key+'.json') for key in candidates}
            if all(p.exists() for p in paths.values()):continue
            resources.boundary();teacher.load();ids=ids[None,:]
            with torch.no_grad():hidden=b.hidden_forward(teacher.model,ids).detach()
            logits=teacher.reference_logits(hidden);b.require(abs(teacher.scores(ids,logits)['KL'])<=1e-10,'Self KL failed')
            for key,path in paths.items():
                if path.exists():b.load_record(path,ident);continue
                wt,_=store.get(root/'candidates'/(key+'.safetensors'))
                with resources.timed('actual_KL',window=c,candidate=key):
                    with teacher.deploy({PLAN['module']:wt['W_deploy']}):score=teacher.scores(ids,logits)
                    b.require(math.isfinite(score['KL']) and score['KL']>=0,'Invalid KL')
                    b.commit(path,ident,scores=score,input_hash=b.mo.digest_tensor(ids[0]),official_split='validation')
                del wt
            del hidden,logits;print('VALIDATION_WINDOW_COMPLETE',c+1,16,flush=True)
    finally:teacher.unload()
    values={key:torch.tensor([b.load_record(root/'scores'/f'w{c:02d}'/(key+'.json'),ident)['scores']['KL'] for c in range(16)],dtype=torch.float64) for key in candidates}
    rows=paired_summary(values);b.commit(root/'summary.json',ident,rows=rows,factor_directions=direction_rows,plan=PLAN,
        caveat='Paired bootstrap over16 windows, not independent articles; one module and one fit budget; no convergence claim',completed=True)
    b.save_csv(root/'summary.csv',rows);print('TOKEN_ROUND_PILOT_COMPLETE',rows,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','validation','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();source=args.source.resolve();validation=args.validation.resolve();root=args.output.resolve()
    b.require(root not in (source,validation) and source not in root.parents and validation not in root.parents,'Use separate output')
    legacy=b.read(source/'manifest.json');b.require(source_identity(legacy['config'])==legacy,'Historical source changed')
    b.require(b.read(source/'status.json')['status']=='COMPLETE','Parent must be complete')
    vm=b.read(validation/'manifest.json');b.require(b.identity(vm['config'])==vm,'Validation source changed')
    freeze=b.load_record(validation/'data/freeze.json',vm['identity'])
    b.checked_files(validation,{k:freeze['files'][k] for k in ('data/validation.safetensors','data/validation.json')})
    config=dict(legacy['config'],cache_GiB=10.,budget_hours=.25,disk_limit_GiB=8.)
    manifest=dict(plan=PLAN,config=config,source=str(source),validation=str(validation),
        parent_manifest_sha256=b.sha_file(source/'manifest.json'),validation_manifest_sha256=b.sha_file(validation/'manifest.json'),
        validation_assets={n:b.sha_file(validation/'data'/n) for n in ('validation.safetensors','validation.json','freeze.json')},
        own={p.name:b.sha_file(p) for p in HERE.iterdir() if p.suffix in ('.py','.md')},borrowed=b.identity(config))
    ident=b.digest(manifest);manifest['identity']=ident;root.mkdir(parents=True,exist_ok=True)
    b.require(torch.cuda.device_count()==2 and all('4090' in torch.cuda.get_device_name(i) for i in range(2)),'Require dual4090')
    torch.set_num_threads(config['cpu_threads']);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    with b.lock(root):
        if (root/'manifest.json').exists():b.require(b.read(root/'manifest.json')==manifest,'Frozen short-test identity differs')
        else:b.save_json(root/'manifest.json',manifest)
        resources=Resources(root,ident,config)
        try:run(source,validation,root,ident,config,resources)
        finally:resources.flush()


if __name__=='__main__':main()
