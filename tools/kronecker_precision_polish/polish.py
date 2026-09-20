#!/usr/bin/env python3
"""Validate FP64 marginal statistics and final iteration after FP32 early rounds.

The failed pure-FP32 pilot stays immutable. Same total iteration count, same
FP64 reference, same effect gates; strict FP64 raw-PSD solver gate is restored.
"""
import argparse
import gc
from pathlib import Path
import sys
import time
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'kronecker_precision_pilot'))
from bootstrap import bridge,PLAN,SharedTeacher,Resources,manifest as base_manifest
from kernels import Ops,solve,factor_comparison


def iterate(samples,a,g,method,device,L,T,full_rounds=8,token_rounds=3,boundary=lambda:None):
    require=bridge.require;require(method in ('Token-joint','Full-fit'),'Only iterative methods need polishing')
    rounds=token_rounds if method=='Token-joint' else full_rounds;history=[]
    require(rounds>=2,'Need early and final rounds')
    if method=='Token-joint':g=torch.eye(len(g),dtype=torch.float64,device=device)
    for i in range(rounds):
        boundary();dtype=torch.float64 if i==rounds-1 else torch.float32;op=Ops(dtype,device)
        if str(device).startswith('cuda'):torch.cuda.synchronize(device)
        started=time.perf_counter()
        if method=='Full-fit':a,g,row=op.full(samples,a,g,T)
        else:a,g=op.token(samples,a,g,T);a,g=bridge.sm.gauge(a,g);row=dict(numeric_round_passed=True)
        if str(device).startswith('cuda'):torch.cuda.synchronize(device)
        history.append(dict(iteration=i+1,dtype=str(dtype),seconds=time.perf_counter()-started,**row))
    a,g=bridge.sm.gauge(a,g);return dict(A=a,G=g),history


def run(source,root,resources,ident,config):
    source_manifest=bridge.read(source/'manifest.json');source_id=source_manifest['identity']
    bridge.require(base_manifest(source_manifest['config'])==source_manifest,'Source pilot code/config changed')
    bridge.require((source/'summary.json').exists(),'Finish original pilot first')
    for name in ('candidate_freeze.json','collection.json','quantized/freeze.json'):
        bridge.checked_files(source,bridge.load_record(source/name,source_id)['files'])
    old=bridge.TensorStore(source_id,0);store=bridge.TensorStore(ident,0)
    bridge.warm_and_probe(['cuda:0','cuda:1'])
    comparisons=[]
    for name in PLAN['modules']:
        device='cuda:1' if name.endswith('down_proj') else 'cuda:0';torch.cuda.set_device(device)
        rows=[]
        for c in PLAN['fit_windows']:
            t,_=old.get(source/'cache'/bridge.slug(name)/f'w{c}.safetensors');rows.append((t['x'].reshape(PLAN['L'],-1).to(device),t['g'].to(device)))
        samples=lambda rows=rows:iter(rows)
        stats,_=old.get(source/'factors'/bridge.slug(name)/'fp64/statistics.safetensors')
        quant,_=old.get(source/'quantized'/(bridge.slug(name)+'.safetensors'));error=(quant['W0'].double()-quant['Wq'].double()).to(device)
        for method in ('Token-joint','Full-fit'):
            cp=root/'candidates'/bridge.slug(name)/(method+'.safetensors');rp=root/'factors'/bridge.slug(name)/(method+'.safetensors')
            result=root/'comparisons'/bridge.slug(name)/(method+'.json')
            if result.exists():comparisons.append(bridge.load_record(result,ident));continue
            if rp.exists():raw,meta=store.get(rp);history=meta['history'];raw={k:v.to(device) for k,v in raw.items()}
            else:
                with resources.timed('mixed_iterations',module=name,method=method,device=device):
                    raw,history=iterate(samples,stats['A'].to(device),stats['G'].to(device),method,device,PLAN['L'],PLAN['T'],
                        full_rounds=PLAN['full_rounds'],token_rounds=PLAN['token_rounds'],boundary=resources.boundary)
                store.put(rp,raw,history=history)
            ref,_=old.get(source/'factors'/bridge.slug(name)/'fp64'/(method+'.safetensors'))
            comparison=factor_comparison(raw['A'].cpu(),raw['G'].cpu(),ref['A'],ref['G']);comparison['iterations_passed']=all(h['numeric_round_passed'] for h in history)
            objective=None;error_message=None
            try:
                if not cp.exists():
                    with resources.timed('strict_FP64_solve',module=name,method=method,device=device):
                        t,audit=solve(error,raw['A'],raw['G'],quant['W0'],quant['Wq'],PLAN['rank'],False)
                        store.put(cp,t,audit=audit,strict_raw_PSD=True);del t
                a=ref['A'].to(device);g=ref['G'].to(device)
                a=a+PLAN['eta']*a.trace()/len(a)*torch.eye(len(a),dtype=torch.float64,device=device)
                g=g+PLAN['eta']*g.trace()/len(g)*torch.eye(len(g),dtype=torch.float64,device=device)
                values={}
                for label,st,p in [('fp64',old,source/'candidates'/bridge.slug(name)/'fp64'/(method+'.safetensors')),('hybrid',store,cp)]:
                    t,_=st.get(p);r=error-t['P64'].to(device)@t['Q64'].to(device);values[label]=bridge.sm.qmetric(r,a,g);del t,r
                relative=abs(values['hybrid']-values['fp64'])/max(abs(values['fp64']),1e-12)
                objective=dict(values=values,relative=relative,passed=relative<=PLAN['tolerances']['reference_objective_relative']);del a,g
            except RuntimeError as exc:error_message=repr(exc);print('HYBRID_CANDIDATE_FAILED',name,method,error_message,flush=True)
            row=bridge.commit(result,ident,module=name,method=method,factors=comparison,objective=objective,error=error_message,history=history)
            comparisons.append(row);del raw,ref;gc.collect();torch.cuda.empty_cache()
        del error,quant,rows,samples,stats;gc.collect();torch.cuda.empty_cache()
    bridge.commit(root/'candidate_freeze.json',ident,files=bridge.file_table(root,[p for d in ('candidates','comparisons','factors') for p in (root/d).rglob('*') if p.is_file()]))
    teacher=SharedTeacher(config,root,ident,resources.timed);ids_all=old.get(source/'data/windows.safetensors')[0]['input_ids']
    try:
        for c in PLAN['check_windows']:
            resources.boundary();teacher.load();ids=ids_all[c:c+1]
            with torch.no_grad():reference=bridge.hidden_forward(teacher.model,ids).detach()
            logits=teacher.reference_logits(reference);bridge.require(abs(teacher.scores(ids,logits)['KL'])<=1e-10,'Self KL failed')
            for row in comparisons:
                name=row['module'];method=row['method'];cp=root/'candidates'/bridge.slug(name)/(method+'.safetensors')
                out=root/'scores'/f'w{c}'/bridge.slug(name)/(method+'.json')
                if not cp.exists() or out.exists():continue
                t,_=store.get(cp)
                with resources.timed('actual_KL',window=c,module=name,method=method):
                    with teacher.deploy({name:t['W_deploy']}):score=teacher.scores(ids,logits)
                    prefix=source/'scores'/f'w{c}'/bridge.slug(name)
                    baseline=bridge.load_record(prefix/'quantized__None.json',source_id)['scores']['KL']
                    reference_kl=bridge.load_record(prefix/('fp64__'+method+'.json'),source_id)['scores']['KL']
                    allowed=max(PLAN['tolerances']['KL_absolute'],baseline*PLAN['tolerances']['KL_over_quantized'])
                    diff=abs(score['KL']-reference_kl)
                    bridge.commit(out,ident,hybrid=score['KL'],reference=reference_kl,baseline=baseline,allowed=allowed,absolute_error=diff,
                        recovery_difference_pp=100*(reference_kl-score['KL'])/baseline if baseline>1e-12 else None,
                        passed=diff<=allowed,input_hash=bridge.mo.digest_tensor(ids[0]),weight_hash=bridge.mo.digest_tensor(t['W_deploy']))
                del t
            del reference,logits
    finally:teacher.unload()
    for row in comparisons:
        row['KL']=[]
        for c in PLAN['check_windows']:
            path=root/'scores'/f'w{c}'/bridge.slug(row['module'])/(row['method']+'.json')
            row['KL'].append(bridge.load_record(path,ident) if path.exists() else dict(passed=False))
        row['passed']=bool(row['factors']['passed'] and row['factors']['iterations_passed'] and row['objective'] and row['objective']['passed'] and all(v['passed'] for v in row['KL']))
    bridge.commit(root/'summary.json',ident,passed=all(r['passed'] for r in comparisons),rows=comparisons,formal_precision_changed=False,
        policy='Marginal/Sequence remain FP64; Token2xFP32+1xFP64; Full7xFP32+1xFP64. Same total rounds, strict FP64 final raw-PSD gate; no clipping or changed damping.',
        scope=PLAN['scope']);print('POLISH_PILOT_COMPLETE',all(r['passed'] for r in comparisons),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    source=args.source.resolve();root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    source_manifest=bridge.read(source/'manifest.json');config=dict(source_manifest['config']);config.update(budget_hours=.5,disk_limit_GiB=32.)
    frozen=dict(parent_manifest_sha256=bridge.sha_file(source/'manifest.json'),parent_summary_sha256=bridge.sha_file(source/'summary.json'),
        source=str(source),config=config,plan=PLAN,policy='FP64 marginal and final fixed iteration; strict PSD; other equations/gates unchanged',
        code={p.name:bridge.sha_file(p) for p in HERE.glob('*') if p.suffix in ('.py','.md')})
    frozen['identity']=bridge.digest(frozen);ident=frozen['identity']
    bridge.require(torch.cuda.device_count()==2 and all('4090' in torch.cuda.get_device_name(i) for i in range(2)),'Require dual4090')
    torch.set_num_threads(config['cpu_threads']);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    with bridge.lock(root):
        if (root/'manifest.json').exists():bridge.require(bridge.read(root/'manifest.json')==frozen,'Source changed; use a new output')
        else:bridge.save_json(root/'manifest.json',frozen)
        resources=Resources(root,ident,config)
        try:run(source,root,resources,ident,config)
        finally:resources.flush()


if __name__=='__main__':main()
