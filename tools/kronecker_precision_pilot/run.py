#!/usr/bin/env python3
"""Bounded fit-only precision pilot; never modifies the formal12-module run."""
import argparse
import gc
import math
from pathlib import Path
import statistics
import time
import traceback
import torch
from bootstrap import bridge,PLAN,manifest,SharedTeacher,Resources
from kernels import Ops,factor_comparison,solve


def prepare(root,config,ident,resources):
    from assets_local import prepare_quantized
    prepare_quantized(root,config,ident)
    store=bridge.TensorStore(ident,0);p=root/'collection.json'
    if p.exists():bridge.checked_files(root,bridge.load_record(p,ident)['files']);return
    parent=Path(config['assets'])/'exp03';t,m=bridge.read_tensors(parent/'data/fit_windows.safetensors')
    from common import PLAN as OLD
    bridge.require(m['identity']==OLD['parent_identity'],'Parent data identity differs')
    ids=t['input_ids'][PLAN['check_windows']]
    store.put(root/'data/windows.safetensors',dict(input_ids=ids),source_sha256=bridge.sha_file(parent/'data/fit_windows.safetensors'))
    paths=[root/'data/windows.safetensors'];teacher=SharedTeacher(config,root,ident,resources.timed)
    try:
        for c in PLAN['fit_windows']:
            lp=parent/'data/fit_samples'/f'w{c:02d}_k000.safetensors';lt,lm=bridge.read_tensors(lp);label=lt['labels']
            bridge.require(lm['identity']==OLD['parent_identity'] and lm['input_hash']==bridge.mo.digest_tensor(ids[c]) and lm['label_hash']==bridge.mo.digest_tensor(label),'Parent label binding differs')
            dest=root/'data'/f'label_{c}.safetensors';store.put(dest,dict(labels=label),source_sha256=bridge.sha_file(lp));paths.append(dest)
            with resources.timed('collect_shared',window=c):
                ref,xs=teacher.reference_all(ids[c:c+1],PLAN['modules']);rows=teacher.gradient_all(ids[c:c+1],PLAN['modules'],ref,xs,label)
                for name,row in rows.items():
                    dest=root/'cache'/bridge.slug(name)/f'w{c}.safetensors';store.put(dest,row,window=c,module=name);paths.append(dest)
                del ref,xs,rows
        bridge.commit(p,ident,files=bridge.file_table(root,paths),fit_windows=PLAN['fit_windows'],test_data_used=False)
    finally:teacher.unload()


def raw_factors(root,ident,name,precision,method,samples,device,resources):
    path=root/'factors'/bridge.slug(name)/precision/(method+'.safetensors');store=bridge.TensorStore(ident,0)
    if path.exists():return store.get(path)
    op=Ops(torch.float64 if precision=='fp64' else torch.float32,device);stats=root/'factors'/bridge.slug(name)/precision/'statistics.safetensors'
    if stats.exists():t,_=store.get(stats);a=t['A'].to(device);g=t['G'].to(device)
    else:
        with resources.timed('marginal',module=name,precision=precision,device=device):a,g=op.marginal(samples,PLAN['L'],PLAN['T'])
        store.put(stats,dict(A=a,G=g))
    history=[]
    if method=='Token-joint':
        g=torch.eye(len(g),dtype=torch.float64,device=device)
        for i in range(PLAN['token_rounds']):
            with resources.timed('token_round',module=name,precision=precision,iteration=i+1,device=device):
                a,g=op.token(samples,a,g,PLAN['T']);a,g=bridge.sm.gauge(a,g)
    elif method=='Sequence-one-step':
        with resources.timed('sequence_moments',module=name,precision=precision,device=device):
            ma,mg=op.moments(samples,PLAN['T']);a,g=bridge.sm.sequence_one_step(ma,mg)
    elif method=='Full-fit':
        for i in range(PLAN['full_rounds']):
            with resources.timed('full_round',module=name,precision=precision,iteration=i+1,device=device):
                a,g,row=op.full(samples,a,g,PLAN['T']);history.append(dict(iteration=i+1,**row))
    else:bridge.require(method=='Marginal','Unknown method')
    a,g=bridge.sm.gauge(a,g);store.put(path,dict(A=a,G=g),module=name,method=method,precision=precision,history=history,
        passed=all(h['numeric_round_passed'] for h in history),fixed_round_pilot=True)
    return store.get(path)


def benchmark(root,ident,name,samples,device,resources):
    path=root/'benchmarks'/(bridge.slug(name)+'.json')
    if path.exists():return bridge.load_record(path,ident)
    op64=Ops(torch.float64,device);a,g=op64.marginal(samples,PLAN['L'],PLAN['T'])
    # Same nativeFP32 GPU-resident inputs and same factors for both kernels.
    # This measures computation+casting, excludes disk/model/SVD.
    values={p:[] for p in ('fp64','fp32')}
    for dtype in (torch.float64,torch.float32):
        warm=Ops(dtype,device).full(samples,a,g,PLAN['T']);del warm
    for repeat in range(3):
        for precision in (('fp64','fp32') if repeat%2==0 else ('fp32','fp64')):
            resources.boundary();torch.cuda.synchronize(device);start=time.perf_counter()
            result=Ops(torch.float64 if precision=='fp64' else torch.float32,device).full(samples,a,g,PLAN['T'])
            torch.cuda.synchronize(device);values[precision].append(time.perf_counter()-start);del result
    med={p:statistics.median(v) for p,v in values.items()}
    row=bridge.commit(path,ident,module=name,seconds=values,median_seconds=med,speedup=med['fp64']/med['fp32'],
        workload='one two-window Full-fit update; GPU-resident nativeFP32 cache; FP64 reductions; no eigenspectrum or SVD')
    print('KERNEL_BENCHMARK',name,row['median_seconds'],'speedup',row['speedup'],flush=True);return row


def fit(root,config,ident,resources):
    bridge.warm_and_probe(['cuda:0','cuda:1']);store=bridge.TensorStore(ident,0)
    for name in PLAN['modules']:
        device='cuda:1' if name.endswith('down_proj') else 'cuda:0';torch.cuda.set_device(device)
        rows=[]
        for c in PLAN['fit_windows']:
            t,_=store.get(root/'cache'/bridge.slug(name)/f'w{c}.safetensors');rows.append((t['x'].reshape(PLAN['L'],-1).to(device),t['g'].to(device)))
        samples=lambda rows=rows:iter(rows);benchmark(root,ident,name,samples,device,resources)
        quant,_=store.get(root/'quantized'/(bridge.slug(name)+'.safetensors'))
        error=(quant['W0'].double()-quant['Wq'].double()).to(device)
        for method in PLAN['methods']:
            factor_paths=[]
            for precision in ('fp64','fp32'):
                raw,meta=raw_factors(root,ident,name,precision,method,samples,device,resources)
                factor_paths.append(root/'factors'/bridge.slug(name)/precision/(method+'.safetensors'))
                cp=root/'candidates'/bridge.slug(name)/precision/(method+'.safetensors')
                failure=cp.with_suffix('.failed.json')
                if cp.exists():store.get(cp);continue
                if failure.exists():bridge.load_record(failure,ident);continue
                try:
                    with resources.timed('FP64_solve',module=name,precision=precision,method=method,device=device):
                        tensors,audit=solve(error,raw['A'].to(device),raw['G'].to(device),quant['W0'],quant['Wq'],PLAN['rank'],precision=='fp32')
                        store.put(cp,tensors,audit=audit,iterations_passed=meta['passed'],raw_sha256=bridge.sha_file(factor_paths[-1]));del tensors
                except RuntimeError as exc:
                    bridge.commit(failure,ident,passed=False,error=repr(exc));print('CANDIDATE_FAILED',name,precision,method,repr(exc),flush=True)
                finally:
                    del raw;gc.collect();torch.cuda.empty_cache()
            ref,rm=store.get(factor_paths[0]);mixed,mm=store.get(factor_paths[1])
            checks=factor_comparison(mixed['A'],mixed['G'],ref['A'],ref['G'])
            checks['iterations_passed']=rm['passed'] and mm['passed']
            bridge.commit(root/'comparisons'/bridge.slug(name)/(method+'.json'),ident,**checks)
            cp64=root/'candidates'/bridge.slug(name)/'fp64'/(method+'.safetensors');cp32=root/'candidates'/bridge.slug(name)/'fp32'/(method+'.safetensors')
            if cp64.exists() and cp32.exists():
                a=ref['A'].to(device);g=ref['G'].to(device)
                a=a+PLAN['eta']*a.trace()/len(a)*torch.eye(len(a),dtype=torch.float64,device=device)
                g=g+PLAN['eta']*g.trace()/len(g)*torch.eye(len(g),dtype=torch.float64,device=device)
                scores={}
                for precision,cp in [('fp64',cp64),('fp32',cp32)]:
                    t,_=store.get(cp);r=error-t['P64'].to(device)@t['Q64'].to(device)
                    scores[precision]=bridge.sm.qmetric(r,a,g);del r,t
                relative=abs(scores['fp32']-scores['fp64'])/max(abs(scores['fp64']),1e-12)
                bridge.commit(root/'objectives'/bridge.slug(name)/(method+'.json'),ident,values=scores,relative=relative,
                    passed=relative<=PLAN['tolerances']['reference_objective_relative']);del a,g
            del ref,mixed
        del error,quant,rows,samples;store.clear();gc.collect();torch.cuda.empty_cache()
    paths=[p for d in ('factors','candidates','comparisons','objectives') for p in (root/d).rglob('*') if p.is_file()]
    bridge.commit(root/'candidate_freeze.json',ident,files=bridge.file_table(root,paths),evaluation_seen=False)


def evaluate(root,config,ident,resources):
    bridge.checked_files(root,bridge.load_record(root/'candidate_freeze.json',ident)['files'])
    store=bridge.TensorStore(ident,0);ids_all=store.get(root/'data/windows.safetensors')[0]['input_ids']
    teacher=SharedTeacher(config,root,ident,resources.timed)
    try:
        for c in PLAN['check_windows']:
            resources.boundary();ids=ids_all[c:c+1]
            teacher.load()
            with torch.no_grad():ref=bridge.hidden_forward(teacher.model,ids).detach()
            logits=teacher.reference_logits(ref)
            bridge.require(abs(teacher.scores(ids,logits)['KL'])<=1e-10,'Self-KL differs')
            for name in PLAN['modules']:
                targets=[('quantized','None',root/'quantized'/(bridge.slug(name)+'.safetensors'))]
                targets += [(precision,method,root/'candidates'/bridge.slug(name)/precision/(method+'.safetensors')) for precision in ('fp64','fp32') for method in PLAN['methods']]
                for precision,method,path in targets:
                    out=root/'scores'/f'w{c}'/bridge.slug(name)/(precision+'__'+method+'.json')
                    if out.exists():bridge.load_record(out,ident);continue
                    if not path.exists():continue
                    t,_=store.get(path);weight=t['Wq' if precision=='quantized' else 'W_deploy']
                    with resources.timed('actual_KL',window=c,module=name,precision=precision,method=method):
                        with teacher.deploy({name:weight}):score=teacher.scores(ids,logits)
                        bridge.require(math.isfinite(score['KL']) and score['KL']>=0,'Invalid KL')
                        bridge.commit(out,ident,scores=score,weight_hash=bridge.mo.digest_tensor(weight),input_hash=bridge.mo.digest_tensor(ids[0]),
                            role='fit' if c in PLAN['fit_windows'] else 'heldout_within_parent_train')
                    del t,weight
            del ref,logits
    finally:teacher.unload()


def report(root,ident):
    rows=[];passed=True
    for name in PLAN['modules']:
        for method in PLAN['methods']:
            row=dict(module=name,method=method)
            checks=bridge.load_record(root/'comparisons'/bridge.slug(name)/(method+'.json'),ident)
            row.update(factors=checks,KL=[]);objective=root/'objectives'/bridge.slug(name)/(method+'.json')
            row['objective']=bridge.load_record(objective,ident) if objective.exists() else dict(passed=False)
            for c in PLAN['check_windows']:
                folder=root/'scores'/f'w{c}'/bridge.slug(name);paths=[folder/(p+'__'+method+'.json') for p in ('fp64','fp32')]
                if not all(p.exists() for p in paths):row['KL'].append(dict(window=c,passed=False,reason='candidate or evaluation absent'));continue
                base=bridge.load_record(folder/'quantized__None.json',ident)['scores']['KL']
                ref,mixed=[bridge.load_record(p,ident)['scores']['KL'] for p in paths]
                allowed=max(PLAN['tolerances']['KL_absolute'],PLAN['tolerances']['KL_over_quantized']*base)
                row['KL'].append(dict(window=c,fp64=ref,fp32=mixed,baseline=base,absolute_error=abs(mixed-ref),allowed=allowed,
                    recovery_difference_pp=100*(ref-mixed)/base if base>1e-12 else None,passed=abs(mixed-ref)<=allowed))
            row['passed']=checks['passed'] and checks['iterations_passed'] and row['objective']['passed'] and all(v['passed'] for v in row['KL'])
            passed=passed and row['passed'];rows.append(row)
    benchmarks=[bridge.load_record(root/'benchmarks'/(bridge.slug(n)+'.json'),ident) for n in PLAN['modules']]
    summary=dict(passed=passed,rows=rows,benchmarks=benchmarks,scope=PLAN['scope'],formal_precision_changed=False)
    bridge.commit(root/'summary.json',ident,**summary);print('PRECISION_PILOT_COMPLETE',passed,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--from-config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();config=bridge.read(args.from_config);config.update(modules=PLAN['modules'],budget_hours=.75,disk_limit_GiB=64.,cache_GiB=0.)
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=True);frozen=manifest(config);ident=frozen['identity']
    bridge.require(torch.cuda.device_count()==2 and all('4090' in torch.cuda.get_device_name(i) for i in range(2)),'Require idle dual4090')
    torch.set_num_threads(config['cpu_threads']);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    with bridge.lock(root):
        if (root/'manifest.json').exists():bridge.require(bridge.read(root/'manifest.json')==frozen,'Pilot source/config changed; use a new output')
        else:bridge.save_json(root/'manifest.json',frozen)
        resources=Resources(root,ident,config)
        try:
            prepare(root,config,ident,resources);fit(root,config,ident,resources);evaluate(root,config,ident,resources);report(root,ident)
            bridge.save_json(root/'status.json',dict(status='COMPLETE',identity=ident))
        except BaseException:
            bridge.save_json(root/'status.json',dict(status='FAILED',identity=ident,traceback=traceback.format_exc()));raise
        finally:resources.flush()


if __name__=='__main__':main()
