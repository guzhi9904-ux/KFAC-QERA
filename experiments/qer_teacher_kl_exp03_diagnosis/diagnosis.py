#!/usr/bin/env python3
"""Read-only two-stage diagnosis of a frozen Exp-3 run."""
import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback
from types import SimpleNamespace

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
PLAN=read(HERE/'plan.json')
sys.path.insert(0,PLAN['exp3_code'])
from experiment import FullFitExperiment
import torch
import torch.nn.functional as F
import math_ops as mo
import ag_math as am
from storage import save_json,save_csv,save_tensors,read_tensors,sha_file,atomic_bytes
from run import Experiment,slug,clean,sync,Paused,request_pause
from model_ops import capture,hidden_forward
from eval_stage import EvalStage
from safetensors import safe_open
sys.path.insert(0,str(HERE))
import diag_math as dm
import diag_analysis as da


class Diagnosis(FullFitExperiment):
    gradient=FullFitExperiment.gradient
    timed=FullFitExperiment.timed

    def __init__(self,output):
        self.root=Path(output).resolve()
        assert str(self.root).startswith('/data2/cck/KFAC-QERA/runs/qer_teacher_kl_exp03_diagnosis/')
        self.root.mkdir(parents=True,exist_ok=True)
        assert not (self.root/'manifest.json').exists(),'Fresh run required; do not silently resume or overwrite'
        self.plan=PLAN;self.exp3=Path(PLAN['exp3_output'])
        self.parent_id=PLAN['exp3_identity'];self.expected=read(HERE/'parent_expected.json')
        pi=read(self.exp3/'identity.json');assert pi['identity']==self.parent_id
        assert hashlib.sha256(json.dumps({k:v for k,v in pi.items() if k!='identity'},sort_keys=True).encode()).hexdigest()==self.parent_id
        self.parent=Path(pi['plan']['parent_output'])
        self.config=read(self.parent/'identity.json')['config']
        self.model=None;self.resources=[];self.directions={};self.checks={'small_matrices':dm.small_checks()}
        source={p.name:sha_file(p) for p in sorted(HERE.iterdir()) if p.suffix in ('.py','.json','.sh','.md')}
        material=dict(plan=PLAN,source=source)
        self.identity=hashlib.sha256(json.dumps(material,sort_keys=True).encode()).hexdigest()
        save_json(self.root/'identity.json',dict(identity=self.identity,**material))
        atomic_bytes(self.root/'protocol.md',(HERE/'protocol.md').read_bytes())
        torch.set_num_threads(self.config['cpu_threads']);torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.manual_seed(PLAN['fit_seed'])
        seeds=[mo.stream_seed(c,k,PLAN['fit_seed']) for c in range(8) for k in range(16)]
        oldseeds=[mo.stream_seed(c,k,2026091803) for c in range(8) for k in range(4)]
        assert len(set(seeds))==128 and not set(seeds)&set(oldseeds)
        save_json(self.root/'manifest.json',dict(identity=self.identity,plan=PLAN,source=source,
            frozen_at=time.time(),parent_pins=self.expected,new_seeds=seeds,old_seeds=oldseeds,
            rng='torch.Generator(device=cuda:0); FP64 torch.rand and full-vocabulary inverse CDF; inherited parent implementation',
            seed_algorithm='int.from_bytes(SHA256(f"{2026091804}:{window}:{replicate}")[:8], little) % (2**63-1)',
            seed_namespace='Exp3-diagnosis base 2026091804, disjoint from fit base 2026091803',
            projection_guard='abs(d1-d2)<=1e-10*sum(abs(S*R)); exact-zero scale requires zero difference',
            J_replay_guard='abs(new-old)<=1e-8*max(1,abs(new),abs(old))',
            storage='128 real label tensors and scalar records; no persistent new S or gradient Gram'))

    def integrity(self,after=False):
        before=read(self.root/'parent_integrity_before.json') if after else None
        files={};tensors={}
        for n,h in self.expected.items():
            assert sha_file(self.exp3/n)==h,n
            files[str(self.exp3/n)]=h
            if not after:atomic_bytes(self.root/'parent_snapshot'/n,(self.exp3/n).read_bytes())
        frozen=read(self.exp3/'evaluation_freeze.json')
        assert frozen['identity']==self.parent_id and frozen['manifest_sha256']==sha_file(self.exp3/'manifest.json')
        for n,h in frozen['files'].items():
            path=self.exp3/n;assert sha_file(path)==h,n;files[str(path)]=h
            if path.suffix=='.safetensors':
                t,m=read_tensors(path);assert m['identity']==self.parent_id
                tensors[str(path)]={k:mo.digest_tensor(v) for k,v in t.items()};del t
        for code,identity in [(Path(PLAN['exp3_code']),read(self.exp3/'identity.json')),
                              (Path(PLAN['parent_code']),read(self.parent/'identity.json'))]:
            for n,h in identity['source'].items():
                path=code/n;assert sha_file(path)==h,str(path);files[str(path)]=h
        samples=read(self.exp3/'data/fit_sample_manifest.json')['samples']
        assert {(r['module'],r['window'],r['replicate']) for r in samples}=={(m,c,k) for m in PLAN['modules'] for c in range(8) for k in range(4)}
        assert len(samples)==64
        for m in PLAN['modules']:
            assert len(list((self.exp3/'cache'/slug(m)).glob('w*.safetensors')))==32
        for r in samples:
            path=self.exp3/r['relative_path'];assert sha_file(path)==r['file_sha256'];files[str(path)]=r['file_sha256']
            side=read(path.with_suffix('.json'))
            for key in ('module','window','replicate','label_hash','S_hash','file_sha256'):assert side[key]==r[key]
            files[str(path.with_suffix('.json'))]=sha_file(path.with_suffix('.json'))
            labels,lm=read_tensors(self.exp3/'data/fit_samples'/f"w{r['window']:02d}_k{r['replicate']:03d}.safetensors")
            assert mo.digest_tensor(labels['labels'])==r['label_hash']==lm['label_hash']
        for name in PLAN['modules']:
            frozen_old=read(self.parent/'directions/frozen.json')
            path=self.parent/'directions'/(slug(name)+'.safetensors')
            assert sha_file(path)==frozen_old['files'][name];files[str(path)]=sha_file(path)
            d,meta=read_tensors(path)
            for k,v in d.items():assert mo.digest_tensor(v)==meta['direction_hashes'][k]
            qpath=self.parent/'quantized'/(slug(name)+'.safetensors')
            assert sha_file(qpath)==meta['quantized_file_hash'];files[str(qpath)]=sha_file(qpath)
            q,qm=read_tensors(qpath)
            for k in ('W0','Wq'):assert mo.digest_tensor(q[k])==qm[k+'_hash']
            self.directions[name]=d
            del q
            for c in range(8):
                path=self.exp3/'cache'/slug(name)/f'x_w{c:02d}.safetensors'
                xt,xm=read_tensors(path);assert xm['identity']==self.parent_id and mo.digest_tensor(xt['x'])==xm['input_hash']
                files[str(path)]=sha_file(path);tensors[str(path)]={'x':xm['input_hash']};del xt
        self.fit,fm=read_tensors(self.exp3/'data/fit_windows.safetensors')
        assert fm['identity']==self.parent_id and self.fit['input_ids'].shape==(8,2048)
        if after:
            assert files==before['files'] and tensors==before['tensor_hashes']
            own=read(self.root/'identity.json')['source']
            for n,h in own.items():assert sha_file(HERE/n)==h,n
        save_json(self.root/('parent_integrity_after.json' if after else 'parent_integrity_before.json'),
                  dict(passed=True,parent_identity=self.parent_id,files=files,tensor_hashes=tensors,cached_samples=64,
                       scope='All frozen parent files/tensors, old labels, x; S and gradient Gram tensor hashes checked during stage A; all cache file hashes before/after',time=time.time()))
        self.samples=samples

    def candidates(self,name):
        proxy=SimpleNamespace(parent=self.parent,root=self.exp3,identity=self.parent_id,directions=self.directions)
        result=EvalStage.deployed_candidates(proxy,name)
        parentrows=read(self.exp3/'eval'/f'{slug(name)}_metric.json')['scores']
        for r in parentrows:assert result[r['candidate']]['R_hash']==r['R_hash']
        return result

    def stage_a(self):
        rows=[];objectives=[];metrics=[];audits=[];contractions=[]
        times=dict(io_and_hash_seconds=0.,projection_seconds=0.,J_contraction_seconds=0.)
        assert self.model is None
        for name in PLAN['modules']:
            candidates=self.candidates(name);rs={k:v['R'].cuda() for k,v in candidates.items()}
            factors={};terms={}
            parentrows=read(self.exp3/'eval'/f'{slug(name)}_metric.json')['scores']
            for method in ('marginal','full_fit'):
                fpath=self.exp3/'factors'/slug(name)/(method+'_solve.safetensors')
                t,meta=read_tensors(fpath)
                raw,_=read_tensors(self.exp3/'factors'/slug(name)/(method+'_raw.safetensors'))
                # Parent solve saving may include a product-preserving gauge; verify unchanged canonical products via scores.
                for variant in ('raw','solve'):
                    src=raw if variant=='raw' else t
                    a,g=src['A_'+variant].cuda(),src['G_'+variant].cuda()
                    factors[method,variant]=(a,g);terms[method,variant]=[]
                    scores={label:am.qmetric(r,a,g) for label,r in rs.items()}
                    for label,q in scores.items():
                        previous=next(r for r in parentrows if (r['metric'],r['variant'],r['candidate'])==(method,variant,label))
                        assert abs(q-previous['q_K'])<=1e-10*max(abs(q),abs(previous['q_K']),1e-12)
                        metrics.append(dict(module=name,metric=method,variant=variant,candidate=label,q_K=q,
                            gamma_K=1-q/scores['None'] if scores['None']>1e-12 else None,
                            R_hash=candidates[label]['R_hash'],factor_file_sha256=sha_file(fpath if variant=='solve' else fpath.with_name(method+'_raw.safetensors')),parent_q_K=previous['q_K']))
                    if variant=='raw':
                        r=rs['None'];other=am.qmetric(r,raw['A_raw'].cuda(),raw['G_raw'].cuda())
                        assert abs(other-scores['None'])<=1e-10*max(abs(other),1e-12)
                del t,raw
            for sample in (r for r in self.samples if r['module']==name):
                tick=time.perf_counter();path=self.exp3/sample['relative_path'];t,meta=read_tensors(path)
                assert meta['identity']==self.parent_id and mo.digest_tensor(t['S'])==sample['S_hash']
                assert mo.digest_tensor(t['gradient_gram'])==meta['gradient_gram_hash']
                assert meta['token_hash']==mo.digest_tensor(self.fit['input_ids'][sample['window']])
                s=t['S'].cuda();assert s.dtype==torch.float64;del t
                sync();times['io_and_hash_seconds']+=time.perf_counter()-tick
                tick=time.perf_counter()
                for label,r in rs.items():
                    d=float((s*r).sum());b=d*d/(2*2047);assert math.isfinite(b)
                    rows.append(dict(stage='old_fit',module=name,window=sample['window'],replicate=sample['replicate'],
                        label_hash=sample['label_hash'],input_hash=meta['token_hash'],candidate=label,R_hash=candidates[label]['R_hash'],
                        d=d,b=b,T=2047,S_hash=sample['S_hash'],source_file=sample['relative_path'],source_sha256=sample['file_sha256']))
                    if (sample['window'],sample['replicate'])==(0,0):
                        check=dm.projection_check(s,r,float(torch.dot(s.flatten(),r.flatten())))
                        audits.append(dict(stage='old_fit',module=name,candidate=label,**check))
                sync();times['projection_seconds']+=time.perf_counter()-tick
                tick=time.perf_counter()
                for (method,variant),(a,g) in factors.items():
                    value=dm.cross(s,a,g);terms[method,variant].append(value)
                    contractions.append(dict(module=name,window=sample['window'],replicate=sample['replicate'],
                        metric=method,variant=variant,trace_GSAS=value,S_hash=sample['S_hash']))
                    if (sample['window'],sample['replicate'])==(0,0):
                        alternate=float((g*(s@a@s.T)).sum())
                        assert abs(value-alternate)<=1e-10*max(abs(value),abs(alternate),1e-12)
                sync();times['J_contraction_seconds']+=time.perf_counter()-tick
                del s
                self.status('STAGE_A',completed_samples=len(rows)//5,target_samples=64,module=name)
            history=read(self.exp3/'factors'/slug(name)/'history.json')['iterations']
            for (method,variant),(a,g) in factors.items():
                norm=dm.norm_product(a,g);cross_mean=math.fsum(terms[method,variant])/32
                j=norm-2*cross_mean/2047
                previous=(history[0]['J_before'] if method=='marginal' else history[-1]['J_after_A']) if variant=='raw' else None
                if previous is not None:assert abs(j-previous)<=1e-8*max(1,abs(j),abs(previous))
                objectives.append(dict(module=name,metric=method,variant=variant,J_old=j,norm_product=norm,
                    mean_trace_GSAS=cross_mean,N=32,T=2047,parent_J=previous,replay_abs_error=None if previous is None else abs(j-previous)))
            del factors,rs,candidates,a,g;clean()
        assert len(rows)==320
        save_csv(self.root/'stage_a/old_fit_scores.csv',rows);save_csv(self.root/'stage_a/metric_objectives.csv',objectives)
        save_csv(self.root/'stage_a/metric_cross_scores.csv',metrics);save_csv(self.root/'stage_a/J_contractions.csv',contractions)
        save_json(self.root/'stage_a/results.json',dict(rows=rows,objectives=objectives,metrics=metrics,timings=times))
        self.checks['stage_a_projections']=audits;self.checks['stage_a_J_replay']=objectives
        self.checks['metric_cross_replay_passed']=True
        save_json(self.root/'stage_a/acceptance.json',dict(passed=True,identity=self.identity,model_loaded=False,
            samples=64,rows=320,checks=self.checks,timings=times))
        return rows,metrics

    def new_samples(self,c,reference):
        paths=[self.root/'stage_b/samples'/f'w{c:02d}_k{k:03d}.safetensors' for k in range(16)]
        if not any(p.exists() for p in paths):
            weight=self.model.lm_head.weight
            generators={k:torch.Generator(device=weight.device).manual_seed(mo.stream_seed(c,k,PLAN['fit_seed'])) for k in range(16)}
            labels={k:[] for k in range(16)}
            with torch.no_grad():
                for start in range(0,2047,self.config['vocab_chunk']):
                    prob=F.linear(reference[0,start:min(2047,start+self.config['vocab_chunk'])],weight).double().softmax(-1)
                    cdf=prob.cumsum(-1);cdf[:,-1]=1
                    for k in range(16):
                        u=torch.rand((len(prob),1),dtype=torch.float64,device=weight.device,generator=generators[k])
                        labels[k].append(torch.searchsorted(cdf,u).flatten().cpu())
            for k,p in enumerate(paths):
                t=torch.cat(labels[k]);save_tensors(p,{'labels':t},dict(identity=self.identity,window=c,replicate=k,
                    seed=mo.stream_seed(c,k,PLAN['fit_seed']),base_seed=PLAN['fit_seed'],label_hash=mo.digest_tensor(t),
                    input_hash=mo.digest_tensor(self.fit['input_ids'][c]),distribution='teacher FP64 full-vocabulary inverse CDF'))
        out=[]
        for p in paths:
            t,m=read_tensors(p)
            assert m['identity']==self.identity and m['input_hash']==mo.digest_tensor(self.fit['input_ids'][c])
            assert mo.digest_tensor(t['labels'])==m['label_hash'] and t['labels'].shape==(2047,)
            out.append((t['labels'],m))
        return out

    def reference(self,name,c):
        with torch.no_grad(),capture(self.model.get_submodule(name)) as state:
            reference=hidden_forward(self.model,self.fit['input_ids'][c:c+1]).detach()
        x=state['x'];old,meta=read_tensors(self.exp3/'cache'/slug(name)/f'x_w{c:02d}.safetensors')
        assert torch.equal(x.cpu(),old['x']) and mo.digest_tensor(reference)==meta['teacher_hidden_hash']
        return reference,x

    def stage_b(self):
        self.load_model()
        a=read(self.root/'teacher_identity.json');b=read(self.exp3/'teacher_identity.json')
        for k in ('tensor_hashes','device_map','checkpoint_manifest_hash'):assert a[k]==b[k]
        a=read(self.root/'environment.json');b=read(self.exp3/'environment.json')
        assert a['torch']==b['torch'] and a['packages']==b['packages']
        self.checks['teacher_and_environment_identical']=True
        allrows=[];pilot=[]
        # Both module first samples are a numerical gate, each retained exactly once.
        for phase in ('pilot','formal'):
            for name in PLAN['modules']:
                candidates=self.candidates(name)
                for c in ([0] if phase=='pilot' else range(8)):
                    reference,x=self.reference(name,c);xd=x.reshape(2048,-1).double()
                    samples=self.new_samples(c,reference)
                    rx={label:xd@v['R'].cuda().T for label,v in candidates.items()}
                    for k in ([0] if phase=='pilot' else range(16)):
                        if phase=='formal' and c==0 and k==0:continue
                        with self.timed('new_label_gradient_projection',module=name,window=c,replicate=k,pilot=phase=='pilot'):
                            g,audit=self.gradient(name,self.fit['input_ids'][c:c+1],reference,x,samples[k][0])
                            scores={}
                            if phase=='pilot':s=g.T@xd
                            for label in da.CANDIDATES:
                                d=float((g*rx[label]).sum());b=d*d/(2*2047);assert math.isfinite(b)
                                scores[label]=dict(d=d,b=b,R_hash=candidates[label]['R_hash'])
                                if phase=='pilot':
                                    check=dm.projection_check(s,candidates[label]['R'].cuda(),d)
                                    pilot.append(dict(module=name,window=c,replicate=k,candidate=label,**check))
                                allrows.append(dict(stage='new_labels',module=name,window=c,replicate=k,
                                    label_hash=samples[k][1]['label_hash'],input_hash=samples[k][1]['input_hash'],candidate=label,T=2047,**scores[label]))
                            record=dict(identity=self.identity,module=name,window=c,replicate=k,
                                label_hash=samples[k][1]['label_hash'],input_hash=samples[k][1]['input_hash'],scores=scores,audit=audit)
                            save_json(self.root/'stage_b/records'/f'{slug(name)}_w{c:02d}_k{k:03d}.json',record)
                            del g
                            if phase=='pilot':del s
                        self.status('STAGE_B_'+phase.upper(),completed_records=len(allrows)//5,target_records=256,module=name,window=c,replicate=k)
                    del reference,x,xd,rx,samples;clean()
                del candidates;clean()
            if phase=='pilot':
                save_json(self.root/'stage_b/pilot.json',dict(passed=True,identity=self.identity,checks=pilot,
                    included_once_in_256=True,projection_implementation='FP64 sum(g*(x@R.T))',frozen_source=read(self.root/'identity.json')['source']))
        self.unload()
        assert len(allrows)==1280
        self.checks['stage_b_projection_pilot']=pilot
        sm=[]
        for p in sorted((self.root/'stage_b/samples').glob('*.safetensors')):
            t,m=read_tensors(p);assert mo.digest_tensor(t['labels'])==m['label_hash']
            sm.append(dict(path=str(p.relative_to(self.root)),sha256=sha_file(p),**m))
        assert len(sm)==128
        save_json(self.root/'stage_b/sample_manifest.json',dict(identity=self.identity,samples=sm))
        save_csv(self.root/'stage_b/new_label_scores.csv',allrows)
        return allrows

    def run(self):
        started=time.time();self.doctor();self.status('VERIFYING_PARENT')
        with self.timed('parent_integrity_before'):self.integrity()
        with self.timed('stage_a_total'):old,metrics=self.stage_a()
        self.status('STAGE_A_ACCEPTED')
        new=self.stage_b()
        self.directions={};clean()
        with self.timed('parent_integrity_after'):self.integrity(after=True)
        self.checks.update(passed=True,old_samples=64,new_records=256,new_labels=128,new_score_rows=1280,
                           parent_unchanged=True,no_new_KL=True,no_factor_updates=True)
        save_json(self.root/'numerical_checks.json',self.checks)
        da.build(self.root,PLAN['modules'],old,new,metrics,read(self.exp3/'analysis.json'))
        save_json(self.root/'resource_usage.json',dict(started=started,completed=time.time(),seconds=time.time()-started,
            records=self.resources,stage_a_timings=read(self.root/'stage_a/results.json')['timings'],
            gpus=1,host_memory_GiB=64,new_S_storage_bytes=0))
        self.status('COMPUTATION_COMPLETE',old_rows=320,new_rows=1280,new_labels=128,new_records=256)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    import fcntl
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    with (root/'.run.lock').open('w') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        signal.signal(signal.SIGTERM,request_pause);signal.signal(signal.SIGINT,request_pause)
        e=Diagnosis(args.output)
        try:e.run()
        except BaseException as error:
            e.status('FAILED',error=repr(error),traceback=traceback.format_exc());raise


if __name__=='__main__':main()
