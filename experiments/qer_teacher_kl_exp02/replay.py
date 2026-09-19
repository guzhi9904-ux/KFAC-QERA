#!/usr/bin/env python3
"""Replay fixed teacher labels; diagnose shared-position and marginal-product structure."""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('HF_DATASETS_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
sys.dont_write_bytecode=True


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();spec=read(args.config);here=Path(__file__).resolve().parent
    parent=Path(spec['parent_output']);extension=Path(spec['extension_output']);code=Path(spec['parent_code'])
    expected=read(here/'parent_expected.json');pi=read(parent/'identity.json')
    material={k:v for k,v in pi.items() if k!='identity'}
    assert hashlib.sha256(json.dumps(material,sort_keys=True).encode()).hexdigest()==spec['parent_identity']==pi['identity']
    for n,h in pi['source'].items():assert digest(code/n)==h, 'Frozen parent source changed: '+n
    for source,base in (('parent',parent),('extension',extension)):
        for n,h in expected[source].items():assert digest(base/n)==h, 'Source asset changed: '+source+'/'+n
    assert read(extension/'identity.json')['identity']==spec['extension_identity']
    sys.path.insert(0,str(code))
    import numpy as np
    import torch
    import math_ops as mo
    from storage import atomic_bytes,save_json,save_tensors,read_tensors,save_csv
    from model_ops import capture,recompute_suffix,hidden_forward
    from run import Experiment,MODULES,DIRECTIONS,OFFICIAL,slug,clean,sync,Paused,request_pause
    from score_ops import full_pos,direct_sep,gram_sep,validate_scores,relative_scalar
    from stats_ops import estimate
    assert spec['K']==64 and spec['L']==2048 and spec['T']==2047 and spec['gpus']==1
    assert spec['bootstrap_count']==2000 and spec['bootstrap_seed']==2026091802
    assert spec['ratio_denominator_floor']==1e-12 and spec['score_error_band_fraction_q0']==.1
    assert spec['benefit_error_band_fraction_delta0']==.2 and spec['gamma_difference_band']==.05

    class StructureExperiment(Experiment):
        def __init__(self):
            self.root=Path(args.output).resolve();self.root.mkdir(parents=True,exist_ok=True)
            for base in (parent,extension):
                if self.root==base.resolve() or base.resolve() in self.root.parents:raise RuntimeError('Separate output required')
            self.config=pi['config'];self.spec=spec;self.model=None;self.cal=None;self.val=None
            self.resources=read(self.root/'resource_records.json') if (self.root/'resource_records.json').exists() else []
            source={p.name:digest(p) for p in sorted(here.iterdir()) if p.suffix in {'.py','.sh','.json','.md'}}
            id_data={'spec':spec,'source':source,'parent_identity':spec['parent_identity'],'extension_identity':spec['extension_identity']}
            self.identity=hashlib.sha256(json.dumps(id_data,sort_keys=True).encode()).hexdigest()
            if (self.root/'identity.json').exists():assert read(self.root/'identity.json')['identity']==self.identity
            else:
                save_json(self.root/'identity.json',dict(identity=self.identity,**id_data))
                atomic_bytes(self.root/'protocol.md',(here/'protocol.md').read_bytes())
                import yaml
                atomic_bytes(self.root/'config.resolved.yaml',yaml.safe_dump(spec,sort_keys=False).encode())
            torch.set_num_threads(self.config['cpu_threads']);torch.set_float32_matmul_precision('highest')
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.manual_seed(20260918)
            self.directions={};self.labels={};self.parent_records={}
            self.impl=read(self.root/'implementation.json') if (self.root/'implementation.json').exists() else {}

        def verify_assets(self):
            self.status('VERIFYING_ASSETS')
            for source,base in (('parent',parent),('extension',extension)):
                for name,sha in expected[source].items():
                    target=self.root/(source+'_snapshot')/name
                    if target.exists():assert digest(target)==sha
                    else:atomic_bytes(target,(base/name).read_bytes())
            self.val,vm=read_tensors(parent/'data/validation.safetensors')
            assert vm['identity']==spec['parent_identity'] and vm['source_split']=='validation'
            assert self.val['input_ids'].shape==(8,2048) and self.val['attention_mask'].all()
            assert [mo.digest_tensor(t) for t in self.val['input_ids']]==vm['window_tensor_hashes']
            frozen=read(parent/'directions/frozen.json');assets={}
            teacher=read(parent/'teacher_identity.json')
            for name in MODULES:
                path=parent/'directions'/(slug(name)+'.safetensors');assert digest(path)==frozen['files'][name]
                tensors,meta=read_tensors(path)
                assert meta['identity']==spec['parent_identity'] and set(tensors)==set(DIRECTIONS)
                for d,r in tensors.items():assert r.dtype==torch.float32 and mo.digest_tensor(r)==meta['direction_hashes'][d];mo.finite(r)
                qp=parent/'quantized'/(slug(name)+'.safetensors');assert digest(qp)==meta['quantized_file_hash']
                quant,qm=read_tensors(qp)
                assert qm['identity']==spec['parent_identity'] and (qm['width'],qm['block_size'],qm['block_axis'])==(3,32,-1)
                assert qm['quantizer_hash']==OFFICIAL['quantize/quantizers/mxint.py']
                for k in ('W0','Wq'):assert mo.digest_tensor(quant[k])==qm[k+'_hash']
                assert qm['W0_hash']==teacher['tensor_hashes'][name+'.weight']['hash']
                self.directions[name]=tensors;assets[name]={'file_hash':frozen['files'][name],'metadata':meta,'quantization':qm}
                del quant
            for c in range(8):
                for k in range(64):
                    path=parent/'samples'/f'w{c:02d}_k{k:03d}.safetensors'
                    tensors,meta=read_tensors(path);labels=tensors['labels']
                    assert meta['identity']==spec['parent_identity'] and (meta['window'],meta['replicate'])==(c,k)
                    assert labels.shape==(2047,) and mo.digest_tensor(labels)==meta['label_hash']
                    self.labels[(c,k)]=(labels,meta['label_hash'])
                    for name in MODULES:
                        r=read(parent/'records/mc'/f'{slug(name)}_w{c:02d}_k{k:03d}.json')
                        assert r['identity']==spec['parent_identity'] and r['label_hash']==meta['label_hash'] and r['T']==2047
                        for d in DIRECTIONS:assert r['directions'][d]['R_hash']==assets[name]['metadata']['direction_hashes'][d]
                        self.parent_records[(name,c,k)]=r
            self.assets=assets
            save_json(self.root/'parent_verification.json',dict(identity=self.identity,parent_identity=spec['parent_identity'],
                extension_identity=spec['extension_identity'],verified_files=expected,labels=512,parent_MC_records=1024,
                directions=assets,validation_metadata=vm))

        def verify_teacher(self):
            old,new=read(parent/'teacher_identity.json'),read(self.root/'teacher_identity.json')
            assert old['tensor_hashes']==new['tensor_hashes'] and old['device_map']==new['device_map']
            assert old['checkpoint_manifest_hash']==new['checkpoint_manifest_hash']
            oldenv,newenv=read(parent/'environment.json'),read(self.root/'environment.json')
            assert oldenv['torch']==newenv['torch'] and oldenv['packages']==newenv['packages']
            save_json(self.root/'teacher_verification.json',dict(identity=self.identity,all_tensor_hashes_match=True,device_map_match=True,versions_match=True))

        def collect_diagnostic_A(self):
            folder=self.root/'diagnostic_A';progress=folder/'progress.safetensors'
            if (folder/'complete.json').exists():
                self.a_audits=read(folder/'complete.json')['modules'];return
            self.status('COLLECTING_DIAGNOSTIC_A')
            if progress.exists():
                sums,meta=read_tensors(progress);assert meta['identity']==self.identity
                assert meta['completed']==list(range(len(meta['completed']))) and meta['count']==len(meta['completed'])*2048
            else:
                sums={slug(n):torch.zeros((4096,4096),dtype=torch.float64) for n in MODULES}
                meta={'identity':self.identity,'completed':[],'count':0,'input_hashes':{n:[] for n in MODULES},'hidden_hashes':[],'SSE_audits':{}}
            for c in range(len(meta['completed']),8):
                with self.timed('diagnostic_A_forward_and_gram',window=c):
                    with torch.no_grad(),contextlib.ExitStack() as stack:
                        states={n:stack.enter_context(capture(self.model.get_submodule(n))) for n in MODULES}
                        hidden=hidden_forward(self.model,self.val['input_ids'][c:c+1])
                        for name,state in states.items():
                            x=state['x'];assert x.shape==(1,2048,4096)
                            sums[slug(name)].add_(mo.gram(x,256,double_product=True).cpu())
                            meta['input_hashes'][name].append(mo.digest_tensor(x))
                            if c==0:
                                sample=x.reshape(-1,4096)[:64].double();metric=sample.T@sample/64
                                audit={}
                                for d,r in self.directions[name].items():
                                    rd=r.to(sample.device).double()
                                    direct=float((sample@rd.T).square().sum()/64)
                                    trace=mo.objective(rd,metric);error=relative_scalar(trace,direct)
                                    if error>1e-10:raise RuntimeError('Diagnostic A SSE/trace failed')
                                    audit[d]={'direct_SSE':direct,'trace':trace,'relative_error':error}
                                meta['SSE_audits'][name]=audit
                                del sample,metric,rd
                        meta['hidden_hashes'].append(mo.digest_tensor(hidden))
                    meta['completed'].append(c);meta['count']+=2048
                    save_tensors(progress,sums,meta)
                    del states,state,x,hidden;clean()
                self.status('COLLECTING_DIAGNOSTIC_A',completed=c+1,count=meta['count'])
            self.a_audits={}
            for name in MODULES:
                raw=sums[slug(name)]/meta['count'];symmetric=(raw+raw.T)/2
                audit={'identity':self.identity,'module':name,'count':meta['count'],'N':8,'L':2048,
                    'dtype':'float64','centered':False,'regularization':'none','symmetry_error':mo.relative(raw,raw.T),
                    'symmetrization_change':mo.relative(symmetric,raw),'input_hashes':meta['input_hashes'][name],
                    'teacher_hidden_hashes':meta['hidden_hashes'],'A_hash':mo.digest_tensor(symmetric),
                    'sum_hash':mo.digest_tensor(sums[slug(name)]),'SSE_audit':meta['SSE_audits'][name]}
                assert audit['count']==16384 and audit['symmetrization_change']<=1e-10
                save_tensors(folder/(slug(name)+'_sum.safetensors'),{'S_A':sums[slug(name)]},{'identity':self.identity,'count':16384})
                save_tensors(folder/(slug(name)+'_A.safetensors'),{'A_diag':symmetric},audit)
                save_json(folder/(slug(name)+'_audit.json'),audit);self.a_audits[name]=audit
            save_json(folder/'counts.json',{'identity':self.identity,'module_counts':{n:16384 for n in MODULES},'unique_windows':list(range(8))})
            save_json(folder/'complete.json',{'identity':self.identity,'modules':self.a_audits})
            del sums,raw,symmetric;clean()

        def construct_metrics(self):
            self.status('CONSTRUCTING_DIRECTION_METRICS')
            for name in MODULES:
                path=self.root/'direction_metrics'/(slug(name)+'.safetensors')
                if path.exists():
                    _,meta=read_tensors(path);assert meta['identity']==self.identity and meta['A_hash']==self.a_audits[name]['A_hash'];continue
                with self.timed('construct_M_R',module=name):
                    device=self.model.get_submodule(name).weight.device
                    tensors,am=read_tensors(self.root/'diagnostic_A'/(slug(name)+'_A.safetensors'))
                    assert am['identity']==self.identity and mo.digest_tensor(tensors['A_diag'])==am['A_hash']==self.a_audits[name]['A_hash']
                    a=tensors['A_diag'].to(device);metrics={};audits={}
                    for d,r in self.directions[name].items():
                        sync();start=time.perf_counter();rd=r.to(device).double();m=mo.finite(rd@a@rd.T);sync()
                        elapsed=time.perf_counter()-start
                        symmetry=mo.relative(m,m.T)
                        if symmetry>1e-10:raise RuntimeError('M_R symmetry audit failed')
                        audits[d]={'shape':list(m.shape),'bytes':m.numel()*m.element_size(),'symmetry_error':symmetry,
                                   'construction_seconds':elapsed,'M_hash':mo.digest_tensor(m),
                                   'R_hash':self.assets[name]['metadata']['direction_hashes'][d]}
                        metrics[d]=m.cpu();del rd,m
                    meta={'identity':self.identity,'module':name,'A_hash':am['A_hash'],'A_count':16384,'dtype':'float64',
                          'formula':'R @ pooled_A_diag @ R.T','directions':audits}
                    save_tensors(path,metrics,meta);save_json(path.with_suffix('.json'),meta)
                    del metrics,tensors,a;clean()

        def context(self,name,window):
            with self.timed('teacher_and_Rx_cache',module=name,window=window):
                reference,x,h=self.teacher_reference(name,window)
                assert mo.digest_tensor(x)==self.a_audits[name]['input_hashes'][window]
                assert mo.digest_tensor(reference)==self.a_audits[name]['teacher_hidden_hashes'][window]
                rx={d:x.reshape(2048,-1).double()@r.to(x.device).double().T for d,r in self.directions[name].items()}
                del h
            return reference,x,rx

        def load_metrics(self,name):
            tensors,meta=read_tensors(self.root/'direction_metrics'/(slug(name)+'.safetensors'))
            assert meta['identity']==self.identity and meta['A_hash']==self.a_audits[name]['A_hash']
            for d,t in tensors.items():assert mo.digest_tensor(t)==meta['directions'][d]['M_hash']
            device=self.model.get_submodule(name).weight.device
            return {d:t.to(device) for d,t in tensors.items()},meta

        def unit(self,name,c,k,reference,x,rx,metrics,metric_meta):
            path=self.root/'records'/f'{slug(name)}_w{c:02d}_k{k:03d}.json'
            labels,label_hash=self.labels[(c,k)]
            if path.exists():
                r=self.read_record(path)
                assert (r['module'],r['window'],r['replicate'],r['label_hash'])==(name,c,k,label_hash)
                assert set(r['directions'])==set(DIRECTIONS) and r['L']==2048 and r['T']==2047
                assert r['implementation']==self.impl[name]['selected']
                for d in DIRECTIONS:
                    assert r['directions'][d]['R_hash']==metric_meta['directions'][d]['R_hash']
                    assert r['directions'][d]['A_hash']==metric_meta['A_hash']
                return r
            with self.timed('replay_unit',module=name,window=c,replicate=k):
                sync();started=time.perf_counter();timings={}
                target=self.model.get_submodule(name);layer=int(name.split('.')[2])
                with recompute_suffix(self.model,layer,True),capture(target,True) as state:
                    tick=time.perf_counter();hidden=hidden_forward(self.model,self.val['input_ids'][c:c+1]);sync()
                    timings['forward_seconds']=time.perf_counter()-tick
                    if not torch.equal(state['x'],x):raise RuntimeError('Replay input changed')
                    hidden_error=mo.relative(hidden.detach(),reference)
                    if hidden_error>1e-7:raise RuntimeError('Replay teacher hidden changed')
                    tick=time.perf_counter()
                    seed=mo.sampled_seed(hidden.detach(),self.model.lm_head.weight,labels,self.config['vocab_chunk']);sync()
                    timings['seed_seconds']=time.perf_counter()-tick
                    tick=time.perf_counter();gradient=torch.autograd.grad(hidden,state['h'],grad_outputs=seed)[0];sync()
                    timings['backward_seconds']=time.perf_counter()-tick
                g=mo.finite(gradient.detach().reshape(2048,-1).double())
                del hidden,seed,gradient,state
                tick=time.perf_counter()
                projected={d:full_pos(g,rx[d],2047) for d in DIRECTIONS};sync()
                timings['full_pos_contraction_seconds']=time.perf_counter()-tick
                if name not in self.impl:
                    times={'direct':[],'gram':[]};answers={}
                    for repeat in range(2):
                        for method,function in (('direct',direct_sep),('gram',gram_sep)):
                            sync();tick=time.perf_counter();answers[method]=function(g,metrics,2047);sync()
                            times[method].append(time.perf_counter()-tick)
                    errors={d:relative_scalar(answers['gram'][d],answers['direct'][d]) for d in DIRECTIONS}
                    if max(errors.values())>1e-8:raise RuntimeError('Actual-gradient sep equivalence failed')
                    selected=min(times,key=lambda method:float(np.median(times[method])))
                    self.impl[name]={'selected':selected,'numeric_version':'exp02_fp64_full_channels_v1',
                        'benchmark_seconds':times,'direct_values':answers['direct'],'gram_values':answers['gram'],
                        'relative_errors':errors,'pilot_window':c,'pilot_replicate':k}
                    save_json(self.root/'implementation.json',self.impl)
                function=direct_sep if self.impl[name]['selected']=='direct' else gram_sep
                tick=time.perf_counter();separated=function(g,metrics,2047);sync()
                timings['sep_contraction_seconds']=time.perf_counter()-tick
                old=self.parent_records[(name,c,k)];scores={}
                for d in DIRECTIONS:
                    signed,full,pos=projected[d];sep=separated[d]
                    validate_scores([full,pos,sep],max(full,pos,abs(sep),1e-12))
                    ep,ed,et=pos-full,sep-pos,sep-full
                    if abs(et-ep-ed)>1e-12*max(abs(full),abs(pos),abs(sep),1e-30):raise RuntimeError('Error decomposition failed')
                    scores[d]={'d':signed,'b_full':full,'b_pos':pos,'b_sep':sep,'e_position':ep,
                        'e_dependence':ed,'e_total':et,'parent_d':old['directions'][d]['d'],
                        'parent_d_absolute_error':abs(signed-old['directions'][d]['d']),
                        'R_hash':metric_meta['directions'][d]['R_hash'],'A_hash':metric_meta['A_hash'],
                        'M_hash':metric_meta['directions'][d]['M_hash']}
                row={'identity':self.identity,'parent_identity':spec['parent_identity'],'module':name,'window':c,'replicate':k,
                    'L':2048,'T':2047,'label_hash':label_hash,'input_hash':self.a_audits[name]['input_hashes'][c],
                    'directions':scores,'loss_reduction':'sum','full_reduction':'square_after_sum_all_module_positions',
                    'pos_reduction':'sum_squared_position_projections','sep_reduction':'sum_g_M_g_div_2T_pooled_A_NL',
                    'implementation':self.impl[name]['selected'],'numeric_version':'exp02_fp64_full_channels_v1',
                    'teacher_hidden_relative_difference':hidden_error,'timings':timings,
                    'elapsed_seconds':time.perf_counter()-started}
                save_json(path,row);del g
            self.status('REPLAYING_K64',module=name,window=c,replicate=k,
                        committed_records=len(list((self.root/'records').glob('*.json'))))
            return row

        def projection_audit(self,records,pilot=False):
            audits=[]
            for name in MODULES:
                selected=sorted((r for r in records if r['module']==name),key=lambda r:(r['window'],r['replicate']))
                expected_keys=[(0,k) for k in range(4)] if pilot else [(c,k) for c in range(8) for k in range(64)]
                assert [(r['window'],r['replicate']) for r in selected]==expected_keys
                for d in DIRECTIONS:
                    actual=np.array([r['directions'][d]['d'] for r in selected],dtype=np.float64)
                    reference=np.array([self.parent_records[(name,r['window'],r['replicate'])]['directions'][d]['d'] for r in selected],dtype=np.float64)
                    norm=float(np.linalg.norm(reference));error=float(np.linalg.norm(actual-reference))
                    relative=error/norm if norm else error
                    row={'module':name,'direction':d,'records':len(selected),'d_relative_l2':relative,
                         'd_absolute_l2':error,'parent_d_norm':norm,'max_d_absolute_error':float(np.max(np.abs(actual-reference))),
                         'passed':relative<=1e-5}
                    if pilot:row['per_item_absolute_errors']=np.abs(actual-reference).tolist()
                    else:
                        full=float(np.mean([r['directions'][d]['b_full'] for r in selected]))
                        baseline=next(r['q_hat'] for r in read(parent/'summary.json') if r['module']==name and r['direction']==d)
                        q_error=relative_scalar(full,baseline)
                        row.update(q_full=full,parent_q_hat=baseline,q_relative_error=q_error,passed=row['passed'] and q_error<=1e-5)
                    audits.append(row)
            return audits

        def pilot_structure(self):
            if (self.root/'pilot.json').exists():
                assert read(self.root/'pilot.json')['passed'];return
            self.status('REAL_GRADIENT_PILOT')
            records=[]
            for name in MODULES:
                metrics,meta=self.load_metrics(name);reference,x,rx=self.context(name,0)
                for k in range(4):records.append(self.unit(name,0,k,reference,x,rx,metrics,meta))
                del metrics,reference,x,rx;clean()
            audits=self.projection_audit(records,True);passed=all(a['passed'] for a in audits)
            estimates={name:512*float(np.mean([r['elapsed_seconds'] for r in records if r['module']==name])) for name in MODULES}
            save_json(self.root/'pilot.json',{'identity':self.identity,'passed':passed,'projection_audits':audits,
                'sep_implementation':self.impl,'estimated_1024_units_seconds':sum(estimates.values()),
                'module_estimates_seconds':estimates,'estimate_scope':'Pilot mean unit time times 512 per module; excludes setup and window reference/cache, includes initial benchmarks.'})
            if not passed:raise RuntimeError('Pilot signed projection replay failed')

        def replay_all(self):
            for name in MODULES:
                metrics,meta=self.load_metrics(name)
                for c in range(8):
                    reference,x,rx=self.context(name,c)
                    for k in range(64):self.unit(name,c,k,reference,x,rx,metrics,meta)
                    del reference,x,rx;clean()
                del metrics;clean()

        def finalize_audit(self):
            records=[self.read_record(p) for p in sorted((self.root/'records').glob('*.json'))]
            assert len(records)==1024
            audits=self.projection_audit(records)
            passed=all(a['passed'] for a in audits)
            save_json(self.root/'numerical_audit.json',{'identity':self.identity,'passed':passed,'formal_projection_audits':audits,
                'sep_equivalence':self.impl,'diagnostic_A':self.a_audits})
            if not passed:raise RuntimeError('Final full projection replay audit failed; structure interpretation withheld')
            for source,base in (('parent',parent),('extension',extension)):
                for n,h in expected[source].items():assert digest(base/n)==h,'Parent asset changed'
            for n,h in pi['source'].items():assert digest(code/n)==h
            for name,a in self.assets.items():
                assert digest(parent/'directions'/(slug(name)+'.safetensors'))==a['file_hash']
                assert digest(parent/'quantized'/(slug(name)+'.safetensors'))==a['metadata']['quantized_file_hash']
            save_json(self.root/'parent_integrity.json',dict(identity=self.identity,before_equals_after=True,
                checked_files=expected,frozen_code_unchanged=True,directions_and_quantized_unchanged=True))
            return records

        def run_structure(self):
            self.doctor();self.verify_assets();self.load_model();self.verify_teacher()
            self.collect_diagnostic_A();self.construct_metrics();self.pilot_structure();self.replay_all()
            self.unload();records=self.finalize_audit()
            from report import write_report
            write_report(self,records)

    return execute(StructureExperiment, args, Paused, request_pause)


def execute(experiment_class,args,Paused,request_pause):
    import fcntl
    root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=True)
    with (root/'.run.lock').open('w') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        signal.signal(signal.SIGTERM,request_pause);signal.signal(signal.SIGINT,request_pause)
        experiment=experiment_class()
        try:experiment.run_structure()
        except Paused as error:
            experiment.status('PAUSED',reason=str(error));return 75
        except BaseException as error:
            experiment.status('FAILED',error=repr(error),traceback=traceback.format_exc());raise
    return 0


if __name__=='__main__':sys.exit(main())
