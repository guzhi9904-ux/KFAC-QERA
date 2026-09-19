#!/usr/bin/env python3
"""Exp-3 staged fitting-side pilot, frozen formal fit, then common evaluation."""
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

os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('HF_DATASETS_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1');sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
PLAN=read(HERE/'plan.json')
sys.path.insert(0,PLAN['parent_code'])
import torch
import torch.nn.functional as F
import math_ops as mo
from storage import atomic_bytes,save_json,save_tensors,read_tensors,save_csv,sha_file
from model_ops import capture,hidden_forward,recompute_suffix
from run import Experiment,slug,clean,sync,Paused,request_pause,DIRECTIONS
import fit_data
from fit_stage import FitStage
from eval_stage import EvalStage


class FullFitExperiment(FitStage,EvalStage,Experiment):
    def __init__(self,output,stage):
        self.root=Path(output).resolve();self.root.mkdir(parents=True,exist_ok=True)
        self.plan=PLAN;self.stage=stage;self.parent=Path(PLAN['parent_output']);self.extension=Path(PLAN['extension_output'])
        self.expected=read(HERE/'parent_expected.json');pi=read(self.parent/'identity.json')
        assert pi['identity']==PLAN['parent_identity']
        assert hashlib.sha256(json.dumps({k:v for k,v in pi.items() if k!='identity'},sort_keys=True).encode()).hexdigest()==pi['identity']
        self.config=pi['config'];self.model=None;self.val=None;self.cal=None;self.fit=None
        self.parent_source=pi['source'];self.directions={};self.assets={}
        self.resources=read(self.root/'resource_records.json') if (self.root/'resource_records.json').exists() else []
        source={p.name:sha_file(p) for p in sorted(HERE.iterdir()) if p.suffix in ('.py','.json','.sh','.md')}
        material={'plan':PLAN,'source':source};self.identity=hashlib.sha256(json.dumps(material,sort_keys=True).encode()).hexdigest()
        if (self.root/'identity.json').exists():assert read(self.root/'identity.json')['identity']==self.identity,'Source changed: create new version'
        else:
            save_json(self.root/'identity.json',dict(identity=self.identity,**material))
            atomic_bytes(self.root/'protocol.md',(HERE/'protocol.md').read_bytes())
            save_json(self.root/'plan.json',PLAN)
        torch.set_num_threads(self.config['cpu_threads']);torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.manual_seed(2026091803)

    @contextlib.contextmanager
    def timed(self,stage,**extra):
        with super().timed(stage,**extra):yield
        base=Path('/sys/fs/cgroup')/Path('/proc/self/cgroup').read_text().split('0::',1)[1].strip().lstrip('/')
        self.resources[-1]['cgroup_memory']={n:(base/n).read_text().strip() for n in ('memory.current','memory.peak','memory.max') if (base/n).exists()}
        save_json(self.root/'resource_records.json',self.resources)

    def check_assets(self,final=False):
        for name,h in self.parent_source.items():assert sha_file(Path(self.plan['parent_code'])/name)==h
        for group,base in (('parent',self.parent),('extension',self.extension)):
            for name,h in self.expected[group].items():
                assert sha_file(base/name)==h,group+'/'+name
                dst=self.root/(group+'_snapshot')/name
                if dst.exists():assert sha_file(dst)==h
                else:atomic_bytes(dst,(base/name).read_bytes())
        assert read(self.extension/'identity.json')['identity']==PLAN['extension_identity']
        self.val,meta=read_tensors(self.parent/'data/validation.safetensors')
        assert meta['identity']==self.plan['parent_identity'] and self.val['input_ids'].shape==(8,2048)
        assert [mo.digest_tensor(x) for x in self.val['input_ids']]==meta['window_tensor_hashes']
        frozen=read(self.parent/'directions/frozen.json')
        for name in self.plan['modules']:
            p=self.parent/'directions'/(slug(name)+'.safetensors');assert sha_file(p)==frozen['files'][name]
            d,dm=read_tensors(p);q,qm=read_tensors(self.parent/'quantized'/(slug(name)+'.safetensors'))
            assert sha_file(self.parent/'quantized'/(slug(name)+'.safetensors'))==dm['quantized_file_hash']
            for k,v in d.items():assert mo.digest_tensor(v)==dm['direction_hashes'][k]
            for k in ('W0','Wq'):assert mo.digest_tensor(q[k])==qm[k+'_hash']
            assert torch.equal((q['W0'].double()-q['Wq'].double()).float(),d['R_none'])
            self.directions[name]=d;self.assets[name]={'direction_file_sha256':sha_file(p),'quantized_file_sha256':dm['quantized_file_hash'],'metadata':dm}
            del q
        save_json(self.root/('parent_integrity.json' if final else 'parent_verification.json'),
                  {'identity':self.identity,'passed':True,'checked_files':self.expected,'assets':self.assets,'final':final})

    def verify_teacher(self):
        a=read(self.parent/'teacher_identity.json');b=read(self.root/'teacher_identity.json')
        assert a['tensor_hashes']==b['tensor_hashes'] and a['device_map']==b['device_map']
        assert a['checkpoint_manifest_hash']==b['checkpoint_manifest_hash']
        a=read(self.parent/'environment.json');b=read(self.root/'environment.json')
        assert a['torch']==b['torch'] and a['packages']==b['packages']
        save_json(self.root/'teacher_verification.json',{'identity':self.identity,'passed':True,'tensor_hashes_equal':True,'versions_equal':True,'device_map_equal':True})

    def fitting_samples(self,c,k,reference):
        folder=self.root/'data/fit_samples';paths=[folder/f'w{c:02d}_k{j:03d}.safetensors' for j in range(k)]
        missing=[j for j,p in enumerate(paths) if not p.exists()]
        if missing:
            weight=self.model.lm_head.weight
            generators={j:torch.Generator(device=weight.device).manual_seed(mo.stream_seed(c,j,self.plan['fit_seed'])) for j in missing}
            labels={j:[] for j in missing}
            with torch.no_grad():
                for start in range(0,2047,self.config['vocab_chunk']):
                    prob=F.linear(reference[0,start:min(2047,start+self.config['vocab_chunk'])],weight).double().softmax(-1)
                    cdf=prob.cumsum(-1);cdf[:,-1]=1
                    for j in missing:
                        u=torch.rand((len(prob),1),dtype=torch.float64,device=weight.device,generator=generators[j])
                        labels[j].append(torch.searchsorted(cdf,u).flatten().cpu())
            for j in missing:
                t=torch.cat(labels[j]);save_tensors(paths[j],{'labels':t},{'identity':self.identity,'window':c,'replicate':j,
                      'seed':mo.stream_seed(c,j,self.plan['fit_seed']),'label_hash':mo.digest_tensor(t),
                      'input_hash':mo.digest_tensor(self.fit['input_ids'][c]),'distribution':'teacher FP64 full-vocabulary inverse CDF'})
        out=[]
        for p in paths:
            t,m=read_tensors(p);assert m['identity']==self.identity and mo.digest_tensor(t['labels'])==m['label_hash']
            out.append((t['labels'],m['label_hash']))
        return out

    def gradient(self,name,ids,reference,x,labels,weight_check=False):
        target=self.model.get_submodule(name);layer=int(name.split('.')[2]);audit={}
        target.weight.requires_grad_(weight_check)
        try:
            with recompute_suffix(self.model,layer,True),capture(target,not weight_check) as state:
                hidden=hidden_forward(self.model,ids)
                assert torch.equal(state['x'],x)
                hidden_error=mo.relative(hidden.detach(),reference)
                assert hidden_error<=1e-7
                seed=mo.sampled_seed(hidden.detach(),self.model.lm_head.weight,labels,self.config['vocab_chunk'])
                inputs=(state['h'],target.weight) if weight_check else (state['h'],)
                grads=torch.autograd.grad(hidden,inputs,grad_outputs=seed)
            g=grads[0].detach().reshape(2048,-1).double()
            if weight_check:
                s=g.T@x.reshape(2048,-1).double()
                difference=mo.relative(s,grads[1].double())
                if difference>self.plan['S_autograd_relative_tolerance']:raise RuntimeError('S shared-weight gradient mismatch')
                audit['S_vs_autograd_weight_relative_error']=difference
            audit['teacher_hidden_relative_error']=hidden_error
            return g,audit
        finally:target.weight.requires_grad_(False)

    def run_stage(self):
        self.doctor();atomic_bytes(self.root/('environment_'+self.stage+'.json'),(self.root/'environment.json').read_bytes())
        self.check_assets();fit_data.prepare(self)
        if self.stage=='pilot':
            self.load_model();self.verify_teacher();self.collect_fit(1,2,pilot=True);self.unload();self.pilot_linear_algebra()
            return
        manifest=read(self.root/'manifest.json')
        assert manifest['identity']==self.identity and manifest['plan']==self.plan and manifest['pilot_sha256']==sha_file(self.root/'pilot.json')
        assert read(self.root/'pilot.json')['passed']
        self.load_model();self.verify_teacher();self.collect_fit(self.plan['N_fit'],self.plan['K_fit']);self.unload()
        self.fit_and_solve();self.freeze_candidates()
        self.load_model();self.verify_teacher();self.evaluate();self.unload();self.check_assets(final=True)
        from analysis import report
        report(self)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--stage',choices=['pilot','formal'],required=True)
    args=p.parse_args();root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (root/'.run.lock').open('w') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        signal.signal(signal.SIGTERM,request_pause);signal.signal(signal.SIGINT,request_pause)
        e=FullFitExperiment(args.output,args.stage)
        try:e.run_stage()
        except Paused as error:e.status('PAUSED',reason=str(error));return 75
        except BaseException as error:e.status('FAILED',error=repr(error),traceback=traceback.format_exc());raise
    return 0


if __name__=='__main__':sys.exit(main())
