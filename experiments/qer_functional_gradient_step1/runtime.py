import contextlib
import math
from pathlib import Path
import time
from bridge import (FullFitExperiment,PLAN,read,source_identity,read_tensors,save_json,
                    torch,mo,clean,capture,hidden_forward,F,save_tensors,sha_file,OFFICIAL,slug)


class BudgetReached(RuntimeError):pass


class Runtime(FullFitExperiment):
    @contextlib.contextmanager
    def timed(self,stage,**extra):
        started=time.time()
        with super().timed(stage,**extra):yield
        self.resources[-1].update(wall_started=started,wall_completed=time.time())
        save_json(self.root/'resource_records.json',self.resources)

    def __init__(self,runroot,name,deadline):
        self.runroot=Path(runroot).resolve();self.name=name;self.deadline=deadline
        self.root=self.runroot/'modules'/slug(name);self.root.mkdir(parents=True,exist_ok=True)
        identity=read(self.runroot/'identity.json');assert identity==source_identity()
        self.identity=identity['identity'];self.plan=PLAN;self.exp3=Path(PLAN['exp3_output'])
        parent=read(self.exp3/'identity.json')['plan'];self.parent=Path(parent['parent_output'])
        self.config=read(self.parent/'identity.json')['config'];assert self.config['gpus']==1
        self.model=None;self.resources=read(self.root/'resource_records.json') if (self.root/'resource_records.json').exists() else []
        self.fit,fm=read_tensors(self.exp3/'data/fit_windows.safetensors')
        assert fm['identity']==PLAN['exp3_identity'] and self.fit['input_ids'].shape==(8,2048)
        self.is_parent=name in PLAN['parent_modules'];self.checks={}
        self._last_disk_check=0.
        torch.set_num_threads(self.config['cpu_threads']);torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.manual_seed(PLAN['sample_seed'])

    def boundary(self):
        if time.time()>=self.deadline:raise BudgetReached('Frozen execution budget reached at an atomic boundary')
        if time.time()-self._last_disk_check>30:
            import shutil
            used=sum(p.stat().st_size for p in self.runroot.rglob('*') if p.is_file())
            if used>PLAN['disk_budget_GiB']*2**30:raise BudgetReached('Frozen per-run disk budget reached')
            if shutil.disk_usage(self.runroot).free<8*2**30:raise BudgetReached('Less than 8 GiB disk headroom; stop at checkpoint')
            self._last_disk_check=time.time()

    def checked_model(self):
        self.boundary()
        if self.model is not None:return
        self.doctor();self.load_model()
        actual=read(self.root/'teacher_identity.json');expected=read(self.exp3/'teacher_identity.json')
        for k in ('tensor_hashes','device_map','checkpoint_manifest_hash'):assert actual[k]==expected[k],k
        old=read(self.exp3/'environment.json');current=read(self.root/'environment.json')
        assert old['torch']==current['torch'] and old['packages']==current['packages']

    def reference(self,c):
        self.checked_model()
        with torch.no_grad(),capture(self.model.get_submodule(self.name)) as state:
            reference=hidden_forward(self.model,self.fit['input_ids'][c:c+1]).detach()
        # The hidden teacher tensor is common to every module; original hashes are pinned.
        p=self.exp3/'cache'/slug(PLAN['parent_modules'][0])/f'x_w{c:02d}.safetensors'
        from safetensors import safe_open
        with safe_open(str(p),framework='pt') as f:meta=__import__('json').loads(f.metadata()['record'])
        assert mo.digest_tensor(reference)==meta['teacher_hidden_hash']
        return reference,state['x'].detach(),state['h'].detach()

    def old_label(self,c,k):
        path=self.exp3/'data/fit_samples'/f'w{c:02d}_k{k:03d}.safetensors'
        t,m=read_tensors(path)
        assert m['identity']==PLAN['exp3_identity'] and m['label_hash']==mo.digest_tensor(t['labels'])
        assert m['input_hash']==mo.digest_tensor(self.fit['input_ids'][c])
        return t['labels'],m

    def new_samples(self,c,reference):
        import fcntl
        folder=self.runroot/'new_labels/samples';folder.mkdir(parents=True,exist_ok=True)
        with (folder/f'.window{c}.lock').open('w') as lock:
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
            paths=[folder/f'w{c:02d}_k{k:03d}.safetensors' for k in range(16)]
            missing=[k for k,p in enumerate(paths) if not p.exists()]
            if missing:
                weight=self.model.lm_head.weight
                generators={k:torch.Generator(device=weight.device).manual_seed(mo.stream_seed(c,k,PLAN['sample_seed'])) for k in missing}
                labels={k:[] for k in missing}
                with torch.no_grad():
                    for start in range(0,2047,self.config['vocab_chunk']):
                        prob=F.linear(reference[0,start:min(2047,start+self.config['vocab_chunk'])],weight).double().softmax(-1)
                        cdf=prob.cumsum(-1);cdf[:,-1]=1
                        for k in missing:
                            uniform=torch.rand((len(prob),1),dtype=torch.float64,device=weight.device,generator=generators[k])
                            labels[k].append(torch.searchsorted(cdf,uniform).flatten().cpu())
                for k in missing:
                    t=torch.cat(labels[k]);meta=dict(identity=self.identity,namespace=PLAN['sample_namespace'],window=c,replicate=k,
                        seed=mo.stream_seed(c,k,PLAN['sample_seed']),base_seed=PLAN['sample_seed'],label_hash=mo.digest_tensor(t),
                        input_hash=mo.digest_tensor(self.fit['input_ids'][c]),distribution='parent FP64 full-vocabulary inverse CDF',positions=2047)
                    save_tensors(paths[k],{'labels':t},meta)
            result=[]
            for k,p in enumerate(paths):
                t,m=read_tensors(p)
                assert m['identity']==self.identity and m['label_hash']==mo.digest_tensor(t['labels'])
                assert m['input_hash']==mo.digest_tensor(self.fit['input_ids'][c]) and m['seed']==mo.stream_seed(c,k,PLAN['sample_seed'])
                result.append((t['labels'],m))
            return result

    def quantized(self):
        path=(self.parent/'quantized'/(slug(self.name)+'.safetensors')) if self.is_parent else self.root/'quantized.safetensors'
        if not path.exists():
            assert not self.is_parent
            self.checked_model();weight=self.model.get_submodule(self.name).weight.detach()
            with self.timed('quantize_new_module'):
                quantizer=self.official('quantize/quantizers/mxint.py').mxint_quantizer
                q=quantizer(weight.float(),width=3,block_size=32,block_axis=-1)
                save_tensors(path,{'W0':weight,'Wq':q},dict(identity=self.identity,module=self.name,W0_hash=mo.digest_tensor(weight),
                    Wq_hash=mo.digest_tensor(q),width=3,block_size=32,block_axis=-1,quantizer_hash=OFFICIAL['quantize/quantizers/mxint.py']))
        t,m=read_tensors(path)
        for key in ('W0','Wq'):assert mo.digest_tensor(t[key])==m[key+'_hash']
        teacher=read(self.exp3/'teacher_identity.json')['tensor_hashes'][self.name+'.weight']
        assert m['W0_hash']==teacher['hash']
        return t,dict(path=str(path),sha256=sha_file(path),**m)

    def s_manifest(self):
        if self.is_parent:
            rows=[dict(r,path=str(self.exp3/r['relative_path'])) for r in read(self.exp3/'data/fit_sample_manifest.json')['samples'] if r['module']==self.name]
        else:rows=read(self.root/'fit_cache_manifest.json')['samples']
        assert len(rows)==32 and {(r['window'],r['replicate']) for r in rows}=={(c,k) for c in range(8) for k in range(4)}
        return rows

    def stream_s(self):
        from safetensors import safe_open
        for row in self.s_manifest():
            self.boundary();path=Path(row['path']);assert sha_file(path)==row['file_sha256']
            with safe_open(str(path),framework='pt') as f:s=f.get_tensor('S')
            assert s.dtype==torch.float64 and mo.digest_tensor(s)==row['S_hash']
            _,label=self.old_label(row['window'],row['replicate']);assert row['label_hash']==label['label_hash']
            yield row,s.cuda()

    def collect_fit(self):
        if self.is_parent:return
        if (self.root/'statistics.safetensors').exists():
            assert (self.root/'fit_cache_manifest.json').exists();return
        self.checked_model();self.quantized()
        progress=self.root/'temporary/fit_progress.safetensors'
        if progress.exists():
            t,m=read_tensors(progress);assert m['identity']==self.identity
            a,g=t['A_sum'].cuda(),t['G_sum'].cuda();start=m['completed_windows'];del t
        else:a=g=None;start=0
        for c in range(start,8):
            self.boundary();reference,x,_=self.reference(c);xd=x.reshape(2048,-1).double()
            with self.timed('fit_window_statistics',module=self.name,window=c):
                aa=xd.T@xd;a=aa if a is None else a+aa;del aa
                for k in range(4):
                    labels,lm=self.old_label(c,k)
                    gradient,audit=self.gradient(self.name,self.fit['input_ids'][c:c+1],reference,x,labels,weight_check=(c==0 and k==0))
                    s=gradient.T@xd;gg=gradient.T@gradient;g=gg if g is None else g+gg
                    path=self.root/'temporary'/f'w{c:02d}_k{k:03d}.safetensors'
                    meta=dict(identity=self.identity,module=self.name,window=c,replicate=k,T=2047,L=2048,
                        S_hash=mo.digest_tensor(s),label_hash=lm['label_hash'],input_hash=mo.digest_tensor(self.fit['input_ids'][c]),
                        x_hash=mo.digest_tensor(x),audit=audit,definition='FP64 S=sum_t g_t x_t.T; sum NLL')
                    if path.exists():
                        t,old=read_tensors(path);assert old['identity']==self.identity and old['S_hash']==meta['S_hash'];del t
                    else:save_tensors(path,{'S':s},meta)
                    save_json(path.with_suffix('.json'),dict(path=str(path),file_sha256=sha_file(path),**meta))
                    del gradient,s,gg
                save_tensors(progress,{'A_sum':a,'G_sum':g},dict(identity=self.identity,completed_windows=c+1))
            self.status('COLLECTING_FIT',module=self.name,windows=c+1,total=8)
            del reference,x,xd;clean()
        rows=[read(self.root/'temporary'/f'w{c:02d}_k{k:03d}.json') for c in range(8) for k in range(4)]
        save_json(self.root/'fit_cache_manifest.json',dict(identity=self.identity,samples=rows,A_count=8*2048,G_count=32*2048,
            reconstruct='Pinned teacher, exact original fit token/mask and saved old labels; gradient() inherited from Exp-3; complete FP64 contractions'))
        save_tensors(self.root/'statistics.safetensors',{'A_sum':a,'G_sum':g},dict(identity=self.identity,A_count=8*2048,G_count=32*2048))
        del a,g;clean()

    def candidates(self):
        frozen=read(self.root/'candidate_freeze.json');assert frozen['identity']==self.identity
        q,_=self.quantized();result={}
        result['None']=dict(W=q['Wq'],R=q['W0'].double()-q['Wq'].double())
        for name,record in frozen['candidates'].items():
            if name=='None':continue
            path=self.root/record['path'];assert sha_file(path)==record['file_sha256']
            t,m=read_tensors(path);assert m['identity']==self.identity
            result[name]=dict(W=t['W_deploy'],R=q['W0'].double()-t['W_deploy'].double())
        for name,v in result.items():
            expected=frozen['candidates'][name]
            v['R_hash']=mo.digest_tensor(v['R']);v['W_hash']=mo.digest_tensor(v['W'])
            assert v['R_hash']==expected['R_hash'] and v['W_hash']==expected['W_hash']
        return result
