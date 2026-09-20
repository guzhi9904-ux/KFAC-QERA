"""Independent exact FP64 fits on the shared sample cache; two device workers."""
import gc
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
import torch
from bridge import LegacyExperiment,TensorStore,read,save_json,save_tensors,read_tensors,mo,sm,require,slug,commit,load_record,file_table,checked_files,warm_and_probe,digest
from common import PLAN
from collection import x_path,g_path
from resources import DeviceResources


def compact_moments(groups,T,device):
    aa=gg=None;count=0
    for x,zs in groups:
        x=x.to(device,dtype=torch.float64);xx=x@x.T;zz=None
        for z in zs:
            z=z.to(device,dtype=torch.float64);term=z@z.T;zz=term if zz is None else zz+term
            term=z.T@xx@z;gg=term if gg is None else gg+term;count+=1
        require(zz is not None,'Empty gradient group');term=x.T@zz@x;aa=term if aa is None else aa+term
    require(count>0,'Empty moment stream');return sm.sym(aa/(count*T)),sm.sym(gg/(count*T)),count


class ModuleFit(LegacyExperiment):
    def __init__(self,root,config,identity,resources,name,device,stop=None):
        self.runroot=root;self.root=root/'modules'/slug(name);self.root.mkdir(parents=True,exist_ok=True)
        self.config=config;self.identity=identity;self.name=name;self.device=device
        self.resources=DeviceResources(resources,device,name,stop);self.store=TensorStore(identity,config['cache_GiB'])
        self.index=read(root/'data/index.json');require(self.index['identity']==identity,'Sample index mismatch')
        self.entries={r['id']:r for r in self.index['fit']}
        self.quantized=self.store.get(root/'quantized'/(slug(name)+'.safetensors'))[0]
        self.error=self.quantized['W0'].double()-self.quantized['Wq'].double()
    def path(self,kind,key):
        if kind=='fit_x':return x_path(self.runroot,self.name,int(key[1:]))
        if kind=='fit_g':return g_path(self.runroot,self.name,key)
        raise RuntimeError('Dense S cache and evaluation gradients are excluded from main workflow')
    def s(self,key,role='fit'):
        require(role=='fit','No evaluation gradient cache')
        row=self.entries[key];x=self.store.get(x_path(self.runroot,self.name,row['window']))[0]['x'].reshape(PLAN['L'],-1)
        g=self.store.get(g_path(self.runroot,self.name,key))[0]['g']
        return g.double().T@x.double()
    def moments(self,keys,tag,role='fit'):
        require(role=='fit','No evaluation gradient cache');p=self.root/'statistics'/('moments_'+tag+'.safetensors')
        if p.exists():return self.store.get(p)[0]
        with self.resources.timed('sequence_moments',budget=tag,samples=len(keys)):
            a,g,n=compact_moments(self.grouped_tokens(keys,role),PLAN['T'],self.device)
            self.store.put(p,dict(StS=a,SSt=g),samples=keys,N=n,T=PLAN['T'])
        return dict(StS=a.cpu(),SSt=g.cpu())
    def candidate(self,key,a,g,budget,method):
        cp=self.root/'corrections'/(key+'.safetensors')
        if cp.exists():self.store.get(cp);return
        with self.resources.timed('weighted_SVD',candidate=key):
            _,c,audit=sm.solve(self.error.to(self.device),a.to(self.device),g.to(self.device),PLAN['rank'],self.quantized['Wq'],self.quantized['W0'])
            # R64/Rdeploy and damped factors can be reconstructed exactly from
            # W0/Wq/P64/Q64 and saved raw factors; main evaluation needs W only.
            self.store.put(cp,{k:c[k] for k in ('P64','Q64','W_deploy')},candidate=key,budget=budget,method=method,
                audit=audit,W_hash=mo.digest_tensor(c['W_deploy']),raw_A_hash=mo.digest_tensor(a),raw_G_hash=mo.digest_tensor(g))
    def raw_factors(self,budget,method):
        # Same update equations/stopping as frozen L10; only checkpoint retention
        # changes. One atomic current state replaces dozens of huge A snapshots.
        folder=self.root/'factors'/budget/method;p=folder/'raw.safetensors';cp=folder/'progress.safetensors'
        if p.exists():return self.store.get(p)[0]
        keys=self.index['budgets'][budget];stats=self.marginal_stats(keys,budget)
        a=stats['A_m'].to(self.device);g=stats['G_canonical'].to(self.device)
        history=[];stable=0;start=1;stop='FIXED_CONSTRUCTION'
        if cp.exists():
            t,m=read_tensors(cp)
            checksum=m.pop('checkpoint_sha256');require(digest(m)==checksum,'Checkpoint metadata changed')
            require(m['identity']==self.identity and m['samples']==keys and m['method']==method and m['module']==self.name and m['budget']==budget,'Checkpoint binding differs')
            require(all(mo.digest_tensor(v)==m['tensor_hashes'].get(k) for k,v in t.items()),'Checkpoint tensor changed')
            a=t['A'].to(self.device);g=t['G'].to(self.device);history=m['history'];stable=m['stable'];start=len(history)+1
        elif method=='Token-joint':g=torch.eye(len(g),dtype=torch.float64,device=self.device)
        if method in ('Token-joint','Full-fit'):
            maximum=PLAN['token_rounds'] if method=='Token-joint' else PLAN['ALS_max_iterations']
            if method=='Full-fit':a,g=sm.gauge(a,g)
            for iteration in range(start,maximum+1):
                if method=='Full-fit' and stable>=PLAN['ALS_stable_rounds']:break
                self.boundary()
                with self.resources.timed('token_joint_round' if method=='Token-joint' else 'full_fit_round',budget=budget,iteration=iteration,samples=len(keys)):
                    if method=='Token-joint':
                        olda,oldg=a,g;a,g=sm.token_step(self.token_stream(keys),a,g,PLAN['T']);a,g=sm.gauge(a,g)
                        change,guard=sm.parent.product_change(a,g,olda,oldg)
                        row=dict(iteration=iteration,product_relative_change=change,distance_roundoff=guard,
                            A_spectrum=sm.parent.spectrum(a),G_spectrum=sm.parent.spectrum(g),synchronous=True)
                    else:
                        a,g,row=sm.full_step(self.stream(keys),a,g,PLAN['T']);row['iteration']=iteration
                        stable=stable+1 if row['relative_J_improvement']<=PLAN['ALS_relative_J_tolerance'] and row['product_relative_change']<=PLAN['ALS_product_tolerance'] else 0
                    history.append(row);t=dict(A=a,G=g)
                    state=dict(identity=self.identity,module=self.name,budget=budget,samples=keys,method=method,history=history,stable=stable,tensor_hashes={k:mo.digest_tensor(v) for k,v in t.items()})
                    save_tensors(cp,t,dict(state,checkpoint_sha256=digest(state)))
            stop=('FIXED_THREE_SYNCHRONOUS_ROUNDS_NOT_CONVERGENCE_CLAIM' if method=='Token-joint' else
                  'TWO_CONSECUTIVE_CONVERGED_CYCLES' if stable>=PLAN['ALS_stable_rounds'] else 'MAX_ITERATIONS_REACHED')
        elif method=='Sequence-one-step':
            m=self.moments(keys,budget);a,g=sm.sequence_one_step(m['StS'].to(self.device),m['SSt'].to(self.device))
        else:require(method=='Marginal','Unknown method')
        sm.parent.spectrum(a);sm.parent.spectrum(g);a,g=sm.gauge(sm.sym(a),sm.sym(g))
        self.store.put(p,dict(A_raw=a,G_raw=g),samples=keys,method=method,budget=budget,history=history,stop=stop)
        save_json(folder/'iterations.json',dict(identity=self.identity,history=history,stop=stop))
        return dict(A_raw=a.cpu(),G_raw=g.cpu())


def candidate_keys(config):return [f'N{n}__{m}' for n in config['budgets'] for m in config['methods']]+['None']


def fit(root,config,identity,resources):
    require(torch.cuda.device_count()==2 and all('4090' in torch.cuda.get_device_name(i) for i in range(2)),'Require dual4090')
    index=read(root/'data/index.json')
    for row in index['fit']:checked_files(root,load_record(root/'cache/commits'/(row['id']+'.json'),identity)['files'])
    devices=[f'cuda:{i}' for i in range(config['offline_workers'])]
    warm_and_probe(devices);jobs=queue.Queue();stop=threading.Event()
    for name in config['modules']:
        for n in config['budgets']:jobs.put((name,f'N{n}'))
    def consume(device):
        torch.cuda.set_device(device)
        try:
            while not stop.is_set():
                try:name,budget=jobs.get_nowait()
                except queue.Empty:return
                worker=ModuleFit(root,config,identity,resources,name,device,stop)
                for method in config['methods']:
                    worker.boundary();t=worker.raw_factors(budget,method)
                    worker.candidate(budget+'__'+method,t['A_raw'],t['G_raw'],budget,method)
                    del t
                worker.store.clear();del worker;gc.collect()
                print('MODULE_BUDGET_COMPLETE',name,budget,flush=True)
        except BaseException:stop.set();raise
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futures=[pool.submit(consume,d) for d in devices]
        for f in futures:f.result()
    paths=[]
    for name in config['modules']:
        paths.append(root/'quantized'/(slug(name)+'.safetensors'))
        for key in candidate_keys(config)[:-1]:paths.append(root/'modules'/slug(name)/'corrections'/(key+'.safetensors'))
    paths.extend([root/'data/freeze.json',root/'data/index.json',root/'manifest.json',root/'quantized/freeze.json'])
    commit(root/'candidate_freeze.json',identity,files=file_table(root,paths),keys=candidate_keys(config),modules=config['modules'],test_selection=False)
