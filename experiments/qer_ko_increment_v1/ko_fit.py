import gc
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
import torch
from ko_common import (read,require,commit,file_table,load_record,checked_files,slug,mo,sm,
                       MixedStore,source_x,NEW,OLD,METHODS)
from ko_assets import new_x,new_g
from fitting import ModuleFit
from bridge import warm_and_probe


class IncrementFit(ModuleFit):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        parent=self.config['parent_run'];pid=read(__import__('pathlib').Path(parent)/'manifest.json')['identity']
        self.store=MixedStore(self.identity,self.config['cache_GiB'],parent,pid)
        # Reuse the exact input moment for old modules and for k via q.
        from pathlib import Path
        source_name=self.name.replace('.k_proj','.q_proj') if self.name.endswith('.k_proj') else self.name
        if self.name in OLD or self.name.endswith('.k_proj'):
            ap=Path(parent)/'modules'/slug(source_name)/'statistics/N256_A.safetensors'
            t,m=self.store.get(ap)
            require(m['windows']==list(range(256)) and m['each_window_counted_once'],'Wrong borrowed input moment')
            dest=self.root/'statistics/N256_A.safetensors'
            self.store.put(dest,t,windows=m['windows'],each_window_counted_once=True,borrowed_source=str(ap),
                           borrowed_A_hash=mo.digest_tensor(t['A_m']))
    def path(self,kind,key):
        if kind=='fit_x':
            window=int(key[1:])
            return new_x(self.runroot,self.name,window) if self.name.endswith('.o_proj') else source_x(self.config['parent_run'],self.name,window)
        if kind=='fit_g':return new_g(self.runroot,self.name,key)
        raise RuntimeError('No dense S or eval gradients')
    def raw_factors(self,budget,method):
        if method!='A-only':return super().raw_factors(budget,method)
        p=self.root/'factors'/budget/method/'raw.safetensors'
        if p.exists():return self.store.get(p)[0]
        ap=self.root/'statistics/N256_A.safetensors'
        if ap.exists():a=self.store.get(ap)[0]['A_m'].to(self.device)
        else:
            a=None
            for window in range(256):
                self.boundary();x=self.store.get(self.path('fit_x',f'w{window:04d}'))[0]['x'].reshape(2048,-1).to(self.device).double()
                term=x.T@x;a=term if a is None else a+term
            a=sm.sym(a/(256*2048));self.store.put(ap,dict(A_m=a),windows=list(range(256)),each_window_counted_once=True)
        g=torch.eye(self.error.shape[0],dtype=torch.float64,device=self.device)
        self.store.put(p,dict(A_raw=a,G_raw=g),method='A-only',budget=budget,samples=self.index['budgets'][budget],
                       definition='Input-only QERA objective with full uncentered A_m; G exactly identity before solver damping; same rank/relative damping/deployment checks')
        return dict(A_raw=a.cpu(),G_raw=g.cpu())


def fit(root,config,ident,resources):
    if (root/'candidate_freeze.json').exists():checked_files(root,load_record(root/'candidate_freeze.json',ident)['files']);return
    for row in read(root/'data/index.json')['fit']:checked_files(root,load_record(root/'cache/commits'/(row['id']+'.json'),ident)['files'])
    devices=['cuda:0','cuda:1'];warm_and_probe(devices);q=queue.Queue();stop=threading.Event()
    for name in config['modules']:q.put(name)
    def consume(device):
        torch.cuda.set_device(device)
        try:
            while not stop.is_set():
                try:name=q.get_nowait()
                except queue.Empty:return
                worker=IncrementFit(root,config,ident,resources,name,device,stop)
                for method in METHODS if name in NEW else ['A-only']:
                    with resources.timed('fit_method',device=device,module=name,method=method):
                        t=worker.raw_factors('N256',method);worker.candidate('N256__'+method,t['A_raw'],t['G_raw'],'N256',method)
                    del t
                worker.store.clear();del worker;gc.collect();print('MODULE_FIT_COMPLETE',name,flush=True)
        except BaseException:stop.set();raise
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(consume,d) for d in devices]
        for f in futures:f.result()
    paths=[root/'modules'/slug(n)/'corrections'/('N256__'+m+'.safetensors') for n in config['modules'] for m in (METHODS if n in NEW else ['A-only'])]
    paths += [root/'quantized/freeze.json',root/'data/index.json',root/'acceptance.json']
    commit(root/'candidate_freeze.json',ident,files=file_table(root,paths),new_corrections=44,methods=METHODS,test_selection=False)
