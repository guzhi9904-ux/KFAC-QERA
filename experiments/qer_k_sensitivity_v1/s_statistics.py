import queue
import threading
from concurrent.futures import ThreadPoolExecutor
import torch
from s_common import (MODULES,PLAN,Source,TensorStore,require,mo,slug,save_json,read_tensors,save_tensors,
                      sha_file,digest,commit,load_record,file_table,checked_files,restore_evidence)
from s_math import weighted_sum,pilot,compare
from s_assets import finalize_assets


def evidence(source):return dict(base=source.base.store.evidence,ko=source.store.evidence,records=source.records)


def verify_evidence(source,ev):
    restore_evidence(source.base.store,ev['base']);restore_evidence(source.store,ev['ko'])
    for p,h in ev['records'].items():require(sha_file(p)==h,'Parent commit changed: '+p)
    source.records=dict(ev['records'])


def checkpoint(folder,ident,name,u,d,rows,source):
    progress=folder/'progress';count=len(rows);p=progress/f'generation_{count:04d}.safetensors'
    t=dict(U=u,D=d.reshape(1));meta=dict(identity=ident,module=name,count=count,windows=list(range(count)),
        rows=rows,evidence=evidence(source),tensor_hashes={k:mo.digest_tensor(v) for k,v in t.items()})
    save_tensors(p,t,meta)
    pointer=dict(identity=ident,module=name,count=count,file=p.name,file_sha256=sha_file(p))
    save_json(progress/'latest.json',dict(pointer,record_sha256=digest(pointer)))
    generations=sorted(progress.glob('generation_*.safetensors'));older=[q for q in generations if q.name<p.name]
    keep={p}|({older[-1]} if older else set())
    for old in generations:
        if old not in keep:
            require(old.resolve().parent==progress.resolve(),'Own checkpoint path escaped');old.unlink()


def restore(folder,ident,name,source,device):
    pointer=folder/'progress/latest.json'
    if not pointer.exists():return torch.zeros(4096,4096,dtype=torch.float64,device=device),torch.zeros(1,dtype=torch.float64,device=device),[]
    row=load_record(pointer,ident);p=pointer.parent/row['file']
    require(p.resolve().parent==pointer.parent.resolve() and row['module']==name and sha_file(p)==row['file_sha256'],'Checkpoint file mismatch')
    t,m=read_tensors(p);require(m['identity']==ident and m['module']==name and m['count']==row['count'],'Checkpoint identity mismatch')
    require(0<m['count']<=256 and m['windows']==list(range(m['count'])) and [r['window'] for r in m['rows']]==m['windows'],'Checkpoint counted windows differ')
    require(all(mo.digest_tensor(v)==m['tensor_hashes'][k] for k,v in t.items()),'Checkpoint content mismatch')
    verify_evidence(source,m['evidence'])
    return t['U'].to(device,copy=True),t['D'].to(device,copy=True),m['rows']


def collect_module(root,ident,name,device,resources,kroot,max_windows=256):
    folder=root/'statistics'/slug(name);store=TensorStore(ident,0);source=Source(kroot)
    if (folder/'complete.json').exists():
        checked_files(root,load_record(folder/'complete.json',ident)['files'])
        verify_evidence(source,load_record(folder/'parent_evidence.json',ident)['evidence']);return
    u,d,rows=restore(folder,ident,name,source,device)
    for window in range(len(rows),max_windows):
        resources.boundary();sample=source.index['fit'][window]
        with resources.timed('read_verify_X_Kg',device=device,module=name,window=window):x,g=source.sample(name,sample)
        with resources.timed('transfer_X_Kg',device=device,module=name,window=window):
            x=x.reshape(2048,4096).to(device,dtype=torch.float64);g=g.to(device,dtype=torch.float64)
        if window<2:
            with resources.timed('offline_math_pilot',device=device,module=name,window=window):checks=pilot(x,g)
            commit(root/'pilot'/slug(name)/f'w{window:04d}.json',ident,passed=True,checks=checks,window=window,module=name)
        with resources.timed('weighted_Gram',device=device,module=name,window=window):
            term,mass,w=weighted_sum(x,g);u.add_(term);d.add_(mass)
            require(torch.isfinite(u).all() and torch.isfinite(d),'Nonfinite accumulators')
            rows.append(dict(window=window,sum_w=float(mass),sum_w2=float(w.square().sum()),max_w=float(w.max()),zero_weights=int((w==0).sum()),finite=True))
        del x,g,term,w
        count=window+1
        if count in (1,2) or count%PLAN['checkpoint_every']==0 or count==256:
            with resources.timed('statistics_checkpoint',device=device,module=name,windows=count):checkpoint(folder,ident,name,u,d,rows,source)
            print('SENSITIVITY_STATISTICS_COMMITTED',name,count,'/256',flush=True)
    if max_windows<256:return
    require(float(d)>0 and torch.isfinite(d),'Degenerate zero/nonfinite gradient mass')
    gt,gm=store.get(root/'borrowed_G'/slug(name)/'canonical.safetensors');g=gt['G']
    check=compare(d.reshape(())/(256*2047),g.trace().to(device))
    a=(u+u.T)*.5/d
    require(mo.digest_tensor(g)==gm['parent_tensor_hash'],'Parent G changed')
    store.put(folder/'raw.safetensors',dict(U=u,D=d,A_raw=a,G_raw=g),module=name,windows=list(range(256)),
              A_definition='sum(||g||^2*x*x.T)/sum(||g||^2)',G_source=gm['parent_file'],G_hash=gm['parent_tensor_hash'],G_normalization='N*T',trace_check=check)
    ss=sum(r['sum_w2'] for r in rows)
    commit(folder/'weight_summary.json',ident,module=name,windows=256,positions=256*2048,D=float(d),sum_w2=ss,
           weight_effective_count=float(d)**2/ss,max_w=max(r['max_w'] for r in rows),zero_weights=sum(r['zero_weights'] for r in rows),
           interpretation='Weight effective count, NOT number of independent texts/gradient samples',per_window=rows,trace_check=check)
    commit(folder/'parent_evidence.json',ident,evidence=evidence(source),X=256,K_gradients=256)
    paths=[folder/n for n in ('raw.safetensors','weight_summary.json','parent_evidence.json')]
    paths += [root/'pilot'/slug(name)/f'w{w:04d}.json' for w in (0,1)]
    commit(folder/'complete.json',ident,windows=256,passed=True,files=file_table(root,paths))
    print('SENSITIVITY_MODULE_STATISTICS_COMPLETE',name,flush=True)


def collect(root,ident,resources,kroot,assets):
    require(torch.cuda.device_count()==2 and all('4090' in torch.cuda.get_device_name(i) for i in range(2)),'Require dual4090')
    # Sequential pilot guarantees L20 first. Its full-window accumulators are retained.
    # Other modules run their own fixed pilot before contributing those windows.
    jobs=queue.Queue();errors=[];mutex=threading.Lock()
    ordered=[MODULES[2],MODULES[0],MODULES[1],MODULES[3]]
    pilot_failed=False
    if assets['modules'][MODULES[2]]['status'] in ('pending_stream_validation','verified'):
        try:collect_module(root,ident,MODULES[2],'cuda:0',resources,kroot,max_windows=2)
        except Exception as exc:
            errors.append(dict(module=MODULES[2],error=repr(exc),status='missing' if isinstance(exc,FileNotFoundError) else 'identity_mismatch'));pilot_failed=True
    for n in ordered:
        if n==MODULES[2] and pilot_failed:continue
        if assets['modules'][n]['status'] in ('pending_stream_validation','verified'):jobs.put(n)
        else:errors.append(dict(module=n,**assets['modules'][n]))
    def worker(device):
        torch.cuda.set_device(device)
        while True:
            try:name=jobs.get_nowait()
            except queue.Empty:return
            try:collect_module(root,ident,name,device,resources,kroot)
            except Exception as exc:
                with mutex:errors.append(dict(module=name,error=repr(exc),status='missing' if isinstance(exc,FileNotFoundError) else 'identity_mismatch'))
                save_json(root/'statistics'/slug(name)/'failure.json',dict(module=name,error=repr(exc)))
                print('SENSITIVITY_MODULE_STOPPED',name,repr(exc),flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(worker,f'cuda:{i}') for i in range(2)]
        for f in futures:f.result()
    if errors:
        save_json(root/'asset_failures.json',dict(identity=ident,modules=errors));raise RuntimeError('Incomplete modules; see asset_failures.json. No teacher backward fallback.')
    finalize_assets(root,ident)
    pilots=[root/'pilot'/slug(n)/f'w{w:04d}.json' for n in MODULES for w in (0,1)]
    for p in pilots:require(load_record(p,ident)['passed'],'Pilot incomplete')
    commit(root/'pilot/math_checks.json',ident,passed=True,windows=[0,1],files=file_table(root,pilots),pilot_counted_once=True)
    print('SENSITIVITY_ALL_STATISTICS_COMPLETE',flush=True)
