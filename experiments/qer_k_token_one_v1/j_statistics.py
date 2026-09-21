import queue
import threading
from concurrent.futures import ThreadPoolExecutor
import torch
from j_common import (MODULES,PLAN,Source,TensorStore,require,mo,slug,save_json,read_tensors,save_tensors,
                      sha_file,digest,commit,load_record,file_table,checked_files,verify_evidence,compare,sm)
from j_math import contract,pilot
from j_assets import finalize_assets


def checkpoint(folder,ident,name,v,rows,parent_hash):
    progress=folder/'progress';count=len(rows);p=progress/f'generation_{count:04d}.safetensors'
    save_tensors(p,dict(V=v),dict(identity=ident,module=name,windows=list(range(count)),rows=rows,
                 parent_evidence_hash=parent_hash,V_hash=mo.digest_tensor(v)))
    pointer=dict(identity=ident,module=name,count=count,file=p.name,file_sha256=sha_file(p))
    save_json(progress/'latest.json',dict(pointer,record_sha256=digest(pointer)))
    generations=sorted(progress.glob('generation_*.safetensors'));older=[q for q in generations if q.name<p.name]
    keep={p}|({older[-1]} if older else set())
    for old in generations:
        if old not in keep:
            require(old.resolve().parent==progress.resolve(),'Checkpoint escaped own output');old.unlink()


def restore(folder,ident,name,parent_hash,device,m=1024):
    pointer=folder/'progress/latest.json'
    if not pointer.exists():return torch.zeros(m,m,dtype=torch.float64,device=device),[]
    row=load_record(pointer,ident);p=pointer.parent/row['file']
    require(p.resolve().parent==pointer.parent.resolve() and row['module']==name and sha_file(p)==row['file_sha256'],'Checkpoint changed')
    t,meta=read_tensors(p)
    require(meta['identity']==ident and meta['module']==name and meta['parent_evidence_hash']==parent_hash,'Checkpoint identity differs')
    require(0<row['count']<=256 and meta['windows']==list(range(row['count'])) and [r['window'] for r in meta['rows']]==meta['windows'],'Window counting differs')
    require(t['V'].shape==(m,m) and mo.digest_tensor(t['V'])==meta['V_hash'],'Checkpoint content differs')
    return t['V'].to(device,copy=True),meta['rows']


def collect_module(root,ident,name,device,resources,kroot,sroot,max_windows=256):
    folder=root/'statistics'/slug(name);store=TensorStore(ident,0);source=Source(kroot,sroot)
    initial,meta=store.get(root/'borrowed'/slug(name)/'initial.safetensors')
    ep=source.sroot/'statistics'/slug(name)/'parent_evidence.json'
    require(str(ep)==meta['sens_evidence'] and sha_file(ep)==meta['sens_evidence_hash'],'Parent cache evidence changed')
    with resources.timed('reuse_cache_evidence',device=device,module=name):verify_evidence(source,load_record(ep,source.sid)['evidence'])
    if (folder/'complete.json').exists():checked_files(root,load_record(folder/'complete.json',ident)['files']);return
    a=initial['A_M'].to(device);norm2=float(a.square().sum());require(norm2>0,'Zero raw A_M')
    v,rows=restore(folder,ident,name,meta['sens_evidence_hash'],device);samples=[];terms=[]
    if len(rows)>=2:require(load_record(root/'pilot'/slug(name)/'one_step.json',ident)['passed'],'Missing pilot acceptance at resume')
    for window in range(len(rows),max_windows):
        resources.boundary()
        with resources.timed('read_bound_X_Kg',device=device,module=name,window=window):x,g=source.sample(name,source.index['fit'][window])
        with resources.timed('transfer_X_Kg',device=device,module=name,window=window):
            x=x.reshape(2048,4096).to(device,dtype=torch.float64);g=g.to(device,dtype=torch.float64)
        with resources.timed('first_G_contraction',device=device,module=name,window=window):
            term,u,row=contract(x,g,a);v.add_(term);row['window']=window;rows.append(row)
        if window<2:samples.append((x,g));terms.append((term,u))
        if window==1:
            require(len(samples)==2,'Pilot windows must be committed together')
            with resources.timed('parent_one_step_equivalence',device=device,module=name):checks=pilot(samples,a,terms,2047)
            commit(root/'pilot'/slug(name)/'one_step.json',ident,passed=True,windows=[0,1],checks=checks,
                   full_A_M=True,original_parent_token_step=True,A_source=meta['A_recovery_checks'],pilot_G_counted_once=True)
            samples.clear();terms.clear();print('TOKEN_ONE_MODULE_PILOT_PASSED',name,flush=True)
        del x,g,term,u
        count=window+1
        if count==2 or count%PLAN['checkpoint_every']==0 or count==256:
            with resources.timed('G_checkpoint',device=device,module=name,windows=count):checkpoint(folder,ident,name,v,rows,meta['sens_evidence_hash'])
            print('TOKEN_ONE_G_COMMITTED',name,count,'/256',flush=True)
    if max_windows<256:return
    g1=sm.sym(v/(256*2047*norm2));a1=initial['A1'].to(device)
    su=sum(r['sum_u'] for r in rows);sug=sum(r['sum_u_g2'] for r in rows)
    checks=dict(input_sum=compare(su,256*2048*norm2),G_trace=compare(g1.trace(),sug/(256*2047*norm2)))
    store.put(folder/'raw.safetensors',dict(A1=a1,G1=g1,V=v),module=name,windows=list(range(256)),N=256,L=2048,T=2047,
              A1_denominator=256*2047*1024,G1_denominator=256*2047*norm2,A_M_norm2=norm2,
              A1_source=meta['U_source'],A1_source_hash=meta['U_hash'],A1_over_sensitivity_A=meta['A_recovery_checks']['A1_over_sensitivity_A'],
              checks=checks,initialization='raw A_M,I',synchronous=True,rounds=1)
    commit(folder/'input_weight_summary.json',ident,windows=256,sum_u=su,sum_u2=sum(r['sum_u2'] for r in rows),sum_u_g2=sug,
           min_u=min(r['min_u'] for r in rows),max_u=max(r['max_u'] for r in rows),negative_u_count=sum(r['negative_u_count'] for r in rows),
           no_clamp=True,per_window=rows,checks=checks)
    paths=[folder/'raw.safetensors',folder/'input_weight_summary.json',root/'pilot'/slug(name)/'one_step.json']
    commit(folder/'complete.json',ident,passed=True,windows=256,files=file_table(root,paths),parent_evidence_hash=meta['sens_evidence_hash'])
    print('TOKEN_ONE_MODULE_STATISTICS_COMPLETE',name,flush=True)


def collect(root,ident,resources,kroot,sroot,assets):
    require(torch.cuda.device_count()==2 and all('4090' in torch.cuda.get_device_name(i) for i in range(2)),'Require dual4090')
    errors=[];mutex=threading.Lock();jobs=queue.Queue();first=MODULES[2];failed_first=False
    if assets['modules'][first]['status']=='pending_stream_validation':
        try:collect_module(root,ident,first,'cuda:0',resources,kroot,sroot,max_windows=2)
        except Exception as exc:errors.append(dict(module=first,error=repr(exc)));failed_first=True
    for name in [MODULES[2],MODULES[0],MODULES[1],MODULES[3]]:
        if name==first and failed_first:continue
        if assets['modules'][name]['status']=='pending_stream_validation':jobs.put(name)
        else:errors.append(dict(module=name,**assets['modules'][name]))
    def worker(device):
        torch.cuda.set_device(device)
        while True:
            try:name=jobs.get_nowait()
            except queue.Empty:return
            try:collect_module(root,ident,name,device,resources,kroot,sroot)
            except Exception as exc:
                with mutex:errors.append(dict(module=name,error=repr(exc)))
                save_json(root/'statistics'/slug(name)/'failure.json',dict(error=repr(exc)))
                print('TOKEN_ONE_MODULE_STOPPED',name,repr(exc),flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(worker,f'cuda:{i}') for i in range(2)]
        for f in futures:f.result()
    if errors:
        save_json(root/'asset_failures.json',dict(identity=ident,modules=errors));raise RuntimeError('Incomplete modules; see asset_failures.json. No automatic recollection.')
    finalize_assets(root,ident)
    paths=[root/'pilot'/slug(n)/'one_step.json' for n in MODULES]
    for p in paths:require(load_record(p,ident)['passed'],'Pilot incomplete')
    commit(root/'pilot/one_step_equivalence.json',ident,passed=True,files=file_table(root,paths))
    print('TOKEN_ONE_ALL_STATISTICS_COMPLETE',flush=True)
