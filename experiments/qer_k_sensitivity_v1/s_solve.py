from concurrent.futures import ThreadPoolExecutor
import torch
from s_common import (MODULES,TensorStore,require,slug,mo,sm,save_json,commit,load_record,checked_files,file_table)
from bridge import warm_and_probe


def solve(root,ident,resources,source):
    freeze=root/'candidate_freeze.json'
    if freeze.exists():checked_files(root,load_record(freeze,ident)['files']);return
    warm_and_probe(['cuda:0','cuda:1'])
    def worker(device,names):
        torch.cuda.set_device(device);store=TensorStore(ident,0)
        for name in names:
            folder=root/'factors'/slug(name);p=root/'corrections'/slug(name)/'sensitivity_weighted.safetensors'
            if (folder/'complete.json').exists():checked_files(root,load_record(folder/'complete.json',ident)['files']);continue
            with resources.timed('full_matrix_rank64_solve',device=device,module=name):
                raw,rm=store.get(root/'statistics'/slug(name)/'raw.safetensors')
                require(mo.digest_tensor(raw['G_raw'])==rm['G_hash'],'G changed between statistics and solve')
                a=raw['A_raw'].to(device);g=raw['G_raw'].to(device)
                require(a.shape==(4096,4096) and g.shape==(1024,1024),'Wrong full matrix dimensions')
                sa=sm.parent.spectrum(a);sg=sm.parent.spectrum(g)
                w=source.store.get(source.root/'quantized'/(slug(name)+'.safetensors'))[0]
                error=w['W0'].to(device).double()-w['Wq'].to(device).double()
                t,c,audit=sm.solve(error,a,g,64,w['Wq'],w['W0'])
                require(audit['rank']==64 and c['P64'].shape==(1024,64) and c['Q64'].shape==(64,4096),'Not a complete-weight rank64 solve')
                audit.update(raw_A_spectrum=sa,raw_G_spectrum=sg,G_parent_hash=rm['G_hash'],
                             full_G_unchanged_before_gauge=True,internal_gauge=True,
                             solver_A_divisor=float(a.norm()),solver_G_multiplier=float(a.norm()),
                             solver_G_after_gauge_hash=mo.digest_tensor(t['G_raw']),only_A_changed=True)
                store.put(p,{k:c[k] for k in ('P64','Q64','W_deploy','R64','R_deploy')},module=name,rank=64,audit=audit,W_hash=mo.digest_tensor(c['W_deploy']))
                store.put(folder/'regularized.safetensors',dict(A_solve=t['A_solve'],G_solve=t['G_solve']),module=name,eta_A=.001,eta_G=.001)
                save_json(folder/'solve_audit.json',audit)
                commit(folder/'complete.json',ident,passed=True,files=file_table(root,[p,folder/'regularized.safetensors',folder/'solve_audit.json']))
            print('SENSITIVITY_MODULE_SOLVE_COMPLETE',name,flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(worker,'cuda:0',MODULES[:2]),pool.submit(worker,'cuda:1',MODULES[2:])]
        for job in jobs:job.result()
    paths=[root/'corrections'/slug(n)/'sensitivity_weighted.safetensors' for n in MODULES]
    paths += [root/'factors'/slug(n)/f for n in MODULES for f in ('regularized.safetensors','solve_audit.json','complete.json')]
    paths += [root/'manifest.json',root/'parent_assets_audit.json',root/'pilot/math_checks.json']
    commit(freeze,ident,files=file_table(root,paths),candidates=4,rank_per_complete_K=64,validation_selection=False)
    print('SENSITIVITY_ALL_CANDIDATES_FROZEN',flush=True)
