from concurrent.futures import ThreadPoolExecutor
import torch
from j_common import MODULES,TensorStore,require,slug,mo,sm,save_json,commit,load_record,checked_files,file_table
from bridge import warm_and_probe


def solve(root,ident,resources,source):
    freeze=root/'candidate_freeze.json'
    if freeze.exists():checked_files(root,load_record(freeze,ident)['files']);return
    warm_and_probe(['cuda:0','cuda:1'])
    def worker(device,names):
        torch.cuda.set_device(device);store=TensorStore(ident,0)
        for name in names:
            folder=root/'factors'/slug(name);p=root/'corrections'/slug(name)/'token_joint_one.safetensors'
            if (folder/'complete.json').exists():checked_files(root,load_record(folder/'complete.json',ident)['files']);continue
            with resources.timed('full_matrix_rank64_solve',device=device,module=name):
                raw,meta=store.get(root/'statistics'/slug(name)/'raw.safetensors')
                a1=raw['A1'].to(device);g1=raw['G1'].to(device)
                require(a1.shape==(4096,4096) and g1.shape==(1024,1024) and meta['rounds']==1 and meta['synchronous'],'Wrong first-round factors')
                sa=sm.parent.spectrum(a1);sg=sm.parent.spectrum(g1);scale=float(a1.norm());a,g=sm.gauge(a1,g1)
                w=source.store.get(source.root/'quantized'/(slug(name)+'.safetensors'))[0]
                error=w['W0'].to(device).double()-w['Wq'].to(device).double()
                t,c,audit=sm.solve(error,a,g,64,w['Wq'],w['W0'])
                require(audit['rank']==64 and c['P64'].shape==(1024,64) and c['Q64'].shape==(64,4096),'Not a full-weight rank64 solve')
                audit.update(A1_spectrum=sa,G1_spectrum=sg,A1_divisor=scale,G1_multiplier=scale,
                    solver_internal_A_divisor=float(a.norm()),solver_internal_G_multiplier=float(a.norm()),
                    A1_hash=mo.digest_tensor(a1),G1_hash=mo.digest_tensor(g1),gauge_A_hash=mo.digest_tensor(a),gauge_G_hash=mo.digest_tensor(g),
                    initialization='A_M,I',rounds=1,synchronous=True)
                store.put(p,{k:c[k] for k in ('P64','Q64','W_deploy','R64','R_deploy')},module=name,rank=64,audit=audit,W_hash=mo.digest_tensor(c['W_deploy']))
                store.put(folder/'raw_and_regularized.safetensors',dict(A1=a1,G1=g1,A_gauge=a,G_gauge=g,A_solve=t['A_solve'],G_solve=t['G_solve']),module=name,eta_A=.001,eta_G=.001,gauge_scale=scale)
                save_json(folder/'solve_audit.json',audit)
                commit(folder/'complete.json',ident,passed=True,files=file_table(root,[p,folder/'raw_and_regularized.safetensors',folder/'solve_audit.json']))
            print('TOKEN_ONE_MODULE_SOLVE_COMPLETE',name,flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(worker,'cuda:0',MODULES[:2]),pool.submit(worker,'cuda:1',MODULES[2:])]
        for job in jobs:job.result()
    paths=[root/'corrections'/slug(n)/'token_joint_one.safetensors' for n in MODULES]
    paths += [root/'factors'/slug(n)/f for n in MODULES for f in ('raw_and_regularized.safetensors','solve_audit.json','complete.json')]
    paths += [root/'manifest.json',root/'parent_assets_audit.json',root/'pilot/one_step_equivalence.json']
    commit(freeze,ident,files=file_table(root,paths),candidates=4,rank_per_complete_K=64,validation_selection=False)
    print('TOKEN_ONE_ALL_CANDIDATES_FROZEN',flush=True)
