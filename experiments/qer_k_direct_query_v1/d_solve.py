from concurrent.futures import ThreadPoolExecutor
import torch
from d_common import MODULES,TensorStore,require,slug,mo,sm,save_json,commit,load_record,checked_files,file_table
from bridge import warm_and_probe


def solve(root,config,ident,resources,source):
    freeze=root/'candidate_freeze.json'
    if freeze.exists():checked_files(root,load_record(freeze,ident)['files']);return
    warm_and_probe(['cuda:0','cuda:1'])
    def worker(device,names):
        torch.cuda.set_device(device);store=TensorStore(ident,0)
        for name in names:
            p=root/'corrections'/slug(name)/'direct_query.safetensors'
            regularized=root/'factors'/slug(name)/'regularized.safetensors'
            audit_path=root/'factors'/slug(name)/'solve_audit.json'
            if p.exists() and regularized.exists() and audit_path.exists():
                store.get(p);store.get(regularized);continue
            with resources.timed('full_matrix_rank64_solve',device=device,module=name):
                raw=store.get(root/'statistics'/slug(name)/'raw.safetensors')[0]
                a=raw['A_raw'].to(device);g=raw['G_raw'].to(device)
                require(a.shape==(4096,4096) and g.shape==(1024,1024),'Wrong full-weight factor dimensions')
                off=g.clone()
                for b in range(8):off[b*128:(b+1)*128,b*128:(b+1)*128]=0
                require(torch.count_nonzero(off)==0,'G is not exact KV-block-diagonal');del off
                sa=sm.parent.spectrum(a);sg=sm.parent.spectrum(g)
                eig_a=torch.linalg.eigvalsh(a);eig_g=torch.linalg.eigvalsh(g)
                ranks=dict(A_positive=int((eig_a>0).sum()),G_positive=int((eig_g>0).sum()),
                           A_above_relative_1e_10=int((eig_a>eig_a[-1]*1e-10).sum()),G_above_relative_1e_10=int((eig_g>eig_g[-1]*1e-10).sum()))
                w=source.store.get(source.root/'quantized'/(slug(name)+'.safetensors'))[0]
                error=w['W0'].to(device).double()-w['Wq'].to(device).double()
                t,c,audit=sm.solve(error,a,g,64,w['Wq'],w['W0'])
                require(audit['rank']==64 and c['P64'].shape==(1024,64) and c['Q64'].shape==(64,4096),'Not one global rank64 solve')
                audit.update(raw_A_spectrum=sa,raw_G_spectrum=sg,raw_rank_diagnostics=ranks,G_KV_block_diagonal=True,
                             omitted_cross_head_covariance=True,shared_A_across_all_query_heads=True)
                store.put(p,{k:c[k] for k in ('P64','Q64','W_deploy','R64','R_deploy')},module=name,rank=64,audit=audit,W_hash=mo.digest_tensor(c['W_deploy']))
                store.put(root/'factors'/slug(name)/'regularized.safetensors',dict(A_solve=t['A_solve'],G_solve=t['G_solve']),module=name,eta_A=1e-3,eta_G=1e-3)
                save_json(root/'factors'/slug(name)/'solve_audit.json',audit)
            print('CHECK2_MODULE_PASSED',name,flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(worker,'cuda:0',MODULES[:2]),pool.submit(worker,'cuda:1',MODULES[2:])]
        for job in jobs:job.result()
    paths=[root/'corrections'/slug(n)/'direct_query.safetensors' for n in MODULES]
    paths += [root/'factors'/slug(n)/filename for n in MODULES for filename in ('regularized.safetensors','solve_audit.json')]
    commit(freeze,ident,files=file_table(root,paths),candidates=4,rank_per_complete_K=64,validation_selection=False)
    print('CHECK2_ALL_CANDIDATES_FROZEN',flush=True)
