#!/usr/bin/env python3
"""Runs only in an explicitly acquired cck lease. Never allocates or releases GPUs."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from bridge import HERE,read,PLAN,sha_file,save_json,atomic_bytes,source_identity,mo,torch,check_parent,slug
from functional_math import small_checks
from functional_analysis import test_statistics


def initialize(root):
    identity=source_identity()
    if (root/'identity.json').exists():assert read(root/'identity.json')==identity,'Code changed: use a new run version'
    else:
        save_json(root/'identity.json',identity);atomic_bytes(root/'protocol.md',(HERE/'protocol.md').read_bytes())
        seeds=[mo.stream_seed(c,k,PLAN['sample_seed']) for c in range(8) for k in range(16)]
        previous={mo.stream_seed(c,k,b) for b,K in [(2026091803,4),(2026091804,16)] for c in range(8) for k in range(K)}
        assert len(set(seeds))==128 and not set(seeds)&previous
        save_json(root/'manifest.json',dict(**identity,frozen_at=time.time(),sample_seeds=seeds,
            seed_derivation='SHA256(f"{base}:{window}:{replicate}") first 8 bytes little endian mod (2**63-1); same parent algorithm',
            sampler='CUDA torch.Generator; FP64 torch.rand + full-vocabulary inverse CDF, parent chunk order',
            bootstrap='numpy.PCG64 seed 2026091902, 2000 paired within-window draws, common across 17 candidates and modules',
            tolerances=dict(roots_SVD=1e-8,contraction=1e-10,hidden_replay=1e-7,parent_C_replay=1e-8,
                deployment_relative_drift=1e-4,factor_condition=1e8,KL_repeat='max(1e-12,1e-7*abs(KL))',
                scalar_guard='max(abs(x),abs(y),32*float64_eps*abs(reference_scale)/tolerance); exact zero denominator demands equality',
                projection_guard='abs(d_S-d_direct)<=1e-10*sum(abs(S*R)); exact zero requires equality'),
            logical_counts=dict(fit_module_samples=896,new_module_samples=3584,new_score_rows=60928,KL_main=1120,KL_repeat=140),
            staging='Pilot numerical samples retained once; after all module constructions, global candidate freeze gates formal evaluation',
            disk_budget_GiB=PLAN['disk_budget_GiB'],temporary_policy='Only current module S per worker; no per-sample G Gram; retain identities and rebuild recipe before deleting own temporary files'))


def assignments(modules,workers,shapes):
    """Fixed cost model, never effect-based. Parent comparisons are dispatched first."""
    slots=[[] for _ in range(workers)];loads=[0.]*workers
    for name in modules:
        out_dim,in_dim=shapes[name+'.weight']['shape'];layer=int(name.split('.')[2])
        cost=(40-layer)/10+(max(out_dim,in_dim)/4096)**3+(out_dim*in_dim/4096**2)
        target=min(range(workers),key=lambda x:(loads[x],x));slots[target].append(name);loads[target]+=cost
    return slots


def run_workers(root,phase,modules,workers,devices,deadline):
    shapes=read(Path(PLAN['exp3_output'])/'teacher_identity.json')['tensor_hashes']
    slots=assignments(modules,workers,shapes);jobs=[];logs=[]
    save_json(root/'workers'/f'{phase}_assignment.json',dict(modules=modules,slots=slots,
        rule='Greedy fixed dimension/depth cost in frozen module order; no observed effect used',devices=devices))
    try:
        for i,names in enumerate(slots):
            if not names:continue
            path=root/'logs'/f'{phase}_{i}_{int(time.time())}.log';path.parent.mkdir(parents=True,exist_ok=True)
            log=path.open('w');logs.append(log)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=devices[i],PYTHONDONTWRITEBYTECODE='1')
            cmd=[sys.executable,'-u','-B',str(HERE/'worker.py'),'--output',str(root),'--phase',phase,
                 '--worker-id',str(i),'--deadline',str(deadline),'--modules',*names]
            jobs.append(subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in jobs):
            if any(p.poll() not in (None,0) for p in jobs):raise RuntimeError(f'{phase}: worker failed; inspect module status/logs')
            if time.time()>deadline+120:raise RuntimeError('Budget overrun beyond atomic-unit grace period')
            time.sleep(2)
        if any(p.returncode for p in jobs):raise RuntimeError(f'{phase}: worker failed')
    finally:
        for p in jobs:
            if p.poll() is None:p.terminate()
        for p in jobs:
            if p.poll() is None:
                try:p.wait(timeout=20)
                except subprocess.TimeoutExpired:p.kill();p.wait()
        for log in logs:log.close()


def memory_limit():
    p=Path('/sys/fs/cgroup')/Path('/proc/self/cgroup').read_text().split('0::',1)[1].strip().lstrip('/')
    limits=[]
    while p.is_relative_to('/sys/fs/cgroup'):
        x=p/'memory.max'
        if x.exists() and x.read_text().strip()!='max':limits.append(int(x.read_text()))
        if str(p)=='/sys/fs/cgroup':break
        p=p.parent
    assert limits
    return min(limits)


def accept_pilot(root,workers,elapsed):
    resources=[]
    for name in PLAN['pilot_modules']:
        folder=root/'modules'/slug(name);result=read(folder/'pilot_complete.json')
        assert result['passed'] and result['complete_shapes'] and result['module']==name
        resources.extend(read(folder/'resource_records.json'))
    gpu=max(max(r['allocated_peak_GiB'].values(),default=0) for r in resources)
    cgroup=max(int(r.get('cgroup_memory',{}).get('memory.peak',0)) for r in resources)
    limit=memory_limit();assert limit>=32*2**30
    assert gpu<PLAN['pilot_gpu_peak_limit_GiB'] and cgroup<PLAN['pilot_host_fraction_limit']*limit,('Resource margin failed',gpu,cgroup,limit)
    concurrency=None
    if workers==2:
        a,b=[read(root/'workers'/f'pilot_{i}.json') for i in range(2)]
        overlap=min(a['finished'],b['finished'])-max(a['started'],b['started'])
        assert overlap>0 and a['completed_modules'] and b['completed_modules']
        concurrency=dict(worker_overlap_seconds=overlap,scope='Real overlapping full-shape workers, shared job cgroup peak measured; not a guaranteed 2x speedup')
    # An empirical estimate for planning; no automatic sample/rank/budget changes.
    means={}
    for stage in ('fit_window_statistics','roots_and_two_dense_SVD','new_gradient_and_17_projections','precompute_17_Rx','actual_KL'):
        values=[r['seconds'] for r in resources if r['stage']==stage]
        means[stage]=sum(values)/len(values) if values else None
    rough=(26*8*(means['fit_window_statistics'] or 0)+28*(means['roots_and_two_dense_SVD'] or 0)
           +3584*(means['new_gradient_and_17_projections'] or 0)+224*(means['precompute_17_Rx'] or 0)
           +1120*(means['actual_KL'] or 0))/workers
    save_json(root/'pilot_acceptance.json',dict(passed=True,identity=read(root/'identity.json')['identity'],workers=workers,
        modules=PLAN['pilot_modules'],seconds=elapsed,GPU_allocated_peak_GiB=gpu,cgroup_peak_GiB=cgroup/2**30,
        host_limit_GiB=limit/2**30,parallel_probe=concurrency,stage_mean_seconds=means,
        rough_formal_seconds=rough,estimate_note='Mixed pilot shapes, nonuniform module depths, cache IO and model reloads: not a promised runtime',
        formal_budget_seconds=PLAN['formal_budget_seconds'][str(workers)],time=time.time()))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--stage',choices=['pilot','formal','report'],required=True)
    parser.add_argument('--workers',type=int,choices=[1,2],default=1);args=parser.parse_args()
    assert os.name=='posix' and os.getuid()==1001 and os.environ.get('USER')=='cck'
    root=Path(args.output).resolve()
    assert root.is_relative_to('/data2/cck/KFAC-QERA/runs/qer_functional_gradient_step1') and root.name!='qer_functional_gradient_step1'
    root.mkdir(parents=True,exist_ok=True)
    with (root/'.controller.lock').open('w') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        initialize(root)
        if args.stage=='report':
            from report import aggregate
            aggregate(root);return
        assert '/labgpu.slice/' in Path('/proc/self/cgroup').read_text()
        devices=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
        assert len(devices)==args.workers and all(devices),'Lease GPU count must match worker count; no idle second GPU'
        assert torch.cuda.device_count()==args.workers
        check_parent(root,after=(root/'parent_integrity_before.json').exists())
        checks=small_checks();test_statistics();save_json(root/'small_checks.json',dict(math=checks,paired_statistics_passed=True))
        history=read(root/'execution_attempts.json') if (root/'execution_attempts.json').exists() else []
        budget=PLAN['pilot_budget_seconds'] if args.stage=='pilot' else PLAN['formal_budget_seconds'][str(args.workers)]
        assert not any(r['status']=='RUNNING' for r in history),'Unclosed previous controller attempt: inspect before resuming; never silently reset its time budget'
        used=sum(r['seconds'] for r in history if r['stage']==args.stage)
        assert used<budget,'Frozen cumulative execution budget exhausted; requires an explicitly revised budget/version'
        started=time.time();deadline=started+budget-used
        attempt=dict(stage=args.stage,workers=args.workers,started=started,host_limit_bytes=memory_limit(),status='RUNNING',seconds=0.)
        history.append(attempt);save_json(root/'execution_attempts.json',history)
        try:
            if args.stage=='pilot':
                run_workers(root,'pilot',PLAN['pilot_modules'],args.workers,devices,deadline)
                accept_pilot(root,args.workers,time.time()-started)
            else:
                gate=read(root/'pilot_acceptance.json');assert gate['passed'] and gate['identity']==source_identity()['identity']
                assert gate['workers']==args.workers,'A matching one/two-worker resource pilot is required'
                assert memory_limit()>=gate['host_limit_GiB']*2**30,'Formal host allocation cannot be smaller than tested pilot'
                remaining=[m for m in PLAN['modules'] if not (root/'modules'/slug(m)/'candidate_freeze.json').exists()]
                if remaining:run_workers(root,'construct',remaining,args.workers,devices,deadline)
                from worker import ModuleExperiment
                freezes={}
                for m in PLAN['modules']:
                    e=ModuleExperiment(root,m,deadline);e.validate_freeze()
                    freezes[m]=sha_file(e.root/'candidate_freeze.json')
                freeze=dict(identity=source_identity()['identity'],modules=freezes)
                fp=root/'global_candidate_freeze.json'
                if fp.exists():assert read(fp)==freeze
                else:save_json(fp,freeze)
                remaining=[m for m in PLAN['modules'] if not (root/'modules'/slug(m)/'complete.json').exists()]
                if remaining:run_workers(root,'evaluate',remaining,args.workers,devices,deadline)
                from verify import verify_run
                verify_run(root,require_complete=True)
            check_parent(root,after=True);attempt['status']='COMPLETE'
        except BaseException as error:
            attempt.update(status='STOPPED_INCOMPLETE',error=repr(error),traceback=traceback.format_exc());raise
        finally:
            attempt.update(finished=time.time(),seconds=time.time()-started);save_json(root/'execution_attempts.json',history)
            from report import aggregate
            aggregate(root)


if __name__=='__main__':main()
