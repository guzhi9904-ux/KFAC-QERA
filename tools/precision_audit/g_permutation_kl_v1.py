#!/usr/bin/env python3
"""Matched output-coordinate permutations of frozen DG/GF; rank64 residual compensation."""
from __future__ import annotations
import argparse
import contextlib
import copy
import hashlib
import math
from pathlib import Path
import shutil
import signal
import sys
import tarfile

sys.dont_write_bytecode = True
import numpy as np
import torch
import local_output_kl_v1 as local

frozen, dual, single, core, prev = local.frozen, local.dual, local.single, local.core, local.prev
parent = frozen.parent
VERSION = 'g_permutation_kl_v1'
REAL = ('full_gi','full_gd','full_gf')
SEEDS = (20260913,20260914,20260915)
PERMUTED = tuple(f'{m}_p{i}' for m in REAL[1:] for i in range(3))
ARMS = REAL + PERMUTED
MODULES, PILOT_MODULES = local.MODULES, local.PILOT_MODULES
WINDOWS, LENGTH = local.WINDOWS, local.LENGTH
HELPERS = ('local_output_kl_v1.py',*local.HELPERS)


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def permutation(name, dimension, index):
    require(index in range(len(SEEDS)) and dimension > 1,'Invalid permutation request')
    digest = hashlib.sha256(f'{VERSION}|{name}|{SEEDS[index]}'.encode()).digest()
    seed = int.from_bytes(digest[:8],'big') % (2**63)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    return torch.randperm(dimension,generator=generator)


def permute_root(root, p):
    require(root.ndim==2 and root.shape[0]==root.shape[1] and torch.isfinite(root).all(), 'Invalid square root')
    require(p.dtype==torch.int64 and p.shape==(root.shape[0],)
            and torch.equal(p.cpu().sort().values,torch.arange(root.shape[0])), 'Not a bijective permutation')
    index=p.to(root.device)
    result=root.index_select(0,index).index_select(1,index).contiguous()
    # Exact inverse permutation certifies a simultaneous row/column relabeling.
    inv=index.argsort()
    require(torch.equal(result.index_select(0,inv).index_select(1,inv),root),'Permutation changed values')
    return result


def signed_stats(rows, positive, negative, metric, draws=2000):
    """Paired circular-block intervals; all seeds share the same sampled windows."""
    delta=np.asarray([sum(r[a+'_'+metric+'_sum'] for a in positive)/len(positive)
                      -sum(r[a+'_'+metric+'_sum'] for a in negative)/len(negative) for r in rows])
    counts=np.asarray([r['tokens'] for r in rows],dtype=float)
    rng=np.random.default_rng(20260913)
    starts=rng.integers(0,len(rows),size=(draws,math.ceil(len(rows)/8)))
    ix=((starts[:,:,None]+np.arange(8))%len(rows)).reshape(draws,-1)[:,:len(rows)]
    low,high=np.quantile(delta[ix].sum(1)/counts[ix].sum(1),[.025,.975])
    return {'delta':float(delta.sum()/counts.sum()),'ci95_low':float(low),'ci95_high':float(high)}


@contextlib.contextmanager
def arm_binding():
    old=local.METHODS
    try:
        local.METHODS=ARMS
        yield
    finally:
        local.METHODS=old


def factor_path(ctx,name,arm):
    return ctx.output/'factors'/arm/(core.safe_name(name)+'.json')


def root_path(ctx,name,method):
    return ctx.output/'roots'/method/(core.safe_name(name)+'.json')


def read_root(ctx,name,method):
    cache=ctx.__dict__.setdefault('root_cache',{})
    if (name,method) in cache: return cache[(name,method)]
    path=root_path(ctx,name,method)
    if not path.exists(): return None
    r=core.read_json(path)
    require(r.get('experiment_identity')==ctx.identity and r.get('module')==name and r.get('method')==method,'Root identity mismatch')
    require(r.get('source_binding_sha256')==core.fingerprint(ctx.tasks[name]),'Root source binding mismatch')
    require(Path(r['file']['path']).resolve()==path.with_suffix('.safetensors').resolve(),'Root path mismatch')
    values=ctx.inputs.tensors(r['file'])
    root=values['root']
    n=ctx.tasks[name]['shape'][0]
    require(set(values)=={'root'} and root.dtype==torch.float32 and root.shape==(n,n)
            and torch.isfinite(root).all(),'Root schema mismatch')
    expected=ctx.factors[method][name]['root_bits']['g']
    require(single.tensor_record(root)==r['root_bits']==expected,'Reconstructed G root differs from historical solve')
    cache[(name,method)]=r
    return r


def prepare_root(ctx,name,method):
    r=read_root(ctx,name,method)
    if r is not None: return r
    path=root_path(ctx,name,method)
    path.parent.mkdir(parents=True,exist_ok=True)
    with torch.no_grad(),core.heartbeat('ROOT frozen reconstruction '+method+' '+name):
        root,info=parent.make_g_root(method,ctx.tasks[name],ctx.dg[name],ctx.inputs,ctx.device)
        bits=single.tensor_record(root)
        require(bits==ctx.factors[method][name]['root_bits']['g'],'Original FP32 G root bits not reproduced; stop')
        r={'experiment_identity':ctx.identity,'module':name,'method':method,
           'source_binding_sha256':core.fingerprint(ctx.tasks[name]),'root_bits':bits,'diagnostics':info,
           'file':single.atomic_tensors(path.with_suffix('.safetensors'),{'root':root}),
           'policy':'Reconstruct original normalized/floored FP32 root and require historical bit equality; no new statistics.'}
        core.atomic_json(path,r)
    return r


def read_factor(ctx,name,arm):
    if arm in REAL: return ctx.rank64_checked[arm][name]
    require(arm in PERMUTED,'Unknown arm')
    cache=ctx.__dict__.setdefault('factor_cache',{})
    if (name,arm) in cache: return cache[(name,arm)]
    path=factor_path(ctx,name,arm)
    if not path.exists(): return None
    r=core.read_json(path)
    method,index=arm.rsplit('_p',1)
    root=read_root(ctx,name,method)
    require(root is not None,'Missing source root')
    require(r.get('experiment_identity')==ctx.identity and r.get('module')==name and r.get('arm')==arm
            and r.get('status')=='PASS' and r.get('rank')==64,'Factor identity mismatch')
    require(r['source_binding_sha256']==core.fingerprint(ctx.tasks[name])
            and r['root_file']==root['file'],'Factor source mismatch')
    require(r['payload_sha256']==core.fingerprint({k:v for k,v in r.items() if k!='payload_sha256'}),'Factor metadata changed')
    require(Path(r['file']['path']).resolve()==path.with_suffix('.safetensors').resolve(),'Factor path mismatch')
    v=ctx.inputs.tensors(r['file'])
    dout,din=ctx.tasks[name]['shape']
    require(set(v)=={'A','B','A_fp64','B_fp64','permutation','singular_values'},'Factor keys mismatch')
    require(torch.equal(v['permutation'],permutation(name,dout,int(index))),'Permutation seed/binding mismatch')
    for k,shape in (('A',(din,64)),('B',(64,dout))):
        require(v[k].dtype==torch.bfloat16 and v[k].shape==shape and v[k+'_fp64'].dtype==torch.float64
                and v[k+'_fp64'].shape==shape and torch.isfinite(v[k+'_fp64']).all()
                and torch.isfinite(v[k]).all() and torch.equal(v[k],v[k+'_fp64'].bfloat16())
                and single.tensor_record(v[k])==r['bits'][k],'Invalid direct BF16 factor')
    s=v['singular_values']
    require(s.dtype==torch.float64 and s.shape==(min(din,dout),) and torch.isfinite(s).all()
            and (s>=0).all() and (s[:-1]>=s[1:]).all(),'Invalid singular values')
    require(abs(r['tail_relative_error'])<=1e-9 and max(r['inverse'].values())<=1e-9,'Numerical certificate failed')
    cache[(name,arm)]=r
    return r


def prepare_factor(ctx,name,arm):
    rec=read_factor(ctx,name,arm)
    if rec is not None: return rec
    method,index=arm.rsplit('_p',1)
    root_rec=prepare_root(ctx,name,method)
    path=factor_path(ctx,name,arm)
    path.parent.mkdir(parents=True,exist_ok=True)
    with torch.no_grad(),core.heartbeat('PERMUTE/SOLVE '+name+' '+arm+' FP64 rank64'):
        e=frozen.residual(ctx,name)
        a=ctx.inputs.tensors(ctx.tasks[name]['root'])['full'].to(ctx.device)
        require(single.tensor_record(a)==ctx.factors[method][name]['root_bits']['a'],'Stored Full-A root changed')
        root=ctx.inputs.tensors(root_rec['file'])['root'].to(ctx.device)
        p=permutation(name,root.shape[0],int(index))
        g=permute_root(root,p)
        left,right,s,inverse=single.solve_fp64(e,a,g,64)
        metrics=prev.metrics(e,a,g,left,right,64)
        tail=core.norm2(s[64:])
        gap=(metrics['sse_after']-tail)/max(metrics['sse_before'],1e-30)
        require(math.isfinite(gap) and abs(gap)<=1e-9,'FP64 rank64 residual disagrees with SVD tail')
        values={'A_fp64':left,'B_fp64':right,'A':left.bfloat16(),'B':right.bfloat16(),
                'permutation':p,'singular_values':s}
        rec={'experiment_identity':ctx.identity,'module':name,'arm':arm,'status':'PASS','rank':64,
             'source_binding_sha256':core.fingerprint(ctx.tasks[name]),'root_file':root_rec['file'],
             'seed':SEEDS[int(index)],'permutation_bits':single.tensor_record(p),
             'original_root_bits':single.tensor_record(root),'permuted_root_bits':single.tensor_record(g),
             'fixed_channels':int((p==torch.arange(len(p))).sum()),'root_relative_change':core.rel(g,root),
             'unchanged_metric_root':torch.equal(g,root),
             'invariance':'Exact root row/column permutation and inverse recovery. R R^T -> P R R^T P^T preserves spectrum, trace and condition number mathematically; no extra eigendecomposition.',
             'inverse':inverse,'metrics':metrics,'svd_tail_sse':tail,'tail_relative_error':gap,
             'bits':{k:single.tensor_record(values[k]) for k in ('A','B')},
             'file':single.atomic_tensors(path.with_suffix('.safetensors'),values)}
        rec['payload_sha256']=core.fingerprint(rec)
        core.atomic_json(path,rec)
    del e,a,root,g,left,right,s,values
    dual.clear_gpu()
    return read_factor(ctx,name,arm)


def prepare(ctx,stop,names):
    for name in names:
        for arm in PERMUTED:
            if read_factor(ctx,name,arm) is not None: continue
            stop.check()
            prepare_factor(ctx,name,arm)
            stop.done+=1
            core.log('COMMITTED FACTOR '+name+' '+arm)


def binding(ctx,name):
    files={a:read_factor(ctx,name,a) for a in ARMS}
    require(all(v is not None for v in files.values()),'Prepare all permutations for this module first')
    return {'experiment_identity':ctx.identity,'module':name,'quant':ctx.tasks[name]['quant'],
            'factors':{a:r['file'] for a,r in files.items()},'historical_state':ctx.old_records[name]}


def state_path(ctx,name,pilot=False):
    return ctx.output/('pilot_windows' if pilot else 'windows')/(core.safe_name(name)+'.json')


def replay(row,reference):
    """Same GPU kernels/protocol expected; compare all three real methods per window."""
    keys=['teacher_nll_sum','reference_energy_sum']+[a+'_'+k+'_sum' for a in REAL for k in ('sse','kl','nll')]
    ratios=[]
    for k in keys:
        a,b=row[k],reference[k]
        require(math.isfinite(a) and math.isfinite(b),'Nonfinite replay metric')
        tolerance=1e-8+1e-10*abs(b)
        ratios.append(abs(a-b)/tolerance)
    maximum=max(ratios)
    require(maximum<=1,'Historical local GI/DG/GF replay failed')
    return maximum


def read_state(ctx,name,pilot=False,verify=True):
    path=state_path(ctx,name,pilot)
    total=2 if pilot else WINDOWS
    if not path.exists(): return {'records':[],'complete':False,'expected_windows':total}
    state=core.read_json(path)
    require(state['binding']==binding(ctx,name) and state['expected_windows']==total,'Window identity mismatch')
    require(state['payload_sha256']==core.fingerprint({k:v for k,v in state.items() if k!='payload_sha256'}),'Window metadata changed')
    rr=state['records']
    require(len(rr)<=total and state['complete']==(len(rr)==total),'Window completion mismatch')
    for i,row in enumerate(rr):
        require(row['window']==i and row['tokens']==LENGTH-1 and 0<=row['self_kl_max']<=1e-9,'Window order/self control mismatch')
        expected=path.parent/core.safe_name(name)/f'window_{i:04d}.safetensors'
        require(Path(row['file']['path']).resolve()==expected.resolve(),'Window file path mismatch')
        if verify:
            with arm_binding(): actual=local.tensor_rows(ctx.inputs.tensors(row['file']),i)
            require(all(row[k]==v for k,v in actual.items()),'Token tensor sums differ from checkpoint')
        maximum=replay(row,ctx.old_rows[name][i])
        require(maximum==row['replay_max_tolerance_ratio'],'Replay certificate mismatch')
    return state


def evaluate_module(ctx,teacher,student,name,stop,pilot=False):
    state=read_state(ctx,name,pilot)
    if state['complete']: return
    state['binding']=binding(ctx,name)
    factors={}
    for arm in ARMS:
        v=ctx.inputs.tensors(read_factor(ctx,name,arm)['file'])
        if arm in REAL:
            raw=ctx.inputs.tensors(ctx.factors[arm][name]['file'])
            require(all(torch.equal(v[k],raw[k+'_fp64'].bfloat16()) for k in ('A','B')),'Historical BF16 factors changed')
            del raw
        factors[arm]=(v['A'],v['B'])
    del v
    q=ctx.inputs.tensors(ctx.tasks[name]['quant'])['weight_q']
    data=frozen.windows(ctx)
    path=state_path(ctx,name,pilot)
    folder=path.parent/core.safe_name(name)
    folder.mkdir(parents=True,exist_ok=True)
    for i in range(len(state['records']),state['expected_windows']):
        stop.check()
        with core.heartbeat(f'G-PERM {name} window={i+1}/{state["expected_windows"]} nine arms'),arm_binding():
            values,self_max=local.compare_window(teacher,student,name,factors,q,
                data['input_ids'][i:i+1],data['attention_mask'][i:i+1],self_check=True)
            row=local.tensor_rows(values,i)
        row['replay_max_tolerance_ratio']=replay(row,ctx.old_rows[name][i])
        dual.freeze_json(ctx.output/'teacher_windows'/f'window_{i:04d}.json',
            {'experiment_identity':ctx.identity,'teacher_nll_bits':single.tensor_record(values['teacher_nll'])})
        row['self_kl_max']=self_max
        row['file']=single.atomic_tensors(folder/f'window_{i:04d}.safetensors',values)
        state['records'].append(row)
        state['complete']=i+1==state['expected_windows']
        state.pop('payload_sha256',None)
        state['payload_sha256']=core.fingerprint(state)
        core.atomic_json(path,state)
        stop.done+=1


def summarize(ctx,verify=True):
    summary,paired,detail,certs=[],[],[],[]
    done=0
    for name in ctx.modules:
        for arm in PERMUTED:
            record=read_factor(ctx,name,arm)
            if record is not None:
                certs.append({'module':name,'arm':arm,**{k:record[k] for k in ('seed','fixed_channels','root_relative_change','unchanged_metric_root','tail_relative_error')},
                              'metadata':single.file_record(factor_path(ctx,name,arm))})
        state=read_state(ctx,name,verify=verify)
        if not state['complete']: continue
        done+=1
        rr=state['records']; count=sum(r['tokens'] for r in rr)
        for arm in ARMS:
            summary.append({'module':name,'arm':arm,'rank':64,'tokens':count,
                            **{k+'_per_token':math.fsum(r[arm+'_'+k+'_sum'] for r in rr)/count for k in ('sse','kl','nll')}})
        for method in REAL[1:]:
            shuffled=[method+f'_p{i}' for i in range(3)]
            for pos,label in [([a],a) for a in shuffled]+[(shuffled,method+'_permutation_mean')]:
                stats={k:signed_stats(rr,pos,[method],k) for k in ('sse','kl','nll')}
                paired.append({'module':name,'contrast':label+'_minus_'+method,
                    'seed_count':len(pos),'positive_means':'real G has lower loss/error',
                    **{k+'_'+field:value for k,record in stats.items() for field,value in record.items()}})
        detail.extend({'module':name,**{k:v for k,v in r.items() if k!='file'}} for r in rr)
    for file,values in [('summary.csv',summary),('paired.csv',paired),('per_window.csv',detail),('factor_audit.csv',certs)]:
        prev.write_csv(ctx.output/file,values)
    status={'experiment_identity':ctx.identity,'complete':done==len(ctx.modules),
        'completed_modules':done,'expected_modules':len(ctx.modules),'completed_factors':len(certs),
        'expected_factors':len(ctx.modules)*len(PERMUTED),'completed_arms':len(summary),
        'note':'Report every module/seed, including negative and null effects. Seed-mean CIs condition on three fixed permutations; not a permutation test, not seed-population uncertainty. No multiple-comparison correction.'}
    core.atomic_json(ctx.output/'status.json',status)
    return status


def pack(ctx):
    require(summarize(ctx)['complete'],'Incomplete experiment cannot produce final summary package')
    names=['experiment.json','source_baselines.json','status.json','summary.csv','paired.csv',
           'per_window.csv','factor_audit.csv','pilot.json','background.json']
    for name in ctx.modules:
        names.extend(str(factor_path(ctx,name,a).relative_to(ctx.output)).replace('\\','/') for a in PERMUTED)
        names.extend(str(root_path(ctx,name,m).relative_to(ctx.output)).replace('\\','/') for m in REAL[1:])
    dest=ctx.output/(VERSION+'_summary.tar.gz')
    with tarfile.open(dest,'w:gz') as archive:
        for name in names: archive.add(ctx.output/name,arcname=name,recursive=False)
    core.log('SUMMARY PACKAGE: '+str(dest))


def setup(args,stop):
    directory=Path(__file__).resolve().parent
    pins={}
    for line in (directory/'SHA256SUMS.g_permutation_kl').read_text().splitlines():
        digest,name=line.split('  ',1)
        require(Path(name).name==name and name not in pins and core.sha(directory/name)==digest,'Bundle checksum mismatch')
        pins[name]=digest
    require(all(n in pins for n in (*HELPERS,Path(__file__).name)),'Incomplete bundle manifest')
    ctx=local.setup(args)
    source_dir=args.source_local_dir.resolve()
    core.disjoint(ctx.output,[source_dir])
    path=source_dir/'experiment.json'
    require(core.read_json(path)==ctx.experiment,'Historical local experiment provenance differs')
    old=copy.copy(ctx); old.output=source_dir
    require(core.read_json(source_dir/'status.json').get('complete') is True,'Historical local experiment incomplete')
    ctx.old_rows={}; ctx.old_records={}
    for name in ctx.modules:
        stop.check()
        state=local.read_state(old,name,verify=True)
        require(state['complete'],'Historical local module incomplete')
        ctx.old_rows[name]=state['records']
        ctx.old_records[name]=single.file_record(local.state_path(old,name))
    ctx.experiment={'version':VERSION,'tool_hashes':pins,'frozen_local_experiment':single.file_record(path),
        'frozen_local_identity':old.identity,'source_states':ctx.old_records,'modules':list(MODULES),
        'arms':list(ARMS),'seeds':list(SEEDS),'rank':64,'a':'unchanged stored Full-A root',
        'permutation':'CPU randperm using SHA256(version|module|seed), same P for DG and GF; root[p][:,p]',
        'solve':'original FP64 full SVD and inverse solves; direct BF16 factors, no solver fallback',
        'g_policy':'original normalized/floored FP32 root reconstructed and bit-checked, then permuted; effective metric G=R R^T',
        'protocol':old.experiment['protocol'],'replay':{'absolute_sum_tolerance':1e-8,'relative_sum_tolerance':1e-10,
            'scope':'every scored window, GI/DG/GF SSE/KL/NLL and teacher NLL/reference energy'},
        'primary':'permutation-minus-real teacher KL; 3 individual seeds and within-window seed mean',
        'bootstrap':{'seed':20260913,'draws':2000,'block_windows':8,'conditional_on_fixed_seeds':True},
        'pilot':{'modules':list(PILOT_MODULES),'windows':2},'posthoc_wikitext_diagnostic':True,
        'scope_limit':'Not a proof of exact Fisher, universal optimality, or isolated off-diagonal utility; no rank allocation or G collection.'}
    ctx.identity=core.fingerprint(ctx.experiment)
    return ctx


def run(ctx,stop,pilot=False):
    names=PILOT_MODULES if pilot else ctx.modules
    if all(read_state(ctx,n,pilot)['complete'] for n in names): return
    teacher=student=None
    try:
        stop.check()
        with core.heartbeat('G-PERM load teacher GPU0 and student GPU1'):
            teacher=frozen.load_kl_model(ctx,0); student=frozen.load_kl_model(ctx,1)
            require(not any(m._forward_hooks or m._forward_pre_hooks for model in (teacher,student) for m in model.modules()),'Unexpected pre-existing model hooks')
            local.certify_background(ctx,teacher,student)
        for name in names:
            evaluate_module(ctx,teacher,student,name,stop,pilot)
            if not pilot: summarize(ctx,verify=False)
        local.certify_background(ctx,teacher,student)
    finally:
        teacher=student=None
        dual.clear_gpu()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('doctor','pilot','prepare','run','summarize','pack'))
    for key in ('run-dir','repo-dir','output-dir','source-fp64-dir','source-rank64-dir','source-local-dir',
                'official-qera-root','harness-source','word-reference-dir'):
        p.add_argument('--'+key,required=True,type=Path)
    p.add_argument('--max-hours',type=float,default=10)
    p.add_argument('--max-new-units',type=int)
    args=p.parse_args(argv)
    if not math.isfinite(args.max_hours) or args.max_hours<=0 or (args.max_new_units is not None and args.max_new_units<1): p.error('Positive finite budgets required')
    stop=prev.Budget(args.max_hours,args.max_new_units)
    signal.signal(signal.SIGINT,stop.signal);signal.signal(signal.SIGTERM,stop.signal)
    try: ctx=setup(args,stop)
    except prev.Paused:
        core.log('PAUSED (75) during source audit; repeat identical command');return 75
    if args.command=='doctor':
        core.log('DOCTOR PASS: frozen sources verified; no new roots, SVD or forwards');return 0
    require(not ctx.output.exists() or (ctx.output/'experiment.json').exists() or not any(ctx.output.iterdir()),'Refusing unknown nonempty output')
    ctx.output.mkdir(parents=True,exist_ok=True)
    gate={'experiment_identity':ctx.identity,'status':'PASS','modules':list(PILOT_MODULES),'windows':2,'arms':list(ARMS)}
    with core.audit_lock(ctx.output):
        dual.freeze_json(ctx.output/'experiment.json',ctx.experiment)
        dual.freeze_json(ctx.output/'source_baselines.json',{'experiment_identity':ctx.identity,'source_states':ctx.old_records,
            'rows':{n:[{k:v for k,v in r.items() if k!='file'} for r in rr] for n,rr in ctx.old_rows.items()}})
        try:
            if args.command=='summarize': summarize(ctx);return 0
            if args.command=='pack':
                require(core.read_json(ctx.output/'pilot.json')==gate,'Matching pilot required');pack(ctx);return 0
            require(shutil.disk_usage(ctx.output).free>=24*2**30,'Need at least 24 GiB free for dense frozen roots, factors and paired token checkpoints')
            if args.command!='pilot': require(core.read_json(ctx.output/'pilot.json')==gate,'Run matching pilot first')
            prepare(ctx,stop,PILOT_MODULES if args.command=='pilot' else ctx.modules)
            if args.command=='prepare':
                summarize(ctx,verify=False);core.log('PREPARATION COMPLETE: all 168 permuted factors; run evaluation next');return 0
            run(ctx,stop,pilot=args.command=='pilot')
            if args.command=='pilot':
                dual.freeze_json(ctx.output/'pilot.json',gate)
                core.log('PILOT COMPLETE: 12 new factors; 2 modules x 9 arms x 2 windows with historical replay')
            else:
                pack(ctx);core.log('G PERMUTATION EXPERIMENT COMPLETE; report all modules/seeds including contrary outcomes')
        except prev.Paused:
            core.log('PAUSED (75): repeat identical command/output; resume at factor or complete paired-window boundary');return 75
        except Exception as exc:
            core.atomic_json(ctx.output/'last_failure.json',{'experiment_identity':ctx.identity,'type':type(exc).__name__,'error':str(exc)})
            raise
    return 0


if __name__=='__main__': raise SystemExit(main())
