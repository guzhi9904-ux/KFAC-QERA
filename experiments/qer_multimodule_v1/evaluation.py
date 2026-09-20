"""Paired validation ablations and jointly deployed official-test KL/PPL."""
import math
import torch
from bridge import TensorStore,read,save_json,save_csv,mo,slug,require,commit,load_record,checked_files,sha_file
from dataset import windows


def evaluation_jobs(names,keys,role):
    if role=='validation':return [(name,key,[name]) for name in names for key in keys]
    require(role=='test','Invalid evaluation role');return [('joint12',key,names) for key in keys]


def evaluate(root,config,identity,resources,teacher):
    frozen=load_record(root/'candidate_freeze.json',identity);checked_files(root,frozen['files'])
    store=TensorStore(identity,config['cache_GiB']);freeze_hash=sha_file(root/'candidate_freeze.json')
    for role in ('validation','test'):
        ids_all=windows(root,role,identity);jobs=evaluation_jobs(config['modules'],frozen['keys'],role)
        for c in range(len(ids_all)):
            resources.boundary();folder=root/'scores'/role/f'w{c:04d}'
            paths=[folder/(slug(scope)+'___'+key+'.json') for scope,key,names in jobs]
            teacher_path=folder/'teacher.json'
            if all(p.exists() for p in paths+[teacher_path]):
                for p in paths+[teacher_path]:load_record(p,identity)
                continue
            ids=ids_all[c:c+1];token_hash=mo.digest_tensor(ids[0])
            with resources.timed('eval_shared_reference',role=role,window=c):
                # No hooks or gradient graph on the evaluation data.
                from bridge import hidden_forward
                teacher.load()
                with torch.no_grad():ref=hidden_forward(teacher.model,ids).detach()
                logits=teacher.reference_logits(ref)
                baseline=teacher.scores(ids,logits);require(abs(baseline['KL'])<=1e-10,'Nonzero teacher self-KL')
                if teacher_path.exists():
                    old=load_record(teacher_path,identity)
                    require(old['token_hash']==token_hash and abs(old['scores']['NLL_sum']-baseline['NLL_sum'])<=1e-7,'Teacher replay differs')
                else:commit(teacher_path,identity,token_hash=token_hash,scores=baseline)
            for (scope,key,names),path in zip(jobs,paths):
                resources.boundary()
                if path.exists():
                    row=load_record(path,identity);require(row['freeze_hash']==freeze_hash and row['token_hash']==token_hash,'Score binding changed');continue
                weights={}
                for name in names:
                    p=root/'quantized'/(slug(name)+'.safetensors') if key=='None' else root/'modules'/slug(name)/'corrections'/(key+'.safetensors')
                    weights[name]=store.get(p)[0]['Wq' if key=='None' else 'W_deploy']
                with resources.timed('actual_KL_PPL',role=role,window=c,scope=scope,candidate=key):
                    with teacher.deploy(weights):
                        result=teacher.scores(ids,logits)
                        require(result['tokens']==config['sequence_length']-1 and math.isfinite(result['KL']+result['NLL_sum']) and result['KL']>=0,'Invalid KL/PPL')
                        repeat=None
                        if c==0:
                            second=teacher.scores(ids,logits);repeat=abs(second['KL']-result['KL'])
                            require(repeat<=max(1e-12,abs(result['KL'])*1e-7) and abs(second['NLL_sum']-result['NLL_sum'])<=1e-7,'Evaluation repeat differs')
                    commit(path,identity,role=role,window=c,scope=scope,candidate=key,modules=names,scores=result,
                        repeat_absolute_error=repeat,token_hash=token_hash,freeze_hash=freeze_hash)
                del weights
            del ref,logits;store.clear()
            print('EVAL_WINDOW_COMPLETE',role,c+1,'/',len(ids_all),flush=True)
    teacher.unload();report(root,config,identity)


def report(root,config,identity):
    frozen=load_record(root/'candidate_freeze.json',identity);checked_files(root,frozen['files'])
    index=read(root/'data/index.json');rows=[]
    for role in ('validation','test'):
        expected=index['windows'][role]
        jobs=evaluation_jobs(config['modules'],frozen['keys'],role)+[('teacher','teacher',[])]
        for scope,key,names in jobs:
            values=[]
            for c in range(expected):
                folder=root/'scores'/role/f'w{c:04d}'
                p=folder/'teacher.json' if scope=='teacher' else folder/(slug(scope)+'___'+key+'.json')
                values.append(load_record(p,identity)['scores'])
            tokens=sum(v['tokens'] for v in values);nll=sum(v['NLL_sum'] for v in values)
            rows.append(dict(role=role,scope=scope,candidate=key,windows=expected,tokens=tokens,
                KL=sum(v['KL']*v['tokens'] for v in values)/tokens,NLL=nll/tokens,PPL=math.exp(nll/tokens)))
    baselines={(r['role'],r['scope']):r['KL'] for r in rows if r['candidate']=='None'}
    for row in rows:
        base=baselines.get((row['role'],row['scope']))
        row['KL_recovery_percent']=None if base is None or base<=1e-12 else 100*(1-row['KL']/base)
    save_csv(root/'summary/results.csv',rows);save_json(root/'summary/results.json',dict(identity=identity,rows=rows))
    commit(root/'complete.json',identity,passed=True,modules=config['modules'],rows=len(rows),
        meaning='Single-target validation; jointly quantize/compensate12 specified targets on official test. Other weights FP32.',
        excludes=['eval_gradients','Gram','curvature_fidelity','empirical_bounds'],
        limitations='Disjoint2048-token PPL protocol; not full-model compression, not sliding-window benchmark. Shared teacher gradients do not account for accumulated quantization shifts.')
    print('EXPERIMENT_COMPLETE',flush=True)
