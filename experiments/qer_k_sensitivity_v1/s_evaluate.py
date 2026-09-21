import torch
from s_common import MODULES,BASELINES,TensorStore,slug,mo,require,load_record,checked_files,sha_file,commit
from bridge import hidden_forward


def evaluate(root,config,ident,resources,teacher,source):
    frozen=load_record(root/'candidate_freeze.json',ident);checked_files(root,frozen['files'])
    freeze_hash=sha_file(root/'candidate_freeze.json');store=TensorStore(ident,0)
    for window in range(16):
        resources.boundary();folder=root/'scores/validation'/f'w{window:04d}'
        targets=[folder/(slug(n)+'.json') for n in MODULES]
        if all(p.exists() for p in targets+[folder/'teacher.json']):
            for p in targets+[folder/'teacher.json']:load_record(p,ident)
            if window==0:load_record(root/'scores/baseline_replay.json',ident)
            continue
        ids=source.validation[window:window+1];token_hash=mo.digest_tensor(ids[0])
        with resources.timed('validation_teacher_reference',window=window):
            teacher.load()
            with torch.no_grad():ref=hidden_forward(teacher.model,ids).detach()
            logits=teacher.reference_logits(ref);score=teacher.scores_hidden(ids,logits,ref)
            require(abs(score['KL'])<=1e-10,'Nonzero teacher self-KL')
            if not (folder/'teacher.json').exists():commit(folder/'teacher.json',ident,token_hash=token_hash,scores=score)
        if window==0:
            checks=[]
            for n in MODULES:
                for method in BASELINES:
                    key='None' if method=='None' else 'N256__'+method
                    p=source.root/'quantized'/(slug(n)+'.safetensors') if method=='None' else source.root/'modules'/slug(n)/'corrections'/(key+'.safetensors')
                    w=source.store.get(p)[0]['Wq' if method=='None' else 'W_deploy']
                    with resources.timed('baseline_KL_replay',module=n,candidate=key),teacher.deploy({n:w}):actual=teacher.scores(ids,logits)
                    expected=load_record(source.root/'scores/validation/w0000'/(slug(n)+'___'+key+'.json'),source.identity)
                    require(expected['token_hash']==token_hash,'Baseline token changed')
                    error=abs(actual['KL']-expected['scores']['KL']);tolerance=max(1e-12,1e-7*abs(expected['scores']['KL']))
                    require(error<=tolerance,'Baseline replay failed; cannot combine old and new scores')
                    checks.append(dict(module=n,candidate=key,absolute_error=error,tolerance=tolerance,passed=True))
            cp=root/'scores/baseline_replay.json'
            if not cp.exists():commit(cp,ident,passed=True,checks=checks)
        for n,p in zip(MODULES,targets):
            if p.exists():
                old=load_record(p,ident);require(old['token_hash']==token_hash and old['freeze_hash']==freeze_hash,'New score binding changed');continue
            w=store.get(root/'corrections'/slug(n)/'sensitivity_weighted.safetensors')[0]['W_deploy']
            with resources.timed('new_actual_KL',window=window,module=n),teacher.deploy({n:w}):
                actual=teacher.scores(ids,logits);repeat=None
                if window==0:
                    second=teacher.scores(ids,logits);repeat=abs(actual['KL']-second['KL'])
                    require(repeat<=max(1e-12,1e-7*abs(actual['KL'])),'New candidate repeat differs')
            require(actual['tokens']==2047 and torch.isfinite(torch.tensor(actual['KL'])) and actual['KL']>=0,'Invalid KL')
            commit(p,ident,module=n,window=window,candidate='Sensitivity-weighted Marginal',token_hash=token_hash,freeze_hash=freeze_hash,scores=actual,repeat_absolute_error=repeat)
        del ref,logits;print('VALIDATION_WINDOW_COMPLETE',window+1,'/16',flush=True)
    teacher.unload();print('CHECK3_ALL64_NEW_KL_COMPLETE',flush=True)
