import math
import torch
from ko_common import (jobs,NEW,METHODS,slug,load_record,checked_files,sha_file,mo,require,TensorStore,
                       commit,save_csv,save_json,file_table)
from ko_replay import capture_blocks,suffix_hidden
from bridge import hidden_forward


def evaluate(root,config,ident,resources,teacher,source,acceptance):
    frozen=load_record(root/'candidate_freeze.json',ident);checked_files(root,frozen['files'])
    freeze_hash=sha_file(root/'candidate_freeze.json');store=TensorStore(ident,config['cache_GiB'])
    for window in range(16):
        resources.boundary();folder=root/'scores/validation'/f'w{window:04d}';teacher_path=folder/'teacher.json'
        tasks=jobs(config);paths=[folder/(slug(n)+'___'+('None' if m=='None' else 'N256__'+m)+'.json') for n,m in tasks]
        if all(p.exists() for p in paths+[teacher_path]):
            for p in paths+[teacher_path]:load_record(p,ident)
            continue
        ids=source.validation[window:window+1];token_hash=mo.digest_tensor(ids[0])
        with resources.timed('validation_reference',window=window):
            teacher.load();ref,states=capture_blocks(teacher.model,ids);logits=teacher.reference_logits(ref)
            score=teacher.scores_hidden(ids,logits,ref)
            old=load_record(source.root/'scores/validation'/f'w{window:04d}'/'teacher.json',source.identity)
            require(old['token_hash']==token_hash and abs(old['scores']['NLL_sum']-score['NLL_sum'])<=1e-7 and abs(score['KL'])<=1e-10,'Parent validation teacher replay differs')
            if not teacher_path.exists():commit(teacher_path,ident,token_hash=token_hash,scores=score)
            else:require(load_record(teacher_path,ident)['token_hash']==token_hash,'Teacher token binding changed')
        for (name,method),path in zip(tasks,paths):
            resources.boundary();key='None' if method=='None' else 'N256__'+method;layer=int(name.split('.')[2])
            if path.exists():
                old=load_record(path,ident);require(old['token_hash']==token_hash and old['freeze_hash']==freeze_hash,'Score binding changed');continue
            weight_path=root/'quantized'/(slug(name)+'.safetensors') if method=='None' else root/'modules'/slug(name)/'corrections'/(key+'.safetensors')
            w=store.get(weight_path)[0]['Wq' if method=='None' else 'W_deploy'];audit=None
            with resources.timed('actual_KL_PPL',window=window,module=name,candidate=key),teacher.deploy({name:w}),torch.no_grad():
                if acceptance['evaluation_mode']=='cached_prefix':
                    h=suffix_hidden(teacher.model,layer,states[layer]);score=teacher.scores_hidden(ids,logits,h)
                    if window==0:
                        full_h=hidden_forward(teacher.model,ids);full=teacher.scores_hidden(ids,logits,full_h)
                        relative=mo.relative(h,full_h)
                        require(relative<=config['forward_tolerance'] and abs(full['KL']-score['KL'])<=max(1e-12,abs(full['KL'])*1e-7) and abs(full['NLL_sum']-score['NLL_sum'])<=1e-7,'Actual candidate prefix check failed')
                        audit=dict(hidden_relative=relative,KL_absolute=abs(full['KL']-score['KL']));del full_h
                    del h
                else:score=teacher.scores(ids,logits)
                require(score['tokens']==2047 and score['KL']>=0 and math.isfinite(score['KL']+score['NLL_sum']),'Invalid score')
            commit(path,ident,role='validation',window=window,scope=name,candidate=key,modules=[name],scores=score,
                   token_hash=token_hash,freeze_hash=freeze_hash,prefix_audit=audit)
        del ref,states,logits;store.clear();print('VALIDATION_WINDOW_COMPLETE',window+1,'/16',flush=True)
    teacher.unload();report(root,config,ident,source)


def report(root,config,ident,source):
    rows=[];bindings=[];freeze_hash=sha_file(root/'candidate_freeze.json')
    old_freeze=sha_file(source.root/'candidate_freeze.json')
    for name in config['modules']:
        for method in METHODS+['None']:
            key='None' if method=='None' else 'N256__'+method;borrowed=name not in NEW and method!='A-only'
            src=source.root if borrowed else root;sid=source.identity if borrowed else ident;values=[]
            for window in range(16):
                p=src/'scores/validation'/f'w{window:04d}'/(slug(name)+'___'+key+'.json');d=load_record(p,sid)
                require(d['token_hash']==mo.digest_tensor(source.validation[window]) and d['freeze_hash']==(old_freeze if borrowed else freeze_hash)
                        and d['scope']==name and d['candidate']==key,'Report score binding differs')
                values.append(d['scores']);bindings.append(dict(path=str(p),sha256=sha_file(p)))
            tokens=sum(v['tokens'] for v in values);nll=sum(v['NLL_sum'] for v in values)/tokens
            rows.append(dict(module=name,method=method,budget=256,windows=16,tokens=tokens,KL=sum(v['KL']*v['tokens'] for v in values)/tokens,
                             PPL=math.exp(nll),source='parent_frozen' if borrowed else 'increment'))
    baseline={r['module']:r['KL'] for r in rows if r['method']=='None'}
    aonly={r['module']:r['KL'] for r in rows if r['method']=='A-only'}
    for r in rows:
        require(baseline[r['module']]>1e-12 and aonly[r['module']]>1e-12,'Degenerate recovery denominator')
        r['KL_recovery_percent']=100*(1-r['KL']/baseline[r['module']])
        r['KL_reduction_vs_A_only_percent']=100*(1-r['KL']/aonly[r['module']])
    save_csv(root/'summary/results.csv',rows)
    save_json(root/'summary/results.json',dict(identity=ident,rows=rows,record_sources=bindings,scope='20 single-module validation comparisons; no joint test; no q_full'))
    commit(root/'complete.json',ident,passed=True,modules=20,new_corrections=44,validation_windows=16,test_windows=0,
           files=file_table(root,[root/'summary/results.csv',root/'summary/results.json']))
    print('INCREMENT_EXPERIMENT_COMPLETE',flush=True)
