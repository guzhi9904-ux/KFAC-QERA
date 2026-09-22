import math
import torch.nn.functional as F
from common import *
from deployment import Deployment
from benchmarks import harness_evaluate,harness_model

def token_ids(ctx,role):
    row=read(ctx.root/'data_manifest.json')['data']['tokens_'+role]
    return load_file(str(ctx.verify(row['path'],row['sha256'])))['input_ids']

def score_hidden(h,ids,weight,chunk=64,reference=None):
    require(h.dtype==weight.dtype==torch.bfloat16,'Token scorer model dtype is not BF16')
    losses=[];kls=[]
    for start in range(0,ids.shape[1]-1,chunk):
        end=min(ids.shape[1]-1,start+chunk)
        logits=F.linear(h[0,start:end],weight)
        require(logits.dtype==torch.bfloat16 and torch.isfinite(logits).all(),'Invalid BF16 logits')
        target=ids[0,start+1:end+1].to(logits.device)
        losses.append(-logits.double().log_softmax(-1).gather(1,target[:,None]).flatten().cpu())
        if reference is not None:
            teacher=F.linear(reference[0,start:end].to(weight.device),weight)
            kls.append(math_ops.stable_kl(teacher,logits).cpu())
    row=dict(NLL_sum=float(torch.cat(losses).sum()),tokens=ids.shape[1]-1)
    if kls:
        kl=torch.cat(kls);require(torch.isfinite(kl).all() and float(kl.min())>=0,'Invalid teacher KL')
        row.update(KL_sum=float(kl.sum()),KL=float(kl.mean()))
    return row

def pilot(ctx):
    require(ctx.done('data/complete.json'),'Prepare before pilot')
    if ctx.done('verification/pilot_complete.json'):return
    dep=Deployment(ctx);checks={};ids=token_ids(ctx,'validation')
    try:
        for state in ctx.config['ppl_states']:
            dep.activate(state);errors={}
            # Independent native module calls exercise correction orientation and BF16 arithmetic.
            with torch.inference_mode():
                for key,(a,b) in dep.bits.items():
                    if int(key.split('.')[2]) not in (0,16):continue
                    module=dep.model.get_submodule(key)
                    x=torch.linspace(-.1,.1,4*module.in_features,device=module.weight.device).reshape(1,4,-1).bfloat16()
                    actual=module(x);reference=F.linear(x,module.weight,module.bias)+(x@a)@b
                    require(torch.equal(actual,reference),'Actual BF16 correction differs: '+key)
                    errors[key]=dict(exact=True,A_dtype=str(a.dtype),B_dtype=str(b.dtype),output_dtype=str(actual.dtype))
                before=dict(dep.hits);h=hidden(dep.model,ids[:1]);row=score_hidden(h,ids[:1],dep.model.lm_head.weight,
                                                reference=h if state=='Teacher' else None)
                require(all(dep.hits[k]>before[k] for k in before),'Pilot skipped a target')
                if state=='Teacher':require(row['KL']==0,'BF16 teacher self-KL failed')
                require(math.isfinite(row['NLL_sum']),'Invalid pilot loss');del h
            checks[state]=dict(module_checks=errors,all_224_executed=True,correction_count=len(dep.bits))
        # Native harness (no custom logits/scoring adapter): validate worst compensated arm and teacher.
        for state in ('Teacher','C4'):
            dep.activate(state)
            with torch.inference_mode():
                lm=harness_model(ctx,dep.model)
                request=[(None,torch.cat([ids[0],ids[1]]).tolist(),ids[2,:1].tolist())]
                score=lm._loglikelihood_tokens(request,disable_tqdm=True)
                require(math.isfinite(score[0][0]),'4096-token native harness pilot failed');del lm
            for task in [*ctx.config['tasks'],'wikitext']:
                result,summary,hits=harness_evaluate(ctx,dep,task,pilot=True)
                checks[state+'_'+task]=dict(summary=summary,all_224_executed=True)
        write(ctx.root/'verification/pilot.json',checks)
        ctx.commit('verification/pilot_complete.json',[ctx.root/'verification/pilot.json'],states=6,
                   BF16_base_and_factors=True,native_harness=True,score_selection=False)
        log('BF16_PILOT_ACCEPTED',states=6,native_harness='0.4.7',context=4096)
    finally:dep.close()

def evaluate_tokens(ctx,dep):
    if ctx.done('tokens/complete.json',states=6):return
    corpora={role:token_ids(ctx,role) for role in ('validation','wikitext2','c4')}
    dep.activate('Teacher')
    with torch.inference_mode():reference=[hidden(dep.model,ids[None]).cpu() for ids in corpora['validation']]
    files=[]
    for state in ctx.config['ppl_states']:
        dep.activate(state)
        for role,ids_all in corpora.items():
            cp=f'tokens/{state}/{role}/complete.json'
            if ctx.done(cp,candidate=dep.candidate,windows=len(ids_all)):
                files.append(ctx.root/cp);continue
            with timed(ctx,'token_evaluation',state=state,corpus=role),torch.inference_mode():
                paths=[];records=[];tick=time.monotonic();new=0
                for window,ids in enumerate(ids_all):
                    path=ctx.root/'tokens'/state/role/f'w{window:04d}.json'
                    if path.exists():
                        row=read(path)
                        require(row['identity']==ctx.identity and row['candidate']==dep.candidate and
                                row['input_hash']==tensor_hash(ids) and row['window']==window and row['tokens']==2047,
                                'Token result resume binding differs')
                    else:
                        h=hidden(dep.model,ids[None]);row=score_hidden(h,ids[None],dep.model.lm_head.weight,
                            ctx.config['vocab_chunk'],reference[window] if role=='validation' else None);del h
                        row.update(identity=ctx.identity,candidate=dep.candidate,state=state,corpus=role,window=window,input_hash=tensor_hash(ids))
                        write(path,row);new+=1
                    paths.append(path);records.append(row)
                    if (window+1)%16==0 or window+1==len(ids_all):
                        log('TOKEN_PROGRESS',state=state,corpus=role,completed=window+1,total=len(ids_all),
                            eta_seconds=(len(ids_all)-window-1)*(time.monotonic()-tick)/max(1,new))
                tokens=sum(r['tokens'] for r in records);nll=math.fsum(r['NLL_sum'] for r in records)
                summary=dict(NLL=nll/tokens,PPL=math.exp(nll/tokens),tokens=tokens)
                if role=='validation':summary['KL']=math.fsum(r['KL_sum'] for r in records)/tokens
                if state=='Teacher' and role=='validation':require(summary['KL']==0,'Teacher KL differs')
                ctx.commit(cp,paths,candidate=dep.candidate,windows=len(ids_all),**summary);files.append(ctx.root/cp)
    ctx.commit('tokens/complete.json',files,states=6)

def evaluate_harness(ctx,dep):
    files=[]
    jobs=[(state,'wikitext') for state in ctx.config['ppl_states']]
    jobs += [(state,task) for state in ctx.config['downstream_states'] for task in ctx.config['tasks']]
    current=None
    for state,task in jobs:
        candidate=read(ctx.root/'candidates'/f'{state}.json')['identity'];cp=f'harness/{state}/{task}/complete.json'
        if ctx.done(cp,candidate=candidate):files.append(ctx.root/cp);continue
        if current!=state:dep.activate(state);current=state
        result,summary,hits=harness_evaluate(ctx,dep,task)
        folder=ctx.root/'harness'/state/task
        samples=result.pop('samples');write(folder/'samples.json',samples);write(folder/'results.json',result)
        ctx.commit(cp,[folder/'samples.json',folder/'results.json'],candidate=candidate,summary=summary,
                   all_224_executed=True,forward_calls=hits)
        files.append(ctx.root/cp);log('HARNESS_COMMITTED',state=state,task=task,summary=summary)
        report(ctx)
    ctx.commit('harness/complete.json',files,word_ppl_states=6,downstream_jobs=25)

def evaluate(ctx):
    require(ctx.done('verification/pilot_complete.json'),'BF16 pilot required')
    if ctx.done('complete.json'):return
    dep=Deployment(ctx)
    try:
        evaluate_tokens(ctx,dep);report(ctx);evaluate_harness(ctx,dep)
    finally:dep.close()
    report(ctx)

def report(ctx):
    import csv
    rows=[];missing=[]
    for state in ctx.config['ppl_states']:
        row=dict(state=state)
        for role in ('validation','wikitext2','c4'):
            p=ctx.root/'tokens'/state/role/'complete.json';record=read(p) if p.exists() else {}
            row[role+'_token_PPL']=record.get('PPL')
            if role=='validation':row['KL16']=record.get('KL')
            if not record:missing.append(f'{state}/tokens/{role}')
        p=ctx.root/'harness'/state/'wikitext/complete.json'
        row['wikitext_word_PPL']=read(p)['summary']['word_PPL'] if p.exists() else None
        if not p.exists():missing.append(f'{state}/word_PPL')
        scores=[]
        for task in ctx.config['tasks']:
            p=ctx.root/'harness'/state/task/'complete.json'
            value=read(p)['summary']['score']*100 if p.exists() else None;row[task+'_percent']=value
            if state in ctx.config['downstream_states']:
                if value is None:missing.append(f'{state}/{task}')
                else:scores.append(value)
        row['SRR_5_mean_percent']=sum(scores)/5 if len(scores)==5 else None;rows.append(row)
    folder=ctx.root/'summary';folder.mkdir(exist_ok=True)
    with (folder/'results.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write(folder/'results.json',dict(identity=ctx.identity,rows=rows,missing=missing))
    if not missing and ctx.done('harness/complete.json',word_ppl_states=6,downstream_jobs=25):
        require(ctx.done('tokens/complete.json',states=6),'Global token completion receipt missing')
        ctx.commit('complete.json',[folder/'results.csv',folder/'results.json',ctx.root/'tokens/complete.json',ctx.root/'harness/complete.json'])
        log('BF16_SRR_EXPERIMENT_COMPLETE')
