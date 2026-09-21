"""Full-model dense FP32 deployment and bounded-vocabulary scoring."""
import gc
import math
import sys
import time
import torch
import torch.nn.functional as F
from fm_common import *
from data import checked_ids, task_dict, task_evidence

class Deployment:
    def __init__(self,ctx):
        self.ctx=ctx;ctx.teacher.load();self.model=ctx.teacher.model
        self.expected=read(ctx.root/'teacher_identity.json')['tensor_hashes']
        self.originals={name(i,k):self.model.get_submodule(name(i,k)).weight.detach().cpu().clone()
                        for i in range(32) for k in KINDS}
        self.state='Teacher'

    def activate(self,state):
        ctx=self.ctx;candidate=read(ctx.root/'candidates'/f'{state}.json')
        for key,row in candidate['modules'].items():
            target=self.model.get_submodule(key).weight
            if state=='Teacher':weight=self.originals[key]
            else:
                require(sha(row['path'])==row['file_sha256'],'Deployment Wq file changed')
                weight=load_file(row['path'])['Wq'].to(target.device)
                require(mo.digest_tensor(weight)==row['Wq_hash'],'Deployment Wq tensor changed')
                if row['family']:
                    require(sha(row['factor'])==row['factor_sha256'],'Deployment factors changed')
                    f=load_file(row['factor'],device=str(target.device))
                    weight=weight+(f['P64']@f['Q64']).float()
                    del f
            with torch.no_grad():target.copy_(weight.to(target.device))
            require(mo.digest_tensor(target)==row['W_deploy_hash'],'Candidate deployed weight hash differs')
        self.verify_non_targets()
        self.state=state
        write(ctx.root/'verification/deployment'/f'{state}.json',dict(passed=True,candidate_identity=candidate['identity'],
              modules=224,all_deployment_hashes_verified=True,non_target_parameters_verified=True))
        log('CANDIDATE_ACTIVE',state=state,identity=candidate['identity'])

    def verify_non_targets(self):
        for key,value in self.model.named_parameters():
            if key.removesuffix('.weight') not in self.originals:
                require(mo.digest_tensor(value)==self.expected[key]['hash'],'Non-target teacher weight changed: '+key)

    def restore(self):
        for key,weight in self.originals.items():
            target=self.model.get_submodule(key).weight
            with torch.no_grad():target.copy_(weight.to(target.device))
            require(mo.digest_tensor(target)==self.expected[key+'.weight']['hash'],'Teacher restoration failed')
        self.verify_non_targets();self.state='Teacher'

def score_hidden(hidden,ids,weight,chunk=64,reference=None):
    losses=[];kls=[]
    with torch.no_grad():
        for start in range(0,ids.shape[1]-1,chunk):
            end=min(ids.shape[1]-1,start+chunk)
            logits=F.linear(hidden[0,start:end],weight)
            require(torch.isfinite(logits).all(),'Nonfinite candidate logits')
            targets=ids[0,start+1:end+1].to(logits.device)
            losses.append(-logits.double().log_softmax(-1).gather(1,targets[:,None]).flatten().cpu())
            if reference is not None:
                teacher_logits=F.linear(reference[0,start:end].to(weight.device),weight)
                kls.append(mo.stable_kl(teacher_logits,logits).cpu())
    row=dict(NLL_sum=float(torch.cat(losses).sum()),tokens=ids.shape[1]-1)
    if kls:
        value=torch.cat(kls);require(torch.isfinite(value).all() and float(value.min())>=0,'Invalid KL')
        row.update(KL_sum=float(value.sum()),KL=float(value.mean()))
    return row

def eval_kl_ppl(ctx):
    require(ctx.done('candidates/complete.json',states=6),'Candidates must be frozen before evaluation')
    if ctx.done('ppl/complete.json',states=6,corpora=2,KL_windows=16):return
    deployment=Deployment(ctx);model=deployment.model
    data={r:checked_ids(ctx,r) for r in ('validation','wikitext2','c4')}
    # 16*2048*4096*4 = 0.5 GiB; hidden-state cache, never full logits.
    reference=[]
    with torch.no_grad():
        for ids in data['validation']:
            reference.append(hidden_forward(model,ids[None]).cpu())
    require(sum(h.numel()*h.element_size() for h in reference)<=512*2**20,'Teacher reference cache bound exceeded')
    files=[]
    try:
        for state in STATES:
            candidate=read(ctx.root/'candidates'/f'{state}.json')
            with ctx.timed('candidate_switch',state=state):deployment.activate(state)
            for role in ('validation','wikitext2','c4'):
                cp=f'{"kl" if role=="validation" else "ppl"}/{state}/{role}_complete.json'
                if ctx.done(cp,candidate=candidate['identity'],windows=len(data[role])):
                    files.append(ctx.root/cp);continue
                paths=[];start=time.monotonic()
                for window,ids in enumerate(data[role]):
                    path=ctx.root/('kl' if role=='validation' else 'ppl')/state/role/f'w{window:04d}.json'
                    if path.exists():
                        old=read(path);require(old['identity']==ctx.identity and old['candidate']==candidate['identity'] and
                            old['input_hash']==mo.digest_tensor(ids) and old['tokens']==T,'Scoring resume identity differs')
                    else:
                        with torch.no_grad():h=hidden_forward(model,ids[None])
                        row=score_hidden(h,ids[None],model.lm_head.weight,ctx.config['vocab_chunk'],
                                         reference[window] if role=='validation' else None)
                        if state=='Teacher' and role=='validation':require(row['KL']<=1e-12,'Teacher self KL gate failed')
                        write(path,dict(identity=ctx.identity,candidate=candidate['identity'],state=state,corpus=role,
                            window=window,input_hash=mo.digest_tensor(ids),**row))
                        del h
                    paths.append(path)
                    if (window+1)%16==0 or window+1==len(data[role]):
                        log('EVAL_PROGRESS',state=state,corpus=role,window=window+1,total=len(data[role]),
                            eta_seconds=(len(data[role])-window-1)*(time.monotonic()-start)/(window+1))
                ctx.commit(cp,paths,candidate=candidate['identity'],windows=len(data[role]));files.append(ctx.root/cp)
        ctx.commit('ppl/complete.json',files,states=6,corpora=2,KL_windows=16)
    finally:
        deployment.restore();del deployment,reference;ctx.teacher.unload();gc.collect()

def make_harness_model(ctx):
    sys.path.insert(0,ctx.config['harness'])
    from lm_eval.models.huggingface import HFLM
    from transformers import AutoTokenizer
    class ChunkedHFLM(HFLM):
        """Identical causal tokenization/truncation, batch1, bounded LM-head tensors."""
        def _loglikelihood_tokens(self,requests,disable_tqdm=False,override_bs=None):
            output=[];tick=time.monotonic()
            for index,(cache_key,context,continuation) in enumerate(requests):
                require(len(context)>0 and 0<len(continuation)<=self.max_length,'Invalid harness token request')
                source=(context+continuation)[-(self.max_length+1):][:-1]
                ids=torch.tensor([source],dtype=torch.long,device=self.model.model.embed_tokens.weight.device)
                with torch.no_grad():
                    h=hidden_forward(self.model,ids);begin=len(source)-len(continuation);parts=[];greedy=True
                    for start in range(0,len(continuation),ctx.config['vocab_chunk']):
                        end=min(len(continuation),start+ctx.config['vocab_chunk'])
                        logits=F.linear(h[0,begin+start:begin+end],self.model.lm_head.weight)
                        require(torch.isfinite(logits).all(),'Nonfinite downstream logits')
                        targets=torch.tensor(continuation[start:end],device=logits.device)
                        lp=F.log_softmax(logits,dim=-1)
                        parts.append(lp.gather(1,targets[:,None]).flatten().cpu())
                        greedy=greedy and bool(torch.equal(lp.argmax(-1),targets))
                    answer=(float(torch.cat(parts).sum()),greedy)
                output.append(answer);self.cache_hook.add_partial('loglikelihood',cache_key,answer)
                if (index+1)%100==0 or index+1==len(requests):
                    log('DOWNSTREAM_REQUESTS',completed=index+1,total=len(requests),
                        eta_seconds=(len(requests)-index-1)*(time.monotonic()-tick)/(index+1))
            return output
    tokenizer=AutoTokenizer.from_pretrained(ctx.config['model'],local_files_only=True)
    return ChunkedHFLM(pretrained=ctx.teacher.model,tokenizer=tokenizer,batch_size=1,max_length=4096,
                       dtype='float32',logits_cache=False,add_bos_token=False,parallelize=False)

def eval_downstream(ctx):
    require(ctx.done('ppl/complete.json',states=6,corpora=2,KL_windows=16),'KL/PPL stage incomplete')
    if ctx.done('downstream/complete.json',model_tasks=25):return
    import os
    os.environ['HF_DATASETS_OFFLINE']='1';os.environ['HF_HUB_OFFLINE']='1'
    from lm_eval import evaluator
    from lm_eval.utils import handle_non_serializable
    expected=read(ctx.root/'eval_manifest.json')['downstream']
    for key,value in expected['source'].items():require(sha(Path(ctx.config['harness'])/key)==value,'Harness source changed')
    deployment=Deployment(ctx);lm=make_harness_model(ctx);files=[]
    try:
        for state in ('Teacher','C0','C2','C3','C4'):
            candidate=read(ctx.root/'candidates'/f'{state}.json')
            with ctx.timed('candidate_switch',state=state):deployment.activate(state)
            for task_name,metric in TASKS.items():
                cp=f'downstream/{state}/{task_name}/complete.json'
                if ctx.done(cp,candidate=candidate['identity']):files.append(ctx.root/cp);continue
                tasks=task_dict(ctx,[task_name]);task=tasks[task_name]
                actual=task_evidence(task,metric)
                require(actual==expected['tasks'][task_name],'Task/data identity changed before scoring: '+task_name)
                with ctx.timed('downstream_task',state=state,task=task_name):
                    result=evaluator.simple_evaluate(model=lm,tasks=[task],num_fewshot=0,batch_size=1,
                        limit=None,bootstrap_iters=0,log_samples=True,use_cache=None,cache_requests=False,
                        apply_chat_template=False,random_seed=20260921,numpy_random_seed=20260921,
                        torch_random_seed=20260921,fewshot_random_seed=20260921)
                count=result['n-samples'][task_name]
                require(count['effective']==count['original']==actual['documents'],'Incomplete downstream task')
                value=result['results'][task_name][metric+',none'];require(math.isfinite(value),'Nonfinite task metric')
                folder=ctx.root/'downstream'/state/task_name;folder.mkdir(parents=True,exist_ok=True)
                # Preserve all options/responses/documents/metrics in the native harness samples.
                samples=result.pop('samples');sample_path=folder/'samples.json'
                write(sample_path,json.loads(json.dumps(samples,default=handle_non_serializable)))
                path=folder/'results.json';write(path,json.loads(json.dumps(result,default=handle_non_serializable)))
                ctx.commit(cp,[path,sample_path],candidate=candidate['identity'],task=task_name,metric=metric,
                    value=float(value),documents=actual['documents'],dataset_sha256=actual['content_sha256'])
                files.append(ctx.root/cp)
        ctx.commit('downstream/complete.json',files,model_tasks=25)
    finally:
        deployment.restore();del lm,deployment;ctx.teacher.unload();gc.collect()
