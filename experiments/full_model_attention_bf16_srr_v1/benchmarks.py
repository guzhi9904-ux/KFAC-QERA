"""Unmodified lm-eval 0.4.7 tasks, prompts, scoring and SRR aggregation."""
from common import *
from deployment import factor_cast

def task_tree(ctx,name):
    from lm_eval.tasks import TaskManager,get_task_dict
    manager=TaskManager(verbosity='WARNING')
    if name=='wikitext':return get_task_dict(['wikitext'],manager)
    return ctx.srr.load_tasks(name,manager)

def evidence(ctx,tasks):return ctx.srr.jsonable(ctx.srr.data_record(tasks))

def prepare(ctx):
    if ctx.done('data/complete.json'):return
    ctx.parent_complete('factors/complete.json',factors=416)
    ctx.parent_complete('candidates/complete.json',states=6)
    files=[];data={}
    for benchmark in [*ctx.config['tasks'],'wikitext']:
        with timed(ctx,'freeze_task',task=benchmark):
            record=evidence(ctx,task_tree(ctx,benchmark))
            old=Path(ctx.config['srr_reference'])/'data'/f'{benchmark}.json'
            if benchmark!='wikitext':require(record==read(old),'Task differs from existing SRR-aligned frozen protocol: '+benchmark)
            path=ctx.root/'data'/f'{benchmark}.json';freeze(path,record);files.append(path)
            data[benchmark]=dict(path=str(path),sha256=sha(path),documents=sum(r['count'] for r in record.values()),
                                 subtasks=len(record))
    parent_data=read(ctx.parent/'data_manifest.json')
    for role,count in [('validation',16),('wikitext2',138),('c4',128)]:
        path=ctx.parent/'data'/f'{role}.safetensors'
        ctx.verify(path,parent_data['datasets'][role]['sha256']);values=load_file(str(path))
        require(values['input_ids'].shape==(count,2048) and torch.equal(values['attention_mask'],torch.ones_like(values['input_ids'])),
                'Frozen token corpus changed')
        data['tokens_'+role]=dict(path=str(path),sha256=sha(path),windows=count,length=2048)
    freeze(ctx.root/'data_manifest.json',dict(identity=ctx.identity,data=data,downstream_fewshot=0,
        harness='0.4.7',chat_template=False,max_length=4096,word_ppl='native harness wikitext word_perplexity',
        token_ppl='exp(sum NLL_sum / sum prediction tokens); historical WT2 and C4 windows unchanged'))
    files.append(ctx.root/'data_manifest.json')
    # Candidate bits are frozen before viewing any BF16 scores.
    teacher=read(ctx.parent/'teacher_identity.json')['tensor_hashes'];quant=read(ctx.parent/'quantization_manifest.json')['modules']
    bases={}
    for key,row in quant.items():
        w=load_file(str(ctx.verify(row['path'],row['file_sha256'])))['Wq']
        require(tensor_hash(w)==row['Wq_hash'] and row['W0_hash']==teacher[key+'.weight']['hash'],'Parent Wq identity differs')
        require(torch.equal(w.bfloat16().float(),w),'MXINT3 Wq not exactly representable in BF16')
        bases[key]=tensor_hash(w.bfloat16())
    for state in ctx.config['ppl_states']:
        old=read(ctx.parent/'candidates'/f'{state}.json');modules={}
        for key,row in old['modules'].items():
            row=dict(row)
            if state=='Teacher':
                # Parent checkpoint originally stores BF16; verify against FP32 teacher below on load.
                from safetensors import safe_open
                index=read(ctx.model_path/'model.safetensors.index.json')
                with safe_open(str(ctx.model_path/index['weight_map'][key+'.weight']),framework='pt') as f:w=f.get_tensor(key+'.weight')
                require(w.dtype==torch.bfloat16 and tensor_hash(w.float())==row['W0_hash'],'Checkpoint teacher differs')
                row['base_BF16_hash']=tensor_hash(w)
            else:row['base_BF16_hash']=bases[key]
            if row['family']:
                a,b=factor_cast(load_file(str(ctx.verify(row['factor'],row['factor_sha256']))),row['shape'])
                row.update(A_BF16_hash=tensor_hash(a),B_BF16_hash=tensor_hash(b))
            modules[key]=row
        material=dict(run_identity=ctx.identity,state=state,parent_candidate=old['identity'],modules=modules,
                      data_sha256=sha(ctx.root/'data_manifest.json'),deployment=ctx.config['deployment'])
        path=ctx.root/'candidates'/f'{state}.json';freeze(path,dict(identity=digest(material),**material));files.append(path)
    ctx.commit('data/complete.json',files,states=6,statistics_reused=True,factors_reused=True)

def harness_model(ctx,model):
    from transformers import AutoTokenizer
    from lm_eval.models.huggingface import HFLM
    class ProgressHFLM(HFLM):
        # Retain all native scoring/cache/batching behavior. Only add finite-input assertions.
        def _loglikelihood_tokens(self,requests,*args,**kwargs):
            for _,context,continuation in requests:
                require(len(context)+len(continuation)<=self.max_length+1,'Request would truncate; freeze a revised protocol first')
            return super()._loglikelihood_tokens(requests,*args,**kwargs)
    tokenizer=AutoTokenizer.from_pretrained(str(ctx.model_path),local_files_only=True)
    return ProgressHFLM(pretrained=model,tokenizer=tokenizer,batch_size=1,max_length=4096,
                        add_bos_token=False,logits_cache=True)

def harness_evaluate(ctx,deployment,name,pilot=False):
    from lm_eval import simple_evaluate
    tasks=task_tree(ctx,name);data=evidence(ctx,tasks)
    require(data==read(ctx.root/'data'/f'{name}.json'),'Task identity changed: '+name)
    lm=harness_model(ctx,deployment.model)
    before=dict(deployment.hits)
    with timed(ctx,'harness',state=deployment.state,task=name,pilot=pilot),torch.inference_mode():
        result=simple_evaluate(model=lm,tasks=list(tasks.values()),num_fewshot=0,batch_size=1,
            limit=2 if pilot else None,bootstrap_iters=0,log_samples=True,use_cache=None,cache_requests=False,
            apply_chat_template=False,random_seed=0,numpy_random_seed=1234,torch_random_seed=1234,fewshot_random_seed=1234)
    result=ctx.srr.jsonable(result)
    hits={key:deployment.hits[key]-before[key] for key in before}
    require(min(hits.values())>0 and len(set(hits.values()))==1,'Missing/uneven target execution')
    if name=='wikitext':
        count=result['n-samples']['wikitext'];expected=min(2,data['wikitext']['count']) if pilot else data['wikitext']['count']
        require(count['effective']==expected,'Incomplete word-PPL')
        value=result['results']['wikitext']['word_perplexity,none']
        require(value>0 and __import__('math').isfinite(value),'Invalid word-PPL')
        summary=dict(word_PPL=value,documents=expected,metric='word_perplexity')
    else:
        summary=ctx.srr.validate_result(result,data,name,pilot)
        pairing=summary.pop('pairing')
        freeze(ctx.root/('pilot_pairing' if pilot else 'pairing')/f'{name}.json',pairing)
    return result,summary,hits
