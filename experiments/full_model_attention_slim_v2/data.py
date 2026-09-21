"""Freeze all data before viewing scores; predictive labels are unique per window."""
import hashlib
import json
import os
from pathlib import Path
import sys
import torch
from fm_common import *

def tokenizer_identity(ctx):
    root = Path(ctx.config['model'])
    files = {p.name: sha(p) for p in sorted(root.iterdir()) if p.is_file() and
             (p.name.startswith('tokenizer') or p.name in ('special_tokens_map.json', 'added_tokens.json', 'vocab.json', 'merges.txt'))}
    require('tokenizer.json' in files, 'Tokenizer asset absent')
    return files

def checked_ids(ctx, role):
    p = ctx.root/'data'/f'{role}.safetensors'
    values = load_file(str(p))
    ids = values['input_ids']; mask = values['attention_mask']
    require(ids.ndim == 2 and ids.shape[1] == L and ids.dtype == torch.int64, 'Invalid frozen token shape/type')
    require(torch.equal(mask, torch.ones_like(ids)), 'Padding/masking ambiguity')
    require(int(ids.min()) >= 0 and int(ids.max()) < 128256, 'Token outside teacher vocabulary')
    return ids

def freeze_existing(ctx, role, path, count=None):
    values = load_file(str(path)); ids = values['input_ids']
    mask = values.get('attention_mask', torch.ones_like(ids))
    require(ids.shape[1] == L and (count is None or len(ids) == count), f'Unexpected {role} shape')
    require(torch.equal(mask, torch.ones_like(ids)), 'Historical tokens contain padding')
    output = ctx.root/'data'/f'{role}.safetensors'
    tensors(output, dict(input_ids=ids, attention_mask=mask))
    return dict(source=str(path), source_sha256=sha(path), file=str(output.relative_to(ctx.root)),
                sha256=sha(output), windows=len(ids), tokens=int(mask.sum()), prediction_tokens=len(ids)*T,
                window_hashes=[mo.digest_tensor(x) for x in ids],
                mask_hash=mo.digest_tensor(mask), preprocessing='unchanged frozen historical tensor order')

def c4_data(ctx, tokenizer):
    out = ctx.root/'data/c4.safetensors'; manifest = ctx.root/'data/c4_manifest.json'
    if manifest.exists():
        row = read(manifest)
        require(sha(out) == row['sha256'] and row['tokenizer'] == tokenizer_identity(ctx), 'C4 frozen identity differs')
        return row
    transfer=ctx.root/'data/c4_transfer'
    if (transfer/'source.json').exists():
        source=read(transfer/'source.json')
        raw_path=transfer/'documents.jsonl'
        require(sha(raw_path)==source['documents_sha256'],'C4 transfer document hash differs')
        require(source['repo']==ctx.config['c4_dataset'] and source['config']=='en' and source['split']=='validation', 'Wrong C4 transfer source')
        require(source['tokenizer']==tokenizer_identity(ctx),'C4 transfer tokenizer identity differs')
        windows=[];origins=[];tail=0;visited=0
        with raw_path.open(encoding='utf-8') as handle:
            for line in handle:
                doc=json.loads(line);visited+=1
                ids=tokenizer(doc['text'],add_special_tokens=True,truncation=False)['input_ids']
                require(mo.digest_tensor(torch.tensor(ids,dtype=torch.int64))==doc['tokens_hash'],'Local/server C4 tokenization differs')
                tail+=len(ids)%L
                for block in range(len(ids)//L):
                    tokens=torch.tensor(ids[block*L:(block+1)*L],dtype=torch.int64)
                    windows.append(tokens);origins.append(dict(shard=doc['shard'],document_index=doc['document_index'],
                        document_sha256=hashlib.sha256(doc['text'].encode()).hexdigest(),document_tokens=len(ids),
                        block=block,token_hash=mo.digest_tensor(tokens)))
                    if len(windows)==128:break
                if len(windows)==128:break
        require(len(windows)==128,'C4 transfer incomplete')
        ids=torch.stack(windows);tensors(out,dict(input_ids=ids,attention_mask=torch.ones_like(ids)))
        row=dict(**source,sha256=sha(out),windows=128,length=L,prediction_tokens=128*T,origins=origins,
                 add_special_tokens=True,cross_document_concat=False,visited_documents=visited,
                 visited_document_short_tail_tokens=tail,tokenizer_cross_runtime_exact=True,
                 rule='sorted validation shards; document order; per-document nonoverlapping full blocks; first128')
        write(ctx.root/'data/c4_source.json',source);write(manifest,row)
        return row
    from huggingface_hub import HfApi
    from datasets import load_dataset
    revision_file = ctx.root/'data/c4_source.json'
    if revision_file.exists():
        source = read(revision_file)
    else:
        info = HfApi().dataset_info(ctx.config['c4_dataset'], revision='main')
        shards = sorted(x.rfilename for x in info.siblings if x.rfilename.startswith('en/c4-validation.') and x.rfilename.endswith('.json.gz'))
        require(shards, 'No C4 English validation shards')
        source = dict(repo=ctx.config['c4_dataset'], revision=info.sha, config='en', split='validation', shards=shards)
        write(revision_file, source)
    windows = []; origins = []; dropped_tail = 0; visited = 0
    for shard in source['shards']:
        stream = load_dataset(source['repo'], data_files={'validation': [shard]}, split='validation',
                              revision=source['revision'], streaming=True)
        for doc_index, doc in enumerate(stream):
            raw = doc['text']; ids = tokenizer(raw, add_special_tokens=True, truncation=False)['input_ids']
            full = len(ids)//L; dropped_tail += len(ids)%L; visited += 1
            for block in range(full):
                token = torch.tensor(ids[block*L:(block+1)*L], dtype=torch.int64)
                windows.append(token)
                origins.append(dict(shard=shard, document_index=doc_index, document_sha256=hashlib.sha256(raw.encode()).hexdigest(),
                                    document_tokens=len(ids), block=block, token_hash=mo.digest_tensor(token)))
                if len(windows) == 128:
                    break
            if len(windows) == 128:
                break
        if len(windows) == 128:
            break
    require(len(windows) == 128, 'C4 fixed128 unavailable')
    ids = torch.stack(windows); tensors(out, dict(input_ids=ids, attention_mask=torch.ones_like(ids)))
    row = dict(**source, sha256=sha(out), windows=128, length=L, prediction_tokens=128*T,
               tokenizer=tokenizer_identity(ctx), add_special_tokens=True, cross_document_concat=False,
               rule='sorted validation shards; document order; per-document nonoverlapping full blocks; first128',
               visited_documents=visited, visited_document_short_tail_tokens=dropped_tail, origins=origins)
    write(manifest, row)
    return row

def task_dict(ctx,names=None):
    sys.path.insert(0, ctx.config['harness'])
    from lm_eval.tasks import get_task_dict, TaskManager
    import datasets
    transfer=ctx.root/'data/arc_easy_transfer'
    original=datasets.load_dataset
    def frozen_load(path,name=None,*args,**kwargs):
        if path=='allenai/ai2_arc' and name=='ARC-Easy' and (transfer/'source.json').exists():
            source=read(transfer/'source.json')
            require(source['repo']=='allenai/ai2_arc' and source['config']=='ARC-Easy' and
                    source['revision']=='210d026faf9955653af8916fad021475a3f00453','ARC-Easy source identity differs')
            splits={}
            for split,row in source['files'].items():
                file=transfer/row['local_file'];require(sha(file)==row['sha256'],'Transferred task file changed')
                splits[split]=datasets.Dataset.from_parquet(str(file),cache_dir=str(ctx.root/'data/task_cache'))
            require(set(splits)=={'train','test','validation'},'ARC-Easy standard splits incomplete')
            return datasets.DatasetDict(splits)
        return original(path,name,*args,**kwargs)
    datasets.load_dataset=frozen_load
    try:return get_task_dict(list(TASKS) if names is None else names, task_manager=TaskManager())
    finally:datasets.load_dataset=original

def task_evidence(task, metric):
    docs = task.eval_docs; h = hashlib.sha256()
    for row in docs:
        payload = json.dumps(row, sort_keys=True, ensure_ascii=False).encode()
        h.update(len(payload).to_bytes(8, 'little')); h.update(payload)
    metrics = task.aggregation()
    require(metric in metrics, f'Frozen task missing required metric {metric}; explicit pre-score resolution required')
    config = task.dump_config()
    return dict(documents=len(docs), content_sha256=h.hexdigest(), fingerprint=getattr(docs, '_fingerprint', None),
                metric=metric, config=config, split=task.config.test_split or task.config.validation_split,
                cache_files=[dict(filename=x['filename'], sha256=sha(x['filename'])) for x in getattr(docs, 'cache_files', [])])

def freeze_tasks(ctx):
    tasks = task_dict(ctx)
    result = {key: task_evidence(tasks[key], metric) for key, metric in TASKS.items()}
    root = Path(ctx.config['harness'])
    paths = [root/'lm_eval/evaluator.py', root/'lm_eval/models/huggingface.py']
    for folder in ('hellaswag','piqa','winogrande','arc'):
        paths.extend(p for p in (root/'lm_eval/tasks'/folder).rglob('*') if p.suffix in ('.yaml','.py'))
    transfer=ctx.root/'data/arc_easy_transfer/source.json'
    return dict(tasks=result, harness_commit='3823cfec41c016378acbcc8616dd1ac92c15edd4',
        source={str(p.relative_to(root)): sha(p) for p in paths}, num_fewshot=0, max_length=4096,
        chat_template=False, batch_size=1, limit=None, response_cache='disabled', precision='FP32',
        offline_dataset_transfer=read(transfer) if transfer.exists() else None)

def prepare(ctx):
    if ctx.done('data/complete.json'):
        return
    from transformers import AutoTokenizer
    from audit import freeze_weights
    tok = AutoTokenizer.from_pretrained(ctx.config['model'], local_files_only=True)
    meta = read(Path(ctx.config['calibration']).with_suffix('.json'))
    require(meta['raw_prefix_rows'] == 5120 and meta['official_dataset_name'] == 'slim_pajama_6b' and
            meta['qera_commit'] == 'bd7fc86a2e44d41f95b9b0421f27f5624dd37064', 'Calibration preprocessing differs')
    historical_config = read(Path(ctx.config['calibration']).parents[1]/'config_resolved.json')
    require(historical_config['num_workers'] == 8 and historical_config['sequence_length'] == 2048, 'Historical worker/length differs')
    datasets = {role: freeze_existing(ctx, role, ctx.config[role], count) for role, count in
                [('calibration',256),('wikitext2',None),('validation',16)]}
    datasets['calibration']['historical_metadata'] = meta
    datasets['calibration']['historical_config_sha256'] = sha(Path(ctx.config['calibration']).parents[1]/'config_resolved.json')
    datasets['wikitext2']['historical_metadata'] = read(Path(ctx.config['wikitext2']).with_suffix('.json'))
    datasets['wikitext2']['discarded_tail_tokens'] = None
    datasets['wikitext2']['tail_note'] = 'Historical official worker/map preprocessing did not save discarded-tail count; all frozen full windows retained unchanged'
    log('PREPARE_C4_START')
    datasets['c4'] = c4_data(ctx, tok)
    log('PREPARE_TASKS_START')
    tasks = freeze_tasks(ctx)
    write(ctx.root/'data_manifest.json', dict(identity=ctx.identity, tokenizer=tokenizer_identity(ctx), datasets=datasets))
    write(ctx.root/'eval_manifest.json', dict(identity=ctx.identity, token_data=datasets, downstream=tasks))
    freeze_weights(ctx)
    files = [ctx.root/'data_manifest.json',ctx.root/'eval_manifest.json',ctx.root/'quantization_manifest.json',
             *[ctx.root/'data'/f'{r}.safetensors' for r in ('calibration','wikitext2','validation','c4')],
             ctx.root/'data/c4_manifest.json',ctx.root/'data/c4_source.json']
    ctx.commit('data/complete.json', files)
    ctx.teacher.unload()

def label(ctx, window, hidden=None):
    path = ctx.root/'data/labels'/f'w{window:04d}.safetensors'; rec = path.with_suffix('.json')
    ids = checked_ids(ctx, 'calibration')[window]
    binding = dict(window=window, input_hash=mo.digest_tensor(ids), seed=mo.stream_seed(window, 0, ctx.config['base_seed']))
    if rec.exists():
        meta = read(rec)
        require(meta['identity'] == ctx.identity and all(meta[k] == v for k, v in binding.items()) and sha(path) == meta['sha256'], 'Predictive label binding differs')
        result = load_file(str(path))['labels']
        require(mo.digest_tensor(result) == meta['label_hash'], 'Label tensor changed')
    else:
        require(hidden is not None, 'Missing frozen labels; do not resample during Pass B')
        result = ctx.teacher.labels(hidden, [binding['seed']])[0]
        tensors(path, {'labels': result})
        write(rec, dict(identity=ctx.identity, **binding, sha256=sha(path), label_hash=mo.digest_tensor(result),
                        algorithm='Full-vocabulary FP64-softmax inverse-CDF; CUDA float64 uniform; CDF last entry=1',
                        seed_derivation='sha256(f"20260921:{window}:0") first8 little endian mod (2**63-1)',
                        teacher_identity_sha256=sha(ctx.root/'teacher_identity.json'), T=T, vocab_chunk=ctx.config['vocab_chunk']))
    require(result.shape == (T,) and result.dtype == torch.int64 and int(result.min()) >= 0 and int(result.max()) < 128256, 'Invalid predictive labels')
    return result
