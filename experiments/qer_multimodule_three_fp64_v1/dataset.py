"""Freeze nested train windows and separate official validation/test streams."""
import hashlib
from pathlib import Path
import torch
from bridge import read,save_json,save_tensors,read_tensors,sha_file,mo,digest,require,commit,load_record,file_table,checked_files


def split_windows(tokens,length,count=None,shuffle=False):
    usable=len(tokens)//length;require(usable>0,'Split shorter than one window')
    order=list(range(usable))
    if shuffle:order.sort(key=lambda i:hashlib.sha256(f'qer-multimodule-v1/{i}'.encode()).digest())
    if count is not None:require(usable>=count,'Insufficient nonoverlapping windows');order=order[:count]
    ids=torch.tensor([tokens[i*length:(i+1)*length] for i in order],dtype=torch.int64)
    rows=[dict(window=j,token_start=i*length,token_stop=(i+1)*length,token_hash=mo.digest_tensor(ids[j])) for j,i in enumerate(order)]
    return ids,rows,dict(total_tokens=len(tokens),complete_windows=usable,discarded_tail_tokens=len(tokens)%length)


def prepare(root,config,identity):
    frozen=root/'data/freeze.json'
    if frozen.exists():
        row=load_record(frozen,identity);checked_files(root,row['files']);return read(root/'data/index.json')
    from datasets import load_from_disk,DatasetDict
    from transformers import AutoTokenizer
    source=Path(config['wikitext']);tokenizer=AutoTokenizer.from_pretrained(config['model'],local_files_only=True,trust_remote_code=False)
    require(source.exists(),'Dataset not available locally')
    raw=None if all((source/s/'state.json').exists() for s in ('train','validation','test')) else load_from_disk(str(source))
    require(raw is None or isinstance(raw,DatasetDict),'Need all three official splits')
    paths=[];metadata={};split_hashes={};window_hashes=set()
    for role,split,count in [('fit','train',max(config['budgets'])),('validation','validation',config['validation_windows']),('test','test',None)]:
        data=load_from_disk(str(source/split)) if raw is None else raw[split]
        text='\n\n'.join(data['text']);text_hash=hashlib.sha256(text.encode()).hexdigest()
        require(text_hash not in split_hashes.values(),'Two named splits contain identical text');split_hashes[split]=text_hash
        tokens=tokenizer(text,add_special_tokens=False,return_attention_mask=False)['input_ids']
        ids,rows,stats=split_windows(tokens,config['sequence_length'],count,shuffle=role!='test')
        hashes={r['token_hash'] for r in rows};require(len(hashes)==len(rows) and not hashes&window_hashes,'Duplicate full window within/across splits');window_hashes|=hashes
        tp=root/'data'/f'{role}.safetensors';jp=tp.with_suffix('.json')
        save_tensors(tp,dict(input_ids=ids),dict(identity=identity,tensor_hash=mo.digest_tensor(ids)))
        save_json(jp,dict(identity=identity,split=split,windows=rows,**stats,text_sha256=text_hash,
            sampling='nonoverlapping blocks from joined text, fixed hash order for train/validation; contiguous test',
            article_independence_claim=False,ppl_protocol='disjoint2048 windows; predict2047 next tokens, drop final incomplete block'))
        paths.extend([tp,jp]);metadata[role]=rows
    entries=[]
    for row in metadata['fit']:
        for k in range(config['labels_per_window']):
            seed=int(digest(['qer-multimodule-v1',row['token_hash'],k])[:16],16)%(2**63-1)
            entries.append(dict(id=f'f{row["window"]:04d}_k{k:02d}',window=row['window'],replicate=k,token_hash=row['token_hash'],seed=seed))
    budgets={f'N{n}':[r['id'] for r in entries if r['window']<n] for n in config['budgets']}
    index=dict(identity=identity,fit=entries,budgets=budgets,windows={r:len(v) for r,v in metadata.items()})
    save_json(root/'data/index.json',index);paths.append(root/'data/index.json')
    # This hashes the actual local dataset and tokenizer files, not a dataset nickname.
    sources={str(p.resolve()):sha_file(p) for p in source.rglob('*') if p.is_file()}
    tokenizer_files={str(p.resolve()):sha_file(p) for p in Path(config['model']).glob('*') if p.is_file() and ('token' in p.name or p.name in ('special_tokens_map.json','config.json'))}
    commit(frozen,identity,files=file_table(root,paths),source_files=sources,tokenizer_files=tokenizer_files,
        old_train_windows_may_recur=True,official_splits=True,labels_shared_across_all_methods=True)
    return index


def windows(root,role,identity):
    t,m=read_tensors(root/'data'/f'{role}.safetensors');require(m['identity']==identity and mo.digest_tensor(t['input_ids'])==m['tensor_hash'],'Frozen window mismatch');return t['input_ids']
