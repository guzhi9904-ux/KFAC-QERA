"""Nested fit views and genuinely unused articles, fixed before any effect evaluation."""
import hashlib
import re
import struct
from pathlib import Path
import torch
from common import PLAN,read,read_tensors,save_json,save_tensors,sha_file,mo,digest,seed,require


def articles(rows):
    start=0;parts=[];title='preamble'
    for i,row in enumerate(rows):
        text=row['text'];line=text.strip()
        if re.fullmatch(r'= [^=].*[^=] =',line):
            if parts:yield start,i,title,'\n'.join(parts)
            start=i;parts=[];title=line
        parts.append(text)
    if parts:yield start,len(rows),title,'\n'.join(parts)


def spans(tokens,width=64):
    raw=struct.pack('<'+'I'*len(tokens),*tokens)
    return {raw[4*i:4*(i+width)] for i in range(len(tokens)-width+1)}


def select(candidates,forbidden,titles,texts,count=40,length=2048):
    accepted=[];rejected=[];selected_spans=set()
    ordering=lambda item:(hashlib.sha256(('qer-ksample-v1/data'+item[0]['article_id']).encode()).hexdigest(),item[0]['article_id'])
    titles=set(titles);texts=set(texts)
    for row,tokens in sorted(candidates,key=ordering):
        reason=None
        if row['article_title'] in titles or row['article_text_sha256'] in texts:reason='used_or_duplicate_document'
        elif len(tokens)<length:reason='short'
        else:
            s=spans(tokens)
            if s&forbidden:reason='historical_64_token_span'
            elif s&selected_spans:reason='new_collection_64_token_span'
        if reason:rejected.append(dict(article_id=row['article_id'],reason=reason));continue
        prefix=torch.tensor(tokens[:length],dtype=torch.int64)
        accepted.append((dict(row,token_start=0,token_stop=length,token_hash=mo.digest_tensor(prefix),article_token_count=len(tokens)),prefix))
        titles.add(row['article_title']);texts.add(row['article_text_sha256']);selected_spans.update(s)
        if len(accepted)==count:break
    return accepted,rejected


def sample_views(fit_rows,eval_rows):
    samples=[];budgets={k:[] for k in PLAN['budgets']}
    for c,row in enumerate(fit_rows):
        count=16 if c<8 else 4
        for k in range(count):
            key=f'f{c:02d}_k{k:02d}'
            source='parent_saved' if c<8 and k<4 else 'new_teacher_sample'
            entry=dict(id=key,role='fit',window=c,replicate=k,token_hash=row['token_hash'],source=source,
                seed=None if source=='parent_saved' else seed('fit',row['token_hash'],k))
            samples.append(entry)
            if c<8 and k<4:budgets['S0'].append(key)
            if c<8:budgets['S1'].append(key)
            if k<4:budgets['S2'].append(key)
    evaluations=[dict(id=f'e{c:02d}_k{k:02d}',role='eval',window=c,replicate=k,token_hash=row['token_hash'],
        source='new_teacher_sample',seed=seed('eval',row['token_hash'],k))
        for c,row in enumerate(eval_rows) for k in range(PLAN['eval_labels'])]
    return dict(fit=samples,eval=evaluations,budgets=budgets,
        encoding='SHA256(JSON compact [namespace/role, token_hash, replicate]), first 8 bytes little endian modulo 2^63-1',
        A_views={'S0':'A8','S1':'A8','S2':'A32'})


def prepare(root,config,identity,assets):
    frozen=root/'data/data_freeze.json'
    if frozen.exists():
        m=read(frozen);require(m['identity']==identity,'Data freeze identity differs')
        for rel,value in m['files'].items():require(sha_file(root/rel)==value,'Frozen data changed')
        return read(root/'data/sample_index.json')
    from datasets import load_from_disk
    from transformers import AutoTokenizer
    from safetensors.torch import load_file
    raw={r:load_from_disk(str(Path(config['wikitext'])/r)) for r in ('train','validation')}
    tokenizer=AutoTokenizer.from_pretrained(config['model'],local_files_only=True,trust_remote_code=False)
    old=Path(config['assets'])/'exp03'
    fit,fitmeta=read_tensors(old/'data/fit_windows.safetensors');metadata=read(old/'data/fit_windows.json')
    require(fitmeta['identity']==PLAN['parent_identity'] and fit['input_ids'].shape==(8,PLAN['L']),'Original fit identity/shape differs')
    titles=set();texts=set();forbidden=set();history=[]
    for path in assets['historical_manifests']:
        for row in read(path)['windows']:titles.add(row['article_title']);texts.add(row['article_text_sha256'])
    for _,_,title,text in articles(raw['validation']):titles.add(title);texts.add(hashlib.sha256(text.encode()).hexdigest())
    for path in assets['historical_windows']:
        t=load_file(path);ids=t['input_ids']
        require(ids.ndim==2 and ids.shape[1]==PLAN['L'],'Historical window shape differs')
        if 'attention_mask' in t:require(bool(t['attention_mask'].eq(1).all()),'Unexpected historical padding')
        for row in ids.tolist():forbidden.update(spans(row))
        history.append(dict(path=path,file_sha256=sha_file(path),windows=len(ids)))
    candidates=[]
    for start,stop,title,text in articles(raw['train']):
        text_hash=hashlib.sha256(text.encode()).hexdigest()
        row=dict(split='train',source_row_start=start,source_row_stop=stop,article_title=title,
            article_text_sha256=text_hash,article_id=digest(['wikitext-train',title,text_hash]))
        tokens=tokenizer(text,add_special_tokens=False,return_attention_mask=False)['input_ids']
        candidates.append((row,tokens))
    selected,rejected=select(candidates,forbidden,titles,texts)
    if len(selected)!=40:
        save_json(root/'data/shortfall.json',dict(required=40,available=len(selected),rejected=rejected))
        raise RuntimeError('Insufficient unused articles; no change to corpus, length or duplicate policy')
    parent_rows=[]
    for c,row in enumerate(metadata['windows']):
        require(mo.digest_tensor(fit['input_ids'][c])==row['token_hash'],'Parent token hash mismatch')
        parent_rows.append(dict(row,window=c,role='fit',article_id=digest(['wikitext-train',row['article_title'],row['article_text_sha256']]),source='parent_saved'))
    fit_rows=parent_rows+[dict(r,window=i+8,role='fit',source='new_article') for i,(r,t) in enumerate(selected[:24])]
    eval_rows=[dict(r,window=i,role='eval',source='new_article') for i,(r,t) in enumerate(selected[24:])]
    fit_ids=torch.cat([fit['input_ids'],torch.stack([t for r,t in selected[:24]])])
    eval_ids=torch.stack([t for r,t in selected[24:]])
    for role,rows,ids in [('fit',fit_rows,fit_ids),('eval',eval_rows,eval_ids)]:
        masks=torch.ones_like(ids)
        save_tensors(root/'data'/f'{role}_windows.safetensors',dict(input_ids=ids,attention_mask=masks),dict(identity=identity,role=role,mask_hash=mo.digest_tensor(masks)))
        save_json(root/'data'/f'{role}_windows.json',dict(identity=identity,windows=rows,mask_hash=mo.digest_tensor(masks)))
    index=sample_views(fit_rows,eval_rows)
    require(len(index['fit'])==224 and len(index['eval'])==256,'Sample budget differs')
    require([len(index['budgets'][b]) for b in ('S0','S1','S2')]==[32,128,128],'Budget nesting differs')
    save_json(root/'data/sample_index.json',dict(identity=identity,**index))
    files={p.relative_to(root).as_posix():sha_file(p) for p in sorted((root/'data').glob('*')) if p.is_file()}
    save_json(frozen,dict(identity=identity,files=files,historical_windows=history,rejected=rejected,
        selection='SHA256(namespace/data + stable article ID), UTF8 hex ascending; 24 additional fit then 16 eval',
        full_article_64_token_exclusion=True,geometry_history_excluded=True,official_test_split_used=False))
    return dict(identity=identity,**index)
