"""Deterministic train-article prefixes, with auditable overlap checks."""
import hashlib
import re
from pathlib import Path
import torch
from storage import read_tensors,save_tensors,save_json,sha_file
import math_ops as mo


def articles(rows):
    start=0;parts=[];title='preamble'
    for i,row in enumerate(rows):
        text=row['text'];line=text.strip()
        if re.fullmatch(r'= [^=].*[^=] =',line):
            if parts:yield start,i,title,'\n'.join(parts)
            start=i;parts=[];title=line
        parts.append(text)
    if parts:yield start,len(rows),title,'\n'.join(parts)


def prepare(experiment):
    e=experiment;p=e.root/'data/fit_windows.safetensors'
    if p.exists():
        data,meta=read_tensors(p);assert meta['identity']==e.identity
        assert [mo.digest_tensor(x) for x in data['input_ids']]==[x['token_hash'] for x in meta['windows']]
        e.fit=data;return
    from transformers import AutoTokenizer
    import datasets
    tokenizer=AutoTokenizer.from_pretrained(e.config['model'],local_files_only=True,trust_remote_code=False)
    source=Path(e.config['wikitext']);raw=datasets.load_from_disk(str(source))
    validation=list(articles(raw['validation']))
    valtitles={r[2] for r in validation};valtexts={hashlib.sha256(r[3].encode()).hexdigest() for r in validation}
    val64={tuple(row[i:i+64]) for row in e.val['input_ids'].tolist() for i in range(2048-63)}
    selected=[];ids=[];rejected=[]
    for start,end,title,text in articles(raw['train']):
        text_hash=hashlib.sha256(text.encode()).hexdigest()
        if title in valtitles or text_hash in valtexts:
            rejected.append({'row':start,'reason':'validation_article_title_or_text'});continue
        tokens=tokenizer(text,add_special_tokens=False,return_attention_mask=False)['input_ids']
        if len(tokens)<2048:continue
        window=tokens[:2048]
        if any(tuple(window[i:i+64]) in val64 for i in range(2048-63)):
            rejected.append({'row':start,'reason':'shared_64_token_segment'});continue
        tensor=torch.tensor(window,dtype=torch.int64);h=mo.digest_tensor(tensor)
        assert h not in {mo.digest_tensor(v) for v in e.val['input_ids']}
        ids.append(tensor)
        selected.append({'window':len(ids)-1,'split':'train','article_title':title,
                         'source_row_start':start,'source_row_stop':end,'article_text_sha256':text_hash,
                         'article_token_count':len(tokens),'token_start':0,'token_stop':2048,'token_hash':h})
        if len(ids)==e.plan['N_fit']:break
    assert len(ids)==e.plan['N_fit']
    source_files={str(f.relative_to(source)):sha_file(f) for split in ('train','validation')
                  for f in (source/split).iterdir() if f.name in ('data-00000-of-00001.arrow','state.json','dataset_info.json')}
    tokenizer_files={f.name:sha_file(f) for f in Path(e.config['model']).glob('*')
                     if f.is_file() and (f.name.startswith('tokenizer') or f.name=='special_tokens_map.json')}
    meta={'identity':e.identity,'dataset_path':str(source),'source_sha256':source_files,
          'fingerprints':{s:raw[s]._fingerprint for s in ('train','validation')},'tokenizer_files':tokenizer_files,
          'selection':e.plan['fit_selection'],'add_special_tokens':False,'join':'newline between original article rows',
          'windows':selected,'rejected':rejected,'D_dev':None,'validation_article_identity_checked':True,
          'shared_64_token_segments_with_eval':0,'eval_window_hashes':[mo.digest_tensor(v) for v in e.val['input_ids']],
          'overlap_scope':'Different split and article/title/text identity; additionally no identical 64-token span with the eight evaluation windows. Not a general semantic contamination guarantee.'}
    data={'input_ids':torch.stack(ids),'attention_mask':torch.ones(len(ids),2048,dtype=torch.int64)}
    save_tensors(p,data,meta);save_json(p.with_suffix('.json'),meta)
    save_json(e.root/'data/eval_windows.json',{'parent_identity':e.plan['parent_identity'],
              'input_file_sha256':sha_file(e.parent/'data/validation.safetensors'),
              'window_hashes':meta['eval_window_hashes'],'role':'historical fixed evaluation; never factor fitting'})
    e.fit=data
