#!/usr/bin/env python3
"""Local network bridge: stream a small prefix, preserve raw documents for server retokenization."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import time
import requests
import torch
from transformers import AutoTokenizer

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def tensor_hash(x):
    h=hashlib.sha256(str((str(x.dtype),tuple(x.shape))).encode());h.update(x.view(torch.uint8).numpy().tobytes());return h.hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tokenizer',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'source.json').exists():
        row=json.loads((out/'source.json').read_text());assert sha(out/'documents.jsonl')==row['documents_sha256'];print('C4_TRANSFER_ALREADY_FROZEN');return
    tok=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True)
    tok_files={p.name:sha(p) for p in sorted(Path(args.tokenizer).iterdir()) if p.is_file() and
        (p.name.startswith('tokenizer') or p.name in ('special_tokens_map.json','added_tokens.json','vocab.json','merges.txt'))}
    response=requests.get('https://huggingface.co/api/datasets/allenai/c4',timeout=30);response.raise_for_status();info=response.json()
    revision_path=out/'revision.json'
    if revision_path.exists():
        info=json.loads(revision_path.read_text())
    else:
        revision_path.write_text(json.dumps(info),encoding='utf-8')
    revision=info['sha'];shards=sorted(p['rfilename'] for p in info['siblings'] if p['rfilename'].startswith('en/c4-validation.') and p['rfilename'].endswith('.json.gz'))
    count=0;documents=0;path=out/'documents.jsonl'
    with path.open('w',encoding='utf-8',newline='\n') as target:
        for shard in shards:
            url=f'https://huggingface.co/datasets/allenai/c4/resolve/{revision}/{shard}'
            with requests.get(url,stream=True,timeout=(30,120)) as stream:
                stream.raise_for_status();stream.raw.decode_content=True
                with gzip.GzipFile(fileobj=stream.raw) as source:
                    for doc_index,line in enumerate(source):
                        raw=json.loads(line);tokens=tok(raw['text'],add_special_tokens=True,truncation=False)['input_ids']
                        row=dict(shard=shard,document_index=doc_index,text=raw['text'],
                                 tokens_hash=tensor_hash(torch.tensor(tokens,dtype=torch.int64)))
                        target.write(json.dumps(row,ensure_ascii=False)+'\n');documents+=1;count+=len(tokens)//2048
                        if documents%100==0:print('C4_PREFIX',documents,'documents',count,'full_windows',flush=True)
                        if count>=128:break
            if count>=128:break
    assert count>=128
    row=dict(repo='allenai/c4',config='en',split='validation',revision=revision,shards=shards,tokenizer=tok_files,
        documents_sha256=sha(path),documents=documents,full_blocks_in_prefix=count,required_windows=128,
        selection='first128 full2048 blocks in sorted shards and original document order, no concatenation',
        created_utc=time.time(),local_transformers=__import__('transformers').__version__)
    (out/'source.json').write_text(json.dumps(row,indent=2),encoding='utf-8');print('C4_TRANSFER_COMPLETE',documents,count,flush=True)
if __name__=='__main__':main()
