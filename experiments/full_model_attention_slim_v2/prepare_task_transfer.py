#!/usr/bin/env python3
"""Fetch the missing official ARC-Easy standard splits at a pinned revision."""
import argparse
import hashlib
import json
from pathlib import Path
import requests

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    source=dict(repo='allenai/ai2_arc',config='ARC-Easy',revision='210d026faf9955653af8916fad021475a3f00453',files={})
    for split in ('train','test','validation'):
        filename=f'{split}-00000-of-00001.parquet';repo_path='ARC-Easy/'+filename
        url=f"https://huggingface.co/datasets/{source['repo']}/resolve/{source['revision']}/{repo_path}"
        response=requests.get(url,timeout=120);response.raise_for_status()
        assert response.content[:4]==b'PAR1'
        (out/filename).write_bytes(response.content)
        source['files'][split]=dict(repo_path=repo_path,local_file=filename,sha256=hashlib.sha256(response.content).hexdigest(),bytes=len(response.content))
        print('ARC_EASY_TRANSFER',split,len(response.content),flush=True)
    (out/'source.json').write_text(json.dumps(source,indent=2),encoding='utf-8')
if __name__=='__main__':main()
