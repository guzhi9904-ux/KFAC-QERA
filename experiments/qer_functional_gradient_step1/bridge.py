"""Read-only imports of the pinned server implementation."""
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode=True
for key in ('HF_HUB_OFFLINE','HF_DATASETS_OFFLINE','TRANSFORMERS_OFFLINE'):
    os.environ.setdefault(key,'1')
HERE=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
PLAN=read(HERE/'plan.json')
sys.path.insert(0,PLAN['exp3_code'])
from experiment import FullFitExperiment
from run import Experiment,slug,clean,sync,OFFICIAL,Paused,request_pause
from storage import save_json,save_csv,save_tensors,read_tensors,sha_file,atomic_bytes
from model_ops import capture,hidden_forward
import math_ops as mo
import torch
import torch.nn.functional as F
sys.path.insert(0,str(HERE))


def source_identity():
    import hashlib
    source={p.name:sha_file(p) for p in sorted(HERE.iterdir()) if p.suffix in ('.py','.json','.sh','.md')}
    material=dict(plan=PLAN,source=source)
    return dict(identity=hashlib.sha256(json.dumps(material,sort_keys=True).encode()).hexdigest(),**material)


def check_parent(root,after=False):
    """Hash verification only; no model, fitting, mutation, or large tensor copy."""
    import hashlib,time
    root=Path(root);exp3=Path(PLAN['exp3_output']);pins=read(HERE/'parent_expected.json')
    files={}
    for n,h in pins.items():
        assert sha_file(exp3/n)==h,n
        files[str(exp3/n)]=h
        if not after:atomic_bytes(root/'parent_snapshot'/n,(exp3/n).read_bytes())
    identity=read(exp3/'identity.json')
    assert identity['identity']==PLAN['exp3_identity']
    material={k:v for k,v in identity.items() if k!='identity'}
    assert hashlib.sha256(json.dumps(material,sort_keys=True).encode()).hexdigest()==identity['identity']
    frozen=read(exp3/'evaluation_freeze.json')
    assert frozen['identity']==identity['identity'] and frozen['manifest_sha256']==sha_file(exp3/'manifest.json')
    for n,h in frozen['files'].items():assert sha_file(exp3/n)==h;files[str(exp3/n)]=h
    for sample in read(exp3/'data/fit_sample_manifest.json')['samples']:
        p=exp3/sample['relative_path'];assert sha_file(p)==sample['file_sha256'];files[str(p)]=sample['file_sha256']
    old=Path(identity['plan']['parent_output']);oldid=read(old/'identity.json')
    for code,record in [(Path(PLAN['exp3_code']),identity),(Path(PLAN['parent_code']),oldid)]:
        for n,h in record['source'].items():assert sha_file(code/n)==h;files[str(code/n)]=h
    directions=read(old/'directions/frozen.json')
    for name in PLAN['parent_modules']:
        p=old/'directions'/(slug(name)+'.safetensors');assert sha_file(p)==directions['files'][name]
        _,meta=read_tensors(p)
        q=old/'quantized'/(slug(name)+'.safetensors');assert sha_file(q)==meta['quantized_file_hash']
        files[str(p)]=sha_file(p);files[str(q)]=sha_file(q)
    if after:assert read(root/'parent_integrity_before.json')['files']==files
    result=dict(passed=True,files=files,parent_identity=identity['identity'],checked=time.time())
    save_json(root/('parent_integrity_after.json' if after else 'parent_integrity_before.json'),result)
    return result
