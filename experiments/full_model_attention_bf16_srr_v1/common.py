"""Independent identity and read-only access to completed FP64 parent artifacts."""
import contextlib
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import time
import torch
from safetensors.torch import load_file

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

def require(ok, message):
    if not ok: raise RuntimeError(message)

def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024**2), b''): h.update(chunk)
    return h.hexdigest()

def digest(value): return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()

def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def freeze(path, value):
    if Path(path).exists(): require(read(path)==value,'Frozen identity differs: '+str(path))
    else: write(path,value)

def log(event, **fields): print(time.strftime('%Y-%m-%dT%H:%M:%S%z'),event,json.dumps(fields),flush=True)

def module_from(path, name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

math_ops=module_from(HERE.parent/'qer_teacher_kl_exp01/math_ops.py','bf16_parent_math')
layout=module_from(HERE.parent/'qer_functional_gradient_4090_v1/device_layout.py','bf16_device_layout')
tensor_hash=math_ops.digest_tensor

@contextlib.contextmanager
def timed(ctx, stage, **fields):
    start=time.monotonic();stop=threading.Event();passed=False
    log('START',stage=stage,**fields)
    def pulse():
        while not stop.wait(30): log('WORKING',stage=stage,elapsed_seconds=time.monotonic()-start,**fields)
    thread=threading.Thread(target=pulse,daemon=True);thread.start()
    try:
        yield
        passed=True
    finally:
        stop.set();thread.join()
        if torch.cuda.is_initialized():
            for d in range(torch.cuda.device_count()):torch.cuda.synchronize(d)
        row=dict(stage=stage,passed=passed,seconds=time.monotonic()-start,**fields)
        if torch.cuda.is_initialized():row['peak_GPU_GiB']=[torch.cuda.max_memory_allocated(d)/2**30 for d in range(2)]
        path=ctx.root/'timings.json';rows=read(path) if path.exists() else [];rows.append(row);write(path,rows);log('END',**row)

class Context:
    def __init__(self, config):
        self.config=read(config);self.root=Path(self.config['run']);self.parent=Path(self.config['parent'])
        self.root.mkdir(parents=True,exist_ok=True);self.verified=set()
        p=read(self.parent/'manifest.json');require(p['identity']==self.config['parent_identity'],'Wrong parent')
        require(digest({k:v for k,v in p.items() if k!='identity'})==p['identity'],'Parent manifest corrupted')
        self.parent_config=p['config'];self.model_path=Path(p['config']['model'])
        require(self.config['model_dtype']==self.config['factor_dtype']=='bfloat16' and self.config['rank']==64,'BF16/rank contract changed')
        require(self.config['tasks']==['hellaswag','winogrande','boolq','mmlu','leaderboard_bbh'],'SRR five-task set changed')
        self.srr_root=Path(self.config['srr_tools']);runtime=self.srr_root/'runtime'
        helper=self.srr_root/'downstream_srr_mxint3_v1.py'
        require(sha(helper)==self.config['srr_helper_sha256'],'SRR helper differs')
        for line in (self.srr_root/'SHA256SUMS.downstream_srr').read_text().splitlines():
            expected,relative=line.split('  ',1)
            if relative.startswith('runtime/'): require(sha(self.srr_root/relative)==expected,'SRR runtime file differs: '+relative)
        sys.path.insert(0,str(runtime));import lm_eval
        require(Path(lm_eval.__file__).resolve()==runtime/'lm_eval/__init__.py','Wrong harness import')
        require(importlib.metadata.version('lm_eval')=='0.4.7','SRR requires lm_eval 0.4.7')
        self.srr=module_from(helper,'frozen_srr_evaluator')
        parents={str(self.parent/x):sha(self.parent/x) for x in ('manifest.json','data_manifest.json','teacher_identity.json',
                 'factors/complete.json','candidates/complete.json','quantization_manifest.json')}
        source={str(f.relative_to(REPO)):sha(f) for f in HERE.iterdir() if f.suffix in ('.py','.json','.md','.sh')}
        for f in (Path(math_ops.__file__),Path(layout.__file__)):source[str(f.relative_to(REPO))]=sha(f)
        material=dict(config=self.config,parents=parents,source=source,srr_release_sha256=sha(self.srr_root/'SHA256SUMS.downstream_srr'),
                      packages={n:importlib.metadata.version(n) for n in ('torch','transformers','accelerate','datasets','safetensors','numpy','tokenizers','lm_eval')})
        self.identity=digest(material);freeze(self.root/'manifest.json',dict(identity=self.identity,**material))

    def verify(self, path, expected):
        key=(str(path),expected)
        if key not in self.verified:
            require(sha(path)==expected,'File changed: '+str(path));self.verified.add(key)
        return Path(path)

    def done(self, relative, **expected):
        path=self.root/relative
        if not path.exists():return False
        row=read(path);require(row['identity']==self.identity and row['passed'],'Completion identity/status differs')
        require(all(row.get(k)==v for k,v in expected.items()),'Completion binding differs')
        for p,h in row['files'].items():self.verify(self.root/p,h)
        return row

    def commit(self, relative, files, **fields):
        write(self.root/relative,dict(identity=self.identity,passed=True,time=time.time(),
              files={str(Path(p).relative_to(self.root)):sha(p) for p in files},**fields))

    def parent_complete(self, relative, **expected):
        row=read(self.parent/relative)
        require(row['identity']==self.config['parent_identity'] and row['passed'],'Parent incomplete: '+relative)
        require(all(row.get(k)==v for k,v in expected.items()),'Parent count mismatch')
        for path,h in row['files'].items():
            self.verify(self.parent/path,h)
            if path.endswith('.complete.json'):self.parent_complete(path)
        return row

def hidden(model, ids):
    device=model.model.embed_tokens.weight.device
    return model.model(input_ids=ids.to(device),attention_mask=torch.ones_like(ids,device=device),
                       use_cache=False,return_dict=True).last_hidden_state.to(model.lm_head.weight.device)
