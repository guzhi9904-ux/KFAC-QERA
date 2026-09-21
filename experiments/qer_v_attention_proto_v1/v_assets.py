from v_common import (Source,ParentStore,Path,read,require,sha_file,mo,slug,commit,load_record,
                      save_json,BASELINES,LAYERS,MODULES)
from ko_common import source_x


class AuditStore(ParentStore):
    def __init__(self,ident):super().__init__(ident,0);self.evidence={}
    def get(self,path):
        t,m=super().get(path);p=Path(path);sig=self.verified[str(p)];receipt=p.with_suffix('.json')
        if str(p) not in self.evidence:
            self.evidence[str(p)]=dict(size=sig[0],mtime_ns=sig[1],file_sha256=sig[2],
                receipt_sha256=sha_file(receipt),tensors={k:dict(shape=list(v.shape),dtype=str(v.dtype),hash=m['tensor_hashes'][k]) for k,v in t.items()},
                binding={k:m[k] for k in ('module','sample','window','token_hash','label_hash','x_hash') if k in m})
        return t,m


def audit_assets(root,config,ident,resources):
    source=Source(config);source.store=AuditStore(source.identity);p=root/'parent_assets_audit.json'
    if p.exists():
        old=load_record(p,ident);require(old['parent_identity']==source.identity,'Wrong parent audit')
        for key,row in old['required_tensors'].items():
            f=Path(key);s=f.stat();require((s.st_size,s.st_mtime_ns)==(row['size'],row['mtime_ns']),'Borrowed asset signature changed')
            require(sha_file(f.with_suffix('.json'))==row['receipt_sha256'],'Borrowed receipt changed')
            source.store.verified[key]=(row['size'],row['mtime_ns'],row['file_sha256'])
        for key,value in old['baseline_record_hashes'].items():require(sha_file(key)==value,'Parent baseline changed')
        source.store.evidence=old['required_tensors'];print('PARENT_AUDIT_RESUMED',len(source.store.verified),flush=True)
        return source
    baseline_hashes={};freeze=sha_file(source.root/'candidate_freeze.json')
    pins=load_record(source.root/'candidate_freeze.json',source.identity)['files']
    teacher_hashes=read(Path(config['assets'])/'exp03/teacher_identity.json')['tensor_hashes']
    for window in range(16):
        for name in MODULES:
            for method in BASELINES:
                key='None' if method=='None' else 'N256__'+method
                f=source.root/'scores/validation'/f'w{window:04d}'/(slug(name)+'___'+key+'.json');row=load_record(f,source.identity)
                require(row['freeze_hash']==freeze and row['token_hash']==mo.digest_tensor(source.validation[window]) and row['scope']==name and row['candidate']==key,'Baseline binding changed')
                baseline_hashes[str(f)]=sha_file(f)
    with resources.timed('required_parent_tensor_audit'):
        for row in source.index['fit']:
            resources.boundary();label=source.label(row);paths=[source.root/'cache/labels'/(row['id']+'.safetensors')]
            for layer,name in zip(LAYERS,MODULES):
                source.x(name,row['window']);source.gradient(name,row,label)
                down=f'model.layers.{layer}.mlp.down_proj';source.gradient(down,row,label)
                paths += [source_x(source.root,name,row['window']),source.root/'cache/g'/slug(name)/(row['id']+'.safetensors'),source.root/'cache/g'/slug(down)/(row['id']+'.safetensors')]
            parent_commit=load_record(source.root/'cache/commits'/(row['id']+'.json'),source.identity)
            require(parent_commit['sample']==row,'Parent cache commit sample changed')
            for f in paths:require(parent_commit['files'][f.relative_to(source.root).as_posix()]==source.store.verified[str(f)][2],'Parent commit/file hash differs')
            if (row['window']+1)%32==0:print('PARENT_ASSET_AUDIT',row['window']+1,'/256',flush=True)
        for name in MODULES:
            qp=source.root/'quantized'/(slug(name)+'.safetensors');qt,qm=source.store.get(qp)
            require(qm['module']==name and mo.digest_tensor(qt['W0'])==teacher_hashes[name+'.weight']['hash'],'Quantized teacher binding changed')
            require(source.store.verified[str(qp)][2]==pins[qp.relative_to(source.root).as_posix()],'Quantization freeze mismatch')
            for method in BASELINES[:-1]:
                source.store.get(source.root/'modules'/slug(name)/'factors/N256'/method/'raw.safetensors')
                cp=source.root/'modules'/slug(name)/'corrections'/('N256__'+method+'.safetensors');source.store.get(cp)
                require(source.store.verified[str(cp)][2]==pins[cp.relative_to(source.root).as_posix()],'Baseline deployment freeze mismatch')
    commit(p,ident,parent_identity=source.identity,parent_config=source.manifest['config'],required_tensors=source.store.evidence,
           baseline_record_hashes=baseline_hashes,baseline_count=192,fit_windows=256,validation_windows=16,
           parent_complete_exists=(source.root/'complete.json').exists(),
           parent_completion_note='Parent fit/candidates and16 validation windows are complete. Original141-window test was deliberately stopped; not required here.',
           audit_scope='Only needed4V/4down/QKV/labels/factors/deployments; not entire216GiB cache')
    save_json(root/'reuse_map.json',dict(parent=str(source.root),reused=['frozen256 train/16 validation','labels','QKV inputs','V/down gradients','W0/Wq','192 baseline scores'],
        replayed=['native attention probabilities','MLP reverse signal','teacher validation logits'],new=['4 A/G','4 rank64 corrections','64 main KL']))
    return source
