import torch
from s_common import (Path,MODULES,BASELINES,require,read,sha_file,mo,slug,commit,load_record,save_json,digest,
                      TensorStore,restore_evidence)
from s_math import scale_relation


def prepare_assets(root,ident,source):
    dest=root/'parent_assets_preflight.json'
    if dest.exists():
        previous=load_record(dest,ident)
        restore_evidence(source.store,previous['small_ko_assets'])
        for key,h in previous['baseline_record_hashes'].items():require(sha_file(key)==h,'Old KL changed')
        if all(m['status']=='pending_stream_validation' for m in previous['modules'].values()):return previous
    hashes={};modules={};pins=load_record(source.root/'candidate_freeze.json',source.identity)['files']
    qpins=load_record(source.root/'quantized/freeze.json',source.identity)['files'];freeze=sha_file(source.root/'candidate_freeze.json')
    teacher=read(Path(source.manifest['config']['assets'])/'exp03/teacher_identity.json')['tensor_hashes']
    store=TensorStore(ident,0)
    for name in MODULES:
        try:
            folder=source.root/'modules'/slug(name)
            p=folder/'statistics/N256.safetensors';t,m=source.store.get(p)
            a=t['A_m'];g=t['G_canonical']
            require(a.shape==(4096,4096) and g.shape==(1024,1024) and a.dtype==g.dtype==torch.float64,'Invalid parent full A/G')
            require(torch.isfinite(a).all() and torch.isfinite(g).all(),'Nonfinite parent A/G')
            require(m['samples']==source.index['budgets']['N256'] and m['normalization']=='A: D*L, G: N*T' and m['A_hash']==mo.digest_tensor(a),'Wrong parent statistics binding')
            ap=folder/'statistics/N256_A.safetensors';at,am=source.store.get(ap)
            require(am['windows']==list(range(256)) and am['each_window_counted_once'] and torch.equal(at['A_m'],a),'Wrong parent input moment')
            raw,rm=source.store.get(folder/'factors/N256/Marginal/raw.safetensors')
            require(rm['samples']==m['samples'] and rm['method']=='Marginal' and rm['budget']=='N256','Wrong Marginal factors')
            relation=scale_relation(raw,a,g)
            qp=source.root/'quantized'/(slug(name)+'.safetensors');qt,qm=source.store.get(qp)
            require(qm['module']==name and mo.digest_tensor(qt['W0'])==teacher[name+'.weight']['hash'],'Teacher W0 mismatch')
            require(qpins[qp.relative_to(source.root).as_posix()]==source.store.verified[str(qp)][2],'Wrong Wq freeze')
            for method in BASELINES:
                key='None' if method=='None' else 'N256__'+method
                if method!='None':
                    cp=folder/'corrections'/(key+'.safetensors');_,cm=source.store.get(cp)
                    require(pins[cp.relative_to(source.root).as_posix()]==source.store.verified[str(cp)][2],'Baseline correction changed')
                    if method=='Marginal':require(cm['raw_G_hash']==mo.digest_tensor(raw['G_raw']) and cm['raw_A_hash']==mo.digest_tensor(raw['A_raw']),'Raw factors not bound to old Marginal candidate')
                for w in range(16):
                    f=source.root/'scores/validation'/f'w{w:04d}'/(slug(name)+'___'+key+'.json');r=load_record(f,source.identity)
                    require(r['freeze_hash']==freeze and r['token_hash']==mo.digest_tensor(source.validation[w]) and r['scope']==name and r['candidate']==key and r['scores']['tokens']==2047,'Old KL binding differs')
                    hashes[str(f)]=sha_file(f)
            store.put(root/'borrowed_G'/slug(name)/'canonical.safetensors',dict(G=g),module=name,
                      parent_file=str(p),parent_tensor_hash=mo.digest_tensor(g),normalization='sum(g.T@g)/(256*2047)',gauge_relation=relation)
            modules[name]=dict(status='pending_stream_validation',G_status='verified',G_source=str(p),G_hash=mo.digest_tensor(g),gauge=relation)
        except (FileNotFoundError,RuntimeError,AssertionError,KeyError) as exc:
            modules[name]=dict(status='missing' if isinstance(exc,FileNotFoundError) else 'identity_mismatch',error=str(exc))
    record=dict(identity=ident,small_ko_assets=source.store.evidence,baseline_record_hashes=hashes,modules=modules,
           optional_direct_query='optional_not_used',audit_and_down_gradients='optional_not_used',
           X_K_gradients='validated once while accumulating; pending until each module commits',base_identity=source.base.identity,ko_identity=source.identity)
    save_json(dest,dict(record,record_sha256=digest(record)))
    return load_record(dest,ident)


def finalize_assets(root,ident):
    p=root/'parent_assets_audit.json';row=load_record(root/'parent_assets_preflight.json',ident);row.pop('identity')
    for name in MODULES:
        done=load_record(root/'statistics'/slug(name)/'complete.json',ident)
        require(done['windows']==256,'Incomplete module')
        row['modules'][name].update(status='verified',stream_audit=str(root/'statistics'/slug(name)/'parent_evidence.json'),X=256,K_gradients=256)
    row['X_K_gradients']='verified all1024 X and1024 K gradients with receipts/commits/sample/label/x hashes'
    require(len(row['baseline_record_hashes'])==320,'Missing old KL points')
    commit(p,ident,**row)
