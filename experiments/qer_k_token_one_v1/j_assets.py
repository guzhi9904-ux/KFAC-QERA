import torch
from j_common import (Path,MODULES,BASELINES,TensorStore,require,read,save_json,digest,sha_file,mo,slug,
                      commit,load_record,restore_evidence)
from j_math import recover_a


def prepare_assets(root,ident,source):
    dest=root/'parent_assets_preflight.json'
    if dest.exists():
        old=load_record(dest,ident)
        restore_evidence(source.store,old['ko_assets']);restore_evidence(source.sstore,old['sens_assets'])
        for p,h in old['baseline_record_hashes'].items():require(sha_file(p)==h,'Old KL changed')
        if all(r['status']=='pending_stream_validation' for r in old['modules'].values()):return old
    pins=load_record(source.root/'candidate_freeze.json',source.identity)['files']
    spins=load_record(source.sroot/'candidate_freeze.json',source.sid)['files']
    qpins=load_record(source.root/'quantized/freeze.json',source.identity)['files']
    kfreeze=sha_file(source.root/'candidate_freeze.json');sfreeze=sha_file(source.sroot/'candidate_freeze.json')
    pa=load_record(source.sroot/'parent_assets_audit.json',source.sid)
    require(pa['ko_identity']==source.identity and pa['base_identity']==source.base.identity,'Sensitivity parent mismatch')
    restore_evidence(source.store,pa['small_ko_assets'])
    teacher=read(Path(source.manifest['config']['assets'])/'exp03/teacher_identity.json')['tensor_hashes']
    store=TensorStore(ident,0);modules={};hashes={}
    for name in MODULES:
        try:
            sfolder=source.sroot/'statistics'/slug(name);done=load_record(sfolder/'complete.json',source.sid)
            require(done['passed'] and done['windows']==256 and pa['modules'][name]['status']=='verified','Sensitivity module incomplete')
            sp=sfolder/'raw.safetensors';st,meta=source.sstore.get(sp)
            require(done['files'][sp.relative_to(source.sroot).as_posix()]==source.sstore.verified[str(sp)][2] and meta['windows']==list(range(256)) and meta['module']==name,'Wrong Sensitivity U binding')
            a1,checks=recover_a(st)
            ep=sfolder/'parent_evidence.json';ev=load_record(ep,source.sid)
            require(ev['X']==ev['K_gradients']==256 and sha_file(ep)==done['files'][ep.relative_to(source.sroot).as_posix()],'Sensitivity source evidence changed')
            kp=source.root/'modules'/slug(name)/'statistics/N256.safetensors';kt,km=source.store.get(kp)
            am=kt['A_m'];gm=kt['G_canonical']
            require(am.shape==a1.shape==(4096,4096) and am.dtype==torch.float64 and gm.shape==(1024,1024),'Invalid raw initialization dimensions')
            require(km['samples']==source.index['budgets']['N256'] and km['normalization']=='A: D*L, G: N*T' and km['A_hash']==mo.digest_tensor(am),'A_M is not frozen raw N256 input moment')
            require(torch.equal(gm,st['G_raw']) and meta['G_hash']==mo.digest_tensor(gm),'Sensitivity full G differs from canonical Marginal')
            tp=source.root/'modules'/slug(name)/'factors/N256/Token-joint/raw.safetensors';tt,tm=source.store.get(tp)
            require(tm['method']=='Token-joint' and tm['samples']==km['samples'] and len(tm['history'])==3 and
                    [h['iteration'] for h in tm['history']]==[1,2,3] and all(h['synchronous'] for h in tm['history']) and
                    tm['stop']=='FIXED_THREE_SYNCHRONOUS_ROUNDS_NOT_CONVERGENCE_CLAIM','Token3 is not original three synchronous rounds')
            qp=source.root/'quantized'/(slug(name)+'.safetensors');qt,qm=source.store.get(qp)
            require(qm['module']==name and mo.digest_tensor(qt['W0'])==teacher[name+'.weight']['hash'] and qpins[qp.relative_to(source.root).as_posix()]==source.store.verified[str(qp)][2],'W0/Wq identity differs')
            for method in BASELINES:
                source.baseline_weight(name,method)
                if method=='Sensitivity-A':
                    cp=source.sroot/'corrections'/slug(name)/'sensitivity_weighted.safetensors'
                    require(spins[cp.relative_to(source.sroot).as_posix()]==source.sstore.verified[str(cp)][2],'Sensitivity candidate changed')
                elif method!='None':
                    key={'Marginal':'Marginal','Token-joint-3':'Token-joint','Sequence':'Sequence-one-step'}[method]
                    cp=source.root/'modules'/slug(name)/'corrections'/('N256__'+key+'.safetensors')
                    require(pins[cp.relative_to(source.root).as_posix()]==source.store.verified[str(cp)][2],'Old correction changed')
                    if method=='Token-joint-3':
                        cm=read(cp.with_suffix('.json'));require(cm['raw_A_hash']==mo.digest_tensor(tt['A_raw']) and cm['raw_G_hash']==mo.digest_tensor(tt['G_raw']),'Token3 factor/candidate mismatch')
                for w in range(16):
                    p,pid=source.baseline_path(name,method,w);r=load_record(p,pid)
                    require(r['token_hash']==mo.digest_tensor(source.validation[w]) and r['scores']['tokens']==2047,'Baseline tokens differ')
                    if method=='Sensitivity-A':require(r['module']==name and r['window']==w and r['candidate']=='Sensitivity-weighted Marginal' and r['freeze_hash']==sfreeze,'Sensitivity KL binding differs')
                    else:
                        expected='None' if method=='None' else 'N256__'+{'Marginal':'Marginal','Token-joint-3':'Token-joint','Sequence':'Sequence-one-step'}[method]
                        require(r['scope']==name and r['candidate']==expected and r['freeze_hash']==kfreeze,'KO KL binding differs')
                    hashes[str(p)]=sha_file(p)
            store.put(root/'borrowed'/slug(name)/'initial.safetensors',dict(A_M=am,A1=a1),module=name,
                      A_M_source=str(kp),A_M_hash=mo.digest_tensor(am),U_source=str(sp),U_hash=meta['tensor_hashes']['U'],
                      A_recovery_checks=checks,sens_evidence=str(ep),sens_evidence_hash=sha_file(ep),
                      A1_denominator=256*2047*1024,G1_denominator=256*2047*float(am.square().sum()),
                      original_initialization='A_M,I',Token3_three_synchronous_rounds=True)
            modules[name]=dict(status='pending_stream_validation',A_reused=True,A_recovery_checks=checks,sens_evidence=str(ep),sens_evidence_hash=sha_file(ep))
        except (FileNotFoundError,RuntimeError,AssertionError,KeyError) as exc:
            modules[name]=dict(status='missing' if isinstance(exc,FileNotFoundError) else 'identity_mismatch',error=str(exc))
    row=dict(identity=ident,modules=modules,ko_assets=source.store.evidence,sens_assets=source.sstore.evidence,
             baseline_record_hashes=hashes,base_identity=source.base.identity,ko_identity=source.identity,sens_identity=source.sid,
             audit_and_direct_query='optional_not_used',A_only='optional_not_used',first_round_extra_snapshot='optional_not_used')
    save_json(dest,dict(row,record_sha256=digest(row)));return load_record(dest,ident)


def finalize_assets(root,ident):
    row=load_record(root/'parent_assets_preflight.json',ident);row.pop('identity')
    for name in MODULES:
        done=load_record(root/'statistics'/slug(name)/'complete.json',ident);require(done['passed'] and done['windows']==256,'Incomplete statistics')
        row['modules'][name].update(status='verified',X=256,K_gradients=256)
    require(len(row['baseline_record_hashes'])==320,'Missing baseline points')
    commit(root/'parent_assets_audit.json',ident,**row)
