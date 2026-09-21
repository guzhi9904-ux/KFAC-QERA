from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'qer_v_attention_proto_v1'))
from v_common import (read,save_json,save_csv,sha_file,mo,require,slug,digest,commit,load_record,
                      checked_files,file_table,lock,Resources,ReplayTeacher)
from ko_common import identity as ko_identity,source_x
from v_common import identity as v_identity
from v_assets import AuditStore
from ko_assets import Source
__all__=['Path','HERE','read','save_json','save_csv','sha_file','mo','require','slug','digest','commit','load_record',
         'checked_files','file_table','lock','Resources','ReplayTeacher','LAYERS','MODULES','WINDOWS','QUERIES','METHODS','PLAN','identity','assets']
LAYERS=[0,10,20,31]
MODULES=[f'model.layers.{i}.self_attn.k_proj' for i in LAYERS]
WINDOWS=[0,32,64,96,128,160,192,224]
QUERIES=[127,383,639,895,1151,1407,1663,1919]
METHODS=['None','A-only','Marginal','Token-joint','Sequence-one-step']
PLAN=dict(layers=LAYERS,windows=WINDOWS,pilot_windows=[0,32],layer_order=[20,0,10,31],
          queries=QUERIES,post_queries=[383,1663],post_head_rule='4*b+(window_list_index%4)',
          spectrum_k=[1,2,4,8],post_k=[1,4],T=2047,L=2048,N=8,rank=64,
          fp32_tolerance=1e-5,fp64_tolerance=1e-10,zero_floor=1e-12,
          softmax_row_identity='sum(e)=mu*(1-sum(p))+kernel_roundoff; mu=sum_FP64(p*dp)',
          softmax_kernel_bound='unchanged 8*eps32*sum_FP64(p*abs(dp))+1e-20',
          softmax_probability_mass_tolerance=1e-5,
          cancellation_rule='FP64 abs bound=1e-10*roundoff_scale only when reference<=1e-12*roundoff_scale',
          methods=METHODS,stages=['A','B'],no_new_method=True,no_new_KL=True,budget_hours=2.,disk_limit_GiB=8.)


def identity(kroot,vroot):
    km=read(kroot/'manifest.json');vm=read(vroot/'manifest.json')
    require(ko_identity(km['config'])==km,'Frozen KO sources changed')
    require(v_identity(vm['config'])==vm,'Frozen V sources changed')
    require(km['config']['parent_run']==vm['config']['parent_run'],'Parents differ')
    material=dict(version='k_structure_audit_v2',plan=PLAN,ko_run=str(kroot),v_run=str(vroot),
                  ko_manifest=km,v_manifest=vm,source={p.name:sha_file(p) for p in HERE.iterdir() if p.suffix in ('.py','.md','.sh')})
    return dict(identity=digest(material),**material)


def assets(root,config,ident,kroot,resources):
    source=Source(config);source.store=AuditStore(source.identity)
    kid=read(kroot/'manifest.json')['identity'];ks=AuditStore(kid)
    require(load_record(kroot/'complete.json',kid)['passed'],'KO incomplete')
    pins=load_record(kroot/'candidate_freeze.json',kid)['files']
    qpins=load_record(kroot/'quantized/freeze.json',kid)['files']
    expected=read(Path(config['assets'])/'exp03/teacher_identity.json')['tensor_hashes']
    baselines=[];baseline_hashes={};residuals={}
    with resources.timed('required_asset_audit'):
        for window in WINDOWS:
            row=source.index['fit'][window];label=source.label(row)
            require(row['window']==window and row['replicate']==0,'Wrong sample')
            parent_commit=load_record(source.root/'cache/commits'/(row['id']+'.json'),source.identity)
            kcommit=load_record(kroot/'cache/commits'/(row['id']+'.json'),kid)
            require(parent_commit['sample']==kcommit['sample']==row,'Cache sample differs')
            lp=source.root/'cache/labels'/(row['id']+'.safetensors')
            require(parent_commit['files'][lp.relative_to(source.root).as_posix()]==source.store.verified[str(lp)][2],'Label commit mismatch')
            for layer,name in zip(LAYERS,MODULES):
                source.x(name,window);source.gradient(f'model.layers.{layer}.mlp.down_proj',row,label)
                gp=kroot/'cache/g'/slug(name)/(row['id']+'.safetensors')
                # Missing K is explicitly reported; no silent N256 reconstruction.
                require(gp.exists(),'Missing K cache; requires explicit 8-window fallback implementation: '+str(gp))
                g,m=ks.get(gp)
                require(m['module']==name and m['sample']==row and m['label_hash']==mo.digest_tensor(label),'K gradient binding differs')
                require(kcommit['files'][gp.relative_to(kroot).as_posix()]==ks.verified[str(gp)][2],'K commit mismatch')
                for p in [source_x(source.root,name,window),source.root/'cache/g'/slug(f'model.layers.{layer}.mlp.down_proj')/(row['id']+'.safetensors')]:
                    require(parent_commit['files'][p.relative_to(source.root).as_posix()]==source.store.verified[str(p)][2],'Parent commit mismatch')
            print('ASSET_WINDOW_VERIFIED',window,flush=True)
        freeze=sha_file(kroot/'candidate_freeze.json')
        for name in MODULES:
            qp=kroot/'quantized'/(slug(name)+'.safetensors');q,m=ks.get(qp)
            require(qpins[qp.relative_to(kroot).as_posix()]==ks.verified[str(qp)][2],'Quantization freeze mismatch')
            require(m['module']==name and mo.digest_tensor(q['W0'])==expected[name+'.weight']['hash'],'Wrong W0')
            residuals[name]={'None':q['W0'].double()-q['Wq'].double()}
            for method in METHODS[1:]:
                ks.get(kroot/'modules'/slug(name)/'factors/N256'/method/'raw.safetensors')
                cp=kroot/'modules'/slug(name)/'corrections'/('N256__'+method+'.safetensors');t,_=ks.get(cp)
                require(pins[cp.relative_to(kroot).as_posix()]==ks.verified[str(cp)][2],'Correction freeze mismatch')
                residuals[name][method]=q['W0'].double()-t['W_deploy'].double()
            for method in METHODS:
                key='None' if method=='None' else 'N256__'+method
                for w in range(16):
                    p=kroot/'scores/validation'/f'w{w:04d}'/(slug(name)+'___'+key+'.json');r=load_record(p,kid)
                    require(r['scope']==name and r['candidate']==key and r['freeze_hash']==freeze
                            and r['token_hash']==mo.digest_tensor(source.validation[w]) and r['scores']['tokens']==2047,'Baseline binding differs')
                    baselines.append(dict(module=name,method=method,window=w,**r['scores']));baseline_hashes[str(p)]=sha_file(p)
    commit(root/'parent_assets_audit.json',ident,passed=True,base_identity=source.identity,ko_identity=kid,
           required_tensors={**source.store.evidence,**ks.evidence},baseline_hashes=baseline_hashes,
           reused_K_gradients=32,reused_down_gradients=32,reused_X=32,reused_labels=8,reused_KL=320,
           no_parent_writes=True,no_full_cache_scan=True)
    save_csv(root/'diagnostics/old_validation_KL.csv',baselines)
    return source,ks,residuals,baselines
