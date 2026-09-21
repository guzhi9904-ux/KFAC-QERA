from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'qer_k_structure_audit_v3'))
import k_common as ac
from k_common import (read,save_json,save_csv,sha_file,mo,require,slug,digest,commit,load_record,
                      checked_files,file_table,lock,Resources,ReplayTeacher,LAYERS,MODULES)
from v_common import TensorStore,sm,read_tensors,save_tensors
from v_assets import AuditStore
from ko_assets import Source as BaseSource
from ko_common import source_x
__all__=['Path','HERE','read','save_json','save_csv','sha_file','mo','require','slug','digest','commit','load_record',
         'checked_files','file_table','lock','Resources','ReplayTeacher','LAYERS','MODULES','TensorStore','sm',
         'read_tensors','save_tensors','METHOD','BASELINES','PLAN','identity','Source','audit_assets']
METHOD='Direct Query-Marginal'
BASELINES=['A-only','Marginal','Token-joint','Sequence-one-step','None']
PLAN=dict(N=256,L=2048,T=2047,Hq=32,Hkv=8,d=128,rank=64,modules=MODULES,
          pilot_windows=[0,1],pilot_layer_order=[20,0,10,31],validation_windows=16,test_windows=0,
          A_denominator='N*L*Hq',G_sum='sum(q_pre_rope.T@q_pre_rope/d) per KV block',
          G_denominator='N*T',Gbar_denominator='N*L*Hq',canonical_multiplier='L*Hq/T',
          ignored_relative_RoPE=True,teacher_RoPE=True,checkpoint_every=1,
          budget_hours=8.,disk_limit_GiB=16.,FP32=1e-5,FP64=1e-10,
          pilot_gram_block=dict(heads=[0,4],queries=list(range(16))),
          zero_e_query_rows_still_count_in_G=True,no_surrogate_vs_true_score_gate=True)


def identity(kroot,auditroot):
    parent=read(auditroot/'manifest.json')
    require(parent==ac.identity(kroot,Path(parent['v_run'])),'Accepted K audit source/config changed')
    require(load_record(auditroot/'complete.json',parent['identity'])['passed'],'K audit incomplete')
    from common import PLAN as solver
    require(solver['eta_A']==solver['eta_G']==1e-3 and solver['rank']==64,'Solver changed')
    material=dict(version='k_direct_query_v1',plan=PLAN,ko_run=str(kroot),audit_run=str(auditroot),
                  audit_manifest=parent,source={p.name:sha_file(p) for p in HERE.iterdir() if p.suffix in ('.py','.md','.sh')})
    return dict(identity=digest(material),**material)


class Source:
    def __init__(self,kroot):
        self.root=kroot;self.manifest=read(kroot/'manifest.json');self.identity=self.manifest['identity']
        self.base=BaseSource(self.manifest['config']);self.base.store=AuditStore(self.base.identity)
        self.store=AuditStore(self.identity);self.index=self.base.index;self.ids=self.base.ids;self.validation=self.base.validation
    def label(self,row):return self.base.label(row)
    def x(self,name,window):return self.base.x(name,window)
    def down(self,layer,row,label):return self.base.gradient(f'model.layers.{layer}.mlp.down_proj',row,label)
    def kgrad(self,name,row,label):
        t,m=self.store.get(self.root/'cache/g'/slug(name)/(row['id']+'.safetensors'))
        require(m['module']==name and m['sample']==row and m['label_hash']==mo.digest_tensor(label),'Wrong frozen K gradient')
        return t['g']


def audit_assets(root,ident,source,resources):
    p=root/'parent_assets_audit.json'
    if p.exists():
        audit=load_record(p,ident)
        for label,store in [('base',source.base.store),('ko',source.store)]:
            for key,row in audit[label].items():
                f=Path(key);s=f.stat()
                require((s.st_size,s.st_mtime_ns)==(row['size'],row['mtime_ns']) and sha_file(f.with_suffix('.json'))==row['receipt_sha256'],'Borrowed asset changed')
                store.verified[key]=(row['size'],row['mtime_ns'],row['file_sha256'])
            store.evidence=audit[label]
        for key,h in audit['baseline_record_hashes'].items():require(sha_file(key)==h,'Old KL record changed')
        print('ASSET_AUDIT_RESUMED',flush=True);return
    pins=load_record(source.root/'candidate_freeze.json',source.identity)['files']
    qpins=load_record(source.root/'quantized/freeze.json',source.identity)['files']
    frozen=sha_file(source.root/'candidate_freeze.json');baselines={}
    with resources.timed('required_asset_audit'):
        for row in source.index['fit']:
            w=row['window'];require(row['replicate']==0,'Require one frozen label sequence per window')
            resources.boundary();label=source.label(row)
            parent_commit=load_record(source.base.root/'cache/commits'/(row['id']+'.json'),source.base.identity)
            paths=[source.base.root/'cache/labels'/(row['id']+'.safetensors')]
            for layer,name in zip(LAYERS,MODULES):
                source.x(name,w);source.down(layer,row,label)
                paths.extend([source_x(source.base.root,name,w),source.base.root/'cache/g'/slug(f'model.layers.{layer}.mlp.down_proj')/(row['id']+'.safetensors')])
                if w<2:
                    source.kgrad(name,row,label)
                    kcommit=load_record(source.root/'cache/commits'/(row['id']+'.json'),source.identity)
                    kp=source.root/'cache/g'/slug(name)/(row['id']+'.safetensors')
                    require(kcommit['sample']==row and kcommit['files'][kp.relative_to(source.root).as_posix()]==source.store.verified[str(kp)][2],'K cache commit mismatch')
            require(parent_commit['sample']==row,'Parent sample mismatch')
            for f in paths:require(parent_commit['files'][f.relative_to(source.base.root).as_posix()]==source.base.store.verified[str(f)][2],'Parent cache hash mismatch')
            if (w+1)%32==0:print('ASSET_AUDIT',w+1,'/256',flush=True)
        expected=read(Path(source.manifest['config']['assets'])/'exp03/teacher_identity.json')['tensor_hashes']
        for name in MODULES:
            qp=source.root/'quantized'/(slug(name)+'.safetensors');q,m=source.store.get(qp)
            require(m['module']==name and mo.digest_tensor(q['W0'])==expected[name+'.weight']['hash'],'Wrong quantized teacher')
            require(qpins[qp.relative_to(source.root).as_posix()]==source.store.verified[str(qp)][2],'Wrong Wq freeze')
            for method in BASELINES:
                key='None' if method=='None' else 'N256__'+method
                if method!='None':
                    cp=source.root/'modules'/slug(name)/'corrections'/(key+'.safetensors');source.store.get(cp)
                    require(pins[cp.relative_to(source.root).as_posix()]==source.store.verified[str(cp)][2],'Old correction changed')
                for w in range(16):
                    f=source.root/'scores/validation'/f'w{w:04d}'/(slug(name)+'___'+key+'.json');r=load_record(f,source.identity)
                    require(r['scope']==name and r['candidate']==key and r['freeze_hash']==frozen and r['token_hash']==mo.digest_tensor(source.validation[w]) and r['scores']['tokens']==2047,'Baseline identity differs')
                    baselines[str(f)]=sha_file(f)
    commit(p,ident,base=source.base.store.evidence,ko=source.store.evidence,baseline_record_hashes=baselines,
           reused_KL=320,fit_windows=256,K_gradient_windows=[0,1],no_parent_writes=True)
    print('REQUIRED_ASSETS_VERIFIED',flush=True)
