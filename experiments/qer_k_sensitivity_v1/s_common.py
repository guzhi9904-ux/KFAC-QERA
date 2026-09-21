from pathlib import Path
import sys
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'qer_ko_increment_v1'))
from ko_common import (read,save_json,save_csv,read_tensors,sha_file,mo,sm,require,slug,digest,
                       commit,load_record,checked_files,file_table,TensorStore,lock,ParentStore,
                       identity as ko_identity,source_x)
from ko_assets import Source as BaseSource
from ko_resources import Resources
from ko_replay import ReplayTeacher
from storage import save_tensors
__all__=['Path','HERE','read','save_json','save_csv','read_tensors','sha_file','mo','sm','require','slug','digest',
         'commit','load_record','checked_files','file_table','TensorStore','lock','ParentStore','source_x',
         'Resources','ReplayTeacher','save_tensors','LAYERS','MODULES','METHOD','BASELINES','PLAN',
         'identity','Source','AuditStore','restore_evidence']
LAYERS=[0,10,20,31]
MODULES=[f'model.layers.{i}.self_attn.k_proj' for i in LAYERS]
METHOD='Sensitivity-weighted Marginal'
BASELINES=['A-only','Marginal','Token-joint','Sequence-one-step','None']
PLAN=dict(N=256,L=2048,T=2047,rank=64,modules=MODULES,FP64_tolerance=1e-10,
          weight='full1024-dimensional squared norm of native K output gradient',
          A='sum(w*x*x.T)/sum(w), one global normalization',G='unchanged full canonical Marginal G',
          eta_A=.001,eta_G=.001,checkpoint_every=8,validation_windows=16,test_windows=0,
          budget_hours=4.,disk_limit_GiB=8.,pilot_windows=[0,1],pilot_tokens=16,pilot_features=16)


def identity(kroot):
    parent=read(kroot/'manifest.json');require(ko_identity(parent['config'])==parent,'Frozen KO/base source/config changed')
    require(load_record(kroot/'complete.json',parent['identity'])['passed'],'KO parent not complete')
    from common import PLAN as solver
    require(solver['rank']==64 and solver['eta_A']==solver['eta_G']==.001,'Solver changed')
    material=dict(version='k_sensitivity_v1',plan=PLAN,ko_run=str(kroot),ko_manifest=parent,
                  source={p.name:sha_file(p) for p in HERE.iterdir() if p.suffix in ('.py','.md','.sh')})
    return dict(identity=digest(material),**material)


class AuditStore(ParentStore):
    def __init__(self,ident):super().__init__(ident,0);self.evidence={}
    def get(self,path):
        t,m=super().get(path);p=Path(path);sig=self.verified[str(p)]
        self.evidence[str(p)]=dict(size=sig[0],mtime_ns=sig[1],file_sha256=sig[2],
            receipt_sha256=sha_file(p.with_suffix('.json')),tensor_hashes=m['tensor_hashes'])
        return t,m


def restore_evidence(store,evidence):
    for key,row in evidence.items():
        f=Path(key);st=f.stat()
        require((st.st_size,st.st_mtime_ns)==(row['size'],row['mtime_ns']),'Borrowed file changed: '+key)
        require(sha_file(f.with_suffix('.json'))==row['receipt_sha256'],'Borrowed receipt changed: '+key)
        store.verified[key]=(row['size'],row['mtime_ns'],row['file_sha256'])
    store.evidence=dict(evidence)


class Source:
    def __init__(self,kroot):
        self.root=kroot;self.manifest=read(kroot/'manifest.json');self.identity=self.manifest['identity']
        self.base=BaseSource(self.manifest['config']);self.base.store=AuditStore(self.base.identity)
        self.store=AuditStore(self.identity);self.index=self.base.index;self.validation=self.base.validation
        self.records={}
        require(len(self.index['fit'])==256 and self.base.ids.shape==(256,2048),'Fit dimensions differ')
        require([r['window'] for r in self.index['fit']]==list(range(256)) and all(r['replicate']==0 for r in self.index['fit']),'Wrong sample ordering')
        index=read(kroot/'data/index.json')
        require(index['identity']==self.identity and index['fit']==self.index['fit'] and index['budgets']['N256']==self.index['budgets']['N256'],'KO/base samples differ')
    def sample(self,name,row):
        label=self.base.label(row);x=self.base.x(name,row['window'])
        gp=self.root/'cache/g'/slug(name)/(row['id']+'.safetensors');t,m=self.store.get(gp);g=t['g']
        require(x.shape==(1,2048,4096) and g.shape==(2048,1024) and x.dtype==g.dtype==torch.float32,'Native X/K-g shape/dtype changed')
        require(torch.isfinite(x).all() and torch.isfinite(g).all(),'Nonfinite cached X/g')
        require(m['sample']==row and m['module']==name and m['loss_reduction']=='sum' and m['label_hash']==mo.digest_tensor(label) and m['x_hash']==mo.digest_tensor(x),'Gradient source binding differs')
        bc=load_record(self.base.root/'cache/commits'/(row['id']+'.json'),self.base.identity)
        kc=load_record(self.root/'cache/commits'/(row['id']+'.json'),self.identity)
        for base in (self.base.root,self.root):
            cp=base/'cache/commits'/(row['id']+'.json');self.records[str(cp)]=sha_file(cp)
        require(bc['sample']==kc['sample']==row and kc['parent_identity']==self.base.identity,'Wrong sample commits')
        for p in (source_x(self.base.root,name,row['window']),self.base.root/'cache/labels'/(row['id']+'.safetensors')):
            require(bc['files'][p.relative_to(self.base.root).as_posix()]==self.base.store.verified[str(p)][2],'Base commit mismatch')
        require(kc['files'][gp.relative_to(self.root).as_posix()]==self.store.verified[str(gp)][2],'K commit mismatch')
        return x,g
