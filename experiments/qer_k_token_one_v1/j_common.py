from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'qer_k_sensitivity_v1'))
from s_common import (read,save_json,save_csv,read_tensors,sha_file,mo,sm,require,slug,digest,
                      commit,load_record,checked_files,file_table,TensorStore,lock,Resources,ReplayTeacher,
                      save_tensors,LAYERS,MODULES,AuditStore,restore_evidence,Source as SensSource,
                      identity as sens_identity)
from s_statistics import verify_evidence,evidence
from s_math import compare
__all__=['Path','HERE','read','save_json','save_csv','read_tensors','sha_file','mo','sm','require','slug','digest',
         'commit','load_record','checked_files','file_table','TensorStore','lock','Resources','ReplayTeacher','save_tensors',
         'LAYERS','MODULES','AuditStore','restore_evidence','verify_evidence','evidence','compare','identity','Source',
         'METHOD','BASELINES','TABLE_METHODS','PLAN']
METHOD='Token-joint-one'
BASELINES=['Marginal','Sensitivity-A','Token-joint-3','Sequence','None']
TABLE_METHODS=['Marginal','Sensitivity-A',METHOD,'Token-joint-3','Sequence']
PLAN=dict(N=256,L=2048,T=2047,n=4096,m=1024,rank=64,modules=MODULES,
          initialization='original raw A_M and I_1024',rounds=1,synchronous=True,
          A1='borrowed Sensitivity U/(N*T*m)',G1='sum((x.T A_M x)*g*g.T)/(N*T*||A_M||F^2)',
          pilot_windows=[0,1],pilot_explicit_positions=16,FP64_tolerance=1e-10,
          eta_A=.001,eta_G=.001,checkpoint_every=8,validation_windows=16,test_windows=0,
          budget_hours=4.,disk_limit_GiB=8.)


def identity(kroot,sroot):
    parent=read(sroot/'manifest.json');require(parent==sens_identity(kroot),'Frozen Sensitivity/KO/base identity changed')
    done=load_record(sroot/'complete.json',parent['identity'])
    require(done['passed'] and done['layers']==4 and done['fit_windows']==256 and done['rank']==64,'Sensitivity incomplete')
    checked_files(sroot,done['files'])
    material=dict(version='k_token_one_v1',plan=PLAN,ko_run=str(kroot),sens_run=str(sroot),sens_manifest=parent,
                  source={p.name:sha_file(p) for p in HERE.iterdir() if p.suffix in ('.py','.sh','.md')})
    return dict(identity=digest(material),**material)


class Source(SensSource):
    def __init__(self,kroot,sroot):
        super().__init__(kroot);self.sroot=sroot;self.sid=read(sroot/'manifest.json')['identity'];self.sstore=AuditStore(self.sid)
    def baseline_path(self,name,method,window):
        if method=='Sensitivity-A':return self.sroot/'scores/validation'/f'w{window:04d}'/(slug(name)+'.json'),self.sid
        key={'Marginal':'N256__Marginal','Token-joint-3':'N256__Token-joint','Sequence':'N256__Sequence-one-step','None':'None'}[method]
        return self.root/'scores/validation'/f'w{window:04d}'/(slug(name)+'___'+key+'.json'),self.identity
    def baseline_weight(self,name,method):
        if method=='Sensitivity-A':return self.sstore.get(self.sroot/'corrections'/slug(name)/'sensitivity_weighted.safetensors')[0]['W_deploy']
        if method=='None':return self.store.get(self.root/'quantized'/(slug(name)+'.safetensors'))[0]['Wq']
        key={'Marginal':'Marginal','Token-joint-3':'Token-joint','Sequence':'Sequence-one-step'}[method]
        return self.store.get(self.root/'modules'/slug(name)/'corrections'/('N256__'+key+'.safetensors'))[0]['W_deploy']
