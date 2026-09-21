from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
BORROWED=HERE.parent/'qer_ko_increment_v1'
sys.path.insert(0,str(BORROWED))
from ko_common import (read,save_json,save_csv,read_tensors,sha_file,mo,sm,require,slug,digest,
                       commit,load_record,checked_files,file_table,TensorStore,lock,parent_identity,ParentStore)
from ko_assets import Source
from ko_resources import Resources
from ko_replay import ReplayTeacher
from storage import save_tensors,atomic_bytes
sys.path.insert(0,str(HERE))
__all__=['Path','HERE','read','save_json','save_csv','read_tensors','sha_file','mo','sm','require','slug','digest',
         'commit','load_record','checked_files','file_table','TensorStore','lock','ParentStore','Source','Resources',
         'ReplayTeacher','save_tensors','atomic_bytes','LAYERS','MODULES','BASELINES','identity']
LAYERS=[0,10,20,31]
MODULES=[f'model.layers.{i}.self_attn.v_proj' for i in LAYERS]
BASELINES=['Marginal','Sequence-one-step','None']


def identity(config):
    p=read(Path(config['parent_run'])/'manifest.json')
    require(parent_identity(p['config'])==p,'Frozen parent code/config changed')
    from common import PLAN
    require(PLAN['eta_A']==PLAN['eta_G']==1e-3 and PLAN['rank']==64,'Wrong frozen solver')
    files={str(x.relative_to(HERE.parent.parent)):sha_file(x) for d in (HERE,BORROWED) for x in d.iterdir() if x.suffix in ('.py','.md','.sh')}
    material=dict(version='v_attention_proto_v1',config=config,parent_identity=p['identity'],source=files,
                  raw_normalization='A=sum_head_effective_xx/(N*L*Hq); G_block=sum_delta_delta/(N*T)',
                  metric='KV-block-diagonal G and one shared A; not exact full GQA Fisher')
    return dict(identity=digest(material),**material)
