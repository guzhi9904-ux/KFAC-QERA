"""Incremental experiment identity; all earlier experiment sources stay frozen."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
PARENT_CODE=HERE.parent/'qer_multimodule_three_fp64_v1'
sys.path.insert(0,str(PARENT_CODE))
from bridge import (read, save_json, save_csv, read_tensors, sha_file, mo, sm, require,
                    slug, digest, commit, load_record, checked_files, file_table,
                    TensorStore, lock, identity as parent_identity)
sys.path.insert(0,str(HERE))
__all__=['read','save_json','save_csv','read_tensors','sha_file','mo','sm','require','slug','digest','commit',
         'load_record','checked_files','file_table','TensorStore','lock','identity','ParentStore','MixedStore',
         'source_x','jobs','LAYERS','NEW','OLD','METHODS']

LAYERS=[0,10,20,31]
NEW=[f'model.layers.{i}.self_attn.{m}_proj' for i in LAYERS for m in ('k','o')]
OLD=[f'model.layers.{i}.{m}' for i in LAYERS for m in ('self_attn.q_proj','self_attn.v_proj','mlp.down_proj')]
METHODS=['A-only','Marginal','Token-joint','Sequence-one-step']


def identity(config):
    parent=read(Path(config['parent_run'])/'manifest.json')
    require(parent_identity(parent['config'])==parent,'Frozen parent source/config changed')
    material=dict(version='qer_ko_increment_v1',config=config,parent_identity=parent['identity'],
                  source={p.name:sha_file(p) for p in HERE.iterdir() if p.suffix in ('.py','.md','.sh')})
    return dict(identity=digest(material),**material)


class ParentStore(TensorStore):
    """Require preexisting receipts so inherited reads can never write to parent."""
    def get(self,path):
        require(Path(path).with_suffix('.json').exists(),'Missing parent receipt: '+str(path))
        return super().get(path)
    def put(self,*args,**kwargs):raise RuntimeError('Parent store is read-only')


class MixedStore(TensorStore):
    def __init__(self,ident,cap,parent,pident):
        super().__init__(ident,cap);self.parent=Path(parent).resolve();self.borrowed=ParentStore(pident,0)
    def get(self,path):
        if Path(path).resolve().is_relative_to(self.parent):return self.borrowed.get(path)
        return super().get(path)


def source_x(parent,name,window):
    group=name.rsplit('.',1)[0]+'.qkv_input' if name.endswith(('.q_proj','.k_proj','.v_proj')) else name+'.input'
    return Path(parent)/'cache/x'/slug(group)/f'w{window:04d}.safetensors'


def jobs(config):
    return [(n,m) for n in config['modules'] for m in (METHODS+['None'] if n in NEW else ['A-only'])]
