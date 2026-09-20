"""Numerical primitives are reused without changing frozen historical sources."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
LEGACY=HERE.parent/'qer_kronecker_l10_v2'
sys.path.insert(0,str(LEGACY))
from common import mo,read,save_json,save_csv,read_tensors,save_tensors,sha_file,digest,require,slug,hidden_forward
from teacher import Teacher
from runtime_io import TensorStore,memory_limits,lock
from experiment import Experiment as LegacyExperiment
from records import commit,load_record,file_table,checked_files
from exact_ops import GroupedStream
import sketch_math as sm
sys.path.insert(0,str(REPO/'tools/kronecker_l10_recovery'))
from resume import warm_and_probe
sys.path.insert(0,str(HERE))
__all__=['mo','read','save_json','save_csv','read_tensors','save_tensors','sha_file','digest','require','slug','hidden_forward',
         'Teacher','TensorStore','memory_limits','lock','LegacyExperiment','commit','load_record','file_table','checked_files',
         'GroupedStream','sm','warm_and_probe','identity','MODULES','METHODS']

METHODS=['Marginal','Token-joint','Sequence-one-step','Full-fit']
MODULES=[f'model.layers.{i}.{suffix}' for i in (0,10,20,31)
         for suffix in ('self_attn.q_proj','self_attn.v_proj','mlp.down_proj')]

def identity(config):
    dirs=[HERE,LEGACY,HERE.parent/'qer_teacher_kl_exp01',HERE.parent/'qer_teacher_kl_exp03',
          HERE.parent/'qer_functional_gradient_4090_v1',REPO/'tools/kronecker_l10_recovery']
    files={str(p.relative_to(REPO)):sha_file(p) for d in dirs for p in sorted(d.iterdir()) if p.suffix in ('.py','.json','.md','.sh')}
    material=dict(version='qer_multimodule_v1',config=config,source=files)
    return dict(identity=digest(material),**material)
