"""Import frozen primitives without changing their source or run identity."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
sys.path.insert(0,str(REPO/'experiments/qer_multimodule_v1'))
import bridge
from shared_teacher import SharedTeacher
from resources import Resources
sys.path.insert(0,str(HERE))
__all__=['HERE','REPO','bridge','SharedTeacher','Resources']

PLAN=dict(version='kronecker_precision_pilot_v1',
    modules=[f'model.layers.10.{s}' for s in ('self_attn.q_proj','self_attn.v_proj','mlp.down_proj')],
    fit_windows=[0,1],check_windows=[0,1,2,3],label_replicate=0,L=2048,T=2047,rank=64,
    methods=['Marginal','Token-joint','Sequence-one-step','Full-fit'],
    token_rounds=3,full_rounds=8,eta=.001,tf32=False,
    tolerances=dict(factor_relative=1e-4,product_relative=1e-4,reference_objective_relative=1e-4,
        KL_over_quantized=1e-3,KL_absolute=1e-8,raw_negative_relative_fp32=1e-5,
        negative_over_damping=.1,round_objective_relative_fp32=1e-5,condition=1e8,
        root=1e-10,svd=1e-8,deployment=1e-4),
    scope='Two real fit windows, eight fixed Full-fit rounds, all four methods and three full module shapes; validation/test splits untouched. Does not establish128/256-window convergence or all-layer equivalence.')

def manifest(config):
    parent=bridge.identity(config)
    own={p.name:bridge.sha_file(p) for p in sorted(HERE.iterdir()) if p.suffix in ('.py','.md','.json','.sh')}
    data=dict(plan=PLAN,config=config,source=own,borrowed=parent)
    return dict(identity=bridge.digest(data),**data)
