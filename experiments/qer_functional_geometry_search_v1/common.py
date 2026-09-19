"""Shared, versioned IO and the frozen parent's numerical primitives."""
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PARENT = HERE.parent/'qer_teacher_kl_exp01'
sys.path.insert(0, str(PARENT))
from storage import save_json, save_csv, save_tensors, read_tensors, sha_file, atomic_bytes
import math_ops as mo
from model_ops import capture, recompute_suffix, hidden_forward as original_hidden_forward
sys.path.insert(0, str(HERE))

PLAN = json.loads((HERE/'plan.json').read_text(encoding='utf8'))
__all__ = ['PLAN','HERE','REPO','PARENT','save_json','save_csv','save_tensors','read_tensors',
           'sha_file','atomic_bytes','mo','capture','recompute_suffix','read','slug','digest',
           'source_identity','seed','hidden_forward','require']
read = lambda p: json.loads(Path(p).read_text(encoding='utf8'))
slug = lambda s: s.replace('.', '__')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def source_identity(config):
    own = {p.name: hashlib.sha256(p.read_text(encoding='utf8').replace('\r\n', '\n').encode()).hexdigest()
           for p in sorted(HERE.iterdir()) if p.suffix in ('.py', '.json', '.md', '.sh')}
    borrowed = {str(p.relative_to(REPO)): sha_file(p) for p in (
        PARENT/'math_ops.py', PARENT/'model_ops.py', PARENT/'storage.py',
        HERE.parent/'qer_teacher_kl_exp03/fit_data.py', HERE.parent/'qer_teacher_kl_exp03/ag_math.py',
        HERE.parent/'qer_functional_gradient_4090_v1/device_layout.py',
        HERE.parent/'qer_functional_gradient_4090_v1/config_identity.py')}
    material = dict(plan=PLAN, config=config, source=own, borrowed=borrowed,
                    source_encoding='new source UTF-8/LF; parent source exact bytes')
    return dict(identity=digest(material), **material)


def seed(role, article_id, replicate):
    key = [PLAN['namespace'], PLAN['seed'], role, article_id, replicate]
    return int.from_bytes(hashlib.sha256(json.dumps(key, separators=(',', ':')).encode()).digest()[:8], 'little') % (2**63-1)


def hidden_forward(model, ids):
    return original_hidden_forward(model, ids).to(model.lm_head.weight.device)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)
