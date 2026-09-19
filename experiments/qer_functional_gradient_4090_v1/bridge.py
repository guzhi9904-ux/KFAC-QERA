"""Versioned portable runtime; original source and parent tensors stay read-only."""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
for key in ('HF_HUB_OFFLINE', 'HF_DATASETS_OFFLINE', 'TRANSFORMERS_OFFLINE'):
    os.environ.setdefault(key, '1')
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
read = lambda p: json.loads(Path(p).read_text(encoding='utf-8'))
CONFIG_PATH = Path(os.environ['QER_PORTABLE_CONFIG']).resolve()
CONFIG = read(CONFIG_PATH)
assert CONFIG['schema'] == 1
ASSETS = Path(CONFIG['assets']).resolve()
EXP01 = ASSETS/'exp01'
EXP03 = ASSETS/'exp03'
PLAN = read(HERE/'plan.json')
PLAN.update(exp3_code=str(HERE.parent/'qer_teacher_kl_exp03'),
            parent_code=str(HERE.parent/'qer_teacher_kl_exp01'), exp3_output=str(EXP03))
sys.path.insert(0, PLAN['parent_code'])
from run import Experiment, slug, clean, sync, OFFICIAL, Paused, request_pause
from storage import save_json, save_csv, save_tensors, read_tensors, sha_file, atomic_bytes
from model_ops import capture, hidden_forward as parent_hidden_forward, recompute_suffix
import math_ops as mo
import torch
import torch.nn.functional as F
sys.path.insert(0, str(HERE))


def hidden_forward(model, ids):
    # Be explicit about the head device, including Accelerate output-routing hooks.
    return parent_hidden_forward(model, ids).to(model.lm_head.weight.device)


def source_identity():
    source = {p.name: sha_file(p) for p in sorted(HERE.iterdir())
              if p.suffix in ('.py', '.json', '.sh', '.md')}
    borrowed = {str(p.relative_to(REPO)): sha_file(p)
                for d in (Path(PLAN['parent_code']), Path(PLAN['exp3_code']))
                for p in sorted(d.iterdir()) if p.suffix in ('.py', '.json', '.sh', '.md')}
    material = dict(plan=PLAN, source=source, borrowed_source=borrowed, portable_config=CONFIG)
    return dict(identity=hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest(), **material)


_sampled_memory_peak = 0


def cgroup_memory():
    global _sampled_memory_peak
    base = Path('/sys/fs/cgroup')
    membership = Path('/proc/self/cgroup').read_text()
    rel = next(line.split(':', 2)[2] for line in membership.splitlines() if line.startswith('0::'))
    candidates = [base, base/rel.lstrip('/')]
    values = {}; limits = []; current = []; peak = []
    for folder in candidates:
        while folder.is_relative_to(base):
            for name in ('memory.max', 'memory.current', 'memory.peak'):
                path = folder/name
                if path.is_file():
                    value = path.read_text().strip(); values[str(path)] = value
                    if value.isdigit():
                        {'memory.max': limits, 'memory.current': current, 'memory.peak': peak}[name].append(int(value))
            if folder == base: break
            folder = folder.parent
    assert limits, 'A finite visible cgroup memory limit is required'
    _sampled_memory_peak = max(_sampled_memory_peak, max(current, default=0))
    return {'memory.max': str(min(limits)), 'memory.current': str(max(current, default=0)),
            'memory.peak': str(max(peak) if peak else _sampled_memory_peak), 'visible_limits': values,
            'peak_measurement': 'kernel' if peak else 'sampled_memory_current_50ms'}


def start_memory_monitor():
    import threading
    stop = threading.Event()
    def monitor():
        while not stop.wait(.05):
            try: cgroup_memory()
            except (OSError, AssertionError): pass  # synchronous stage checks still fail on invalid state
    threading.Thread(target=monitor, daemon=True).start()
    cgroup_memory()
    return stop


class FullFitExperiment(Experiment):
    """Only the original gradient primitive and timing are needed by step 1."""
    @contextlib.contextmanager
    def timed(self, stage, **extra):
        with super().timed(stage, **extra): yield
        self.resources[-1]['cgroup_memory'] = cgroup_memory()
        save_json(self.root/'resource_records.json', self.resources)

    def gradient(self, name, ids, reference, x, labels, weight_check=False):
        target = self.model.get_submodule(name); layer = int(name.split('.')[2]); audit = {}
        assert target.weight.device == x.device
        target.weight.requires_grad_(weight_check)
        try:
            with recompute_suffix(self.model, layer, True), capture(target, not weight_check) as state:
                hidden = hidden_forward(self.model, ids)
                assert torch.equal(state['x'], x)
                hidden_error = mo.relative(hidden.detach(), reference)
                assert hidden_error <= 1e-7
                seed = mo.sampled_seed(hidden.detach(), self.model.lm_head.weight, labels, self.config['vocab_chunk'])
                inputs = (state['h'], target.weight) if weight_check else (state['h'],)
                grads = torch.autograd.grad(hidden, inputs, grad_outputs=seed)
            g = grads[0].detach().reshape(2048, -1).double()
            if weight_check:
                difference = mo.relative(g.T@x.reshape(2048, -1).double(), grads[1].double())
                assert difference <= self.plan['S_autograd_relative_tolerance']
                audit['S_vs_autograd_weight_relative_error'] = difference
            audit['teacher_hidden_relative_error'] = hidden_error
            return g, audit
        finally:
            target.weight.requires_grad_(False)


def check_parent(root, after=False):
    """Use relocated paths, preserve exact source/file identities."""
    import time
    root = Path(root); files = {}; pins = read(HERE/'parent_expected.json')
    migration = ASSETS/'migration_manifest.json'
    assert sha_file(migration) == CONFIG['migration_manifest_sha256']
    migration_files = read(migration)['files']
    for name, digest in pins.items():
        path = EXP03/name
        assert sha_file(path) == digest, name
        files[str(path)] = digest
        if not after: atomic_bytes(root/'parent_snapshot'/name, path.read_bytes())
    identity = read(EXP03/'identity.json')
    assert identity['identity'] == PLAN['exp3_identity']
    old_identity = read(EXP01/'identity.json')
    assert old_identity['identity'] == identity['plan']['parent_identity']
    for record in (identity, old_identity):
        material = {k: v for k, v in record.items() if k != 'identity'}
        assert hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest() == record['identity']
    frozen = read(EXP03/'evaluation_freeze.json')
    assert frozen['identity'] == identity['identity'] and frozen['manifest_sha256'] == sha_file(EXP03/'manifest.json')
    for name, digest in frozen['files'].items():
        assert sha_file(EXP03/name) == digest; files[str(EXP03/name)] = digest
    for sample in read(EXP03/'data/fit_sample_manifest.json')['samples']:
        path = EXP03/sample['relative_path']; assert sha_file(path) == sample['file_sha256']
        files[str(path)] = sample['file_sha256']
    # Per-window x is used for cross-hardware replay; keep its source file hash too.
    for name in PLAN['parent_modules']:
        for c in range(8):
            path = EXP03/'cache'/slug(name)/f'x_w{c:02d}.safetensors'
            files[str(path)] = sha_file(path)
            assert files[str(path)] == migration_files['exp03/'+str(path.relative_to(EXP03)).replace('\\', '/')]['sha256']
    for code, record in ((Path(PLAN['exp3_code']), identity), (Path(PLAN['parent_code']), old_identity)):
        for name, digest in record['source'].items():
            assert sha_file(code/name) == digest, str(code/name)
            files[str(code/name)] = digest
    directions = read(EXP01/'directions/frozen.json')
    for name in PLAN['parent_modules']:
        path = EXP01/'directions'/(slug(name)+'.safetensors')
        assert sha_file(path) == directions['files'][name]
        _, meta = read_tensors(path)
        quant = EXP01/'quantized'/(slug(name)+'.safetensors')
        assert sha_file(quant) == meta['quantized_file_hash']
        files[str(path)] = sha_file(path); files[str(quant)] = meta['quantized_file_hash']
    if after: assert read(root/'parent_integrity_before.json')['files'] == files
    result = dict(passed=True, files=files, parent_identity=identity['identity'], checked=time.time(),
                  relocated=True, source_identities_preserved=True)
    save_json(root/('parent_integrity_after.json' if after else 'parent_integrity_before.json'), result)
    return result
