"""Frozen run identity, checked atomic records and bounded resource accounting."""
from __future__ import annotations
import contextlib
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time
import torch
from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / 'qer_multimodule_three_fp64_v1'))
from bridge import mo, sm, Teacher, hidden_forward, warm_and_probe
from shared_teacher import shared_gradient
sys.path.insert(0, str(HERE.parent / 'qer_v_attention_proto_v1'))
from v_math import effective, normalize, compare
sys.path.insert(0, str(HERE.parent / 'qera_original_a_isolation'))
sys.path.insert(0, str(HERE))

N, L, T, RANK = 256, 2048, 2047, 64
KINDS = ('q', 'k', 'v', 'o', 'gate', 'up', 'down')
STATES = ('Teacher', 'C0', 'C1', 'C2', 'C3', 'C4')
TASKS = {'hellaswag': 'acc_norm', 'piqa': 'acc_norm', 'winogrande': 'acc',
         'arc_easy': 'acc_norm', 'arc_challenge': 'acc_norm'}
GROUP_KEYS = {'qkv': 'q', 'o': 'o', 'gate_up': 'gate', 'down': 'down'}

def require(ok, message):
    if not ok:
        raise RuntimeError(message)

def name(layer, kind):
    return f'model.layers.{layer}.{"self_attn" if kind in "qkvo" else "mlp"}.{kind}_proj'

def a_group(kind):
    return 'qkv' if kind in ('q', 'k', 'v') else ('gate_up' if kind in ('gate', 'up') else kind)

def method(state, kind):
    if state in ('Teacher', 'C0') or (state == 'C1' and kind in ('q', 'k', 'v', 'o')):
        return None
    if kind in ('gate', 'up', 'down') or state == 'C2':
        return 'A-only'
    if state == 'C4' and kind == 'k':
        return 'Token-joint-one'
    if state == 'C4' and kind == 'v':
        return 'Attention-aware'
    return 'Marginal'

def slug(value):
    return value.replace('.', '__')

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, allow_nan=False).encode()).hexdigest()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()

def write(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8', newline='\n') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

def tensors(path, values):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    save_file({k: v.detach().cpu().contiguous() for k, v in values.items()}, str(tmp))
    with tmp.open('r+b') as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)

def log(event, **fields):
    print(time.strftime('%Y-%m-%dT%H:%M:%S%z'), event, json.dumps(fields, ensure_ascii=False), flush=True)

def source_files():
    dirs = [HERE, *[HERE.parent / d for d in (
        'qer_multimodule_three_fp64_v1', 'qer_kronecker_l10_v2', 'qer_teacher_kl_exp01',
        'qer_teacher_kl_exp03', 'qer_functional_gradient_4090_v1', 'qer_v_attention_proto_v1',
        'qera_original_a_isolation')], REPO / 'tools/kronecker_l10_recovery']
    result={str(p.relative_to(REPO)): sha(p) for d in dirs for p in sorted(d.glob('*'))
            if p.is_file() and p.suffix in ('.py', '.yaml', '.json', '.md', '.sh')}
    for module in list(sys.modules.values()):
        filename=getattr(module,'__file__',None)
        if filename:
            p=Path(filename).resolve()
            if REPO in p.parents and p.suffix=='.py':result[str(p.relative_to(REPO))]=sha(p)
    return result

class Context:
    def __init__(self, config_path):
        self.config = read(config_path)
        self.root = Path(self.config['run']); self.base = Path(self.config['base'])
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock(); self.started = time.monotonic()
        parents = {str(p): sha(p) for p in [
            Path(self.config['assets']) / 'exp03/teacher_identity.json',
            Path(self.config['assets']) / 'exp03/environment.json',
            Path(self.config['calibration']).with_suffix('.json'),
            Path(self.config['calibration']), Path(self.config['wikitext2']),
            Path(self.config['wikitext2']).with_suffix('.json'), Path(self.config['validation']),
            Path(self.config['ko_run']) / 'manifest.json', Path(self.config['ko_run']) / 'quantized/freeze.json']}
        material = dict(version=HERE.name, config=self.config, source=source_files(), parents=parents)
        self.identity = digest(material)
        manifest = dict(identity=self.identity, **material)
        path = self.root / 'manifest.json'
        if path.exists():
            require(read(path) == manifest, 'Run/source/data/teacher identity mismatch; refusing resume')
        else:
            write(path, manifest); write(self.root / 'frozen_config.json', self.config)
            shutil.copyfile(HERE / 'protocol.md', self.root / 'protocol.md')
        self.resources_path = self.root / 'resources/timings.json'
        self.resources = read(self.resources_path) if self.resources_path.exists() else {'stages': [], 'sessions': []}
        self.resources['sessions'].append(dict(pid=os.getpid(), started=time.time()))
        self.teacher = Teacher(self.config, self.root, self.identity, self.timed)

    def done(self, path, **expected):
        path = self.root / path
        if not path.exists():
            return False
        row = read(path)
        require(row['identity'] == self.identity and row['passed'], 'Completion identity/status differs')
        require(all(row.get(k) == v for k, v in expected.items()), f'Completion count/binding differs: {path}')
        for filename, value in row['files'].items():
            require(sha(self.root / filename) == value, f'Completed file changed: {filename}')
            if filename.endswith('.json'):
                child=read(self.root/filename)
                if isinstance(child,dict) and child.get('passed') is True and 'files' in child and 'identity' in child:
                    require(filename!=str(path.relative_to(self.root)),'Recursive completion record')
                    self.done(filename)
        return row

    def commit(self, path, files=(), **details):
        row = dict(identity=self.identity, passed=True, time=time.time(),
                   files={str(Path(p).relative_to(self.root)): sha(p) for p in files}, **details)
        write(self.root / path, row)
        return row

    def check(self):
        for limit_path, current_path in [('/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory.current')]:
            if Path(limit_path).exists():
                value = Path(limit_path).read_text().strip()
                if value != 'max':
                    require(int(Path(current_path).read_text()) < int(value) * .9, 'Cgroup memory exceeds 90%')
        require(shutil.disk_usage(self.root).free > 16 * 2**30, 'Filesystem recovery headroom exhausted')

    def resource_snapshot(self):
        import psutil
        row = dict(rss_GiB=psutil.Process().memory_info().rss / 2**30,
                   io=psutil.Process().io_counters()._asdict())
        if torch.cuda.is_initialized():
            row['peak_GPU_GiB'] = [torch.cuda.max_memory_allocated(i) / 2**30 for i in range(2)]
        return row

    @contextlib.contextmanager
    def timed(self, stage, **details):
        self.check(); start = time.monotonic(); passed = False
        log('START', stage=stage, **details)
        try:
            yield
            passed = True
        finally:
            if torch.cuda.is_initialized():
                for d in ([details['device']] if 'device' in details else range(torch.cuda.device_count())):
                    torch.cuda.synchronize(d)
            row = dict(stage=stage, wall_seconds=time.monotonic()-start, passed=passed,
                       resources=self.resource_snapshot(), **details)
            with self.lock:
                self.resources['stages'].append(row)
                self.resources['sessions'][-1]['active_seconds'] = time.monotonic()-self.started
                write(self.resources_path, self.resources)
            log('END', **row)

    def stat(self, layer, family):
        return self.root / 'statistics' / f'L{layer:02d}' / (family + '.safetensors')

    def factor(self, layer, kind, family):
        return self.root / 'factors' / slug(name(layer, kind)) / (family + '.safetensors')

    def cleanup_temporary(self, path):
        path = Path(path).resolve(); boundary = (self.root / 'temporary').resolve()
        require(path != boundary and boundary in path.parents and not path.is_symlink(), 'Unsafe temporary cleanup')
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()

class Accumulator:
    """Current/previous atomic generations. Uncommitted windows are replayed once."""
    def __init__(self, ctx, key, shapes, binding):
        self.ctx, self.binding = ctx, binding
        self.folder = ctx.root / 'temporary' / key
        self.folder.mkdir(parents=True, exist_ok=True)
        self.count = 0; self.values = None
        pointer = self.folder / 'current.json'
        if pointer.exists():
            row = read(pointer)
            require(row['identity'] == ctx.identity and row['binding'] == binding, 'Checkpoint identity differs')
            path = self.folder / row['file']
            require(sha(path) == row['sha256'], 'Checkpoint hash differs')
            # copy mmap-backed inputs before any mutation or removal.
            self.values = {k: v.clone() for k, v in load_file(str(path)).items()}
            self.count = row['count']
        else:
            self.values = {k: torch.zeros(s, dtype=torch.float64) for k, s in shapes.items()}
        require({k: tuple(v.shape) for k, v in self.values.items()} == shapes, 'Checkpoint schema differs')

    def add(self, terms, window):
        require(window == self.count and set(terms) == set(self.values), 'Duplicate/missing statistics window')
        for k, value in terms.items():
            require(value.dtype == torch.float64 and torch.isfinite(value).all(), 'Invalid statistics increment')
            self.values[k].add_(value.cpu())
        self.count += 1

    def save(self):
        path = self.folder / f'w{self.count:04d}.safetensors'
        tensors(path, self.values)
        pointer = self.folder / 'current.json'
        previous = read(pointer) if pointer.exists() else None
        if previous:
            write(self.folder / 'previous.json', previous)
        row = dict(identity=self.ctx.identity, binding=self.binding, count=self.count,
                   file=path.name, sha256=sha(path))
        write(pointer, row)
        keep = {row['file'], previous['file'] if previous else ''}
        for old in self.folder.glob('w*.safetensors'):
            if old.name not in keep:
                self.ctx.cleanup_temporary(old)
        log('CHECKPOINT', group=str(self.folder), windows=self.count)
