#!/usr/bin/env python3
"""Bounded, stdlib-only inventory. Does not import torch or open tensor payloads."""
import argparse
import datetime
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import struct
import subprocess
import sys

GIB = 2**30
CONFIG_KEYS = ('model_type', 'architectures', 'hidden_size', 'intermediate_size',
               'num_hidden_layers', 'num_attention_heads', 'num_key_value_heads',
               'vocab_size', 'tie_word_embeddings')
DTYPE_BYTES = {'F64': 8, 'F32': 4, 'F16': 2, 'BF16': 2, 'I64': 8,
               'I32': 4, 'I16': 2, 'I8': 1, 'U8': 1, 'BOOL': 1}


def read_json(path):
    if path.stat().st_size > 8 * 2**20:
        raise ValueError('Metadata exceeds 8 MiB: ' + str(path))
    return json.loads(path.read_text(encoding='utf-8'))


def command(args):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=15)
        return {'returncode': p.returncode, 'stdout': p.stdout[:20000], 'stderr': p.stderr[:2000]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'error': str(exc)}


def scan(root, wanted, max_depth=4, max_entries=6000):
    """Do not follow directory symlinks or recurse into tensor/cache contents."""
    result = []; stack = [(root, 0)]; count = 0; errors = []
    while stack and count < max_entries:
        folder, depth = stack.pop()
        try:
            with os.scandir(folder) as entries:
                for entry in entries:
                    count += 1
                    if count > max_entries: break
                    if entry.is_file() and entry.name == wanted:
                        result.append(Path(entry.path))
                    if entry.is_dir(follow_symlinks=False) and depth < max_depth:
                        if entry.name not in ('.git', '__pycache__', '.ipynb_checkpoints', 'blobs'):
                            stack.append((Path(entry.path), depth + 1))
        except OSError as exc:
            errors.append(str(exc))
    return sorted(result), {'visited_entries': count, 'truncated': bool(stack) or count > max_entries,
                            'depth_limit': max_depth, 'errors': errors}


def tensor_header(path):
    size = path.stat().st_size
    with path.open('rb') as stream:
        n = stream.read(8)
        if len(n) != 8: raise ValueError('Truncated safetensors prefix')
        length = struct.unpack('<Q', n)[0]
        if length > 8 * 2**20 or length + 8 > size:
            raise ValueError('Invalid safetensors header size')
        header = json.loads(stream.read(length))
    tensors = {}
    for name, value in header.items():
        if name == '__metadata__': continue
        shape = value['shape']; start, end = value['data_offsets']; dtype = value['dtype']
        if any(not isinstance(v, int) or v < 0 for v in shape): raise ValueError('Invalid shape')
        if not 0 <= start <= end <= size - 8 - length: raise ValueError('Invalid offsets')
        if dtype in DTYPE_BYTES and end - start != math.prod(shape) * DTYPE_BYTES[dtype]:
            raise ValueError('Tensor byte length mismatch')
        tensors[name] = {'shape': shape, 'dtype': dtype}
    return tensors


def model_info(config_path):
    folder = config_path.parent; config = read_json(config_path)
    result = {'path': str(folder), 'config': {k: config.get(k) for k in CONFIG_KEYS},
              'weights_content_verified': False, 'errors': []}
    index = folder / 'model.safetensors.index.json'; mapping = None
    if index.exists():
        mapping = read_json(index)['weight_map']; filenames = sorted(set(mapping.values()))
    elif (folder / 'model.safetensors').exists(): filenames = ['model.safetensors']
    else:
        result['errors'].append('No standard safetensors checkpoint; tensor metadata not checked')
        return result
    tensors = {}; total = 0
    for filename in filenames:
        if Path(filename).name != filename or not filename.endswith('.safetensors'):
            raise ValueError('Unsafe checkpoint index filename')
        path = folder / filename
        part = tensor_header(path); total += path.stat().st_size
        if tensors.keys() & part.keys(): raise ValueError('Duplicate tensor name')
        if mapping is not None:
            if set(part) != {k for k, v in mapping.items() if v == filename}:
                raise ValueError('Index/header disagreement')
        tensors.update(part)
    result.update(weight_files=len(filenames), checkpoint_GiB=total/GIB,
                  tensor_count=len(tensors), source_dtypes=sorted(set(v['dtype'] for v in tensors.values())),
                  fp32_parameter_GiB=sum(math.prod(v['shape']) * 4 for v in tensors.values())/GIB)
    manifest = folder / 'DOWNLOAD_MANIFEST.json'
    result['download_manifest_present'] = manifest.is_file()
    return result


def memory_info():
    result = {'host': {}, 'cgroup': [], 'effective_limit_GiB': None}
    meminfo = Path('/proc/meminfo')
    if meminfo.exists():
        for line in meminfo.read_text().splitlines():
            key, value = line.split(':', 1)
            if key in ('MemTotal', 'MemAvailable', 'SwapTotal', 'SwapFree'):
                result['host'][key + '_GiB'] = int(value.strip().split()[0]) * 1024 / GIB
    limits = [result['host']['MemTotal_GiB']] if 'MemTotal_GiB' in result['host'] else []
    cgroup = Path('/proc/self/cgroup')
    if cgroup.exists():
        text = cgroup.read_text(); result['membership'] = text
        # Check both namespace root and process path, and all visible ancestors.
        roots = [Path('/sys/fs/cgroup')]
        for line in text.splitlines():
            _, controllers, rel = line.split(':', 2)
            if controllers == '': roots.append(Path('/sys/fs/cgroup') / rel.lstrip('/'))
            elif 'memory' in controllers.split(','):
                roots.extend([Path('/sys/fs/cgroup/memory'), Path('/sys/fs/cgroup/memory')/rel.lstrip('/')])
        seen = set()
        for root in roots:
            while str(root).startswith('/sys/fs/cgroup'):
                if root not in seen:
                    seen.add(root)
                    for name in ('memory.max', 'memory.current', 'memory.peak', 'memory.limit_in_bytes',
                                 'memory.usage_in_bytes', 'cpu.max', 'cpuset.cpus.effective'):
                        p = root / name
                        if not p.is_file(): continue
                        value = p.read_text().strip()
                        result['cgroup'].append({'file': str(p), 'value': value})
                        if name in ('memory.max', 'memory.limit_in_bytes') and value.isdigit():
                            limits.append(int(value) / GIB)
                if root == Path('/sys/fs/cgroup'): break
                root = root.parent
    if limits: result['effective_limit_GiB'] = min(limits)
    result['note'] = 'Visible cgroup limits only; shared filesystem quota is not inferred.'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=Path.cwd().parent)
    parser.add_argument('--model', type=Path, help='Explicit checkpoint directory if outside modelzoo')
    parser.add_argument('--assets', type=Path, help='Migrated parent asset directory, if already present')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); base = args.base.resolve()
    versions = {}
    for name in ('torch', 'transformers', 'datasets', 'numpy', 'safetensors', 'accelerate'):
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = None
    disk = shutil.disk_usage(base)
    result = {'schema': 1, 'time_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'python': sys.executable, 'python_version': platform.python_version(), 'base': str(base),
              'packages': versions,
              'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'gpu': command(['nvidia-smi', '--query-gpu=index,name,uuid,memory.total,memory.used,memory.free', '--format=csv']),
              'gpu_processes': command(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv']),
              'memory': memory_info(), 'disk': {'free_GiB': disk.free/GIB, 'total_GiB': disk.total/GIB,
                    'per_user_quota_checked': False}, 'models': [], 'assets': [],
              'compute_started': False, 'torch_imported': False, 'ready_for_experiment': False}
    configs, scanned = scan(base/'modelzoo', 'config.json', max_depth=5)
    result['model_scan'] = scanned
    if args.model: configs = sorted(set(configs + [args.model.resolve()/'config.json']))
    for config in configs:
        try: result['models'].append(model_info(config))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            result['models'].append({'path': str(config.parent), 'error': str(exc), 'weights_content_verified': False})
    for root in [base/'qera_runs', base/'teacher_kl_assets'] + ([args.assets.resolve()] if args.assets else []):
        identities, scanned = scan(root, 'identity.json', max_depth=4)
        records = []
        for path in identities:
            try: records.append({'path': str(path), 'identity': read_json(path).get('identity')})
            except (OSError, ValueError) as exc: records.append({'path': str(path), 'error': str(exc)})
        result['assets'].append({'root': str(root), 'scan': scanned, 'identities': records})
    result['parent_identity_verified'] = False
    result['gates_remaining'] = ['Complete parent file checksum validation', 'Full FP32 teacher tensor identity',
        'Versioned two-GPU implementation and numerical replay', 'Full-shape GPU/memory pilot']
    output = args.output.resolve()
    # A report is the only file written. Refuse accidental replacement of an earlier report.
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2); stream.write('\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print('\nReport saved to:', output)


if __name__ == '__main__':
    main()
