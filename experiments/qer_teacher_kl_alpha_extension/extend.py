#!/usr/bin/env python3
"""Forward-only continuation, using verified frozen parent implementation/assets."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('HF_DATASETS_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
sys.dont_write_bytecode = True


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''): h.update(block)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    spec = read(args.config)
    parent, code = Path(spec['parent_output']), Path(spec['parent_code'])
    here = Path(__file__).resolve().parent
    expected = read(here/'parent_expected.json')
    parent_identity = read(parent/'identity.json')
    material = {k: v for k, v in parent_identity.items() if k != 'identity'}
    assert hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest() == spec['parent_identity']
    assert parent_identity['identity'] == spec['parent_identity']
    for name, sha in parent_identity['source'].items():
        assert digest(code/name) == sha, 'Parent source changed: '+name
    for name, sha in expected['files'].items():
        assert digest(parent/name) == sha, 'Parent artifact changed: '+name
    sys.path.insert(0, str(code))
    import torch
    import math_ops as mo
    from model_ops import capture, hidden_forward
    from storage import atomic_bytes, save_json, read_json, read_tensors, sha_file
    from run import Experiment, MODULES, DIRECTIONS, OFFICIAL, slug, clean, sync, Paused, request_pause
    # Parent directory contains analyze.py, while extension uses analysis.py.
    from analysis import NEW, write_report
    assert spec['new_alpha'] == list(NEW) and spec['gpus'] == parent_identity['config']['gpus'] == 1
    assert spec['pilot_relative_tolerance'] == .01

    class Extension(Experiment):
        def __init__(self):
            self.root = Path(args.output).resolve()
            if self.root == parent.resolve() or parent.resolve() in self.root.parents:
                raise RuntimeError('Use a separate extension directory')
            self.root.mkdir(parents=True, exist_ok=True)
            self.config = parent_identity['config']
            self.model = None; self.cal = None; self.val = None
            self.resources = read_json(self.root/'resource_records.json') if (self.root/'resource_records.json').exists() else []
            source = {p.name: digest(p) for p in sorted(here.iterdir()) if p.suffix in {'.py', '.sh', '.json', '.md'}}
            identity_data = {'extension_config': spec, 'source': source, 'parent_identity': spec['parent_identity']}
            self.identity = hashlib.sha256(json.dumps(identity_data, sort_keys=True).encode()).hexdigest()
            identity_path = self.root/'identity.json'
            if identity_path.exists():
                assert read(identity_path)['identity'] == self.identity, 'Extension identity mismatch'
            else:
                save_json(identity_path, {'identity': self.identity, **identity_data})
                save_json(self.root/'config.json', spec)
                atomic_bytes(self.root/'protocol.md', (here/'protocol.md').read_bytes())
            self.direction_cache = {}
            torch.set_num_threads(self.config['cpu_threads'])
            torch.set_float32_matmul_precision('highest')
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.manual_seed(20260918)

        def verify_assets(self):
            self.status('VERIFYING_PARENT_ASSETS')
            # Small evidence snapshots make analysis reproducible without parent writes.
            for name, sha in expected['files'].items():
                target = self.root/'parent_snapshot'/name
                if target.exists(): assert digest(target) == sha
                else: atomic_bytes(target, (parent/name).read_bytes())
            self.val, data_meta = read_tensors(parent/'data/validation.safetensors')
            assert data_meta['identity'] == spec['parent_identity']
            assert data_meta['source_split'] == 'validation'
            assert self.val['input_ids'].shape == (8, 2048) and self.val['attention_mask'].all()
            assert [mo.digest_tensor(t) for t in self.val['input_ids']] == data_meta['window_tensor_hashes']
            frozen = read(parent/'directions/frozen.json')
            assets = {'data_hash': digest(parent/'data/validation.safetensors'), 'directions': {}}
            for name in MODULES:
                path = parent/'directions'/(slug(name)+'.safetensors')
                assert digest(path) == frozen['files'][name]
                directions, meta = read_tensors(path)
                assert meta['identity'] == spec['parent_identity'] and set(directions) == set(DIRECTIONS)
                for d, r in directions.items():
                    assert r.dtype == torch.float32 and mo.digest_tensor(r) == meta['direction_hashes'][d]
                    mo.finite(r)
                quant_path = parent/'quantized'/(slug(name)+'.safetensors')
                assert digest(quant_path) == meta['quantized_file_hash']
                quant, qm = read_tensors(quant_path)
                assert qm['identity'] == spec['parent_identity']
                assert (qm['width'], qm['block_size'], qm['block_axis']) == (3, 32, -1)
                assert qm['quantizer_hash'] == OFFICIAL['quantize/quantizers/mxint.py']
                for key in ('W0', 'Wq'): assert mo.digest_tensor(quant[key]) == qm[key+'_hash']
                assert torch.equal((quant['W0'].double()-quant['Wq'].double()).float(), directions['R_none'])
                assert qm['W0_hash'] == read(parent/'teacher_identity.json')['tensor_hashes'][name+'.weight']['hash']
                self.direction_cache[name] = directions
                assets['directions'][name] = {'file_sha256': frozen['files'][name], 'metadata': meta, 'quantization': qm}
                del quant
            # CSV and JSON must describe the identical fixed small-alpha baseline.
            with (parent/'summary.csv').open(newline='') as f: csv_rows = list(csv.DictReader(f))
            summaries = read(parent/'summary.json')
            for r in csv_rows:
                match = next(x for x in summaries if (x['module'], x['direction']) == (r['module'], r['direction']))
                assert float(r['q_KL']) == match['q_KL']
            self.assets = assets
            save_json(self.root/'asset_verification.json', {'identity': self.identity, 'parent_identity': spec['parent_identity'], **assets})

        def verify_teacher(self):
            original, actual = read(parent/'teacher_identity.json'), read(self.root/'teacher_identity.json')
            assert original['tensor_hashes'] == actual['tensor_hashes'], 'Teacher tensors changed'
            assert original['device_map'] == actual['device_map'], 'Device mapping changed'
            assert original['checkpoint_manifest_hash'] == actual['checkpoint_manifest_hash']
            previous = read(parent/'environment.json')
            current = read(self.root/'environment.json')
            assert previous['torch'] == current['torch'] and previous['packages'] == current['packages']
            assert all(not p.requires_grad for p in self.model.parameters())
            save_json(self.root/'teacher_verification.json', {'identity': self.identity, 'all_tensor_hashes_match': True,
                      'device_map_match': True, 'environment_versions_match': True, 'parent_identity': spec['parent_identity']})

        def kl_point(self, name, direction, window, alpha, reference, x, h, residual, self_floor, pilot=False):
            path = self.root/('pilot' if pilot else 'records/kl')/f'{slug(name)}_{direction}_w{window:02d}_a{alpha:g}.json'
            if path.exists():
                row = self.read_record(path)
                assert row['R_hash'] == mo.digest_tensor(residual) and row['T'] == 2047
                assert (row['module'], row['direction'], row['window'], row['alpha']) == (name, direction, window, alpha)
                return row
            with self.timed('pilot_KL' if pilot else 'KL', module=name, direction=direction, window=window, alpha=alpha):
                start = time.perf_counter()
                module = self.model.get_submodule(name)
                with mo.intervention(module, residual, alpha) as weight_audit:
                    with torch.no_grad(), capture(module) as actual:
                        candidate = hidden_forward(self.model, self.val['input_ids'][window:window+1])
                    same = torch.equal(x, actual['x'])
                    wanted = -alpha*(x.double()@residual.to(x.device).double().T)
                    output_audit = mo.comparison(actual['h'].detach().double()-h.double(), wanted)
                    kl = self.kl_hidden(reference, candidate)
                reasons = []
                if not same: reasons.append('module_input_changed')
                if alpha == 0:
                    if kl['KL_mean'] > 1e-10: reasons.append('self_KL_floor')
                else:
                    if weight_audit['relative_l2'] > .001 or weight_audit['cosine'] < .999999: reasons.append('weight_path')
                    if output_audit['relative_l2'] > .02 or output_audit['cosine'] < .999: reasons.append('output_path')
                    if kl['KL_mean'] <= 100*max(self_floor, 1e-12): reasons.append('below_numerical_floor')
                sync()
                row = {'identity': self.identity, 'parent_identity': spec['parent_identity'], 'module': name,
                    'direction': direction, 'window': window, 'alpha': alpha, 'T': 2047, **kl,
                    'self_KL': self_floor, 'weight_audit': weight_audit, 'output_audit': output_audit,
                    'input_identical': same, 'R_hash': mo.digest_tensor(residual), 'restoration_verified': True,
                    'valid': not reasons, 'invalid_reasons': reasons, 'elapsed_seconds': time.perf_counter()-start}
                # allow_nan=False enforces finite serialized diagnostics as well as KL.
                save_json(path, row)
            return row

        def pilots(self):
            self.status('REPRODUCING_PILOTS')
            old = read(parent/'pilot/complete.json')
            audits = []
            for name in MODULES:
                with self.timed('pilot_teacher', module=name): reference, x, h = self.teacher_reference(name, 0)
                residual = self.direction_cache[name]['R_A64']
                zero = self.kl_point(name, 'R_A64', 0, 0., reference, x, h, residual, 0., True)
                point = self.kl_point(name, 'R_A64', 0, .1, reference, x, h, residual, zero['KL_mean'], True)
                previous = next(r['alpha_0.1'] for r in old['modules'] if r['module'] == name)
                difference = abs(point['KL_mean']-previous['KL_mean'])/previous['KL_mean']
                audit = {'module': name, 'zero': zero, 'alpha_0.1': point,
                         'parent_KL_mean': previous['KL_mean'], 'relative_reproduction_error': difference}
                save_json(self.root/'pilot'/(slug(name)+'_reproduction.json'), audit)
                if not zero['valid'] or not point['valid'] or difference > .01:
                    raise RuntimeError('Pilot reproduction failed: '+name)
                audits.append(audit)
                del reference, x, h; clean()
            save_json(self.root/'pilot/complete.json', {'identity': self.identity, 'modules': audits, 'tolerance': .01})

        def measure_extension(self):
            for name in MODULES:
                for window in range(8):
                    needed = [(d, a) for d in DIRECTIONS for a in NEW if not
                              (self.root/'records/kl'/f'{slug(name)}_{d}_w{window:02d}_a{a:g}.json').exists()]
                    if not needed: continue
                    with self.timed('KL_teacher', module=name, window=window): reference, x, h = self.teacher_reference(name, window)
                    control = self.kl_point(name, 'control', window, 0., reference, x, h,
                                            self.direction_cache[name]['R_none'], 0.)
                    if not control['valid']: raise RuntimeError('Invalid self-KL control')
                    for d, a in needed:
                        self.kl_point(name, d, window, a, reference, x, h, self.direction_cache[name][d], control['KL_mean'])
                        self.status('MEASURING_EXTENSION', module=name, window=window, direction=d, alpha=a,
                                    committed_records=len(list((self.root/'records/kl').glob('*.json'))))
                    del reference, x, h; clean()

        def run_extension(self):
            self.doctor()
            self.verify_assets()
            self.load_model()
            self.verify_teacher()
            self.pilots()
            self.measure_extension()
            self.unload()
            # Check that read-only parent assets and frozen source are unchanged.
            after = {name: digest(parent/name) for name in expected['files']}
            assert after == expected['files'], 'Parent metadata/input changed'
            for name, sha in parent_identity['source'].items(): assert digest(code/name) == sha
            for name in MODULES:
                assert digest(parent/'directions'/(slug(name)+'.safetensors')) == self.assets['directions'][name]['file_sha256']
                assert digest(parent/'quantized'/(slug(name)+'.safetensors')) == self.assets['directions'][name]['metadata']['quantized_file_hash']
            save_json(self.root/'parent_integrity.json', {'identity': self.identity, 'parent_identity': spec['parent_identity'],
                      'before_equals_after': True, 'verified_files': after, 'frozen_code_unchanged': True,
                      'frozen_direction_and_quantized_files_unchanged': True})
            write_report(self)

    import fcntl
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=True)
    with (output/'.run.lock').open('w') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX|fcntl.LOCK_NB)
        signal.signal(signal.SIGTERM, request_pause); signal.signal(signal.SIGINT, request_pause)
        experiment = Extension()
        try: experiment.run_extension()
        except Paused as error:
            experiment.status('PAUSED', reason=str(error)); return 75
        except BaseException as error:
            experiment.status('FAILED', error=repr(error), traceback=traceback.format_exc()); raise
    return 0


if __name__ == '__main__': sys.exit(main())
