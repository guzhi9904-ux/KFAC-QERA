"""CPU coverage for portable imports and exact host-staged Rx contractions."""
import contextlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import torch


class PortableTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='qer-portable-test-')
        self.root = Path(self.temp.name).resolve()
        self.assertEqual(self.root.parent, Path(tempfile.gettempdir()).resolve())
        self.addCleanup(self.temp.cleanup)

    def test_runtime_import_and_source_identity(self):
        config = self.root/'config.json'
        config.write_text(json.dumps({'schema': 1, 'assets': str(self.root/'assets'),
                                     'model': str(self.root/'model'), 'output_parent': str(self.root/'runs')}))
        code = """
import os,sys,types
if os.name=='nt':sys.modules['resource']=types.SimpleNamespace(RUSAGE_SELF=0)
import bridge,runtime,portable_model
r=bridge.source_identity()
assert r['portable_config']['schema']==1
assert len(r['identity'])==64 and len(r['borrowed_source'])>=20
assert runtime.Runtime.__mro__[1] is portable_model.PortableModel
print('Portable runtime import and source identity PASS; no GPU operation')
"""
        result = subprocess.run([sys.executable, '-B', '-c', code], cwd=Path(__file__).parent,
            env=dict(os.environ, QER_PORTABLE_CONFIG=str(config)), capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)

    def test_host_staged_projection_and_resume_identity(self):
        from functional_analysis import CANDIDATES
        def read(p): return json.loads(Path(p).read_text())
        def save(path, value):
            path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value))
        fake = types.ModuleType('bridge')
        for name in ('read_tensors', 'save_csv', 'sha_file', 'capture', 'hidden_forward'): setattr(fake, name, lambda *a: None)
        fake.read = read; fake.save_json = save; fake.torch = torch; fake.mo = None; fake.clean = lambda: None
        with patch.dict(sys.modules, {'bridge': fake}):
            spec = importlib.util.spec_from_file_location('portable_eval_test', Path(__file__).with_name('evaluation.py'))
            module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        rng = torch.Generator().manual_seed(4090)
        x = torch.randn(1, 2048, 4, generator=rng, dtype=torch.float64)
        g = torch.randn(2048, 3, generator=rng, dtype=torch.float64)
        candidates = {n: {'R': torch.randn(3, 4, generator=rng, dtype=torch.float64), 'R_hash': n} for n in CANDIDATES}
        rx = {n: (x[0]@v['R'].T).cpu() for n,v in candidates.items()}
        class Fixture(module.Evaluation):
            root = self.root; identity = 'fixture'; name = 'module'
            fit = {'input_ids': torch.zeros(8, 2048, dtype=torch.long)}
            def boundary(self): pass
            def timed(self, *a, **kw): return contextlib.nullcontext()
            def gradient(self, *a, **kw): return g, {'passed': True}
            def status(self, *a, **kw): pass
        e = Fixture(); sample = (torch.zeros(2047), {'label_hash': 'label', 'input_hash': 'input'})
        with patch.object(torch.Tensor, 'cuda', lambda self: self):
            record = e.new_unit(0, 0, None, x, rx, candidates, sample, True)
        self.assertEqual(len(record['scores']), 17)
        for n in CANDIDATES:
            expected = float(((g.T@x[0])*candidates[n]['R']).sum())
            self.assertAlmostEqual(record['scores'][n]['d'], expected, places=9)
        candidates['None']['R_hash'] = 'changed'
        with self.assertRaises(AssertionError): e.new_unit(0, 0, None, x, rx, candidates, sample, True)


if __name__ == '__main__': unittest.main(verbosity=2)
