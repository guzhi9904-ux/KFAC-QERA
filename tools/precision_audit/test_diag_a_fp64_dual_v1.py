import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
import diag_a_fp64_dual_v1 as new


class DiagonalPrecisionTests(unittest.TestCase):
    def tensors(self):
        torch.manual_seed(2026)
        e = torch.randn(12, 10)
        a = torch.rand(12) + .3
        x = torch.randn(10, 10)
        g = x @ x.T + torch.eye(10)
        return e, a, g

    def test_dense_reference_and_rank_tails(self):
        e, a, g = self.tensors()
        left, right, rows, inverse = new.solve_diagonal(e, a, g, (2, 4, 8))
        u, s, vh = torch.linalg.svd(torch.diag(a.double()) @ e.double() @ g.double(), full_matrices=True)
        expected = torch.linalg.solve(torch.diag(a.double()), u[:, :8]) @ torch.linalg.solve(g.double().T, (s[:8, None]*vh[:8]).T).T
        torch.testing.assert_close(left @ right, expected, rtol=1e-10, atol=1e-10)
        self.assertEqual([r['rank'] for r in rows], [2, 4, 8])
        self.assertLess(max(inverse.values()), 1e-9)
        for row in rows:
            self.assertLess(abs(row['relative_tail_difference']), 1e-9)

    def test_identity_and_diagonal_g(self):
        e, a, _ = self.tensors()
        for g in (torch.eye(10), torch.diag(torch.linspace(.2, 2., 10))):
            _, _, rows, _ = new.solve_diagonal(e, a, g, (2, 8))
            self.assertLess(rows[-1]['sse_after_fp64'], rows[0]['sse_after_fp64'])

    def test_nonpositive_root_stops_instead_of_changing_floor(self):
        e, a, g = self.tensors()
        for value in (0., -1., float('nan')):
            aa = a.clone(); aa[0] = value
            with self.assertRaisesRegex(RuntimeError, 'Nonpositive'):
                new.solve_diagonal(e, aa, g, (2,))

    def test_small_positive_root_not_clamped(self):
        e, a, g = self.tensors(); a[0] = 1e-9
        left, right, _, _ = new.solve_diagonal(e, a, g, (2,))
        u, s, vh = torch.linalg.svd((a.double()[:, None]*e.double()) @ g.double(), full_matrices=True)
        torch.testing.assert_close((a.double()[:, None]*left) @ right @ g.double(), u[:, :2] @ (s[:2, None]*vh[:2]), rtol=1e-10, atol=1e-10)

    def test_bad_rank_rejected(self):
        with self.assertRaises(ValueError):
            new.solve_diagonal(*self.tensors(), ranks=(64,))


class RoutingTests(unittest.TestCase):
    def ctx(self):
        return SimpleNamespace(tasks={'layer': {'quant': 'wq', 'baseline_factors': {'diag_gi': 'old_da', 'full_gd': 'old_fa'}}})

    def test_new_and_old_routes_restore(self):
        old = new.dual.artifact_routes
        with patch.object(new, 'read_checked', return_value={'file': 'new_da'}):
            with new.evaluation_binding():
                self.assertEqual(new.dual.artifact_routes(self.ctx(), 'diag_gi', 8)['layer']['correction'], 'new_da')
                self.assertEqual(new.dual.artifact_routes(self.ctx(), 'full_gd', 8, old=True)['layer']['correction'], 'old_fa')
                with new.evaluation_binding(old_token=True):
                    route = new.dual.artifact_routes(self.ctx(), 'old_diag_gi', 8)
                    self.assertEqual(route['layer']['correction'], 'old_da')
                self.assertEqual(new.dual.METHODS, new.METHODS)
        self.assertIs(new.dual.artifact_routes, old)

    def test_install_old_token_flag_and_exception_restore(self):
        with patch.object(new.dual, 'install', return_value='ok') as install:
            with self.assertRaises(ValueError):
                with new.evaluation_binding():
                    with new.evaluation_binding(old_token=True):
                        new.dual.install(None, {}, 8, None)
                        raise ValueError('stop')
            install.assert_called_once_with(None, {}, 8, None, old=True)
            self.assertIs(new.dual.install, install)

    def test_no_unbound_old_token_route(self):
        with new.evaluation_binding():
            with self.assertRaises(RuntimeError):
                new.dual.artifact_routes(self.ctx(), 'old_diag_gi', 8)


class CheckpointTests(unittest.TestCase):
    def test_prepare_one_commits_and_resumes_without_new_solve(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            torch.manual_seed(17)
            weight = torch.randn(70, 80).bfloat16()
            quant = (weight.float()*.98).bfloat16()
            b = {'shape': [70, 80],
                 'model': new.single.atomic_tensors(root/'model.safetensors', {'layer.weight': weight}),
                 'quant': new.single.atomic_tensors(root/'q.safetensors', {'weight_q': quant}),
                 'root': new.single.atomic_tensors(root/'root.safetensors', {'diag': torch.rand(80)+.5})}
            ctx = SimpleNamespace(output=root/'new', identity='id', inputs=new.core.Inputs(),
                                  checked={}, tasks={'layer': b}, dg={'layer': None}, device='cpu',
                                  g_references={'diag_gi': {'layer': {'bits': new.single.tensor_record(torch.eye(70))}}})
            with patch.object(new.parent, 'make_g_root', return_value=(torch.eye(70)*2, {})):
                with self.assertRaisesRegex(RuntimeError, 'root bits differ'):
                    new.prepare_one(ctx, 'layer', 'diag_gi')
            self.assertFalse(new.record_path(ctx, 'diag_gi', 'layer').exists())
            with patch.object(new.parent, 'make_g_root', return_value=(torch.eye(70), {})):
                new.prepare_one(ctx, 'layer', 'diag_gi')
            r = new.read_checked(ctx, 'diag_gi', 'layer')
            self.assertEqual(r['solve_rank'], 64)
            self.assertEqual([x['rank'] for x in r['metrics']], list(new.RANKS))
            # A new process must validate both saved tensor files before adoption.
            ctx.checked.clear(); ctx.inputs = new.core.Inputs()
            with patch.object(new, 'solve_diagonal', side_effect=AssertionError('unexpected recompute')):
                new.prepare_one(ctx, 'layer', 'diag_gi')
            self.assertEqual(new.core.sha(b['root']['path']), b['root']['sha256'])
            self.assertEqual(new.core.sha(b['model']['path']), b['model']['sha256'])

    def test_rank64_cast_resume_and_tampering(self):
        with tempfile.TemporaryDirectory() as folder:
            ctx = SimpleNamespace(output=Path(folder), identity='id', inputs=new.core.Inputs(), checked={}, tasks={'layer': {'shape': [70, 80]}})
            path = new.record_path(ctx, 'diag_gi', 'layer'); path.parent.mkdir(parents=True)
            values = {'A': torch.randn(80, 64).double(), 'B': torch.randn(64, 70).double()}
            rounded = {k: v.bfloat16() for k, v in values.items()}
            r = {'experiment_identity': 'id', 'status': 'PASS', 'method': 'diag_gi', 'module': 'layer',
                 'binding_sha256': new.core.fingerprint(ctx.tasks['layer']), 'metrics': [{'rank': r} for r in new.RANKS],
                 'bits': {k: new.single.tensor_record(v) for k, v in rounded.items()},
                 'file': new.single.atomic_tensors(path.with_suffix('.bf16.safetensors'), rounded),
                 'fp64_file': new.single.atomic_tensors(path.with_suffix('.fp64.safetensors'), values)}
            new.core.atomic_json(path, r)
            self.assertIsNotNone(new.read_checked(ctx, 'diag_gi', 'layer'))
            for rank in new.RANKS:
                self.assertTrue(torch.equal(rounded['A'][:, :rank], values['A'][:, :rank].bfloat16()))
                self.assertTrue(torch.equal(rounded['B'][:rank], values['B'][:rank].bfloat16()))
            ctx.checked.clear(); ctx.inputs = new.core.Inputs()
            changed = copy.deepcopy(r); changed['experiment_identity'] = 'other'
            new.core.atomic_json(path, changed)
            with self.assertRaises(RuntimeError):
                new.read_checked(ctx, 'diag_gi', 'layer')
            new.core.atomic_json(path, r)
            rounded['A'][0, 0] += 1
            r['file'] = new.single.atomic_tensors(path.with_suffix('.bf16.safetensors'), rounded)
            new.core.atomic_json(path, r)
            with self.assertRaises(RuntimeError):
                new.read_checked(ctx, 'diag_gi', 'layer')


if __name__ == '__main__':
    unittest.main()
