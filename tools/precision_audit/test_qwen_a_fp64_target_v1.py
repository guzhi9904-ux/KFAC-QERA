import copy
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
import qwen_a_fp64_target_v1 as new
import full_g_precision_r8_v1 as single
import qwen_full_a_audit_v1 as h


class MathTests(unittest.TestCase):
    def inputs(self, rows=12, cols=8):
        torch.manual_seed(91)
        x = torch.randn(rows, rows)
        return x @ x.T + torch.eye(rows), torch.randn(rows, cols)

    def test_full_fp64_and_prefix_reference(self):
        root, error = self.inputs()
        root_before, error_before = root.clone(), error.clone()
        with patch.object(torch.linalg, 'svd', wraps=torch.linalg.svd) as svd:
            with patch.object(torch.linalg, 'solve', wraps=torch.linalg.solve) as solve:
                values, rows, info = new.solve(root, error, (2, 4, 8))
        self.assertEqual(svd.call_args.args[0].dtype, torch.float64)
        self.assertIs(svd.call_args.kwargs['full_matrices'], True)
        self.assertEqual(solve.call_count, 2)
        for call in solve.call_args_list:
            self.assertTrue(all(t.dtype == torch.float64 for t in call.args))
        u, s, vh = torch.linalg.svd(root.double() @ error.double(), full_matrices=True)
        for row in rows:
            r = row['rank']
            expected = torch.linalg.solve(root.double(), u[:, :r]) @ (s[:r, None]*vh[:r])
            actual = values['A_fp64'][:, :r] @ values['B_fp64'][:r]
            torch.testing.assert_close(actual, expected)
            self.assertLess(abs(row['relative_tail_difference']), 1e-9)
        self.assertTrue(info['numerical_gate_passed'])
        self.assertTrue(torch.equal(root, root_before))
        self.assertTrue(torch.equal(error, error_before))

    def test_bad_dtype_and_rank_rejected(self):
        root, error = self.inputs()
        with self.assertRaises(RuntimeError):
            new.solve(root.double(), error, (2,))
        for ranks in ((8, 2), (), (16,), (2, 2)):
            with self.assertRaises(ValueError):
                new.solve(root, error, ranks)

    def test_singular_root_no_fallback(self):
        with self.assertRaises(torch.linalg.LinAlgError):
            new.solve(torch.zeros(4, 4), torch.ones(4, 3), (2,))

    def test_failed_inverse_gate_retained_not_relaxed(self):
        root, error = self.inputs()
        original = torch.linalg.solve
        def perturbed(a, b):
            result = original(a, b)
            return result + .01 if a.shape[0] == root.shape[0] else result
        with patch.object(torch.linalg, 'solve', side_effect=perturbed):
            _, rows, checks = new.solve(root, error, (2, 4))
        self.assertFalse(checks['numerical_gate_passed'])
        self.assertEqual(new.outcome_code({'checks': checks}), 2)
        self.assertEqual(len(rows), 2)
        self.assertEqual(checks['inverse_tolerance'], 1e-9)

    def test_bf16_proxy_is_direct_factor_rounding(self):
        root, error = self.inputs()
        values, rows, _ = new.solve(root, error, (2, 4))
        for row in rows:
            r = row['rank']
            product = values['A_bf16'][:, :r].double() @ values['B_bf16'][:r].double()
            expected = new.norm2(root.double() @ (error.double()-product))
            self.assertAlmostEqual(expected, row['bf16_rounded_sse_fp64_proxy'])

    def test_bf16_overflow_not_reported_success(self):
        _, rows, checks = new.solve(torch.eye(3)*1e-39, torch.randn(3, 2), (1,))
        self.assertFalse(checks['bf16_factors_finite'])
        self.assertIsNone(rows[0]['bf16_rounded_sse_fp64_proxy'])
        self.assertEqual(new.outcome_code({'checks': checks}), 2)


class StorageTests(unittest.TestCase):
    def test_commit_and_read_resume_tamper_rejected(self):
        torch.manual_seed(1)
        values, rows, checks = new.solve(torch.eye(72), torch.randn(72, 68))
        new.validate_candidate(values, (68, 72))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            candidate = single.atomic_tensors(output/'candidate_factors.safetensors', values)
            single.previous.write_csv(output/'rank_metrics.csv', rows)
            report = {'experiment_identity': 'id', 'status': 'DIAGNOSTIC_COMPLETE', 'rank_metrics': rows,
                      'candidate_file': candidate, 'metrics_file': single.file_record(output/'rank_metrics.csv'),
                      'factor_bits': {k: single.tensor_record(v) for k,v in values.items()}, 'checks': checks}
            single.core.atomic_json(output/'report.json', report)
            self.assertEqual(new.resume(output, 'id', (68, 72), single, h), report)
            with self.assertRaises(RuntimeError):
                new.resume(output, 'different', (68, 72), single, h)
            bad = copy.deepcopy(report); bad['factor_bits']['A_fp64']['sha256'] = 'changed'
            single.core.atomic_json(output/'report.json', bad)
            with self.assertRaises(RuntimeError):
                new.resume(output, 'id', (68, 72), single, h)

    def test_output_disjoint(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            for output in (base, base/'old'/'new'):
                with self.assertRaises(RuntimeError):
                    single.core.disjoint(output, [base/'old'])
            single.core.disjoint(base/'new', [base/'old'])

    def test_frozen_libraries(self):
        directory = Path(__file__).parent
        loaded, _, _ = new.load_libraries(directory, directory/'qwen_full_a_audit_v2.py')
        self.assertEqual(loaded.core.sha(loaded.__file__), new.HELPERS['full_g_precision_r8_v1.py'])

    def test_release_checksums_and_lf(self):
        directory = Path(__file__).parent
        raw = (directory/'SHA256SUMS.qwen_fp64_target').read_bytes()
        self.assertNotIn(b'\r', raw)
        for line in raw.decode('ascii').splitlines():
            digest, name = line.split('  ', 1)
            self.assertEqual(Path(name).name, name)
            self.assertEqual(hashlib.sha256((directory/name).read_bytes()).hexdigest(), digest)


if __name__ == '__main__':
    unittest.main()
