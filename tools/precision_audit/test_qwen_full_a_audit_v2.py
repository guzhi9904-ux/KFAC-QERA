import hashlib
from pathlib import Path
import unittest
import torch
import qwen_full_a_audit_v2 as v2


class EnvironmentTests(unittest.TestCase):
    def test_release_manifest_lf_and_hashes(self):
        root = Path(__file__).parent
        raw = (root/'SHA256SUMS.qwen_v2').read_bytes()
        self.assertNotIn(b'\r', raw)
        for line in raw.decode('ascii').splitlines():
            digest, name = line.split('  ', 1)
            self.assertEqual(Path(name).name, name)
            self.assertEqual(hashlib.sha256((root/name).read_bytes()).hexdigest(), digest)

    def versions(self):
        return dict(zip(v2.PACKAGES, ('2.3.0', '4.44.2', '0.33.0', '2.21.0', '0.19.1', '0.4.5', '1.26.4', '1.14.1')))

    def test_matching_distributions_with_runtime_suffix_pass(self):
        r = v2.environment_report(self.versions(), self.versions(), '2.3.0+cu121', '12.1')
        self.assertEqual(r['status'], 'PASS')
        self.assertEqual(r['distribution_mismatches'], {})

    def test_each_real_distribution_change_rejected(self):
        for package in v2.PACKAGES:
            current = self.versions(); current[package] += '.changed'
            r = v2.environment_report(self.versions(), current, '2.3.0+cu121', '12.1')
            self.assertEqual(r['status'], 'FAIL')
            self.assertIn(package, r['distribution_mismatches'])

    def test_runtime_suffix_not_discarded(self):
        for runtime, cuda in (('2.3.0+cu118', '11.8'), ('2.3.0+cpu', None), ('2.3.0+cu121', '12.4')):
            r = v2.environment_report(self.versions(), self.versions(), runtime, cuda)
            self.assertEqual(r['status'], 'FAIL')

    def test_missing_metadata_rejected(self):
        current = self.versions(); del current['scipy']
        with self.assertRaises(RuntimeError):
            v2.environment_report(self.versions(), current, '2.3.0+cu121', '12.1')

    def test_frozen_helper_not_modified_and_math_reused(self):
        path = Path(v2.__file__).with_name('qwen_full_a_audit_v1.py')
        before = path.read_bytes(); runtime = torch.__version__
        helper = v2.load_helper(path)
        torch.manual_seed(1)
        s = torch.eye(6); error = torch.randn(6, 4); result = {}
        helper.replay(s, error, 2, torch.device('cpu'), lambda k, v: result.update({k: v}))
        self.assertIn('fp32_inverse', result)
        self.assertIn('fp64_inverse_cast_fp32', result)
        self.assertLess(result['fp64_inverse']['inverse_relative_residual_fp64'], 1e-12)
        self.assertEqual(torch.__version__, runtime)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(hashlib.sha256(before).hexdigest(), v2.HELPER_SHA)

    def test_changed_helper_rejected(self):
        with self.assertRaises(RuntimeError):
            v2.load_helper(Path(__file__))


if __name__ == '__main__':
    unittest.main()
