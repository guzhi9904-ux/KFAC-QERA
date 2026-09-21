import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
import resume_eval  # Establish the same frozen-module import path as the launcher.
import subset_eval as sub
from bridge import commit, read, sha_file, mo, slug


class Tests(unittest.TestCase):
    def test_prefix_keeps_validation_and_restores_original_functions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); ids = torch.arange(64).reshape(32, 2)
            original_report = sub.evaluation.report
            with patch.object(sub.evaluation, 'windows', return_value=ids) as original:
                with sub.limited_test(16):
                    self.assertEqual(len(sub.evaluation.windows(root, 'test', 'id')), 16)
                    self.assertEqual(len(sub.evaluation.windows(root, 'validation', 'id')), 32)
                    # Resume uses exactly the same frozen prefix receipt.
                    self.assertEqual(len(sub.evaluation.windows(root, 'test', 'id')), 16)
                    self.assertEqual(read(root/'evaluation_subsets/test_first_16.json')['test_window_indices'], list(range(16)))
                self.assertIs(sub.evaluation.windows, original)
                self.assertIs(sub.evaluation.report, original_report)
            self.assertFalse((root/'complete.json').exists())

    def test_report_requires_complete_bound_records_and_never_claims_full_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); ident = 'id'; name = 'model.layers.10.self_attn.q_proj'
            config = dict(modules=[name]); keys = ['N128__Marginal', 'None']
            commit(root/'candidate_freeze.json', ident, keys=keys, files=[])
            fh = sha_file(root/'candidate_freeze.json'); ids = torch.arange(64).reshape(32, 2)
            def windows(root, role, identity):return ids[:16] if role == 'validation' else ids
            for role in ('validation', 'test'):
                for i in range(16):
                    folder = root/'scores'/role/f'w{i:04d}'; token = mo.digest_tensor(ids[i])
                    commit(folder/'teacher.json', ident, token_hash=token, scores=dict(KL=0., tokens=1, NLL_sum=1.))
                    for scope, key, names in sub.evaluation.evaluation_jobs([name], keys, role):
                        commit(folder/(slug(scope)+'___'+key+'.json'), ident,
                               token_hash=token, freeze_hash=fh, role=role, window=i, scope=scope,
                               candidate=key, modules=names, scores=dict(KL=2. if key=='None' else 1., tokens=1, NLL_sum=1.))
            path = root/'scores/test/w0015/joint12___N128__Marginal.json'
            saved = path.read_bytes(); path.unlink()
            with self.assertRaises(Exception):sub.subset_report(root, config, ident, 16, windows)
            self.assertFalse((root/'summary/test_subset_16/complete.json').exists())
            path.write_bytes(saved)
            # Extra scores outside the chosen prefix cannot contaminate the result.
            extra = root/'scores/test/w0016'; extra.mkdir(); (extra/'teacher.json').write_text('invalid, intentionally ignored')
            sub.subset_report(root, config, ident, 16, windows)
            report = read(root/'summary/test_subset_16/results.json')
            self.assertTrue(all(r['windows']==16 for r in report['rows']))
            self.assertTrue(all(r['KL_recovery_percent']==50. for r in report['rows'] if r['candidate']=='N128__Marginal'))
            self.assertFalse(report['full_test_complete']); self.assertFalse((root/'complete.json').exists())
            self.assertEqual(path.read_bytes(), saved)
            bad = read(path); bad.pop('record_sha256'); bad.pop('identity'); bad['token_hash']='wrong'
            path.unlink(); commit(path, ident, **bad)
            with self.assertRaisesRegex(RuntimeError, 'token binding'):sub.subset_report(root, config, ident, 16, windows)


if __name__ == '__main__':unittest.main(verbosity=2)
