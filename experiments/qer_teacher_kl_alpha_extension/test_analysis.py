import unittest
from analysis import analyze, MODULES, DIRECTIONS, SMALL, NEW


def fixture():
    parent, extension, baseline = [], [], []
    for m in MODULES:
        for d, scale in zip(DIRECTIONS, (4., 2., 1.)):
            q0 = sum((w+1)*(w+1)*scale for w in range(8))/sum(range(1, 9))
            baseline.append(dict(module=m, direction=d, q_KL=q0, local_alphas=list(SMALL),
                                 q_hat=q0*1.1, ci_low=q0*.8, ci_high=q0*1.4))
            for w in range(8):
                for a in SMALL+NEW:
                    mean = a*a*(w+1)*scale
                    (parent if a in SMALL else extension).append(dict(module=m, direction=d, window=w,
                        alpha=a, T=w+1, KL_mean=mean, KL_sum=mean*(w+1), valid=True, invalid_reasons=[]))
    return parent, extension, baseline


class AnalysisTests(unittest.TestCase):
    def test_window_baseline_and_token_weighting(self):
        result = analyze(*fixture())
        for r in result['pooled']+result['windows']:
            self.assertAlmostEqual(r['rho'], 1.)
        self.assertTrue(all(r['A64_wins'] == 8 for r in result['advantages']))

    def test_large_points_cannot_refit_q0_and_reversal(self):
        p, e, b = fixture()
        for r in e:
            if r['direction'] == 'R_A64' and r['alpha'] == 1:
                r['KL_mean'] *= 3; r['KL_sum'] *= 3
        result = analyze(p, e, b)
        for r in result['pooled']:
            if r['direction'] == 'R_A64' and r['alpha'] == 1:
                self.assertAlmostEqual(r['rho'], 3.)
        for r in result['advantages']:
            if r['alpha'] == 1:
                self.assertTrue(r['rank_reversal']); self.assertEqual(r['SVD64_wins'], 8)

    def test_invalid_excluded_from_interpretation_and_duplicates_rejected(self):
        p, e, b = fixture()
        for r in e:
            if r['direction'] == 'R_A64' and r['alpha'] == 1 and r['window'] == 0:
                r['valid'] = False; r['invalid_reasons'] = ['output_path']
        result = analyze(p, e, b)
        for r in result['pooled']:
            if r['direction'] == 'R_A64' and r['alpha'] == 1:
                self.assertIsNone(r['rho'])
        for r in result['advantages']:
            if r['alpha'] == 1:
                self.assertIsNone(r['rank_reversal']); self.assertEqual(r['invalid_windows'], 1)
        with self.assertRaises(ValueError): analyze(p, e+[e[0]], b)


if __name__ == '__main__': unittest.main(verbosity=2)
