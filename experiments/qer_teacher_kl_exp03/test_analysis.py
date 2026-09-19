import unittest
import numpy as np
from analysis import estimate,decision,ratio


class Inference(unittest.TestCase):
    def test_paired_variance(self):
        x=np.arange(8*64).reshape(8,64).astype(float)
        self.assertEqual(estimate((x+3)-x)['SE'],0)
        self.assertEqual(estimate((x+3)-x)['mean'],3)
    def test_states_and_denominator(self):
        self.assertEqual(decision(.03,.06,.02),'IMPROVED')
        self.assertEqual(decision(-.06,-.03,.02),'DEGRADED')
        self.assertEqual(decision(-.01,.01,.02),'SMALL_WITHIN_BUDGET')
        self.assertEqual(decision(.01,.04,.02),'UNRESOLVED_AT_K64')
        self.assertEqual(ratio(1,np.ones(3),1,np.array([1,0,1]),1e-12)['status'],'RATIO_UNRESOLVED')
    def test_bootstrap_scale_and_pairing(self):
        rng=np.random.default_rng(1);x=rng.uniform(1,3,(8,64))
        ids=rng.integers(0,64,(2000,8,64));b=sum(x[c][ids[:,c]].mean(1) for c in range(8))/8
        a=ratio(x.mean()*.2,b*.2,x.mean(),b,1e-12)
        self.assertAlmostEqual(a['mean'],.2);self.assertAlmostEqual(a['ci_low'],.2);self.assertAlmostEqual(a['ci_high'],.2)


if __name__=='__main__':unittest.main()
