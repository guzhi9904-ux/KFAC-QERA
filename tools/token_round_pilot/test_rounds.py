import unittest
import torch
from run import directions,paired_summary


class Tests(unittest.TestCase):
    def test_direction_ignores_global_scale(self):
        a=torch.diag(torch.tensor([1.,2.],dtype=torch.float64));g=torch.diag(torch.tensor([3.,1.],dtype=torch.float64))
        row=directions(a*1e8,g*1e-3,a,g)
        self.assertLess(row['A_direction_relative'],1e-14);self.assertLess(row['G_direction_relative'],1e-14)
        self.assertAlmostEqual(row['Kronecker_direction_cosine'],1.)
        self.assertAlmostEqual(row['product_norm']/row['reference_product_norm'],1e5)
    def test_paired_improvement_sign_and_baseline(self):
        x=torch.tensor([1.,2.,3.,4.],dtype=torch.float64)
        rows=paired_summary({'None':2*x,'round3':x,'round5':.9*x})
        row=next(r for r in rows if r['candidate']=='round5')
        self.assertAlmostEqual(row['KL_reduction_vs_round3_percent'],10.)
        self.assertAlmostEqual(row['recovery_change_vs_round3_pp'],5.)
        self.assertTrue(all(abs(v-10)<1e-10 for v in row['paired_bootstrap95_KL_reduction_percent']))


if __name__=='__main__':unittest.main(verbosity=2)
