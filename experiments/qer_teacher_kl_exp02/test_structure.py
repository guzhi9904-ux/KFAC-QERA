import unittest
import numpy as np
import torch
from score_ops import full_pos,direct_sep,gram_sep,validate_scores
from stats_ops import estimate,differences,classify,ranking,ratio_summary,bootstrap_indices,bootstrap_means,record_array


class StructureTests(unittest.TestCase):
    def setUp(self): torch.manual_seed(613); torch.set_num_threads(2)
    def close(self,a,b):
        if abs(b)>1e-20:self.assertLessEqual(abs(a-b)/abs(b),1e-10)
        else:self.assertLessEqual(abs(a-b),1e-10)

    def test_signed_cross_terms(self):
        for signs in ((1.,1.),(1.,-1.)):
            g=torch.tensor(signs,dtype=torch.float64).reshape(2,1)
            rx=torch.ones_like(g)
            d,full,pos=full_pos(g,rx,1)
            cross=sum(float(g[t,0]*g[u,0]) for t in range(2) for u in range(t+1,2))
            self.close(full-pos,cross)
            self.assertEqual(np.sign(full-pos),np.sign(signs[1]))

    def test_sep_trace_kronecker_and_all_independent_pairs(self):
        n,k,L,T,inc,outc=2,3,3,2,3,2
        x=torch.randn(n,L,inc,dtype=torch.float64)
        g=torch.randn(n,k,L,outc,dtype=torch.float64)
        r=torch.randn(outc,inc,dtype=torch.float64)
        xf=x.reshape(-1,inc);gf=g.reshape(-1,outc)
        A=xf.T@xf/(n*L);G=gf.T@gf/(n*k*L);M=r@A@r.T
        trace=float(torch.trace(G@M))*L/(2*T)
        kron=float(r.reshape(-1)@torch.kron(G.contiguous(),A.contiguous())@r.reshape(-1))*L/(2*T)
        contraction=np.mean([direct_sep(g[c,j],{'R':M},T)['R'] for c in range(n) for j in range(k)])
        pairs=float((gf@r@xf.T).square().mean())*L/(2*T)
        self.close(kron,trace);self.close(contraction,trace);self.close(pairs,trace)
        self.close(gram_sep(g[0,0],{'R':M},T)['R'],direct_sep(g[0,0],{'R':M},T)['R'])

    def test_paired_se_and_decomposition(self):
        base=np.arange(64,dtype=float).reshape(8,8)/11
        values=np.stack((base,base+2,base-3),axis=-1)
        e=differences(values)
        np.testing.assert_allclose(e[...,2],e[...,0]+e[...,1],atol=1e-14)
        self.close(estimate(e[...,0])['mean'],2.)
        self.assertLess(estimate(e[...,0])['SE'],1e-14)
        self.assertGreater(estimate(base)['SE'],0)

    def test_bootstrap_pairing_and_common_scaling(self):
        rng=np.random.default_rng(7)
        base=rng.uniform(.5,2,size=(8,64,3,1))
        values=base*np.array([1.,2.,5.])[None,None,None,:]
        draws=bootstrap_indices()
        again=bootstrap_indices();np.testing.assert_array_equal(draws,again)
        samples=bootstrap_means(values,draws)
        np.testing.assert_allclose(samples[...,1],samples[...,0]*2,rtol=1e-14)
        rows=ratio_summary(values,draws)
        for r in rows:
            if r['kind'].endswith('difference'):
                self.assertLess(abs(r['mean']),1e-12)
                self.assertLess(abs(r['ci_low']),1e-12);self.assertLess(abs(r['ci_high']),1e-12)

    def test_classification_ranking_and_zero_denominator(self):
        self.assertEqual(classify(-.09,.09,.1),'SMALL_WITHIN_BUDGET')
        self.assertEqual(classify(.11,.3,.1),'MATERIAL_DIFFERENCE')
        self.assertEqual(classify(-.3,-.11,.1),'MATERIAL_DIFFERENCE')
        self.assertEqual(classify(-.02,.2,.1),'UNRESOLVED_AT_K64')
        self.assertEqual(ranking(-.3,-.01),'SVD64_PREFERRED')
        self.assertEqual(ranking(-.01,.03),'RANKING_UNRESOLVED')
        self.assertEqual(ranking(.01,.3),'A64_PREFERRED')
        rows=ratio_summary(np.zeros((8,64,3,3)),bootstrap_indices())
        self.assertTrue(all(r['status']=='RATIO_UNRESOLVED' for r in rows))
        with self.assertRaises(ValueError):record_array([{'module':'m','window':0,'replicate':0}]*2,'m')

    def test_psd_guard_keeps_roundoff_without_clipping(self):
        validate_scores([0.,1.,-1e-15],1.)
        with self.assertRaises(FloatingPointError):validate_scores([-1e-3],1.)
        with self.assertRaises(FloatingPointError):validate_scores([float('nan')],1.)


if __name__=='__main__':unittest.main(verbosity=2)
