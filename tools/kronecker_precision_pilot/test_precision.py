import unittest
import torch
from bootstrap import bridge
from kernels import Ops,factor_comparison,damp,solve


class Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(233);self.rows=[(torch.randn(13,11),torch.randn(13,7)) for _ in range(3)]
        for x,g in self.rows:g[-1]=0
        self.samples=lambda:iter(self.rows);self.T=12
    def test_fp64_equations_match_frozen_dense_reference(self):
        op=Ops(torch.float64,'cpu');a,g=op.marginal(self.samples,13,12)
        dense=lambda:(z.double().T@x.double() for x,z in self.rows)
        for side,factor in [('A',g),('G',a)]:
            actual=op.contraction(self.samples,factor,side,12)
            expected,n=bridge.sm.parent.contraction(dense,factor,side,12)
            torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
        ma,mg=op.moments(self.samples,12);ea,eg,n=bridge.sm.sequence_moments(dense,12,'cpu')
        torch.testing.assert_close(ma,ea,rtol=1e-12,atol=1e-12);torch.testing.assert_close(mg,eg,rtol=1e-12,atol=1e-12)
        ta,tg=op.token(self.samples,a,g,12);ea,eg=bridge.sm.token_step(self.samples,a,g,12)
        torch.testing.assert_close(ta,ea,rtol=1e-12,atol=1e-12);torch.testing.assert_close(tg,eg,rtol=1e-12,atol=1e-12)
        aa,gg,h=op.full(self.samples,a,g,12);ea,eg,eh=bridge.sm.full_step(dense,a,g,12)
        self.assertTrue(h['numeric_round_passed']);torch.testing.assert_close(aa,ea,rtol=1e-12,atol=1e-12);torch.testing.assert_close(gg,eg,rtol=1e-12,atol=1e-12)
    def test_fp32_preserves_dtype_policy_and_matches_reference(self):
        single=Ops(torch.float32,'cpu');double=Ops(torch.float64,'cpu')
        a,g=single.marginal(self.samples,13,12);ra,rg=double.marginal(self.samples,13,12)
        self.assertEqual(a.dtype,torch.float64);self.assertTrue(factor_comparison(a,g,ra,rg)['passed'])
        for _ in range(8):
            a,g,h=single.full(self.samples,a,g,12);ra,rg,rh=double.full(self.samples,ra,rg,12)
            self.assertTrue(h['numeric_round_passed']);self.assertTrue(factor_comparison(a,g,ra,rg)['passed'])
        bad=factor_comparison(a,g*1.1,ra,rg);self.assertFalse(bad['passed'])
    def test_solver_matches_frozen_and_raw_allowance_is_bounded(self):
        a,g=Ops(torch.float64,'cpu').marginal(self.samples,13,12);w0=torch.randn(7,11);wq=w0*.9;error=w0.double()-wq.double()
        t,audit=solve(error,a,g,w0,wq,3,False)
        old,c,old_audit=bridge.sm.solve(error,a,g,3,wq,w0)
        torch.testing.assert_close(t['W_deploy'],c['W_deploy'],rtol=1e-6,atol=1e-7)
        tiny=torch.diag(torch.tensor([-1e-8,0.,1.],dtype=torch.float64))
        with self.assertRaisesRegex(RuntimeError,'PSD'):damp(tiny,False)
        _,_,audit=damp(tiny,True);self.assertFalse(audit['clipping'])
        with self.assertRaisesRegex(RuntimeError,'PSD'):damp(torch.diag(torch.tensor([-.01,1.,1.],dtype=torch.float64)),True)


if __name__=='__main__':unittest.main(verbosity=2)
