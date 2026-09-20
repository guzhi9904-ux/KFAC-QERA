import unittest
import torch
from polish import iterate,Ops,bridge,factor_comparison


class Tests(unittest.TestCase):
    def test_same_round_count_and_strict_final_precision(self):
        torch.manual_seed(991);rows=[(torch.randn(13,11),torch.randn(13,7)) for _ in range(2)]
        samples=lambda:iter(rows);op=Ops(torch.float64,'cpu');a,g=op.marginal(samples,13,12)
        for method,total in [('Token-joint',3),('Full-fit',8)]:
            actual,history=iterate(samples,a,g,method,'cpu',13,12)
            self.assertEqual(len(history),total);self.assertEqual(history[-1]['dtype'],'torch.float64')
            self.assertTrue(all(h['dtype']=='torch.float32' for h in history[:-1]))
            ra,rg=a,g
            if method=='Token-joint':rg=torch.eye(len(g),dtype=torch.float64)
            for i in range(total):
                if method=='Full-fit':ra,rg,_=op.full(samples,ra,rg,12)
                else:ra,rg=op.token(samples,ra,rg,12);ra,rg=bridge.sm.gauge(ra,rg)
            self.assertTrue(factor_comparison(actual['A'],actual['G'],ra,rg)['passed'])
            for factor in actual.values():bridge.sm.parent.spectrum(factor)


if __name__=='__main__':unittest.main(verbosity=2)
