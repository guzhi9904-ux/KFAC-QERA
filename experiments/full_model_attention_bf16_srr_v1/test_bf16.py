import math
import tempfile
import unittest
from common import *
from deployment import factor_cast,correction_hook
from evaluation import score_hidden

class BF16Tests(unittest.TestCase):
    def test_direct_cast_orientation_and_real_module_path(self):
        torch.manual_seed(41)
        p=torch.randn(11,64,dtype=torch.float64)*.02;q=torch.randn(64,9,dtype=torch.float64)*.02
        a,b=factor_cast(dict(P64=p,Q64=q),(11,9))
        self.assertTrue(torch.equal(a,q.T.bfloat16()) and torch.equal(b,p.T.bfloat16()))
        module=torch.nn.Linear(9,11,bias=False).bfloat16();x=torch.randn(2,3,9).bfloat16();hits={'m':0}
        h=module.register_forward_hook(correction_hook(a,b,hits,'m'))
        actual=module(x);expected=torch.nn.functional.linear(x,module.weight)+(x@a)@b
        self.assertTrue(torch.equal(actual,expected));self.assertEqual(hits['m'],1)
        self.assertEqual(actual.dtype,torch.bfloat16);h.remove()
        # A merged BF16 dense correction is a different finite-precision computation.
        merged=torch.nn.functional.linear(x,(module.weight.double()+p@q).bfloat16())
        self.assertFalse(torch.equal(actual,merged))

    def test_reject_fp32_factor_or_activation(self):
        p=torch.zeros(11,64,dtype=torch.float64);q=torch.zeros(64,9,dtype=torch.float64)
        with self.assertRaisesRegex(RuntimeError,'Invalid FP64'):
            factor_cast(dict(P64=p.float(),Q64=q),(11,9))
        a,b=factor_cast(dict(P64=p,Q64=q),(11,9));hook=correction_hook(a,b,{'m':0},'m')
        with self.assertRaisesRegex(RuntimeError,'not BF16'):
            hook(None,(torch.zeros(1,9),),torch.zeros(1,11))

    def test_token_loss_and_self_kl(self):
        torch.manual_seed(17);h=torch.randn(1,11,9).bfloat16();w=torch.randn(23,9).bfloat16();ids=torch.randint(23,(1,11))
        result=score_hidden(h,ids,w,chunk=10,reference=h)
        logits=torch.nn.functional.linear(h[0,:-1],w).double()
        expected=torch.nn.functional.cross_entropy(logits,ids[0,1:],reduction='sum')
        self.assertAlmostEqual(result['NLL_sum'],float(expected),places=12)
        self.assertEqual(result['tokens'],10);self.assertEqual(result['KL'],0)

    def test_frozen_record_cannot_change(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'record.json';freeze(p,{'identity':'x'});freeze(p,{'identity':'x'})
            with self.assertRaisesRegex(RuntimeError,'Frozen identity'):
                freeze(p,{'identity':'y'})

if __name__=='__main__':unittest.main()
