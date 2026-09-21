import unittest
import torch
from d_math import contract_grams,normalize
from k_math import compare,rotate
from d_common import sm


class DirectTests(unittest.TestCase):
    def test_signed_gram_gqa_and_zero_gradient_rows(self):
        torch.manual_seed(9);length,n,d=9,13,4;mapping=[0,0,1,1]
        x=torch.randn(length,n,dtype=torch.float64)
        ds=torch.randn(4,length,length,dtype=torch.float64);ds[:,-1]=0
        q=torch.randn(4,length,d,dtype=torch.float64)
        a,g=contract_grams(ds,q,x,mapping,2)
        explicit=sum((ds[i]@x).T@(ds[i]@x) for i in range(4))
        compare(a,explicit,1e-10)
        expected=torch.stack([sum(q[h].T@q[h]/d for h,b in enumerate(mapping) if b==kv) for kv in range(2)])
        compare(g,expected,1e-10)
        qdrop=q.clone();qdrop[:,-1]=0
        self.assertGreater(float((g-contract_grams(ds,qdrop,x,mapping,2)[1]).norm()),1e-4)
        self.assertGreater(float((a-contract_grams(ds.abs(),q,x,mapping,2)[0]).norm()),1.)
        c=torch.randn(length,d,dtype=torch.float64);s=torch.randn_like(c)
        self.assertGreater(float((g-contract_grams(ds,rotate(q,c,s),x,mapping,2)[1]).norm()),1.)

    def test_normalization_and_full_block_shape(self):
        a=torch.eye(13,dtype=torch.float64)*5;g=torch.stack([torch.eye(4,dtype=torch.float64)*3,torch.eye(4,dtype=torch.float64)*7])
        ar,gr,gb=normalize(a,g,256,2048,32)
        compare(ar,a/(256*2048*32),1e-10)
        compare(gr,gb*(2048*32/2047),1e-10)
        self.assertEqual(gr.shape,(8,8));self.assertEqual(torch.count_nonzero(gr[:4,4:]),0)

    def test_internal_surrogate_identity_and_solver_scalar_invariance(self):
        torch.manual_seed(44)
        x=torch.randn(9,13,dtype=torch.float64);q=torch.randn(9,8,dtype=torch.float64);e=torch.randn(9,9,dtype=torch.float64)
        compare(q.T@(e@x)/8**.5,((e.T@q)/8**.5).T@x,1e-10)
        a=torch.randn(13,13,dtype=torch.float64);a=a.T@a+torch.eye(13)
        g=torch.randn(8,8,dtype=torch.float64);g=g.T@g+torch.eye(8)
        error=torch.randn(8,13,dtype=torch.float64)
        t,_=sm.parent.weighted_svd(error,a,g,3,.001,.001)
        scaled,_=sm.parent.weighted_svd(error,a,g*(2048*32/2047),3,.001,.001)
        compare(t['C64'],scaled['C64'],1e-10)


if __name__=='__main__':unittest.main()
