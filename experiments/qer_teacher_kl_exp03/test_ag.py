import unittest
import torch
import ag_math as am


class Identities(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(613)
        self.x=torch.randn(3,4,5,dtype=torch.float64)
        self.g=torch.randn(3,2,4,3,dtype=torch.float64)
        self.ss=[self.g[c,k].T@self.x[c] for c in range(3) for k in range(2)]
        self.a=sum(x.T@x for x in self.x)/12
        self.gg=sum(self.g[c,k].T@self.g[c,k] for c in range(3) for k in range(2))/24
        self.r=torch.randn(3,5,dtype=torch.float64);self.T=3
    def close(self,a,b):self.assertTrue(torch.allclose(torch.as_tensor(a,dtype=torch.float64),torch.as_tensor(b,dtype=torch.float64),rtol=1e-10,atol=1e-10),(a,b))
    def test_explicit_H_vec_and_blocks(self):
        s=torch.stack([v.T.reshape(-1) for v in self.ss]);h=s.T@s/len(s)/self.T
        v=self.r.T.reshape(-1)
        self.close(v@h@v/2,sum((z*self.r).sum()**2 for z in self.ss)/len(s)/2/self.T)
        k=torch.kron(self.a.contiguous(),self.gg.contiguous())
        self.close(v@k@v/2,am.qmetric(self.r,self.a,self.gg))
        ug,_=am.contraction(lambda:iter(self.ss),self.a,'G',self.T)
        ua,_=am.contraction(lambda:iter(self.ss),self.gg,'A',self.T)
        self.close((self.gg*ug).sum(),(self.a*ua).sum())
        j=am.qmetric(self.r,self.a,self.gg)
        self.assertGreater(j,0)
        a,g,hist,_=am.als(lambda:iter(self.ss),self.a,self.gg,self.T,12,1e-8,1e-6)
        for row in hist:
            self.assertLessEqual(row['J_after_G'],row['J_before']+1e-10)
            self.assertLessEqual(row['J_after_A'],row['J_after_G']+1e-10)
        explicit=float((h-torch.kron(a.contiguous(),g.contiguous())).square().sum()-h.square().sum())
        self.close(explicit,hist[-1]['J_after_A'])
    def test_canonical_marginal_and_cross_position(self):
        sep=sum(((g@self.r@self.a)* (g@self.r)).sum() for row in self.g for g in row)/6/(2*self.T)
        self.close(sep,am.qmetric(self.r,self.a,4/self.T*self.gg))
        independent=sum((gv@self.r@xv)**2 for row in self.g for g in row for gv in g for x in self.x for xv in x)/24/12*4/(2*self.T)
        self.close(sep,independent)
        a,g=am.gauge(self.a,self.gg);self.close(am.qmetric(self.r,a,g),am.qmetric(self.r,self.a,self.gg))
    def test_svd_and_scale_equivariance(self):
        out,audit=am.weighted_svd(self.r,self.a,self.gg,1,.001,.001)
        self.assertLess(audit['tail_relative_error'],1e-10)
        for fa,fg in [(13,1/13),(13,7),(.002,.003)]:
            z,_=am.weighted_svd(self.r,self.a*fa,self.gg*fg,1,.001,.001)
            self.close(out['C64'],z['C64'])
            self.close(am.qmetric(self.r,z['A_solve'],z['G_solve']),fa*fg*am.qmetric(self.r,out['A_solve'],out['G_solve']))
        self.close(out['P64']@out['Q64'],out['C64'])
    def test_invalid_psd_and_zero(self):
        with self.assertRaises(RuntimeError):am.damp(-torch.eye(3,dtype=torch.float64),.001)
        with self.assertRaises(RuntimeError):am.damp(torch.zeros(3,3,dtype=torch.float64),.001)


if __name__=='__main__':unittest.main()
