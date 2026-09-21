import math
import unittest
import torch
from k_math import rotate,transpose_rotate,compare,spectrum,softmax_row_audit,centered_formula_audit


class AlgebraTests(unittest.TestCase):
    def test_centered_algebra_and_native_edge_are_separate_checks(self):
        torch.manual_seed(111)
        p=torch.randn(17,17).softmax(-1);delta=torch.randn(17,8);v=torch.randn(17,8)
        dp=delta@v.T;z=p@v;e=torch._softmax_backward_data(dp,p,-1,torch.float32)
        result=centered_formula_audit(p,dp,e,delta,v,z)
        self.assertTrue(result['algebra_FP64']['passed'])
        self.assertTrue(result['native_edge_vs_FP64']['passed'])
        corrupted=e.clone();corrupted[0,0]+=.01
        with self.assertRaises(RuntimeError):centered_formula_audit(p,dp,corrupted,delta,v,z)
        with self.assertRaises(RuntimeError):centered_formula_audit(p,dp,e,delta,v,z+.1)

    def test_rounded_probability_mass_explains_nonzero_row_sum(self):
        p=torch.tensor([[.25,.25,.25,.25]],dtype=torch.float32)*(1-2e-6)
        dp=torch.tensor([[.9,1.,1.1,1.]],dtype=torch.float32)
        e=torch._softmax_backward_data(dp,p,-1,torch.float32)
        audit=softmax_row_audit(p,dp,e)
        self.assertEqual(audit['old_zero_row_check_failures'],1)
        self.assertTrue(audit['passed'])
        # The revised check still rejects actual gradient corruption.
        broken=e.clone();broken[0,0]+=.001
        with self.assertRaises(RuntimeError):softmax_row_audit(p,dp,broken)
        with self.assertRaises(RuntimeError):softmax_row_audit(p*.99,dp,e)

    def test_transpose_is_not_assumed_inverse(self):
        torch.manual_seed(7)
        x=torch.randn(5,8,dtype=torch.float64,requires_grad=True)
        c=torch.randn_like(x);s=torch.randn_like(x);y=torch.randn_like(x)
        grad=torch.autograd.grad(rotate(x,c,s),x,y)[0]
        compare(transpose_rotate(y,c,s),grad,1e-10)
        self.assertGreater(float((transpose_rotate(y,c,s)-rotate(y,c,-s)).norm()),1.)

    def test_real_rope_gqa_source_query_weight_and_direction(self):
        torch.manual_seed(19);L,n,d,hq,hkv=7,11,8,4,2
        x=torch.randn(L,n,dtype=torch.float64)
        w=torch.randn(hkv*d,n,dtype=torch.float64,requires_grad=True)
        q=torch.randn(hq,L,d,dtype=torch.float64)
        theta=torch.randn(L,d//2,dtype=torch.float64);c=theta.cos().repeat(1,2);s=theta.sin().repeat(1,2)
        qt=rotate(q,c,s);kt=rotate((x@w.T).reshape(L,hkv,d).permute(1,0,2),c,s)
        v=torch.randn(hkv,L,d,dtype=torch.float64);delta=torch.randn(hq,L,d,dtype=torch.float64);delta[:,-1]=0
        mapping=[0,0,1,1];mask=torch.ones(L,L,dtype=torch.bool).triu(1)
        logits=(qt@kt[mapping].transpose(1,2))/math.sqrt(d)
        p=logits.masked_fill(mask,-torch.inf).softmax(-1);z=p@v[mapping]
        expected=torch.autograd.grad((z*delta).sum(),w)[0]
        p=p.detach();dp=delta@v[mapping].transpose(1,2);e=p*(dp-(p*dp).sum(-1,keepdim=True))
        compare(e.sum(-1),torch.zeros(hq,L,dtype=torch.float64),1e-10,scale=float(e.norm()))
        g=torch.zeros(hkv,L,d,dtype=torch.float64);query=torch.zeros(L,dtype=torch.float64)
        r=torch.randn_like(w);rx=(x@r.T).reshape(L,hkv,d)
        msum=torch.zeros_like(w)
        for a,b in enumerate(mapping):
            g[b]+=transpose_rotate(e[a].T@qt[a]/math.sqrt(d),c,s)
            query+=(e[a]*(qt[a]@rotate(rx[:,b],c,s).T)/math.sqrt(d)).sum(-1)
            for t in range(L):
                f=(transpose_rotate(qt[a,t].expand(L,-1),c,s)*e[a,t,:,None]/math.sqrt(d)).T
                msum[b*d:(b+1)*d]+=f@x
        gs=g.permute(1,0,2).reshape(L,-1)
        compare(gs.T@x,expected,1e-10);compare(msum,expected,1e-10)
        source=(gs*rx.reshape(L,-1)).sum(-1)
        compare(query.sum(),source.sum(),1e-10)
        compare(query.sum(),(expected*r).sum(),1e-10)
        self.assertGreater(float((source-query).norm()),1e-6)

    def test_zero_rank_and_post_activation_projection(self):
        torch.manual_seed(3);q=torch.randn(8,dtype=torch.float64);e=torch.randn(13,dtype=torch.float64)
        spec,u=spectrum(q[:,None]*e[None,:]);self.assertAlmostEqual(spec['rho1'],1)
        spec0,_=spectrum(torch.zeros(8,13,dtype=torch.float64));self.assertIsNone(spec0['rho1'])
        f=torch.randn(8,13,dtype=torch.float64);x=torch.randn(13,17,dtype=torch.float64)
        _,u=spectrum(f);uk=u[:,-4:]
        compare((uk@uk.T@f)@x,uk@(uk.T@(f@x)),1e-10)


if __name__=='__main__':unittest.main()
