import itertools
import gc
import tempfile
from pathlib import Path
import unittest

import torch

from math_ops import (gram, relative, objective, choose_metric, low_rank, stable_kl,
                      sampled_seed, intervention, monte_carlo, plateau)
from storage import read_tensors, save_tensors


class MathTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(613);torch.set_num_threads(2)

    def test_gram_and_cholesky_orientation(self):
        x=torch.randn(37,9,dtype=torch.float64)
        a=gram(x,7,True)/len(x)
        e=torch.randn(6,9,dtype=torch.float64)
        self.assertAlmostEqual(float((x@e.T).square().sum()/len(x)),objective(e,a),places=11)
        solve,l,s,_=choose_metric(a)
        p,q,_,audit=low_rank(e,l,3,True)
        p2,q2,_,audit2=low_rank(e,s,3)
        j=objective(e-p@q,solve)
        self.assertAlmostEqual(j,audit["tail_objective"],places=10)
        self.assertAlmostEqual(j,audit2["tail_objective"],places=10)
        self.assertLess(relative(p@q,p2@q2),1e-10)
        wrong=e@l.T
        self.assertGreater(abs(float(wrong.square().sum())-objective(e,a)),1e-3)

    def test_score_enumeration_matches_kl_hessian(self):
        x=torch.randn(2,3,dtype=torch.float64)
        w=torch.randn(2,3,dtype=torch.float64)
        r=torch.randn_like(w)
        mixing=torch.randn(4,6,dtype=torch.float64)
        def logits(alpha):return torch.tanh((x@(w-alpha*r).T).flatten()@mixing).reshape(2,3)
        ref=logits(torch.tensor(0.)).detach();p=ref.softmax(-1)
        a=torch.tensor(0.,dtype=torch.float64,requires_grad=True)
        d=(p*(ref.log_softmax(-1)-logits(a).log_softmax(-1))).sum()/2
        first=torch.autograd.grad(d,a,create_graph=True)[0]
        q=float(torch.autograd.grad(first,a)[0])/2
        expectation=0.;wrong=0.
        for ys in itertools.product(range(3),repeat=2):
            h=(x@w.T).detach().requires_grad_()
            z=torch.tanh(h.flatten()@mixing).reshape(2,3)
            loss=-z.log_softmax(-1)[torch.arange(2),torch.tensor(ys)].sum()
            g=torch.autograd.grad(loss,h)[0]
            projections=(g*(x@r.T)).sum(-1)
            prob=float(p[0,ys[0]]*p[1,ys[1]])
            expectation+=prob*float(projections.sum().square())/4
            wrong+=prob*float(projections.square().sum())/4
        self.assertAlmostEqual(expectation,q,places=11)
        self.assertGreater(abs(wrong-q),1e-4)
        self.assertAlmostEqual(float(stable_kl(ref,logits(1e-4)).mean())/1e-8,q,delta=2e-3*max(q,1))

    def test_seed_matches_fp64_probability_autograd(self):
        hidden=torch.randn(1,9,7,requires_grad=True)
        weight=torch.randn(13,7);labels=torch.randint(13,(8,))
        logits=hidden[0,:-1]@weight.T
        loss=-logits.double().log_softmax(-1)[torch.arange(8),labels].sum()
        direct=torch.autograd.grad(loss,hidden)[0]
        analytic=sampled_seed(hidden.detach(),weight,labels,3)
        self.assertLess(relative(analytic,direct),1e-6)
        self.assertEqual(float(analytic[0,-1].abs().max()),0)

    def test_small_kl_and_restore_on_exception(self):
        z=torch.randn(5,11,dtype=torch.float64);delta=torch.randn_like(z)*1e-7
        values=stable_kl(z,z+delta)
        p=z.softmax(-1);q=.5*(p*(delta-(p*delta).sum(-1,keepdim=True)).square()).sum(-1)
        self.assertLess(relative(values,q),1e-6)
        layer=torch.nn.Linear(4,3,bias=False);old=layer.weight.detach().clone()
        with self.assertRaisesRegex(RuntimeError,"injected"):
            with intervention(layer,torch.randn_like(old),.1):raise RuntimeError("injected")
        self.assertTrue(torch.equal(old,layer.weight))

    def test_atomic_progress_and_mc_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"stats.safetensors"
            save_tensors(p,{"sum":torch.ones(3,3,dtype=torch.float64)},{"completed":[0,1],"N":4})
            save_tensors(p,{"sum":torch.full((3,3),2.,dtype=torch.float64)},{"completed":[0,1,2],"N":6})
            tensors,meta=read_tensors(p)
            self.assertEqual(meta["N"],6);self.assertEqual(float(tensors["sum"][0,0]),2.)
            del tensors
            gc.collect()
        records=[{"window":c,"replicate":k,"T":2,"directions":{"R":{"b":v}}}
                 for c in range(2) for k,v in enumerate([1.,3.])]
        summary=monte_carlo(records,"R")
        self.assertEqual(summary["q_hat"],2.)
        self.assertAlmostEqual(summary["MC_SE"],2**-.5)
        with self.assertRaises(RuntimeError):monte_carlo(records+[records[0]],"R")

    def test_platform_cannot_skip_invalid_alpha(self):
        rows=[{"alpha":a,"valid":True,"KL_mean":2*a*a} for a in [.025,.05,.1,.2]]
        self.assertEqual(plateau(rows)["alphas"],[.025,.05,.1])
        rows[1]["valid"]=False
        self.assertIsNone(plateau(rows))


if __name__=="__main__":unittest.main(verbosity=2)
