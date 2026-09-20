"""Connected multi-target gradients must equal independent interventions."""
import contextlib
import unittest
from types import SimpleNamespace
import torch
from torch import nn
from shared_teacher import shared_gradient,shared_capture,SharedTeacher
from bridge import hidden_forward


class Block(nn.Module):
    def __init__(self):
        super().__init__();self.q=nn.Linear(5,5,bias=False);self.v=nn.Linear(5,5,bias=False);self.down=nn.Linear(5,5,bias=False)
    def forward(self,x):
        # Both parallel branches and a downstream target, with token mixing.
        h=torch.tanh(self.q(x))*self.v(x);h=h.cumsum(1)/torch.arange(1,x.shape[1]+1).to(x)[None,:,None]
        return x+self.down(h).tanh()


class Body(nn.Module):
    def __init__(self):
        super().__init__();self.embed_tokens=nn.Embedding(17,5);self.layers=nn.ModuleList([Block(),Block()])
    def forward(self,input_ids,**kwargs):
        x=self.embed_tokens(input_ids)
        for block in self.layers:x=block(x)
        return SimpleNamespace(last_hidden_state=x)


class Tiny(nn.Module):
    def __init__(self):
        super().__init__();self.model=Body();self.lm_head=nn.Linear(5,17,bias=False)


class Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(81);self.model=Tiny().eval().requires_grad_(False)
        self.ids=torch.tensor([[1,2,5,3,8,2,6]])
        self.names=[f'model.layers.{i}.{s}' for i in (0,1) for s in ('q','v','down')]
    def test_shared_matches_independent_and_weight_autograd(self):
        seed=torch.randn(1,7,5);expected={}
        for name in self.names:
            mod=self.model.get_submodule(name);mod.weight.requires_grad_(True);state={}
            def hook(m,args,h):state.update(x=args[0].detach(),h=h)
            handle=mod.register_forward_hook(hook)
            try:
                h=hidden_forward(self.model,self.ids)
                g,wg=torch.autograd.grad(h,(state['h'],mod.weight),seed)
                x=state['x'].reshape(7,5);torch.testing.assert_close(g.reshape(7,5).T@x,wg)
                expected[name]=g.reshape(7,5)
            finally:handle.remove();mod.weight.requires_grad_(False)
        with torch.no_grad(),shared_capture(self.model,self.names) as states:
            ref=hidden_forward(self.model,self.ids);xs={n:states[n]['x'].clone() for n in self.names}
        for checkpointed in (False,True):
            actual=shared_gradient(self.model,self.names,self.ids,lambda h:seed,ref,xs,checkpointed)
            for name in self.names:
                torch.testing.assert_close(actual[name]['g'],expected[name],rtol=1e-6,atol=1e-7)
                self.assertTrue(torch.equal(actual[name]['x'],xs[name]))
    def test_exception_restores_hooks_and_forwards(self):
        forwards=[b.forward for b in self.model.model.layers]
        with self.assertRaisesRegex(ValueError,'injected'):
            with shared_capture(self.model,self.names,gradient=True):raise ValueError('injected')
        self.assertEqual(forwards,[b.forward for b in self.model.model.layers])
        self.assertFalse(any(m._forward_hooks for m in self.model.modules()))
    def test_cached_logits_scores_and_weight_restore(self):
        teacher=SharedTeacher({'vocab_chunk':2},None,'fixture',lambda *a,**k:contextlib.nullcontext());teacher.model=self.model
        ref=hidden_forward(self.model,self.ids).detach();cached=teacher.reference_logits(ref)
        original={n:self.model.get_submodule(n).weight.detach().clone() for n in self.names}
        weights={n:w+.03 for n,w in original.items()}
        with teacher.deploy(weights):
            actual=hidden_forward(self.model,self.ids).detach();scores=teacher.scores(self.ids,cached)
            self.assertAlmostEqual(scores['KL'],teacher.kl(ref,actual),places=14)
            logits=self.model.lm_head(actual)[0,:-1].double()
            nll=-logits.log_softmax(-1).gather(1,self.ids[0,1:,None]).sum()
            self.assertAlmostEqual(scores['NLL_sum'],float(nll),places=6)
        for n in self.names:self.assertTrue(torch.equal(self.model.get_submodule(n).weight,original[n]))
        with self.assertRaisesRegex(ValueError,'injected'):
            with teacher.deploy(weights):raise ValueError('injected')
        self.assertEqual(teacher.scores(self.ids,cached)['KL'],0.)


if __name__=='__main__':unittest.main(verbosity=2)
