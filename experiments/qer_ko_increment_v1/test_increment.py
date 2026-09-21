import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from ko_common import NEW,OLD,METHODS,jobs,source_x,ParentStore,TensorStore,sm,mo
from ko_replay import capture_blocks,local_gradient,suffix_hidden
from shared_teacher import shared_gradient
from bridge import hidden_forward


class Block(nn.Module):
    def __init__(self):
        super().__init__();self.norm=nn.LayerNorm(6)
        self.self_attn=nn.ModuleDict({n:nn.Linear(6,6,bias=False) for n in ('q_proj','k_proj','v_proj','o_proj')})
        self.mlp=nn.ModuleDict({'up_proj':nn.Linear(6,9,bias=False),'down_proj':nn.Linear(9,6,bias=False)})
    def forward(self,hidden_states,position_ids=None,**kwargs):
        x=self.norm(hidden_states);a=self.self_attn
        scores=a['q_proj'](x)@a['k_proj'](x).transpose(-1,-2)/6**.5
        mask=torch.ones_like(scores,dtype=torch.bool).triu(1)
        h=hidden_states+a['o_proj'](scores.masked_fill(mask,-1e9).softmax(-1)@a['v_proj'](x))
        return (h+self.mlp['down_proj'](self.mlp['up_proj'](self.norm(h)).tanh()),)


class Body(nn.Module):
    def __init__(self):
        super().__init__();self.embed_tokens=nn.Embedding(19,6);self.layers=nn.ModuleList([Block(),Block()]);self.norm=nn.LayerNorm(6)
    def forward(self,input_ids,**kwargs):
        h=self.embed_tokens(input_ids)
        for b in self.layers:h=b(h,position_ids=torch.arange(h.shape[1])[None],use_cache=False)[0]
        return SimpleNamespace(last_hidden_state=self.norm(h))


class Tiny(nn.Module):
    def __init__(self):super().__init__();self.model=Body();self.lm_head=nn.Linear(6,19,bias=False)


class Tests(unittest.TestCase):
    def test_local_down_seed_matches_connected_qkvo_gradients(self):
        torch.manual_seed(5);model=Tiny().eval().requires_grad_(False);ids=torch.tensor([[1,4,6,3,7]])
        names=[f'model.layers.{i}.self_attn.{p}_proj' for i in (0,1) for p in ('q','k','v','o')]
        downs=[f'model.layers.{i}.mlp.down_proj' for i in (0,1)]
        seed=torch.randn(1,5,6);full=shared_gradient(model,names+downs,ids,lambda _:seed,recompute=False)
        ref,states=capture_blocks(model,ids,[0,1])
        for i in (0,1):
            selected=[n for n in names if n.startswith(f'model.layers.{i}.')]
            got,error=local_gradient(model,i,states[i],full[downs[i]]['g'],selected)
            self.assertLessEqual(error,1e-7)
            for n in selected:torch.testing.assert_close(got[n]['g'],full[n]['g'],rtol=1e-5,atol=1e-7)
            # Nontrivial candidate, not merely teacher self replay.
            target=model.get_submodule(selected[-1]);before=target.weight.detach().clone()
            with torch.no_grad():
                target.weight.add_(.031);fast=suffix_hidden(model,i,states[i]);slow=hidden_forward(model,ids)
                torch.testing.assert_close(fast,slow,rtol=1e-6,atol=1e-7);target.weight.copy_(before)
        self.assertFalse(any(m._forward_hooks or m._forward_pre_hooks for m in model.modules()))

    def test_scope_is_44_corrections_832_new_eval_and_no_old_recompute(self):
        c=dict(modules=NEW+OLD)
        self.assertEqual(len(jobs(c))*16,832)
        self.assertEqual(sum(len(METHODS) if n in NEW else 1 for n in c['modules']),44)
        self.assertEqual([(n,m) for n,m in jobs(c) if n in OLD],[(n,'A-only') for n in OLD])
        self.assertEqual(source_x(Path('parent'),'model.layers.10.self_attn.k_proj',0),source_x(Path('parent'),'model.layers.10.self_attn.q_proj',0))

    def test_parent_store_cannot_create_missing_receipt_or_mutate(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'x.safetensors';TensorStore('old',0).put(p,dict(x=torch.ones(2,3)))
            reader=ParentStore('old',0);before=p.read_bytes();reader.get(p)
            with self.assertRaisesRegex(RuntimeError,'read-only'):reader.put(p,dict(x=torch.zeros(2,3)))
            p.with_suffix('.json').unlink()
            with self.assertRaisesRegex(RuntimeError,'Missing parent receipt'):reader.get(p)
            self.assertEqual(p.read_bytes(),before);self.assertFalse(p.with_suffix('.json').exists())

    def test_a_only_matches_input_weighted_svd(self):
        torch.manual_seed(9);x=torch.randn(40,12,dtype=torch.float64);a=x.T@x/len(x)
        w0=torch.randn(10,12);wq=(w0*4).round()/4;e=w0.double()-wq.double()
        t,c,audit=sm.solve(e,a,torch.eye(10,dtype=torch.float64),4,wq,w0)
        vals,vectors=torch.linalg.eigh(t['A_solve']);root=(vectors*vals.sqrt())@vectors.T
        u,s,vh=torch.linalg.svd(e@root,full_matrices=False)
        direct=(u[:,:4]*s[:4])@vh[:4]@torch.linalg.inv(root)
        self.assertLess(mo.relative(t['C64'],direct),1e-10)


if __name__=='__main__':unittest.main(verbosity=2)
