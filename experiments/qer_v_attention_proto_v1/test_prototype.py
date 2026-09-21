import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from v_math import effective,compare,normalize
from v_capture import local_delta
from v_statistics import checkpoint,restore
from v_common import LAYERS
from v_report import paired


class Tests(unittest.TestCase):
    def test_gqa_weight_gradient_and_fast_moments_independent_references(self):
        torch.manual_seed(13);L=7;n=6;hq=4;hkv=2;d=3;mapping=[0,0,1,1]
        x=torch.randn(L,n,dtype=torch.float64);w=torch.randn(hkv*d,n,dtype=torch.float64,requires_grad=True)
        logits=torch.randn(hq,L,L,dtype=torch.float64);prob=logits.masked_fill(torch.ones(L,L,dtype=torch.bool).triu(1),-torch.inf).softmax(-1)
        delta=torch.randn(L,hq,d,dtype=torch.float64);delta[-1]=0
        v=(x@w.T).reshape(L,hkv,d).permute(1,0,2).repeat_interleave(2,dim=0)
        z=(prob@v).transpose(0,1);loss=(z*delta).sum();grad=torch.autograd.grad(loss,w)[0]
        got=effective(prob,x,delta.reshape(L,hq*d),mapping,hkv,'cpu',explicit=True)
        compare(got['S_source'],grad,1e-10);compare(got['S_effective'],grad,1e-10)
        compare(got['A_sum'],got['explicit_A_sum'],1e-10)
        a,g=normalize(got['A_sum'],got['G_blocks_sum'],1,L,hq)
        embedded=[]
        for t in range(L):
            for h,b in enumerate(mapping):
                v=torch.zeros(hkv*d,dtype=torch.float64);v[b*d:(b+1)*d]=delta[t,h];embedded.append(v)
        expected=torch.stack(embedded).T@torch.stack(embedded)/(L-1)
        compare(g,expected,1e-10);self.assertEqual(torch.count_nonzero(g[:d,d:]),0)
        self.assertEqual(a.shape,(n,n));self.assertEqual(g.shape,(hkv*d,hkv*d))

    def test_local_mlp_vjp_keeps_identity_residual(self):
        torch.manual_seed(91);norm=nn.LayerNorm(6);mlp=nn.Module();mlp.up_proj=nn.Linear(6,9);mlp.down_proj=nn.Linear(9,6)
        mlp.forward=lambda x:mlp.down_proj(mlp.up_proj(x).tanh())
        block=SimpleNamespace(mlp=mlp,post_attention_layernorm=norm,self_attn=SimpleNamespace(o_proj=nn.Linear(6,6,bias=False)))
        model=SimpleNamespace(model=SimpleNamespace(layers=[block]));r=torch.randn(1,7,6,requires_grad=True);seed=torch.randn_like(r)
        out=mlp(norm(r));expected=torch.autograd.grad(r+out,r,seed)[0]
        lam,delta=local_delta(model,0,r.detach(),seed,out.detach());compare(lam,expected,1e-5)
        compare(delta,expected@block.self_attn.o_proj.weight,1e-5)
        self.assertGreater(float((lam-seed).norm()),.1)

    def test_atomic_all_layers_checkpoint_crash_does_not_double_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);sums={i:dict(A_sum=torch.ones(3,3)*4,G_blocks_sum=torch.ones(2,2,2)*4) for i in LAYERS}
            checkpoint(root,'id',sums,4)
            next_sums={i:{k:v*2 for k,v in row.items()} for i,row in sums.items()}
            with patch('v_statistics.save_json',side_effect=RuntimeError('crash before pointer')):
                with self.assertRaisesRegex(RuntimeError,'crash'):checkpoint(root,'id',next_sums,8)
            count,got=restore(root,'id',{i:'cpu' for i in LAYERS});self.assertEqual(count,4)
            for i in LAYERS:torch.testing.assert_close(got[i]['A_sum'],sums[i]['A_sum'])
            checkpoint(root,'id',next_sums,8);count,got=restore(root,'id',{i:'cpu' for i in LAYERS});self.assertEqual(count,8)
            self.assertEqual(len(list((root/'statistics/progress').glob('generation_*.safetensors'))),2)

    def test_paired_bootstrap_uses_ratio_of_means_and_handles_degenerate_none(self):
        b=np.arange(1,17,dtype=float);a=.2*b;m=.3*b;s=.21*b;ix=np.random.default_rng(1).integers(0,16,(2000,16))
        r=paired(a,m,s,b,ix);self.assertAlmostEqual(r['vs_Marginal']['point'],.1)
        self.assertAlmostEqual(r['vs_Marginal']['low'],.1);self.assertTrue(r['vs_Sequence']['within_practical_band'])
        self.assertIsNone(paired(a,m,s,np.zeros(16),ix)['vs_Marginal']['point'])
        compare(torch.zeros(3),torch.zeros(3),1e-10)
        with self.assertRaises(RuntimeError):compare(torch.ones(3)*1e-13,torch.zeros(3),1e-10)


if __name__=='__main__':unittest.main(verbosity=2)
