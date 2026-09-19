"""Bounded CPU Llama test: real eval-mode checkpoint recomputation and gradients."""
import unittest
import torch
from transformers import LlamaConfig,LlamaForCausalLM
from math_ops import relative,sampled_seed
from model_ops import capture,recompute_suffix,hidden_forward


class ModelTest(unittest.TestCase):
    def test_explicit_checkpoint_and_hook_scope(self):
        torch.manual_seed(613);torch.set_num_threads(2)
        config=LlamaConfig(vocab_size=17,hidden_size=32,intermediate_size=64,num_hidden_layers=4,
                           num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=64,
                           attention_dropout=0.,use_cache=False)
        config._attn_implementation="eager"
        model=LlamaForCausalLM(config).float().eval().requires_grad_(False)
        ids=torch.randint(17,(1,16));labels=torch.randint(17,(15,))
        target=model.model.layers[1].self_attn.v_proj
        answers=[]
        for enabled in (False,True):
            with recompute_suffix(model,1,enabled),capture(target,True) as state:
                hidden=hidden_forward(model,ids)
                seed=sampled_seed(hidden.detach(),model.lm_head.weight,labels,4)
                grad=torch.autograd.grad(hidden,state["h"],grad_outputs=seed)[0]
                answers.append((hidden.detach(),grad))
        self.assertLess(relative(answers[0][0],answers[1][0]),1e-7)
        self.assertLess(relative(answers[0][1],answers[1][1]),1e-6)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertFalse(target._forward_hooks)


if __name__=="__main__":unittest.main(verbosity=2)
