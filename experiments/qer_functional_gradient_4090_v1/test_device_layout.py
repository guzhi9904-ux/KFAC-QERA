"""Small offline Llama regression; optional real two-GPU forward/backward test."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest
import torch
from device_layout import model_device_map, mapped_device, align_and_check_devices


def small_model():
    from transformers import LlamaConfig, LlamaForCausalLM
    config = LlamaConfig(vocab_size=19, hidden_size=32, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=128, use_cache=False, attention_dropout=0.)
    config._attn_implementation = 'eager'
    torch.manual_seed(4090)
    return LlamaForCausalLM(config).float().eval().requires_grad_(False)


class LayoutTests(unittest.TestCase):
    def test_global_and_nested_rotary_mapping(self):
        mapping = model_device_map()
        self.assertEqual(mapped_device('model.rotary_emb.inv_freq', mapping), torch.device('cuda:0'))
        self.assertEqual(mapped_device('model.layers.10.self_attn.rotary_emb.inv_freq', mapping), torch.device('cuda:0'))
        self.assertEqual(mapped_device('model.layers.31.self_attn.rotary_emb.inv_freq', mapping), torch.device('cuda:1'))
        with self.assertRaises(RuntimeError): mapped_device('unmapped.buffer', mapping)

    def test_nonpersistent_buffers_and_values(self):
        model = small_model(); mapping = model_device_map(4, 'cpu', 'cpu')
        rope = model.model.rotary_emb
        self.assertNotIn('model.rotary_emb.inv_freq', model.state_dict())
        before = rope.inv_freq.clone(); original = rope.original_inv_freq.clone()
        audit = align_and_check_devices(model, mapping)
        self.assertTrue(audit['passed'])
        self.assertTrue(torch.equal(before, rope.inv_freq))
        self.assertTrue(torch.equal(original, rope.original_inv_freq))
        model.register_buffer('unknown_nonpersistent', torch.zeros(1), persistent=False)
        with self.assertRaises(RuntimeError): align_and_check_devices(model, mapping)

    def test_wrong_device_and_meta_rejected(self):
        model = small_model(); mapping = model_device_map(4, 'cpu', 'cpu')
        mapping['lm_head'] = 'meta'
        with self.assertRaises(RuntimeError): align_and_check_devices(model, mapping)
        model.model.rotary_emb.inv_freq = torch.empty(4, device='meta')
        with self.assertRaises(RuntimeError): align_and_check_devices(model, model_device_map(4, 'cpu', 'cpu'))


def cuda_regression():
    from transformers import AutoModelForCausalLM
    assert torch.cuda.device_count() == 2, 'Two visible GPUs required for this small regression'
    # Use the exact same capture/recomputation path as the production runner.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'qer_teacher_kl_exp01'))
    from model_ops import capture, hidden_forward, recompute_suffix
    import math_ops as mo
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with tempfile.TemporaryDirectory(prefix='qer-rope-small-') as tmp:
        root = Path(tmp).resolve()
        assert root.parent == Path(tempfile.gettempdir()).resolve()
        small_model().save_pretrained(root)
        mapping = model_device_map(4)
        old_mapping = {k: v for k, v in mapping.items() if k != 'model.rotary_emb'}
        broken = AutoModelForCausalLM.from_pretrained(root, torch_dtype=torch.float32,
            attn_implementation='eager', local_files_only=True, low_cpu_mem_usage=True, device_map=old_mapping)
        reproduced = False
        try:
            with torch.no_grad(): hidden_forward(broken, torch.ones(1, 16, dtype=torch.long))
        except RuntimeError as exc:
            reproduced = 'same device' in str(exc) and 'cpu' in str(exc) and 'cuda' in str(exc)
            if not reproduced: raise
        del broken
        for i in range(2):
            with torch.cuda.device(i): torch.cuda.empty_cache()
        assert reproduced, 'The expected missing-shared-RoPE failure was not reproduced'
        model = AutoModelForCausalLM.from_pretrained(root, torch_dtype=torch.float32,
            attn_implementation='eager', local_files_only=True, low_cpu_mem_usage=True, device_map=mapping)
        model.eval(); model.requires_grad_(False); model.config.use_cache = False
        placement = align_and_check_devices(model, mapping)
        ids = torch.arange(16)[None] % 19
        checks = []
        for name in ('model.layers.0.self_attn.v_proj', 'model.layers.3.self_attn.q_proj'):
            target = model.get_submodule(name); target.weight.requires_grad_(True); answers = []
            for checkpoint in (False, True):
                with recompute_suffix(model, int(name.split('.')[2]), checkpoint), capture(target) as state:
                    hidden = hidden_forward(model, ids).to(model.lm_head.weight.device)
                    labels = ids[0, 1:]
                    seed = mo.sampled_seed(hidden.detach(), model.lm_head.weight, labels, chunk=4)
                    g, dw = torch.autograd.grad(hidden, (state['h'], target.weight), grad_outputs=seed)
                    s = g.reshape(16, -1).double().T@state['x'].reshape(16, -1).double()
                    error = mo.relative(s, dw.double())
                    assert error <= 1e-5
                    answers.append((hidden.detach().cpu(), g.detach().cpu()))
            forward_error = mo.relative(answers[0][0], answers[1][0])
            gradient_error = mo.relative(answers[0][1], answers[1][1])
            assert forward_error <= 1e-7 and gradient_error <= 1e-6
            target.weight.requires_grad_(False)
            checks.append(dict(module=name, checkpoint_forward_relative_error=forward_error,
                checkpoint_gradient_relative_error=gradient_error, S_vs_autograd_relative_error=error))
        print(json.dumps(dict(passed=True, original_failure_reproduced=True, placement=placement,
            checks=checks, model='random 4-layer hidden-size-32 Llama, 16 tokens',
            real_two_GPU_test=True, full_8B_pilot_executed=False), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--cuda', action='store_true'); args = parser.parse_args()
    if args.cuda: cuda_regression()
    else: unittest.main(argv=[sys.argv[0]], verbosity=2)
