"""CPU numerical/intervention/resume tests; server CUDA pilot remains required."""
import contextlib
import gc
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
import local_output_kl_v1 as exp


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(7, 64, dtype=torch.bfloat16)
        self.proj = torch.nn.Linear(64, 64, bias=False, dtype=torch.bfloat16)
        self.head = torch.nn.Linear(64, 7, bias=False, dtype=torch.bfloat16)

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(logits=self.head(self.proj(self.embedding(input_ids))))


class LocalTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(902)

    def fixture(self):
        teacher, student = Tiny().eval(), Tiny().eval()
        student.load_state_dict(teacher.state_dict())
        factors = {m: ((torch.randn(64, 64)*.01).bfloat16(),
                       (torch.randn(64, 64)*.01).bfloat16()) for m in exp.METHODS}
        ids = torch.arange(9).reshape(1, -1) % 7
        quant = (teacher.proj.weight.float()+.01).bfloat16()
        return teacher, student, factors, quant, ids, torch.ones_like(ids)

    def context(self, directory):
        t, s, factors, q, ids, mask = self.fixture()
        files, raw = {}, {}
        for m, (a, b) in factors.items():
            files[m] = {"proj": {"file": exp.single.atomic_tensors(directory/(m+'.safetensors'), {'A': a, 'B': b})}}
            raw[m] = {"proj": {"file": exp.single.atomic_tensors(directory/(m+'_raw.safetensors'),
                {'A_fp64': a.double(), 'B_fp64': b.double()})}}
        quant = exp.single.atomic_tensors(directory/'q.safetensors', {'weight_q': q})
        ctx = SimpleNamespace(output=directory/'result', identity='test', modules=('proj',),
            tasks={'proj': {'quant': quant, 'shape': [64,64]}}, rank64_checked=files,
            factors=raw, inputs=exp.core.Inputs())
        ctx.output.mkdir()
        data = {'input_ids': ids.repeat(3,1), 'attention_mask': mask.repeat(3,1)}
        return ctx, t, s, data

    def test_prespecified_coverage(self):
        self.assertEqual(len(exp.MODULES), 28)
        self.assertEqual(len(set(exp.MODULES)), 28)
        for layer in exp.LAYERS:
            self.assertEqual(sum(n.startswith(f'model.layers.{layer}.') for n in exp.MODULES), 7)
        self.assertTrue(set(exp.PILOT_MODULES) <= set(exp.MODULES))

    def test_actual_output_sse_excludes_final_position(self):
        ref = torch.randn(1,9,64).bfloat16()
        out = ref.clone(); out[:,:-1] += .1; out[:,-1] = 10000
        error, energy = exp.local_tokens(ref, out, 2)
        self.assertTrue(torch.equal(error, (out[:,:-1].double()-ref[:,:-1].double()).square().sum(-1)[0]))
        self.assertTrue(torch.equal(energy, ref[:,:-1].double().square().sum(-1)[0]))
        with self.assertRaises(ValueError): exp.local_tokens(ref, out[:,:,:2])

    def test_nll_is_shifted_fp32_ce_with_fp64_reduction(self):
        ids = torch.tensor([[0,1,2,3,4,5,6,0,1]])
        logits = torch.randn(1,9,7).bfloat16()
        actual = exp.nll_tokens(logits, ids, 2)
        expected = F.cross_entropy(logits[0,:-1].float(), ids[0,1:], reduction='none').double()
        self.assertTrue(torch.equal(actual, expected))
        logits[0,-1] = float('nan')
        self.assertTrue(torch.equal(actual, exp.nll_tokens(logits, ids)))

    def test_hook_sees_corrected_output_and_restores_on_exception(self):
        teacher, student, factors, q, ids, _ = self.fixture()
        before = {k: v.clone() for k,v in student.state_dict().items()}
        a,b = factors['full_gf']; x = student.embedding(ids)
        expected = F.linear(x, q)+(x@a)@b
        with self.assertRaisesRegex(RuntimeError, 'injected'):
            with exp.replacement(student.proj, q, a, b):
                with exp.capture(student.proj) as observed:
                    value = student.proj(x)
                self.assertTrue(torch.equal(observed['y'], expected))
                self.assertTrue(torch.equal(value, expected))
                raise RuntimeError('injected')
        self.assertFalse(student.proj._forward_hooks)
        self.assertTrue(all(torch.equal(v, before[k]) for k,v in student.state_dict().items()))

    def test_complete_window_keeps_common_background_and_matches_direct_kl(self):
        teacher, student, factors, q, ids, mask = self.fixture()
        before = {k: v.clone() for k,v in student.state_dict().items()}
        values, maximum = exp.compare_window(teacher, student, 'proj', factors, q, ids, mask, True)
        self.assertLess(maximum, 1e-9)
        exp.tensor_rows(values, 0, 8)
        with torch.no_grad():
            a,b = factors['full_gf']
            with exp.replacement(student.proj, q, a, b):
                sl = student(ids).logits.double()[:,:-1]
            tl = teacher(ids).logits.double()[:,:-1]
            expected = (tl.softmax(-1)*(tl.log_softmax(-1)-sl.log_softmax(-1))).sum(-1)[0]
        self.assertTrue(torch.allclose(values['full_gf_kl'], expected, atol=2e-15, rtol=1e-10))
        self.assertTrue(all(torch.equal(v, before[k]) for k,v in student.state_dict().items()))
        self.assertFalse(teacher.proj._forward_hooks)
        self.assertFalse(student.proj._forward_hooks)

    def test_contaminated_background_rejected(self):
        teacher, student, factors, q, ids, mask = self.fixture()
        with torch.no_grad(): student.embedding.weight.add_(1)
        with self.assertRaisesRegex(RuntimeError, 'control'):
            exp.compare_window(teacher, student, 'proj', factors, q, ids, mask, True)
        self.assertFalse(student.proj._forward_hooks)
        with torch.no_grad(): student.proj.weight.add_(1)
        with self.assertRaisesRegex(RuntimeError, 'background'):
            exp.compare_window(teacher, student, 'proj', factors, q, ids, mask, True)

    def test_resume_triplet_checkpoint_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()), \
             patch.object(exp, 'WINDOWS', 3), patch.object(exp, 'LENGTH', 9):
            ctx,t,s,data = self.context(Path(tmp))
            before = exp.deployment(s)
            with patch.object(exp.frozen, 'windows', return_value=data):
                stop = exp.prev.Budget(10, 1)
                with self.assertRaises(exp.prev.Paused): exp.evaluate_module(ctx,t,s,'proj',stop)
                self.assertEqual(len(exp.read_state(ctx,'proj')['records']), 1)
                self.assertEqual(before, exp.deployment(s))
                with patch.object(exp, 'compare_window', wraps=exp.compare_window) as measured:
                    exp.evaluate_module(ctx,t,s,'proj',exp.prev.Budget(10,None))
                    self.assertEqual(measured.call_count, 2)
                del measured
                gc.collect()  # Release mock-held safetensors mappings before Windows cleanup.
                with patch.object(exp, 'compare_window', side_effect=AssertionError('completed rerun')):
                    exp.evaluate_module(ctx,t,s,'proj',exp.prev.Budget(10,None))
            exp.summarize(ctx)
            status = exp.core.read_json(ctx.output/'status.json')
            self.assertTrue(status['complete']); self.assertEqual(status['completed_arms'], 3)
            self.assertEqual(before, exp.deployment(s))
            exp.pack_summary(ctx)
            self.assertTrue((ctx.output/(exp.VERSION+'_summary.tar.gz')).exists())

    def test_failed_method_does_not_commit_partial_window(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()), \
             patch.object(exp,'WINDOWS',3), patch.object(exp,'LENGTH',9):
            ctx,t,s,data = self.context(Path(tmp)); original = exp.local_tokens
            calls = 0
            def fail_second(*args):
                nonlocal calls
                calls += 1
                if calls == 2: raise RuntimeError('injected method failure')
                return original(*args)
            before = exp.deployment(s)
            with patch.object(exp.frozen,'windows',return_value=data), patch.object(exp,'local_tokens',side_effect=fail_second):
                with self.assertRaisesRegex(RuntimeError,'injected'):
                    exp.evaluate_module(ctx,t,s,'proj',exp.prev.Budget(10,None))
            self.assertFalse(exp.state_path(ctx,'proj').exists())
            self.assertEqual(before, exp.deployment(s))
            self.assertFalse(s.proj._forward_hooks)

    def test_resume_rejects_tampered_payload_and_identity(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()), \
             patch.object(exp,'WINDOWS',3), patch.object(exp,'LENGTH',9):
            ctx,t,s,data = self.context(Path(tmp))
            with patch.object(exp.frozen,'windows',return_value=data):
                with self.assertRaises(exp.prev.Paused):
                    exp.evaluate_module(ctx,t,s,'proj',exp.prev.Budget(10,1))
            state = exp.read_state(ctx,'proj'); ctx.identity = 'different'
            with self.assertRaisesRegex(RuntimeError,'identity'): exp.read_state(ctx,'proj')
            ctx.identity = 'test'
            file = Path(state['records'][0]['file']['path'])
            with file.open('r+b') as f: f.seek(-1,2); f.write(b'!')
            ctx.inputs = exp.core.Inputs()
            with self.assertRaisesRegex(RuntimeError,'hash/size'): exp.read_state(ctx,'proj')

    def test_paired_stats_direction_and_token_count(self):
        rows = [{'tokens': i+1, 'full_gf_kl_sum': 2*(i+1), 'full_gi_kl_sum': 3*(i+1)} for i in range(16)]
        stats = exp.paired_stats(rows,'full_gi','kl',draws=100)
        self.assertEqual(stats, {'delta':-1.,'ci95_low':-1.,'ci95_high':-1.})

    def test_same_norm_can_have_different_downstream_kl(self):
        # Small error in a sensitive direction can be worse than a larger error elsewhere.
        ref = torch.zeros(1,3,2)
        gi = ref.clone(); gi[:,:,0] = .1
        gf = ref.clone(); gf[:,:,1] = .2
        downstream = torch.tensor([[10.,-10.],[.1,-.1]])
        rgi,_ = exp.local_tokens(ref,gi); rgf,_ = exp.local_tokens(ref,gf)
        kgi,_ = exp.frozen.kl_tokens(ref@downstream,gi@downstream)
        kgf,_ = exp.frozen.kl_tokens(ref@downstream,gf@downstream)
        self.assertGreater(float(rgf.mean()),float(rgi.mean()))
        self.assertLess(float(kgf.mean()),float(kgi.mean()))


if __name__ == '__main__': unittest.main()
