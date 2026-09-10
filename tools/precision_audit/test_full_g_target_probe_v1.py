"""CPU fixtures only: no server artifacts or model downloads."""
import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file
import full_g_target_probe_v1 as probe


class ProbeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        torch.set_num_threads(2)

    def test_re_solve_and_optimal_tail(self):
        error = torch.randn(12, 10)
        a = torch.diag(torch.linspace(.5, 2, 12))
        z = torch.randn(10, 10)
        g = z @ z.T + torch.eye(10)
        u,s,vh = torch.linalg.svd(a @ error @ g, full_matrices=True)
        left = torch.linalg.solve(a, u[:, :5])
        right = torch.linalg.solve(g.T, (s[:5,None]*vh[:5]).T).T
        rows = probe.numerical_method(error, a, g, left, right, ranks=(1,3,5))
        self.assertEqual(len(rows), 3)
        for row in rows:
            self.assertLess(abs(row['saved_excess_over_tail_over_before']), 1e-5)
            self.assertLess(abs(row['resolved_excess_over_tail_over_before']), 1e-12)
            self.assertLess(row['a_inverse_residual_fp64'], 1e-12)
            self.assertLess(row['a_inverse_solution_fp32_vs_fp64'], 1e-5)

    def test_large_compensation_is_not_hidden(self):
        e = torch.eye(4)
        a = torch.diag(torch.tensor([1e-5,1.,1.,1.]))
        g = torch.eye(4)
        left = torch.zeros(4,1)
        left[0,0] = 100
        right = torch.ones(1,4)
        row = probe.metrics(e,a,g,left,right,1)
        self.assertGreater(row['correction_over_error_norm'], 10)

    def test_diagonal_uses_original_definition(self):
        raw = torch.tensor([0.,1.,2.],dtype=torch.float64)
        root = probe.diagonal_root(raw)
        torch.testing.assert_close(root, torch.tensor([.001,1.,2**.5]))
        with self.assertRaises(RuntimeError):
            probe.diagonal_root(torch.zeros(3))

    def test_hybrid_changes_only_target_without_mutating_inputs(self):
        base = {'x': {'quant': 'q', 'correction': 'fg'}, 'y': {'quant': 'z', 'correction': 'fg2'}}
        replacement = {'x': {'quant': 'q', 'correction': 'gd'}}
        result = probe.route_hybrid(base,replacement,'x')
        self.assertEqual(result['y'],base['y'])
        self.assertEqual(base['x']['correction'],'fg')
        self.assertEqual(result['x']['correction'],'gd')
        with self.assertRaises(RuntimeError):
            probe.route_hybrid(base,{'x': {'quant': 'other', 'correction': 'gd'}},'x')
        with self.assertRaises(RuntimeError):
            probe.route_hybrid(base,base,'x')

    def test_factor_hook_exact_order_and_rank(self):
        x = torch.randn(2,5,7)
        left, right = torch.randn(7,6), torch.randn(6,4)
        original = torch.randn(2,5,4)
        hook = probe.correction_hook(left[:,:3],right[:3])
        torch.testing.assert_close(hook(None,(x,),original),original+(x @ left[:,:3]) @ right[:3])

    def test_control_blocks_drift_and_incomplete(self):
        rows = [{'window':i,'tokens':2047,'nll_sum':2047.} for i in range(138)]
        self.assertEqual(probe.control_check(rows,rows,1e-5)['status'],'PASS')
        changed = [dict(r,nll_sum=3000.) for r in rows]
        self.assertEqual(probe.control_check(changed,rows,1e-5)['status'],'FAIL')
        with self.assertRaises(RuntimeError):
            probe.control_check(rows[:8],rows,1e-5)

    def test_checkpoint_validity(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'eval.json'
            rows=[{'window':i,'tokens':2047,'nll_sum':10.} for i in range(8)]
            probe.core.atomic_json(path,{'probe_identity':'abc','records':rows,'complete':False})
            self.assertEqual(probe.get_records(path,'abc'),rows)
            with self.assertRaises(RuntimeError):
                probe.get_records(path,'changed')
            rows[0]['window']=1
            with self.assertRaises(RuntimeError):
                probe.valid_records(rows,138)

    def test_evaluation_batch_resume_and_source_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            source=root/'source'; source.mkdir()
            output=root/'output'; output.mkdir()
            def save(name,tensors):
                path=source/name
                save_file(tensors,str(path))
                return {'path':str(path),'bytes':path.stat().st_size,'sha256':probe.core.sha(path)}
            artifacts={'proj': {'quant':save('q.safetensors',{'weight_q':torch.ones(3,4).bfloat16()}),
                'correction':save('ab.safetensors',{'A':torch.ones(4,2),'B':torch.ones(2,3)})}}
            source_hashes={p.name:probe.core.sha(p) for p in source.iterdir()}
            calls=[]
            class Model(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.proj=torch.nn.Linear(4,3,bias=False).bfloat16()
                    self.hf_device_map={'proj':0}
                def forward(self,input_ids,attention_mask,use_cache):
                    calls.append(int(input_ids[0,0]))
                    x=torch.ones(input_ids.shape+(4,),dtype=torch.bfloat16)
                    return SimpleNamespace(logits=self.proj(x))
            legacy=SimpleNamespace(load_model=lambda *args:Model(),_input_device=lambda m:'cpu',
                _chunked_window_nll=lambda logits,ids,mask,chunk:[(float(t[0].sum()),2047) for t in logits])
            ids=torch.arange(138)[:,None].expand(138,2048).clone()
            windows={'input_ids':ids,'attention_mask':torch.ones_like(ids)}
            manifest={'payload':{'config':{'eval_batch_size':8,'eval_max_memory':{},'eval_ce_chunk_tokens':256},'source_config':{}}}
            inputs=probe.core.Inputs()
            with contextlib.redirect_stdout(io.StringIO()), patch.object(torch.cuda,'empty_cache'):
                with self.assertRaises(probe.Paused):
                    probe.evaluate_one(legacy,manifest,artifacts,2,'TEST',windows,inputs,output,'id',probe.Budget(batches=1))
                self.assertEqual(len(probe.get_records(output/'TEST.json','id')),8)
                rows=probe.evaluate_one(legacy,manifest,artifacts,2,'TEST',windows,inputs,output,'id',probe.Budget())
                before_calls=len(calls)
                probe.evaluate_one(legacy,manifest,artifacts,2,'TEST',windows,inputs,output,'id',probe.Budget())
                self.assertEqual(before_calls,len(calls))
            self.assertEqual(calls,list(range(0,138,8)))
            self.assertEqual(len(rows),138)
            self.assertEqual(source_hashes,{p.name:probe.core.sha(p) for p in source.iterdir()})

    def test_summary_only_includes_completed_pairs(self):
        with tempfile.TemporaryDirectory() as folder:
            dest=Path(folder)
            probe.summarize_evaluation(dest,'id')
            self.assertEqual(probe.core.read_json(dest/'status.json')['completed_ranks'],[])
            probe.core.atomic_json(dest/'paired_r8.json',{'probe_identity':'id','rank':8,'control_ppl':12.,'hybrid_ppl':10.,
                'delta_ppl':-2.,'delta_mean_nll':-.1,'improved_windows':80,'records':[]})
            probe.summarize_evaluation(dest,'id')
            self.assertEqual(probe.core.read_json(dest/'status.json')['completed_ranks'],[8])
            with self.assertRaises(RuntimeError):
                probe.summarize_evaluation(dest,'changed')

    def test_diagnostics_all_methods_and_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            source=root/'source'; source.mkdir()
            output=root/'output'; output.mkdir()
            def record(path):
                return {'path':str(path),'bytes':path.stat().st_size,'sha256':probe.core.sha(path)}
            def save(name,tensors):
                path=source/(name+'.safetensors')
                save_file(tensors,str(path))
                return record(path)
            size=72
            weight=torch.randn(size,size).bfloat16()
            quant=(weight.float()*.9).bfloat16()
            error=(weight.float()-quant.float()).T
            a=torch.diag(torch.linspace(.5,2,size))
            raw=torch.diag(torch.linspace(.2,2,size).double())
            gf,_=probe.core.full_root(raw,256*2047)
            gd=torch.diag(probe.diagonal_root(raw.diagonal()))
            state={'windows_completed':256,'prediction_tokens':256*2047,
                   'file':save('original_dg',{probe.TARGET:raw.diagonal().contiguous()})}
            state_path=source/'dg_state.json'; probe.core.atomic_json(state_path,state)
            manifest={'payload':{'full_g_protocol':{'diagonal_g_state':state,'diagonal_g_state_file':record(state_path)}}}
            bindings={'root':save('roots',{'full':a}),
                'gram':save('raw',{'gram':raw,'diagonal':raw.diagonal().contiguous()}),
                'model':save('model',{probe.TARGET+'.weight':weight}),
                'quant':save('quant',{'weight_q':quant}), 'baseline_factors':{}}
            for method,g in zip(probe.METHODS,[torch.eye(size),gd,gf]):
                u,s,vh=torch.linalg.svd(a @ error @ g)
                l=torch.linalg.solve(a,u[:,:64]).contiguous()
                b=torch.linalg.solve(g.T,(s[:64,None]*vh[:64]).T).T.contiguous()
                bindings['baseline_factors'][method]=save(method,{'A':l,'B':b})
            hashes={p.name:probe.core.sha(p) for p in source.iterdir()}
            with contextlib.redirect_stdout(io.StringIO()), patch.object(torch.cuda,'empty_cache'):
                probe.diagnostics(manifest,bindings,probe.core.Inputs(),'cpu',output,'id',probe.Budget())
                probe.diagnostics(manifest,bindings,probe.core.Inputs(),'cpu',output,'id',probe.Budget())
            self.assertEqual(probe.core.read_json(output/'diagnostics/status.json')['rows'],12)
            self.assertEqual(hashes,{p.name:probe.core.sha(p) for p in source.iterdir()})
            self.assertTrue((output/'diagnostics/comparison.csv').exists())


if __name__=='__main__':
    unittest.main()
