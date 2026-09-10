"""Synthetic CPU tests; never read production data or modify frozen helpers."""
import contextlib
import copy
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file
import full_a_all_precision_r8_v1 as exp


class AllPrecisionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41); torch.set_num_threads(2)

    def save(self,folder,name,tensors):
        path=folder/(name+'.safetensors')
        save_file({k:v.contiguous() for k,v in tensors.items()},str(path))
        return exp.single.file_record(path)

    def context(self,root,d_in=72,d_out=72):
        source=root/'source'; source.mkdir()
        output=root/'new'; output.mkdir()
        tasks={}; dg={}
        for name in ('one','two'):
            w=torch.randn(d_out,d_in).bfloat16()
            q=(w.float()*.9).bfloat16()
            a=torch.diag(torch.linspace(.4,2,d_in))
            x=torch.randn(d_out,d_out).double()
            gram=x.T@x+torch.eye(d_out).double()
            dg[name]=gram.diagonal().clone()
            b={'shape':[d_out,d_in],'root':self.save(source,name+'_root',{'full':a}),
               'model':self.save(source,name+'_model',{name+'.weight':w}),
               'quant':self.save(source,name+'_q',{'weight_q':q}),
               'gram':self.save(source,name+'_gram',{'gram':gram,'diagonal':dg[name]}),'baseline_factors':{}}
            for method in exp.METHODS:
                if method=='full_gi': g=torch.eye(d_out)
                elif method=='full_gd': g=torch.diag(exp.previous.diagonal_root(dg[name]))
                else: g,_=exp.core.full_root(gram,256*2047)
                e=(w.float()-q.float()).T
                u,s,vh=torch.linalg.svd(a@e@g)
                l=torch.linalg.solve(a,u[:,:64]); r=torch.linalg.solve(g.T,(s[:64,None]*vh[:64]).T).T
                b['baseline_factors'][method]=self.save(source,name+'_'+method,{'A':l,'B':r})
            tasks[name]=b
        return SimpleNamespace(tasks=tasks,dg=dg,output=output,inputs=exp.core.Inputs(),identity='id',device='cpu')

    def test_helper_frozen(self):
        self.assertEqual(exp.core.sha(exp.single.__file__),exp.SINGLE_SHA)
        self.assertEqual(exp.core.sha(exp.previous.__file__),exp.single.PREVIOUS_SHA)

    def test_rectangular_gate_and_down_orientations(self):
        for d_in,d_out in ((72,88),(88,72)):
            with tempfile.TemporaryDirectory() as folder:
                ctx=self.context(Path(folder),d_in,d_out)
                with contextlib.redirect_stdout(io.StringIO()),patch.object(exp,'memory',return_value={}):
                    exp.solve_all(ctx,exp.Budget(),['one'])
                    for method in exp.METHODS:
                        record=exp.solved_record(ctx.output,method,'one',ctx.tasks['one'],'id',ctx.inputs)
                        values=ctx.inputs.tensors(record['file'])
                        self.assertEqual(values['A_fp64'].shape,(d_in,64))
                        self.assertEqual(values['B_fp64'].shape,(64,d_out))
                        self.assertLess(abs(record['relative_tail_difference']),1e-9)
                        del values

    def test_g_roots_original_definitions_and_mismatch_gate(self):
        with tempfile.TemporaryDirectory() as folder:
            ctx=self.context(Path(folder)); b=ctx.tasks['one']; dg=ctx.dg['one']
            with contextlib.redirect_stdout(io.StringIO()):
                gi,_=exp.make_g_root('full_gi',b,dg,ctx.inputs,'cpu')
                gd,_=exp.make_g_root('full_gd',b,dg,ctx.inputs,'cpu')
                gf,info=exp.make_g_root('full_gf',b,dg,ctx.inputs,'cpu')
                self.assertTrue(torch.equal(gi,torch.eye(72)))
                self.assertTrue(torch.equal(gd,torch.diag(exp.previous.diagonal_root(dg))))
                expected,_=exp.core.full_root(ctx.inputs.tensors(b['gram'])['gram'],256*2047)
                self.assertTrue(torch.equal(gf,expected))
                self.assertEqual(info['original_dg_drift'],0)
                with self.assertRaises(RuntimeError): exp.make_g_root('full_gf',b,dg*2,ctx.inputs,'cpu')

    def test_solve_each_method_resume_and_source_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); ctx=self.context(root)
            before={p.name:exp.core.sha(p) for p in (root/'source').iterdir()}
            with contextlib.redirect_stdout(io.StringIO()),patch.object(exp,'memory',return_value={'test':'CPU'}):
                stop=exp.Budget(solves=2)
                with self.assertRaises(exp.previous.Paused): exp.solve_all(ctx,stop)
                status=exp.write_solve_summary(ctx)
                self.assertEqual(status['completed'],{'full_gi':1,'full_gd':1,'full_gf':0})
                exp.solve_all(ctx,exp.Budget())
                self.assertEqual(exp.write_solve_summary(ctx)['status'],'COMPLETE')
                with patch.object(exp.single,'solve_fp64',side_effect=AssertionError('must not repeat')):
                    exp.solve_all(ctx,exp.Budget())
                for name,b in ctx.tasks.items():
                    hashes=[]
                    for method in exp.METHODS:
                        rec=exp.solved_record(ctx.output,method,name,b,ctx.identity,ctx.inputs)
                        hashes.append(rec['root_bits']['a'])
                        self.assertLess(abs(rec['relative_tail_difference']),1e-9)
                        self.assertEqual(rec['rank'],8)
                    self.assertEqual(hashes[0],hashes[1]); self.assertEqual(hashes[1],hashes[2])
            self.assertEqual(before,{p.name:exp.core.sha(p) for p in (root/'source').iterdir()})

    def test_failure_not_committed_and_no_fallback(self):
        with tempfile.TemporaryDirectory() as folder:
            ctx=self.context(Path(folder))
            with contextlib.redirect_stdout(io.StringIO()),patch.object(exp.single,'solve_fp64',side_effect=RuntimeError('singular')) as solve:
                with self.assertRaisesRegex(RuntimeError,'singular'):
                    exp.prepare_one('one','full_gi',ctx.tasks['one'],ctx.dg['one'],ctx.inputs,ctx.output,'id','cpu')
                self.assertEqual(solve.call_count,1)
                self.assertFalse(exp.factor_paths(ctx.output,'full_gi','one')[1].exists())

    def test_binding_tamper_and_incomplete_route(self):
        with tempfile.TemporaryDirectory() as folder:
            ctx=self.context(Path(folder))
            with contextlib.redirect_stdout(io.StringIO()),patch.object(exp,'memory',return_value={}):
                with self.assertRaises(RuntimeError): exp.routes(ctx,'full_gi','FP64')
                exp.prepare_one('one','full_gi',ctx.tasks['one'],ctx.dg['one'],ctx.inputs,ctx.output,'id','cpu')
                with self.assertRaises(RuntimeError):
                    exp.solved_record(ctx.output,'full_gi','one',ctx.tasks['one'],'wrong',ctx.inputs)
                path=exp.factor_paths(ctx.output,'full_gi','one')[0]
                with path.open('wb') as stream: stream.write(b'corrupt')
                with self.assertRaises(RuntimeError):
                    exp.solved_record(ctx.output,'full_gi','one',ctx.tasks['one'],'id',exp.core.Inputs())

    def test_parameter_and_coverage_invariance(self):
        old={'parameter_bits':{'w':'a'},'buffer_bits':{},'device_map':{'one':0},'correction_bits':{'one':'a','two':'b'}}
        new=copy.deepcopy(old); new['correction_bits']={'one':'new','two':'new2'}
        exp.assert_common_deployment(old,new)
        new['parameter_bits']={'w':'b'}
        with self.assertRaises(RuntimeError): exp.assert_common_deployment(old,new)
        new=copy.deepcopy(old); del new['correction_bits']['two']
        with self.assertRaises(RuntimeError): exp.assert_common_deployment(old,new)

    def test_failed_control_blocks_following_methods(self):
        with tempfile.TemporaryDirectory() as folder:
            ctx=SimpleNamespace(output=Path(folder),identity='id',manifest={'payload':{'config':{'control_ppl_tolerance':.01}}})
            ctx.references={m:{'records':[{'window':i,'tokens':2047,'nll_sum':2047.} for i in range(138)]} for m in exp.METHODS}
            exp.eval_folder(ctx,'full_gi').mkdir(parents=True)
            rows=[{'window':i,'tokens':2047,'nll_sum':4094.} for i in range(138)]
            with contextlib.redirect_stdout(io.StringIO()),patch.object(exp,'evaluate',return_value={'records':rows}) as evaluate:
                with self.assertRaises(RuntimeError): exp.controls(ctx,None,exp.Budget())
                self.assertEqual(evaluate.call_count,1)
                self.assertEqual(evaluate.call_args.args[3],'OLD')

    def test_all_six_evaluations_resume_and_compare(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); ctx=self.context(root)
            with contextlib.redirect_stdout(io.StringIO()),patch.object(exp,'memory',return_value={}):
                exp.solve_all(ctx,exp.Budget())
            source=root/'source'
            data=self.save(source,'windows',{'input_ids':torch.zeros(138,2048,dtype=torch.long),
                                             'attention_mask':torch.ones(138,2048,dtype=torch.long)})
            ctx.manifest={'payload':{'config':{'eval_max_memory':{},'control_ppl_tolerance':1e-5},'source_config':{},'data':{'wikitext2':data}}}
            calls=[]
            class Model(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.one=torch.nn.Linear(72,72,bias=False).bfloat16()
                    self.two=torch.nn.Linear(72,72,bias=False).bfloat16()
                    self.register_buffer('fixed',torch.tensor([1.]))
                    self.hf_device_map={'one':0,'two':1}
                def forward(self,input_ids,attention_mask,use_cache):
                    calls.append(len(input_ids))
                    # Expand identical token logits; exercise both actual correction hooks.
                    x=torch.ones(len(input_ids),1,72).bfloat16()*.01
                    logits=self.two(self.one(x)).float().expand(-1,input_ids.shape[1],-1)
                    return SimpleNamespace(logits=logits)
            legacy=SimpleNamespace(load_model=lambda *args:Model(),_input_device=lambda m:'cpu',
                                   _chunked_window_nll=lambda l,i,m,c:exp.single.nll_with_tokens(l,i,m,c)[0])
            original={p.name:exp.core.sha(p) for p in source.iterdir()}
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(exp.previous.Paused): exp.evaluate(ctx,legacy,'full_gi','OLD',exp.Budget(batches=1))
                self.assertEqual(len(exp.single.read_state(exp.eval_folder(ctx,'full_gi'),'OLD','id',ctx.inputs)['records']),8)
                ctx.references={}
                for m in exp.METHODS:
                    ctx.references[m]=exp.evaluate(ctx,legacy,m,'OLD',exp.Budget())
                exp.controls(ctx,legacy,exp.Budget())
                for m in exp.METHODS: exp.evaluate(ctx,legacy,m,'FP64',exp.Budget())
                exp.verify_evaluations(ctx)
                result=exp.summarize(ctx)
                self.assertEqual(result['status'],'COMPLETE')
                self.assertEqual(len(result['completed_configurations']),6)
                self.assertEqual(len(calls),18*6)
                for m in exp.METHODS:
                    exp.evaluate(ctx,legacy,m,'OLD',exp.Budget())
                    exp.evaluate(ctx,legacy,m,'FP64',exp.Budget())
                self.assertEqual(len(calls),18*6)
            self.assertEqual(original,{p.name:exp.core.sha(p) for p in source.iterdir()})
            import csv
            with (ctx.output/'evaluation/comparisons.csv').open() as stream:
                comparisons=list(csv.DictReader(stream))
            self.assertEqual(len(comparisons),6)
            self.assertIn('full_gf_FP64_minus_full_gd_FP64',{r['comparison'] for r in comparisons})
            # A forged deployment route cannot be silently adopted.
            path=exp.eval_folder(ctx,'full_gi')/'deployment_FP64.json'
            record=exp.core.read_json(path); record['route_sha256']='bad'; exp.core.atomic_json(path,record)
            with contextlib.redirect_stdout(io.StringIO()),self.assertRaises(RuntimeError): exp.verify_evaluations(ctx)


if __name__=='__main__': unittest.main()
