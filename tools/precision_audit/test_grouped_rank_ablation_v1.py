"""CPU numerical and transaction tests. No model download or GPU claim."""
import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file
import grouped_rank_ablation_v1 as exp


class GroupTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(311)

    def context(self, root, count=138, length=2048):
        data=root/'windows.safetensors'
        save_file({'input_ids':torch.zeros(count,length,dtype=torch.long),
                   'attention_mask':torch.ones(count,length,dtype=torch.long)},str(data))
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy=torch.nn.Parameter(torch.zeros(1,dtype=torch.bfloat16))
                self.hf_device_map={'model':0}
            def forward(self,input_ids,**kwargs):
                return SimpleNamespace(logits=torch.zeros(*input_ids.shape,3))
        legacy=SimpleNamespace(load_model=lambda *args:Model(),_input_device=lambda m:'cpu',
            _chunked_window_nll=lambda logits,ids,mask,n:exp.single.nll_with_tokens(logits,ids,mask,n)[0])
        expected={'deployment':{'parameter_bits':{'dummy':exp.single.tensor_record(Model().dummy)},'buffer_bits':{}}}
        return SimpleNamespace(output=root,identity='test',inputs=exp.core.Inputs(),tasks={},legacy=legacy,
            rank64_deployments={'teacher':expected,'wq':expected},
            device='cpu',manifest={'payload':{'data':{'wikitext2':exp.single.file_record(data)},
            'source_config':{},'config':{'eval_max_memory':{}}}}),Model

    def test_configurations_and_disjoint_groups(self):
        self.assertEqual(len(exp.CONFIGS),38)
        self.assertEqual(len({exp.label(*c) for c in exp.CONFIGS}),38)
        self.assertEqual([i for group in exp.GROUPS for i in group],list(range(64)))
        with self.assertRaises(ValueError): exp.label('teacher',1)

    def test_plain_svd_same_residual_and_tail_both_orientations(self):
        for di,do in ((72,88),(88,72)):
            w=torch.randn(do,di).bfloat16(); q=(w.float()+.05*torch.randn(do,di)).bfloat16()
            e=(w.float()-q.float()).T
            values,cert=exp.plain_solve(e)
            l,b=values['A_fp64'],values['B_fp64']
            self.assertEqual(l.shape,(di,64));self.assertEqual(b.shape,(64,do))
            self.assertLess(cert['relative_tail_error'],1e-12)
            self.assertTrue(torch.allclose(l.T@l,torch.eye(64).double(),atol=1e-12,rtol=0))
            self.assertTrue(torch.equal(values['A'],l.bfloat16()))
            self.assertTrue(torch.equal(values['B'],b.bfloat16()))
            self.assertTrue(bool((values['singular_values'][1:]<=values['singular_values'][:-1]).all()))
        with self.assertRaises(ValueError):exp.plain_solve(torch.zeros(63,72))

    def test_plain_checkpoint_resume_source_immutability_and_tamper(self):
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp);ctx,_=self.context(root)
            w=torch.randn(72,80).bfloat16();q=(w.float()+.1*torch.randn(72,80)).bfloat16()
            save_file({'proj.weight':w},str(root/'w.safetensors'))
            save_file({'weight_q':q},str(root/'q.safetensors'))
            ctx.tasks={'proj':{'shape':[72,80],'model':exp.single.file_record(root/'w.safetensors'),
                              'quant':exp.single.file_record(root/'q.safetensors')}}
            before={k:exp.core.sha(root/(k+'.safetensors')) for k in ('w','q')}
            rec=exp.prepare_plain(ctx,'proj')
            with patch.object(exp,'plain_solve',side_effect=AssertionError('must resume')):
                self.assertEqual(rec,exp.prepare_plain(ctx,'proj'))
            for k,h in before.items():self.assertEqual(h,exp.core.sha(root/(k+'.safetensors')))
            with Path(rec['file']['path']).open('r+b') as f:
                f.seek(-1,2);f.write(b'!')
            ctx.inputs=exp.core.Inputs()
            with self.assertRaises(RuntimeError):exp.read_plain(ctx,'proj')

    def test_mask_independent_preserves_shapes_and_original_bits(self):
        a=torch.randn(72,64).bfloat16();b=torch.randn(64,80).bfloat16()
        before=(exp.single.tensor_record(a),exp.single.tensor_record(b))
        for group in range(9):
            l,r=exp.masked_factors(a,b,group)
            self.assertEqual(l.shape,a.shape);self.assertEqual(r.shape,b.shape)
            expected=b.clone()
            if group:expected[8*(group-1):8*group]=0
            self.assertTrue(torch.equal(r,expected))
            self.assertTrue(torch.equal(l,a))
        self.assertEqual(before,(exp.single.tensor_record(a),exp.single.tensor_record(b)))
        with self.assertRaises(ValueError):exp.masked_factors(a,b,9)
        with self.assertRaises(ValueError):exp.masked_factors(a.float(),b,1)

    def test_import_existing_factors_without_new_svd_or_certificate_solve(self):
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp);ctx,_=self.context(root)
            values,_=exp.plain_solve(torch.randn(72,80))
            raw=exp.single.atomic_tensors(root/'raw.safetensors',{'A_fp64':values['A_fp64'],'B_fp64':values['B_fp64']})
            saved=exp.single.atomic_tensors(root/'checked.safetensors',{k:values[k] for k in ('A','B')})
            rec={'experiment_identity':'source','module':'proj','method':'full_gi','source_factor':raw,
                 'status':'PASS','metrics':[{'rank':r} for r in exp.dual.RANKS],
                 'bits':{k:exp.single.tensor_record(values[k]) for k in ('A','B')},'file':saved}
            ctx.tasks={'proj':{'shape':[80,72]}};ctx.factors={'full_gi':{'proj':{'file':raw}}}
            ctx.rank64_checked={'full_gi':{'proj':rec}};ctx.rank64_checked_metadata={'full_gi':{'proj':{'sha256':'source'}}}
            before=exp.core.sha(Path(saved['path']))
            with patch.object(torch.linalg,'svd',side_effect=AssertionError('no SVD')), \
                 patch.object(exp.dual,'check_one',side_effect=AssertionError('no repeated solve')):
                result=exp.import_checked(ctx,'full_gi','proj')
                self.assertEqual(result,exp.import_checked(ctx,'full_gi','proj'))
            self.assertEqual(result['bits'],rec['bits']);self.assertEqual(before,exp.core.sha(Path(saved['path'])))
            ctx.output=root/'broken';ctx.output.mkdir()
            bad=exp.single.atomic_tensors(root/'bad.safetensors',{'A':values['A']+1,'B':values['B']})
            ctx.rank64_checked['full_gi']['proj']={**rec,'file':bad}
            with self.assertRaisesRegex(RuntimeError,'direct FP64 rounding'):exp.import_checked(ctx,'full_gi','proj')

    def test_geometry_matches_direct_residual_cross_terms(self):
        e=torch.randn(72,80).double();a=torch.randn(72,72).double()/8
        l=torch.randn(72,64).bfloat16();b=torch.randn(64,80).bfloat16()
        rows=exp.group_geometry(e,a,l,b)
        full=l.double()@b.double();err=e-full
        for row in rows:
            g=row['group'];d=l[:,8*(g-1):8*g].double()@b[8*(g-1):8*g].double()
            expected={'removed_weight_fro2':exp.core.norm2(d),
                      'removed_output_sse_calibration_A':exp.core.norm2(a@d),
                      'delta_total_weight_sse':exp.core.norm2(err+d)-exp.core.norm2(err),
                      'delta_total_output_sse_calibration_A':exp.core.norm2(a@(err+d))-exp.core.norm2(a@err)}
            for k,v in expected.items():self.assertAlmostEqual(row[k]/max(abs(v),1),v/max(abs(v),1),places=10)
        # Removing an intentionally over-correcting component can REDUCE residual error.
        self.assertTrue(any(r['delta_total_weight_sse']<0 for r in rows))

    def test_install_exact_two_gemm_and_hook_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);q=torch.randn(72,72).bfloat16();a=torch.randn(72,64).bfloat16();b=torch.randn(64,72).bfloat16()
            save_file({'weight_q':q},str(root/'q.safetensors'));save_file({'A':a,'B':b},str(root/'ab.safetensors'))
            routes={'proj':{'quant':exp.single.file_record(root/'q.safetensors'),
                            'correction':exp.single.file_record(root/'ab.safetensors')}}
            x=torch.randn(2,3,72).bfloat16()
            for g in (0,1,8):
                model=torch.nn.Module();model.proj=torch.nn.Linear(72,72,bias=False).bfloat16();model.hf_device_map={'proj':0}
                handles,dep=exp.install(model,routes,g,exp.core.Inputs())
                l,r=exp.masked_factors(a,b,g)
                expected=torch.nn.functional.linear(x,q)+(x@l)@r
                self.assertTrue(torch.equal(model.proj(x),expected))
                self.assertEqual(dep['correction_bits']['proj']['B'],exp.single.tensor_record(r))
                for h in handles:h.remove()
                self.assertTrue(torch.equal(model.proj(x),torch.nn.functional.linear(x,q)))

    def test_kl_matches_definition_full_vocabulary_and_chunking(self):
        p=torch.randn(2,17,101).double()*5;q=torch.randn_like(p)*5
        logp=p[:,:-1].log_softmax(-1);logq=q[:,:-1].log_softmax(-1)
        expected=(logp.exp()*(logp-logq)).sum(-1)
        for chunk in (1,5,128):
            actual,_=exp.kl_tokens(p,q,chunk)
            self.assertTrue(torch.allclose(actual,expected,atol=1e-12,rtol=1e-12))
        own,_=exp.kl_tokens(p,p)
        self.assertLess(float(own.max()),1e-12)
        translated,_=exp.kl_tokens(p+200,q-500)
        self.assertTrue(torch.allclose(translated,expected,atol=1e-11,rtol=1e-11))
        q[:,-1]=float('nan') # Unscored final position must not enter KL.
        self.assertTrue(torch.allclose(exp.kl_tokens(p,q)[0],expected,atol=1e-12,rtol=1e-12))
        q[:,0,0]=float('nan')
        with self.assertRaises(RuntimeError):exp.kl_tokens(p,q)
        with self.assertRaises(ValueError):exp.kl_tokens(p,p,0)

    def test_token_checkpoint_pause_resume_and_no_repeat(self):
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            ctx,_=self.context(Path(tmp))
            with self.assertRaises(exp.prev.Paused):exp.token_evaluate(ctx,'teacher',0,exp.prev.Budget(batches=1))
            state=exp.single.read_state(ctx.output/'token_ppl','BF16',ctx.identity,ctx.inputs)
            self.assertEqual(len(state['records']),8)
            first=exp.core.sha(Path(state['batches'][0]['file']['path']))
            state=exp.token_evaluate(ctx,'teacher',0,exp.prev.Budget())
            self.assertTrue(state['complete']);self.assertEqual(first,exp.core.sha(Path(state['batches'][0]['file']['path'])))
            with patch.object(ctx.legacy,'load_model',side_effect=AssertionError('no re-forward')):
                exp.token_evaluate(ctx,'teacher',0,exp.prev.Budget())
            exp.summarize(ctx)
            self.assertEqual(exp.core.read_json(ctx.output/'token_ppl/status.json')['completed'],1)

    def test_kl_checkpoint_pause_resume_tamper_and_self_gate(self):
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()), \
             patch.multiple(exp,WINDOWS=3,LENGTH=5,TOKENS=12):
            ctx,Model=self.context(Path(tmp),3,5)
            teacher,student=Model(),Model()
            with self.assertRaises(exp.prev.Paused):exp.kl_evaluate(ctx,teacher,student,'teacher',0,exp.prev.Budget(batches=1))
            state=exp.kl_state(ctx,'BF16');self.assertEqual(len(state['records']),1)
            first=exp.core.sha(Path(state['records'][0]['file']['path']))
            state=exp.kl_evaluate(ctx,teacher,student,'teacher',0,exp.prev.Budget())
            self.assertTrue(state['complete']);self.assertEqual(first,exp.core.sha(Path(state['records'][0]['file']['path'])))
            with patch.object(teacher,'forward',side_effect=AssertionError('no repeat')):
                exp.kl_evaluate(ctx,teacher,student,'teacher',0,exp.prev.Budget())
            exp.summarize(ctx)
            self.assertEqual(exp.core.read_json(ctx.output/'teacher_kl/status.json')['completed'],1)
            file=Path(state['records'][0]['file']['path'])
            with file.open('r+b') as f:f.seek(-1,2);f.write(b'!')
            ctx.inputs=exp.core.Inputs()
            with self.assertRaises(RuntimeError):exp.kl_state(ctx,'BF16')
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()), \
             patch.multiple(exp,WINDOWS=3,LENGTH=5,TOKENS=12):
            ctx,Model=self.context(Path(tmp),3,5)
            teacher,student=Model(),Model()
            logits=torch.zeros(1,5,3);logits[:,:,0]=2
            with patch.object(student,'forward',return_value=SimpleNamespace(logits=logits)) as forward:
                with self.assertRaisesRegex(RuntimeError,'self-KL'):
                    exp.kl_evaluate(ctx,teacher,student,'teacher',0,exp.prev.Budget())
                forward.reset_mock() # Release mmap-backed input views on Windows.
            self.assertFalse((ctx.output/'teacher_kl/BF16.json').exists())

    def test_damage_is_paired_mean_nll_negative_retained(self):
        base=[{'window':i,'tokens':2047,'loss':4000.} for i in range(138)]
        removed=[{**x,'loss':x['loss']-204.7} for x in base]
        result=exp.damage_stats(base,removed,'loss')
        self.assertAlmostEqual(result['damage_per_token'],-.1)
        self.assertAlmostEqual(result['ci95_low'],-.1)
        self.assertEqual(result['negative_damage_windows'],138)
        self.assertEqual(result,exp.damage_stats(base,removed,'loss'))
        removed[1]['window']=0
        with self.assertRaises(RuntimeError):exp.damage_stats(base,removed,'loss')

    def test_control_failure_prevents_any_group_deletion(self):
        with patch.object(exp,'token_evaluate',return_value={}) as evaluate, \
             patch.object(exp,'token_gate',side_effect=RuntimeError('bad replay')), \
             patch.object(exp,'summarize'):
            with self.assertRaisesRegex(RuntimeError,'bad replay'):exp.run_token(SimpleNamespace(),exp.prev.Budget())
            self.assertEqual([(x.args[1],x.args[2]) for x in evaluate.call_args_list],[('teacher',0)])

    def test_cross_protocol_parameter_drift_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx,Model=self.context(Path(tmp))
            _,deployed=exp.install(Model(),{},0,ctx.inputs)
            exp.bind_deployment(ctx,ctx.output,'BF16',deployed,{},'teacher',0)
            deployed['parameter_bits']['dummy']={'changed':True}
            with self.assertRaisesRegex(RuntimeError,'frozen BF16/Wq'):
                exp.bind_deployment(ctx,ctx.output,'BF16',deployed,{},'teacher',0)

    def test_control_compares_bits_and_window_loss(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'token_ppl';folder.mkdir()
            rr=[{'window':i,'tokens':2047,'nll_sum':4000.} for i in range(138)]
            dep={'deployment':{'bits':'same'}}
            exp.core.atomic_json(folder/'deployment_FULL_GI_R64.json',dep)
            ctx=SimpleNamespace(output=root,identity='id',rank64_states={'full_gi':{'records':rr}},
                                rank64_deployments={'full_gi':dep})
            exp.token_gate(ctx,'full_gi',{'records':rr})
            ctx.rank64_deployments={'full_gi':{'deployment':{'bits':'different'}}}
            with self.assertRaisesRegex(RuntimeError,'deployment differs'):exp.token_gate(ctx,'full_gi',{'records':rr})


if __name__=='__main__':
    unittest.main()
