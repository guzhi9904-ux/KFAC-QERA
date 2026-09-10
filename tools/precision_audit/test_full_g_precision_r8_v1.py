"""CPU synthetic tests. No production artifacts/GPU/model downloads required."""
import contextlib
import copy
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file
import full_g_precision_r8_v1 as exp


class PrecisionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(2)

    def test_helper_hashes(self):
        self.assertEqual(exp.core.sha(exp.previous.__file__),exp.PREVIOUS_SHA)
        self.assertEqual(exp.core.sha(exp.core.__file__),exp.previous.HELPER_SHA)

    def test_tensor_hash_is_bitwise_including_signed_zero(self):
        self.assertNotEqual(exp.tensor_record(torch.tensor([0.])),exp.tensor_record(torch.tensor([-0.])))
        self.assertEqual(exp.tensor_record(torch.ones(4).bfloat16()),exp.tensor_record(torch.ones(4).bfloat16()))
        self.assertNotEqual(exp.tensor_record(torch.ones(4)),exp.tensor_record(torch.ones(4).bfloat16()))
        self.assertEqual(exp.tensor_record(torch.tensor(1.)),exp.tensor_record(torch.tensor(1.)))

    def test_deployment_only_target_changes(self):
        old={'parameter_bits':{'w':'a'},'buffer_bits':{},'device_map':{'x':0},
             'correction_bits':{exp.TARGET:'old','other':'same'}}
        changed=copy.deepcopy(old); changed['correction_bits'][exp.TARGET]='new'
        exp.assert_deployment(old,changed,'FP64')
        changed['parameter_bits']['w']='bad'
        with self.assertRaises(RuntimeError): exp.assert_deployment(old,changed,'FP64')
        changed=copy.deepcopy(old); changed['correction_bits']['other']='bad'
        with self.assertRaises(RuntimeError): exp.assert_deployment(old,changed,'FP64')
        changed=copy.deepcopy(old); changed['correction_bits'][exp.TARGET]={'mode':'zero_no_hook'}
        exp.assert_deployment(old,changed,'ZERO')

    def test_nll_matches_frozen_helper_exactly(self):
        experiments=Path(__file__).parent/'qera_mxint4_full_ag/experiments'
        sys.path.insert(0,str(experiments))
        try:
            from qera_original_a_isolation.pipeline import _chunked_window_nll
            for dtype in (torch.float32,torch.bfloat16):
                logits=torch.randn(3,25,31).to(dtype)
                ids=torch.randint(0,31,(3,25)); mask=torch.ones_like(ids)
                metrics,tokens=exp.nll_with_tokens(logits,ids,mask,7)
                self.assertEqual(metrics,_chunked_window_nll(logits,ids,mask,7))
                self.assertEqual(tokens.shape,(3,24))
                self.assertEqual(tokens.dtype,dtype)
        finally:
            sys.path.pop(0)

    def test_solve_fp64_tail(self):
        e=torch.randn(12,10)
        a=torch.diag(torch.linspace(.3,2,12)); g=torch.diag(torch.linspace(.7,2,10))
        l,b,s,info=exp.solve_fp64(e,a,g,solve_rank=8)
        loss=exp.core.norm2(a.double() @ (e.double()-l @ b) @ g.double())
        self.assertAlmostEqual(loss,exp.core.norm2(s[8:]),places=9)
        self.assertLess(info['a_inverse_residual_fp64'],1e-12)
        self.assertEqual(l.dtype,torch.float64)

    def test_prepare_freezes_factors_and_does_not_re_solve(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); source=root/'source'; output=root/'output'; source.mkdir(); output.mkdir()
            def save(name,values):
                path=source/(name+'.safetensors'); save_file(values,str(path)); return exp.file_record(path)
            size=72
            w=torch.randn(size,size).bfloat16(); q=(w.float()*.9).bfloat16()
            a=torch.diag(torch.linspace(.5,2,size)); raw=torch.diag(torch.linspace(.3,2,size).double())
            g,_=exp.core.full_root(raw,256*2047)
            e=(w.float()-q.float()).T
            u,s,vh=torch.linalg.svd(a @ e @ g)
            l=torch.linalg.solve(a,u[:,:64]).contiguous()
            b=torch.linalg.solve(g.T,(s[:64,None]*vh[:64]).T).T.contiguous()
            binding={'shape':[size,size],'root':save('root',{'full':a}),
                'gram':save('raw',{'gram':raw,'diagonal':raw.diagonal().contiguous()}),
                'model':save('model',{exp.TARGET+'.weight':w}),'quant':save('q',{'weight_q':q}),
                'corrections':{'full_gf':{'file':save('old',{'A':l,'B':b})}}}
            original={p.name:exp.core.sha(p) for p in source.iterdir()}
            inputs=exp.core.Inputs()
            with contextlib.redirect_stdout(io.StringIO()),patch.object(torch.cuda,'empty_cache'):
                rec=exp.prepare_factors(binding,inputs,output,'id','cpu')
                with patch.object(exp,'solve_fp64',side_effect=AssertionError('must not re-solve')):
                    self.assertEqual(rec,exp.prepare_factors(binding,inputs,output,'id','cpu'))
            exp.check_factors(inputs.tensors(rec['file']),size,size)
            self.assertEqual(original,{p.name:exp.core.sha(p) for p in source.iterdir()})
            with self.assertRaises(RuntimeError): exp.prepare_factors(binding,inputs,output,'changed','cpu')

    def test_three_arms_resume_token_checkpoints_and_zero_semantics(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); source=root/'source'; output=root/'output'; source.mkdir(); output.mkdir()
            def save(name,values):
                path=source/(name+'.safetensors'); save_file(values,str(path)); return exp.file_record(path)
            base={}
            for name,din,dout in ((exp.TARGET,4,6),('other',6,5)):
                base[name]={'quant':save(name+'_q',{'weight_q':torch.randn(dout,din).bfloat16()*.1}),
                    'correction':save(name+'_ab',{'A':torch.randn(din,64)*.03,'B':torch.randn(64,dout)*.03})}
            windows={'input_ids':torch.zeros(138,2048,dtype=torch.long),'attention_mask':torch.ones(138,2048,dtype=torch.long)}
            data=save('windows',windows)
            manifest={'payload':{'config':{'eval_max_memory':{},'control_ppl_tolerance':1e-5},
                                  'source_config':{},'data':{'wikitext2':data}}}
            generated={'A_bf16_r8':torch.randn(4,8).bfloat16()*.01,'B_bf16_r8':torch.randn(8,6).bfloat16()*.01}
            # Fixed constructors for exact parameter/buffer comparisons across loads.
            calls=[]
            class Model(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.model=torch.nn.Module(); self.model.layers=torch.nn.ModuleList([torch.nn.Module()])
                    layer=self.model.layers[0]; layer.self_attn=torch.nn.Module()
                    layer.self_attn.o_proj=torch.nn.Linear(4,6,bias=False).bfloat16()
                    self.other=torch.nn.Linear(6,5,bias=False).bfloat16()
                    self.register_buffer('fixed_buffer',torch.tensor([1.]))
                    self.hf_device_map={'model':0,'other':1}
                def forward(self,input_ids,attention_mask,use_cache):
                    calls.append(len(input_ids))
                    x=torch.ones(input_ids.shape+(4,),dtype=torch.bfloat16)
                    return SimpleNamespace(logits=self.other(self.model.layers[0].self_attn.o_proj(x)).float())
            def legacy_nll(logits,ids,mask,chunk):
                return exp.nll_with_tokens(logits,ids,mask,chunk)[0]
            legacy=SimpleNamespace(load_model=lambda *args:Model(),_input_device=lambda m:'cpu',_chunked_window_nll=legacy_nll)
            inputs=exp.core.Inputs(); before={p.name:exp.core.sha(p) for p in source.iterdir()}
            with contextlib.redirect_stdout(io.StringIO()),patch.object(torch.cuda,'empty_cache'):
                with self.assertRaises(exp.previous.Paused):
                    exp.evaluate_arm(legacy,manifest,base,generated,'OLD',inputs,output,'id',exp.previous.Budget(batches=1))
                state=exp.read_state(output/'evaluation','OLD','id',inputs)
                self.assertEqual(len(state['records']),8)
                self.assertEqual(len(state['batches']),1)
                old=exp.evaluate_arm(legacy,manifest,base,generated,'OLD',inputs,output,'id',exp.previous.Budget())
                self.assertEqual(len(old['records']),138)
                exp.run_arms(legacy,manifest,base,generated,old,inputs,output,'id',exp.previous.Budget())
                exp.verify_deployment_reports(output,'id')
                result=exp.summarize(output,'id',inputs)
                prior_calls=len(calls)
                exp.run_arms(legacy,manifest,base,generated,old,inputs,output,'id',exp.previous.Budget())
                self.assertEqual(len(calls),prior_calls)
            self.assertEqual(result['status'],'COMPLETE')
            self.assertEqual(len(calls),18*3)
            self.assertEqual(before,{p.name:exp.core.sha(p) for p in source.iterdir()})
            deployments={a:exp.core.read_json(output/'evaluation'/('deployment_'+a+'.json'))['deployment'] for a in exp.ARMS}
            self.assertEqual(deployments['ZERO']['correction_bits'][exp.TARGET],{'mode':'zero_no_hook'})
            self.assertEqual(deployments['OLD']['parameter_bits'],deployments['FP64']['parameter_bits'])
            self.assertTrue((output/'evaluation/paired_windows.csv').exists())
            # Corrupted token payload must be rejected, including on resume.
            path=Path(old['batches'][0]['file']['path']); path.write_bytes(b'bad')
            with contextlib.redirect_stdout(io.StringIO()),self.assertRaises(RuntimeError):
                exp.read_state(output/'evaluation','OLD','id',exp.core.Inputs())

    def test_control_failure_blocks_both_interventions(self):
        records=[{'window':i,'tokens':2047,'nll_sum':2047.} for i in range(138)]
        reference={'records':[dict(r,nll_sum=4094.) for r in records]}
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder); (output/'evaluation').mkdir()
            manifest={'payload':{'config':{'control_ppl_tolerance':1e-5}}}
            with patch.object(exp,'evaluate_arm',return_value={'records':records}) as evaluate,patch.object(exp,'summarize'):
                with self.assertRaises(RuntimeError):
                    exp.run_arms(None,manifest,{},None,reference,None,output,'id',exp.previous.Budget())
                self.assertEqual(evaluate.call_count,1)
                self.assertEqual(evaluate.call_args.args[4],'OLD')


if __name__=='__main__': unittest.main()
