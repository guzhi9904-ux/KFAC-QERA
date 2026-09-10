"""CPU synthetic checks; no server model, harness download, or source mutation."""
import contextlib
import copy
import io
import math
import sys
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file
import full_a_all_ranks_dual_ppl_v1 as exp
import test_full_a_all_precision_r8_v1 as old_tests


class DualTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(61); torch.set_num_threads(2)

    def factors(self, di=72, do=88):
        a = torch.diag(torch.linspace(.2,2,di)).double()
        g = torch.diag(torch.linspace(.3,3,do)).double()
        e = torch.randn(di,do).double()
        l,b,s,_=exp.single.solve_fp64(e,a,g)
        return e,a,g,l,b,exp.core.norm2(s[8:])

    def test_source_hashes_and_config_count(self):
        self.assertEqual(exp.core.sha(exp.parent.__file__),exp.PARENT_SHA)
        self.assertEqual(len(exp.CONFIGS),14)
        self.assertEqual(len({exp.label(*x) for x in exp.CONFIGS}),14)

    def test_all_ranks_rectangular_objectives_without_svd(self):
        for di,do in ((72,88),(88,72)):
            args=self.factors(di,do)
            with patch.object(torch.linalg,'svd',side_effect=AssertionError('No second SVD')):
                rows=exp.check_rank_metrics(*args)
            self.assertEqual([x['rank'] for x in rows],[8,16,32,64])
            self.assertTrue(all(abs(x['relative_tail_difference'])<1e-8 for x in rows))
            self.assertTrue(all(y['sse_after_fp64']<=x['sse_after_fp64'] for x,y in zip(rows,rows[1:])))

    def test_bad_high_rank_component_and_order_rejected(self):
        e,a,g,l,b,tail=self.factors()
        broken=l.clone(); broken[:,50]*=1.01
        with self.assertRaises(RuntimeError): exp.check_rank_metrics(e,a,g,broken,b,tail)
        order=torch.arange(64); order[15]=40; order[40]=15
        with self.assertRaisesRegex(RuntimeError,'descending'):
            exp.check_rank_metrics(e,a,g,l[:,order],b[order],tail)
        with self.assertRaisesRegex(RuntimeError,'tail mismatch'):
            exp.check_rank_metrics(e,a,g,l,b,tail+1)

    def test_direct_prefix_r8_and_invalid_rank(self):
        _,_,_,a,b,_=self.factors()
        v={'A_fp64':a,'B_fp64':b,'A_bf16_r8':a[:,:8].bfloat16(),'B_bf16_r8':b[:8].bfloat16()}
        for r in exp.RANKS:
            l,rr=exp.prefix(v,r)
            self.assertTrue(torch.equal(l,a[:,:r].bfloat16()))
            self.assertEqual(rr.shape,(r,b.shape[1]))
        with self.assertRaises(ValueError):exp.prefix(v,7)
        v['A_bf16_r8']=v['A_bf16_r8']+1
        with self.assertRaises(RuntimeError):exp.prefix(v,8)

    def test_check_transactions_resume_and_immutable_sources(self):
        helper=old_tests.AllPrecisionTests(); helper.setUp()
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            ctx=helper.context(Path(tmp))
            with patch.object(exp.parent,'memory',return_value={}):
                exp.parent.solve_all(ctx,exp.parent.Budget())
            source=ctx.output
            ctx.factors={m:{} for m in exp.METHODS};ctx.factor_metadata={m:{} for m in exp.METHODS}
            for m in exp.METHODS:
                for n,b in ctx.tasks.items():
                    ctx.factors[m][n]=exp.parent.solved_record(source,m,n,b,ctx.identity,ctx.inputs)
                    ctx.factor_metadata[m][n]=exp.single.file_record(exp.parent.factor_paths(source,m,n)[1])
            before={str(p):exp.core.sha(p) for p in Path(tmp).rglob('*') if p.is_file()}
            ctx.identity='new'; ctx.output=Path(tmp)/'dual';ctx.output.mkdir()
            with self.assertRaises(exp.prev.Paused):exp.check_all(ctx,exp.prev.Budget(batches=2))
            self.assertEqual(exp.write_check_summary(ctx)['checked_module_methods'],2)
            exp.check_all(ctx,exp.prev.Budget())
            self.assertEqual(exp.write_check_summary(ctx)['status'],'COMPLETE')
            with patch.object(exp,'check_rank_metrics',side_effect=AssertionError('No repeated checks')):
                exp.check_all(ctx,exp.prev.Budget())
            for p,h in before.items():self.assertEqual(exp.core.sha(p),h)
            record=exp.read_checked(ctx,'full_gf','one')
            path=Path(record['file']['path'])
            with path.open('r+b') as f:f.seek(-1,2);f.write(b'!')
            ctx.inputs=exp.core.Inputs()
            with self.assertRaises(RuntimeError):exp.read_checked(ctx,'full_gf','one')

    def test_word_aggregation_uses_task_words_and_checks_completeness(self):
        task=SimpleNamespace(process_results=lambda doc,res:{'word_perplexity':(res[0],doc['words'])})
        helper=SimpleNamespace(extract_word_ppl=lambda result:result['results']['wikitext']['word_perplexity,none'])
        result={'results':{'wikitext':{'word_perplexity,none':math.exp(6/3)}},
                'n-samples':{'wikitext':{'original':2,'effective':2}},'samples':{'wikitext':[
                    {'doc_id':0,'doc':{'words':1},'filtered_resps':[-2.]},
                    {'doc_id':1,'doc':{'words':2},'filtered_resps':[-4.]}]}}
        summary,docs=exp.word_documents(result,task,helper)
        self.assertEqual(summary['scored_words'],3);self.assertEqual(summary['nll_sum'],6)
        broken=copy.deepcopy(result);broken['samples']['wikitext'].pop()
        with self.assertRaises(RuntimeError):exp.word_documents(broken,task,helper)
        broken=copy.deepcopy(result);broken['samples']['wikitext'][1]['doc_id']=0
        with self.assertRaisesRegex(RuntimeError,'Duplicate'):exp.word_documents(broken,task,helper)
        broken=copy.deepcopy(result);broken['samples']['wikitext'][0]['filtered_resps']=[-3.]
        with self.assertRaisesRegex(RuntimeError,'reconciled'):exp.word_documents(broken,task,helper)

    def test_freeze_refuses_identity_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'a.json'
            exp.freeze_json(path,{'identity':'a'});exp.freeze_json(path,{'identity':'a'})
            with self.assertRaises(RuntimeError):exp.freeze_json(path,{'identity':'b'})

    def test_token_checkpoint_resume_and_export(self):
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp);source=root/'windows.safetensors'
            save_file({'input_ids':torch.zeros(138,2048,dtype=torch.long),'attention_mask':torch.ones(138,2048,dtype=torch.long)},str(source))
            class Model(torch.nn.Module):
                def __init__(self):super().__init__();self.hf_device_map={'model':0}
                def forward(self,input_ids,**kwargs):
                    return SimpleNamespace(logits=torch.zeros(*input_ids.shape,3))
            legacy=SimpleNamespace(load_model=lambda *args:Model(),_input_device=lambda m:'cpu',
                _chunked_window_nll=lambda logits,ids,mask,n:exp.single.nll_with_tokens(logits,ids,mask,n)[0])
            ctx=SimpleNamespace(output=root,identity='identity',inputs=exp.core.Inputs(),tasks={},word=None,
                manifest={'payload':{'data':{'wikitext2':exp.single.file_record(source)},'source_config':{},'config':{'eval_max_memory':{}}}})
            with self.assertRaises(exp.prev.Paused):exp.token_evaluate(ctx,legacy,'teacher',None,exp.prev.Budget(batches=1))
            state=exp.single.read_state(root/'token_ppl','BF16',ctx.identity,ctx.inputs)
            self.assertEqual(len(state['records']),8)
            first_hash=exp.core.sha(root/'token_ppl/tokens/BF16/batch_0000.safetensors')
            exp.token_evaluate(ctx,legacy,'teacher',None,exp.prev.Budget())
            self.assertEqual(first_hash,exp.core.sha(root/'token_ppl/tokens/BF16/batch_0000.safetensors'))
            with patch.object(legacy,'load_model',side_effect=AssertionError('already complete')):
                self.assertTrue(exp.token_evaluate(ctx,legacy,'teacher',None,exp.prev.Budget())['complete'])
            exp.summarize(ctx)
            self.assertEqual(exp.core.read_json(root/'token_ppl/status.json')['completed'],1)

    def test_install_same_bf16_prefix_and_wq_across_ranks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);q=torch.randn(72,72).bfloat16();a=torch.randn(72,64).bfloat16();b=torch.randn(64,72).bfloat16()
            save_file({'weight_q':q},str(root/'q.safetensors'));save_file({'A':a,'B':b},str(root/'ab.safetensors'))
            routes={'proj':{'quant':exp.single.file_record(root/'q.safetensors'),'correction':exp.single.file_record(root/'ab.safetensors')}}
            for rank in exp.RANKS:
                model=torch.nn.Module();model.proj=torch.nn.Linear(72,72,bias=False).bfloat16();model.hf_device_map={'proj':0}
                handles,d=exp.install(model,routes,rank,exp.core.Inputs())
                self.assertTrue(torch.equal(model.proj.weight,q))
                self.assertEqual(d['correction_bits']['proj']['A'],exp.single.tensor_record(a[:,:rank]))
                for h in handles:h.remove()

    def test_word_gate_failure_is_not_scientific_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ctx=SimpleNamespace(output=root,identity='id',word=SimpleNamespace(references={'BF16':{'state':{'word_ppl':7.55}}}))
            with self.assertRaisesRegex(RuntimeError,'replay failed'):
                exp.word_gate(ctx,'BF16',{'summary':{'ppl':7.8}})
            self.assertEqual(exp.core.read_json(root/'word_ppl/BF16_control.json')['status'],'FAIL')

    def test_word_atomic_configuration_resume_and_tamper(self):
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp);result={'results':{'wikitext':{'word_perplexity,none':math.exp(2)}},
                'n-samples':{'wikitext':{'original':1,'effective':1}},
                'samples':{'wikitext':[{'doc_id':0,'doc':{'words':2},'filtered_resps':[-4.]}]}}
            task=SimpleNamespace(process_results=lambda doc,res:{'word_perplexity':(res[0],doc['words'])})
            helper=SimpleNamespace(extract_word_ppl=lambda r:r['results']['wikitext']['word_perplexity,none'])
            ctx=SimpleNamespace(output=root,identity='id',tasks={},inputs=exp.core.Inputs(),
                manifest={'payload':{'source_config':{'model_path':'unused'}}},
                word=SimpleNamespace(task=task,helper=helper,manager=None,data={'documents':1}))
            class Model(torch.nn.Module):
                def __init__(self):
                    super().__init__();self.config=SimpleNamespace(use_cache=True);self.hf_device_map={'model':0}
            calls=[]
            def harness(**kwargs):
                calls.append(kwargs);return copy.deepcopy(result)
            modules={'transformers':SimpleNamespace(AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *a,**k:Model())),
                     'accelerate':SimpleNamespace(dispatch_model=lambda m,**kw:m),
                     'qera.utils':SimpleNamespace(create_device_map=lambda m,x:{'model':0}),
                     'lm_eval.models.huggingface':SimpleNamespace(HFLM=lambda m:SimpleNamespace(max_length=4096,batch_size=1)),
                     'lm_eval.evaluator':SimpleNamespace(simple_evaluate=harness)}
            with patch.dict(sys.modules,modules):
                state=exp.word_evaluate(ctx,None,'teacher',None,exp.prev.Budget())
                self.assertEqual(state['summary']['documents'],1)
                self.assertEqual(calls[0]['limit'],None);self.assertTrue(calls[0]['log_samples'])
                exp.word_evaluate(ctx,None,'teacher',None,exp.prev.Budget())
                self.assertEqual(len(calls),1)
            doc=root/'word_ppl/BF16/documents.json'
            exp.core.atomic_json(doc,[]);ctx.inputs=exp.core.Inputs()
            with self.assertRaises(RuntimeError):exp.word_existing(ctx,'BF16')

    def test_interrupted_word_does_not_commit_partial_results(self):
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp);ctx=SimpleNamespace(output=root,identity='id',tasks={},inputs=exp.core.Inputs(),
                manifest={'payload':{'source_config':{'model_path':'unused'}}},word=SimpleNamespace(task=None,manager=None))
            class Model(torch.nn.Module):
                def __init__(self):super().__init__();self.config=SimpleNamespace(use_cache=True);self.hf_device_map={'m':0}
            def interrupted(**kw):raise exp.prev.Paused()
            modules={'transformers':SimpleNamespace(AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *a,**k:Model())),
                     'accelerate':SimpleNamespace(dispatch_model=lambda m,**kw:m),
                     'qera.utils':SimpleNamespace(create_device_map=lambda *a:{'m':0}),
                     'lm_eval.models.huggingface':SimpleNamespace(HFLM=lambda m:SimpleNamespace(max_length=4096,batch_size=1)),
                     'lm_eval.evaluator':SimpleNamespace(simple_evaluate=interrupted)}
            with patch.dict(sys.modules,modules),self.assertRaises(exp.prev.Paused):
                exp.word_evaluate(ctx,None,'teacher',None,exp.prev.Budget())
            self.assertFalse((root/'word_ppl/BF16/complete.json').exists())


if __name__=='__main__':unittest.main()
