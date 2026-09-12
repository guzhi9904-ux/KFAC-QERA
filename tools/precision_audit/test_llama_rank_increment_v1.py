import copy
from contextlib import nullcontext
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import llama_rank_increment_v1 as m


class PolicyTests(unittest.TestCase):
    def test_partition_and_fixed_baseline(self):
        for arm in m.ARMS[2:]:
            self.assertEqual(sum(m.rank_for(arm,n)==32 for n in m.MODULES),56)
        for name in m.MODULES:
            self.assertEqual(sum(m.rank_for(a,name)==32 for a in m.ARMS[2:]),1)
            self.assertEqual(m.rank_for(m.ARMS[0],name),16)
            self.assertEqual(m.rank_for(m.ARMS[1],name),32)
        for arm,name in [('bad',m.MODULES[0]),(m.ARMS[0],'model.layers.32.mlp.down_proj'),
                         (m.ARMS[0],'model.layers.0.other.q_proj')]:
            with self.assertRaises(RuntimeError):m.rank_for(arm,name)

    def test_control_checks_units_not_only_cancelled_total(self):
        ref=[{'window':i,'tokens':2047,'nll_sum':100.} for i in range(2)]
        changed=copy.deepcopy(ref)
        changed[0]['nll_sum']+=1;changed[1]['nll_sum']-=1
        self.assertEqual(m.control(ref,ref,'token')['status'],'PASS')
        self.assertEqual(m.control(changed,ref,'token')['status'],'FAIL')
        self.assertAlmostEqual(m.compare_rows(changed,ref,'token')['delta_nll'],0)
        changed[0]['tokens']=10
        with self.assertRaises(RuntimeError):m.control(changed,ref,'token')

    def test_word_pair_identity_and_counts(self):
        ref=[{'document':0,'words':100,'nll_sum':10.,'document_sha256':'doc'}]
        self.assertEqual(m.control(ref,ref,'word')['status'],'PASS')
        for key,value in [('document_sha256','other'),('words',101),('nll_sum',float('nan'))]:
            bad=copy.deepcopy(ref);bad[0][key]=value
            with self.assertRaises(RuntimeError):m.control(bad,ref,'word')

    def test_nonadditivity_uses_nll_and_tracks_parameter_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx=SimpleNamespace(output=Path(tmp),identity='id',tasks={n:{'shape':[5,3]} for n in m.MODULES})
            rows={a:[{'window':0,'tokens':2047,'nll_sum':100.+i}] for i,a in enumerate(m.ARMS)}
            m.export_results(ctx,'token',rows)
            effect=m.core.read_json(ctx.output/'token/nonadditivity.json')
            self.assertEqual(effect['joint_delta_nll'],1.)
            self.assertEqual(effect['sum_group_delta_nll'],14.)
            self.assertEqual(effect['joint_minus_sum'],-13.)
            import csv
            with (ctx.output/'token/ppl_summary.csv').open() as f: summaries=list(csv.DictReader(f))
            self.assertEqual(int(summaries[2]['upgraded_modules']),56)
            self.assertEqual(int(summaries[2]['added_correction_parameters_vs_r16']),56*16*8)


def fixture(directory):
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model=torch.nn.Module();self.model.layers=torch.nn.ModuleList()
            for i in range(9):
                layer=torch.nn.Module()
                if i in (0,8):
                    layer.mlp=torch.nn.Module()
                    layer.mlp.down_proj=torch.nn.Linear(4,4,bias=True,dtype=torch.bfloat16)
                self.model.layers.append(layer)
            self.hf_device_map={'':'cpu'}
        def forward(self,x):
            for i in (0,8):x=self.model.layers[i].mlp.down_proj(x)
            return x
    model=Toy();tasks={};factors={};saved={};correction={16:{},32:{}}
    for i in (0,8):
        name=f'model.layers.{i}.mlp.down_proj';mod=model.get_submodule(name)
        q=torch.randn(4,4).bfloat16();a=(torch.randn(4,64)/32).bfloat16();b=(torch.randn(64,4)/32).bfloat16()
        qr=m.single.atomic_tensors(directory/f'q{i}.safetensors',{'weight_q':q})
        fr=m.single.atomic_tensors(directory/f'f{i}.safetensors',{'A':a,'B':b})
        tasks[name]={'quant':qr,'shape':[4,4]};factors[name]={'file':fr}
        saved[name]=(q,a,b,mod.bias.detach().clone())
        with torch.no_grad():mod.weight.copy_(q)
        for r in (16,32):correction[r][name]={'A':m.single.tensor_record(a[:,:r].contiguous()),'B':m.single.tensor_record(b[:r].contiguous())}
    common={'parameter_bits':{n:m.single.tensor_record(t) for n,t in model.named_parameters()},
            'buffer_bits':{n:m.single.tensor_record(t) for n,t in model.named_buffers()},'device_map':model.hf_device_map}
    historical={(k,r):{'deployment':{**common,'correction_bits':correction[r]}} for k in ('token','word') for r in (16,32)}
    ctx=SimpleNamespace(output=directory,identity='fixture',tasks=tasks,factors=factors,
                        historical=historical,inputs=m.core.Inputs())
    return model,ctx,saved


class HookTests(unittest.TestCase):
    def test_real_mixed_rank_two_gemm_and_no_source_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            model,ctx,saved=fixture(Path(tmp));handles=[]
            hashes={n:m.core.sha(r['file']['path']) for n,r in ctx.factors.items()}
            try:
                deployed,audit=m.install(ctx,model,m.ARMS[2],'token',m.prev.Budget(),handles)
                x=torch.randn(3,4).bfloat16();expected=x
                for name,(q,a,b,bias) in saved.items():
                    rank=m.rank_for(m.ARMS[2],name)
                    expected=torch.nn.functional.linear(expected,q,bias)+(expected@a[:,:rank].contiguous())@b[:rank].contiguous()
                self.assertTrue(torch.equal(model(x),expected))
                self.assertEqual(audit.snapshot()['successful_forwards'],1)
                self.assertEqual(deployed,m.expected_deployment(ctx,m.ARMS[2],'token'))
                for n,r in ctx.factors.items():self.assertEqual(m.core.sha(r['file']['path']),hashes[n])
                with self.assertRaises(RuntimeError):m.install(ctx,model,m.ARMS[0],'token',m.prev.Budget(),[])
            finally:
                for h in handles:h.remove()
            self.assertFalse(any(x._forward_hooks or x._forward_pre_hooks for x in model.modules()))

    def test_missing_correction_and_duplicate_forward_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            model,ctx,_=fixture(Path(tmp));handles=[]
            try:
                m.install(ctx,model,m.ARMS[0],'token',m.prev.Budget(),handles)
                handles[1].remove()
                with self.assertRaises(RuntimeError):model(torch.ones(1,4,dtype=torch.bfloat16))
            finally:
                for h in handles:h.remove()
        audit=m.ForwardAudit(['a'],m.prev.Budget());audit.begin(None,None);audit.hit('a')
        with self.assertRaises(RuntimeError):audit.hit('a')
        audit.begin(None,None);audit.hit('a');audit.end(None,None,None)
        self.assertEqual(audit.snapshot()['aborted_probe_forwards'],1)

    def test_deployment_checks_weight_prefix_and_device_map(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,ctx,_=fixture(Path(tmp));expected=m.expected_deployment(ctx,m.ARMS[2],'token')
            for field in ('parameter_bits','correction_bits','device_map'):
                bad=copy.deepcopy(expected);bad[field]={}
                with self.assertRaises(RuntimeError):m.validate_deployment(ctx,m.ARMS[2],'token',bad)


class ResumeTests(unittest.TestCase):
    def make_context(self,out):
        name=m.MODULES[0]
        data=m.single.atomic_tensors(out/'data.safetensors',{'input_ids':torch.ones(138,2048,dtype=torch.int64),
                                                         'attention_mask':torch.ones(138,2048,dtype=torch.int64)})
        ref=[{'window':i,'tokens':2047,'nll_sum':100.} for i in range(138)]
        dep={'parameter_bits':{},'buffer_bits':{},'device_map':{'':'cpu'},'correction_bits':{name:'prefix'}}
        return SimpleNamespace(output=out,identity='id',tasks={name:{'shape':[2,2]}},inputs=m.core.Inputs(),
            historical={('token',r):{'deployment':dep} for r in (16,32)},refs={('token',r):ref for r in (16,32)},
            manifest={'payload':{'data':{'wikitext2':data},'source_config':{},'config':{'eval_max_memory':{}}}})

    def test_real_checkpoint_loop_pause_resume_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx=self.make_context(Path(tmp));stop=m.prev.Budget();loads=[];call_count=[0]
            class Model:
                def __call__(self,**kwargs):
                    self.audit.begin(None,None)
                    for n in self.audit.names:self.audit.hit(n)
                    self.audit.end(None,None,None)
                    return SimpleNamespace(logits=torch.zeros(kwargs['input_ids'].shape[0],1))
            def load(*args):loads.append(1);return Model()
            def install(ctx,model,arm,kind,stop,handles):
                model.audit=m.ForwardAudit(ctx.tasks,stop)
                return m.expected_deployment(ctx,arm,kind),model.audit
            def evaluate(logits,ids,mask,chunk):
                call_count[0]+=1
                if call_count[0]==1:stop.requested=True
                values=torch.zeros(len(logits),2047,dtype=torch.float32);values[:,0]=100
                return [(100.,2047)]*len(logits),values
            ctx.legacy=SimpleNamespace(load_model=load,_input_device=lambda model:'cpu',
                        _chunked_window_nll=lambda logits,*a:[(100.,2047)]*len(logits))
            with patch.object(m,'install',install),patch.object(m.dual,'resident'),patch.object(m.single,'nll_with_tokens',evaluate):
                with self.assertRaises(m.prev.Paused):m.token_arm(ctx,m.ARMS[0],stop)
                first=m.read_token(ctx,m.ARMS[0]);self.assertEqual(len(first['records']),8)
                result=m.token_arm(ctx,m.ARMS[0],m.prev.Budget())
                self.assertEqual(len(result),138)
                final=m.read_token(ctx,m.ARMS[0]);self.assertTrue(final['complete'])
                self.assertEqual(final['batches'][0],first['batches'][0])
                self.assertEqual(final['batches'][-1]['end'],138)
                self.assertEqual(len(loads),2)
                m.token_arm(ctx,m.ARMS[0],m.prev.Budget());self.assertEqual(len(loads),2)
                bad=copy.deepcopy(final);bad['batches'][0]['successful_forward_delta']=0
                bad.pop('payload_sha256');bad['payload_sha256']=m.core.fingerprint(bad)
                m.core.atomic_json(ctx.output/'token'/f'{m.ARMS[0]}.json',bad)
                with self.assertRaises(RuntimeError):m.read_token(ctx,m.ARMS[0])

    def test_checkpoint_changes_do_not_pass_hash_or_tensor_sum_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx=self.make_context(Path(tmp));root=ctx.output/'pilot';root.mkdir()
            dep=root/(m.ARMS[0]+'_deployment.json');m.core.atomic_json(dep,m.expected_deployment(ctx,m.ARMS[0],'token'))
            values=torch.ones(8,2047);path=root/'tokens'/m.ARMS[0]/'batch_0000.safetensors';path.parent.mkdir(parents=True)
            f=m.single.atomic_tensors(path,{'nll':values})
            s={'experiment_identity':ctx.identity,'arm':m.ARMS[0],'windows':8,'complete':True,
               'records':[{'window':i,'tokens':2047,'nll_sum':2047.,'token_nll_sum_fp64':2047.} for i in range(8)],
               'deployment_file':m.single.file_record(dep),
               'batches':[{'start':0,'end':8,'file':f,'successful_forward_delta':1,'execution':{
                   'status':'PASS','targets':1,'successful_forwards':1,'exactly_once_each_target_per_successful_forward':True}}]}
            def write(s):
                s.pop('payload_sha256',None);s['payload_sha256']=m.core.fingerprint(s)
                m.core.atomic_json(root/(m.ARMS[0]+'.json'),s)
            write(s);m.read_token(ctx,m.ARMS[0],True)
            s['records'][0]['token_nll_sum_fp64']=2048.;write(s)
            with self.assertRaises(RuntimeError):m.read_token(ctx,m.ARMS[0],True)
            self.assertFalse(m.read_token(ctx,m.ARMS[0],False)['complete'])

    def test_word_reader_requires_full_raw_result_and_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,ctx,_=fixture(Path(tmp));ctx.word=SimpleNamespace(task=None,helper=None)
            folder=ctx.output/'word'/m.ARMS[0];folder.mkdir(parents=True)
            rows=[{'document':i,'document_sha256':str(i),'nll_sum':float(i),'words':1} for i in range(62)]
            summary={'documents':62,'scored_words':241335}
            for name,obj in [('results.json',{'raw':True}),('documents.json',rows),
                             ('deployment.json',m.expected_deployment(ctx,m.ARMS[0],'word'))]:m.core.atomic_json(folder/name,obj)
            state={'experiment_identity':ctx.identity,'arm':m.ARMS[0],'status':'PASS','summary':summary,
                'execution':{'status':'PASS','targets':2,'successful_forwards':62,'exactly_once_each_target_per_successful_forward':True},
                **{k:m.single.file_record(folder/n) for k,n in [('results_file','results.json'),('documents_file','documents.json'),('deployment_file','deployment.json')]}}
            m.core.atomic_json(folder/'complete.json',state)
            with patch.object(m.dual,'word_documents',return_value=(summary,rows)):
                self.assertEqual(m.read_word(ctx,m.ARMS[0]),rows)
                state['execution']['targets']=1;m.core.atomic_json(folder/'complete.json',state)
                with self.assertRaises(RuntimeError):m.read_word(ctx,m.ARMS[0])

    def test_failed_endpoint_blocks_groups_in_main(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            ctx=SimpleNamespace(output=out,identity='id',experiment={},source_audit={})
            m.core.atomic_json(out/'pilot_gate.json',{'experiment_identity':'id','status':'PASS','arms':list(m.ARMS),'token_windows_per_arm':8,'separate_from_main':True})
            m.core.atomic_json(out/'experiment.json',{})
            calls=[]
            def token(ctx,arm,stop):
                calls.append(arm)
                if arm==m.ARMS[1]:raise RuntimeError('endpoint failed')
                return []
            argv=['run','--protocol','token']
            for key in ('run-dir','repo-dir','source-fp64-dir','source-da-dir','output-dir','official-qera-root','harness-source','word-reference-dir'):
                argv.extend(['--'+key,str(out)])
            with patch.object(m,'setup',return_value=ctx),patch.object(m.core,'audit_lock',return_value=nullcontext()),\
                 patch.object(m,'token_arm',side_effect=token),patch.object(m,'export_results'):
                with self.assertRaises(RuntimeError):m.main(argv)
            self.assertEqual(calls,list(m.ARMS[:2]))

    def test_pack_requires_both_protocols_and_has_expected_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx=SimpleNamespace(output=Path(tmp))
            with patch.object(m,'summarize',return_value={'complete':False}):
                with self.assertRaises(RuntimeError):m.pack(ctx)
            names=['experiment.json','source_audit.json','status.json','pilot_gate.json']
            for kind in ('token','word'):
                names += [f'{kind}/{n}' for n in ('ppl_summary.csv','per_unit.csv','comparisons.csv','paired_deltas.csv','nonadditivity.json','status.json')]
                names += [f'{kind}/{a}_control.json' for a in m.ARMS[:2]]
            for name in names:
                p=ctx.output/name;p.parent.mkdir(exist_ok=True);p.write_text('{}')
            with patch.object(m,'summarize',return_value={'complete':True}):m.pack(ctx)
            import tarfile
            with tarfile.open(ctx.output/(m.VERSION+'_summary.tar.gz')) as t:self.assertEqual(set(t.getnames()),set(names))


if __name__=='__main__':unittest.main()
