import contextlib
import gc
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from bridge import LegacyExperiment,TensorStore,save_json,save_tensors,mo,sm,slug,commit,load_record
from common import PLAN
from collection import collect,x_path,g_path
from fitting import ModuleFit,compact_moments
from dataset import split_windows
from planning import make_plan
from evaluation import evaluation_jobs,report,evaluate
from shared_teacher import SharedTeacher
from test_shared import Tiny


class Resources:
    def boundary(self):pass
    def timed(self,*args,**kwargs):return contextlib.nullcontext()


class FakeTeacher:
    calls=0
    def reference_all(self,ids,names):return ids,{n:torch.ones(1,7,5) for n in names}
    def labels(self,ref,seeds):return [torch.ones(6,dtype=torch.long) for _ in seeds]
    def gradient_all(self,ids,names,ref,xs,label):
        self.calls+=1;return {n:dict(x=xs[n],g=torch.ones(7,3)) for n in names}
    def unload(self):pass


class Tests(unittest.TestCase):
    def test_shared_collection_partial_resume_and_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ident='test';names=['model.layers.0.self_attn.q_proj','model.layers.0.self_attn.v_proj']
            ids=torch.arange(7)[None,:];save_tensors(root/'data/fit.safetensors',dict(input_ids=ids),dict(identity=ident,tensor_hash=mo.digest_tensor(ids)))
            row=dict(id='f0000_k00',window=0,replicate=0,seed=1,token_hash=mo.digest_tensor(ids[0]))
            save_json(root/'data/index.json',dict(fit=[row],identity=ident));teacher=FakeTeacher()
            config=dict(modules=names,cache_GiB=0)
            # A process can stop after one module file but before the all-module receipt.
            store=TensorStore(ident,0);store.put(g_path(root,names[0],row['id']),dict(g=torch.ones(7,3)))
            collect(root,config,ident,Resources(),teacher);self.assertEqual(teacher.calls,1)
            collect(root,config,ident,Resources(),teacher);self.assertEqual(teacher.calls,1)
            self.assertEqual(x_path(root,names[0],0),x_path(root,names[1],0))
            p=g_path(root,names[1],row['id']);p.write_bytes(p.read_bytes()+b'corrupt')
            with self.assertRaisesRegex(RuntimeError,'Frozen file changed'):collect(root,config,ident,Resources(),teacher)
    def test_four_fits_equal_frozen_dense_and_checkpoint_resume(self):
        torch.manual_seed(52)
        with tempfile.TemporaryDirectory() as tmp,patch.dict(PLAN,L=7,T=6,rank=2,ALS_max_iterations=4):
            root=Path(tmp);name='model.layers.0.self_attn.q_proj';store=TensorStore('test',0);keys=['f0000_k00','f0001_k00']
            save_json(root/'data/index.json',dict(identity='test',fit=[dict(id=k,window=i) for i,k in enumerate(keys)],budgets={'N2':keys}))
            w0=torch.randn(3,5);store.put(root/'quantized'/(slug(name)+'.safetensors'),dict(W0=w0,Wq=w0*.9))
            dense=[];groups=[]
            for i,k in enumerate(keys):
                x=torch.randn(7,5);g=torch.randn(7,3);g[-1]=0
                store.put(x_path(root,name,i),dict(x=x));store.put(g_path(root,name,k),dict(g=g))
                groups.append((x,[g]));dense.append(g.double().T@x.double())
            a,g,n=compact_moments(iter(groups),6,'cpu');ad,gd,nd=sm.sequence_moments(lambda:iter(dense),6,'cpu')
            torch.testing.assert_close(a,ad,rtol=1e-12,atol=1e-12);torch.testing.assert_close(g,gd,rtol=1e-12,atol=1e-12);self.assertEqual(n,nd)
            config=dict(cache_GiB=0);fast=ModuleFit(root,config,'test',Resources(),name,'cpu')
            slow=ModuleFit(root,config,'test',Resources(),name,'cpu');slow.root=root/'dense_oracle'
            # Frozen legacy method uses independent dense moment products.
            def dense_moments(keys,tag,role='fit'):
                a,g,n=sm.sequence_moments(lambda:iter(dense),6,'cpu');return dict(StS=a,SSt=g)
            slow.moments=dense_moments
            for method in ('Marginal','Token-joint','Sequence-one-step','Full-fit'):
                actual=fast.raw_factors('N2',method);expected=LegacyExperiment.raw_factors(slow,'N2',method)
                for k in actual:torch.testing.assert_close(actual[k],expected[k],rtol=1e-10,atol=1e-10)
                fast.candidate('N2__'+method,actual['A_raw'],actual['G_raw'],'N2',method)
                deployed=fast.store.get(fast.root/'corrections'/('N2__'+method+'.safetensors'))[0]
                _,old,audit=sm.solve(fast.error,expected['A_raw'],expected['G_raw'],2,w0*.9,w0)
                torch.testing.assert_close(deployed['W_deploy'],old['W_deploy'],rtol=1e-6,atol=1e-7)
            # Interrupt immediately after the first atomic ALS snapshot, then
            # actually execute the remaining rounds on resume.
            resumed=ModuleFit(root,config,'test',Resources(),name,'cpu');resumed.root=root/'resume'
            def persist_then_interrupt(path,tensors,metadata):
                save_tensors(path,tensors,metadata);raise RuntimeError('simulated interruption')
            with patch('fitting.save_tensors',side_effect=persist_then_interrupt):
                with self.assertRaisesRegex(RuntimeError,'simulated interruption'):resumed.raw_factors('N2','Full-fit')
            actual=resumed.raw_factors('N2','Full-fit');expected=fast.raw_factors('N2','Full-fit')
            for k in actual:torch.testing.assert_close(actual[k],expected[k],rtol=1e-10,atol=1e-10)
            del actual,expected,deployed,fast,slow,resumed;gc.collect()
    def test_data_budget_and_scope_accounting(self):
        ids,rows,stats=split_windows(list(range(100)),7,6,True)
        spans=[set(range(r['token_start'],r['token_stop'])) for r in rows]
        self.assertEqual(len(set.union(*spans)),42)
        ids2,rows2,_=split_windows(list(range(100)),7,3,True);self.assertTrue(torch.equal(ids[:3],ids2))
        names=['model.layers.0.self_attn.q_proj','model.layers.0.self_attn.v_proj'];shapes={names[0]:(5,5),names[1]:(3,5)}
        plan=make_plan(shapes,test_windows=3)
        self.assertEqual(plan['shared_fit_backward_passes'],256);self.assertEqual(plan['separate_fit_backward_passes'],512)
        self.assertEqual(plan['main_eval_backward_passes'],0)
        self.assertEqual(len(evaluation_jobs(names,['x','None'],'validation')),4)
        self.assertEqual(len(evaluation_jobs(names,['x','None'],'test')),2)
    def test_report_demands_all_windows_and_weights_tokens(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);name='one';config=dict(modules=[name]);ident='test'
            commit(root/'candidate_freeze.json',ident,files={},keys=['None'],modules=[name])
            save_json(root/'data/index.json',dict(windows=dict(validation=2,test=2)))
            for role,scope in [('validation',name),('test','joint12')]:
                for c in range(2):
                    scores=dict(KL=.1,NLL_sum=2*(c+1),tokens=c+1)
                    folder=root/'scores'/role/f'w{c:04d}'
                    commit(folder/(scope+'___None.json'),ident,scores=scores)
                    commit(folder/'teacher.json',ident,scores=scores)
            report(root,config,ident);self.assertTrue(load_record(root/'complete.json',ident)['passed'])
            (root/'scores/test/w0001/joint12___None.json').unlink()
            with self.assertRaises(FileNotFoundError):report(root,config,ident)
    def test_real_evaluation_flow_and_resume_without_forward(self):
        torch.manual_seed(61)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ident='fixture';model=Tiny().eval().requires_grad_(False)
            names=['model.layers.0.q','model.layers.1.down'];config=dict(modules=names,cache_GiB=0,vocab_chunk=2,sequence_length=7)
            teacher=SharedTeacher(config,root,ident,Resources().timed);teacher.model=model
            store=TensorStore(ident,0);original={n:model.get_submodule(n).weight.detach().clone() for n in names}
            for n in names:
                store.put(root/'quantized'/(slug(n)+'.safetensors'),dict(W0=original[n],Wq=original[n]+.03))
                store.put(root/'modules'/slug(n)/'corrections/fitted.safetensors',dict(W_deploy=original[n]+.01))
            save_json(root/'data/index.json',dict(windows=dict(validation=2,test=2)))
            for role in ('validation','test'):
                ids=torch.randint(0,17,(2,7));save_tensors(root/'data'/f'{role}.safetensors',dict(input_ids=ids),dict(identity=ident,tensor_hash=mo.digest_tensor(ids)))
            commit(root/'candidate_freeze.json',ident,files={},keys=['fitted','None'],modules=names)
            evaluate(root,config,ident,Resources(),teacher)
            for n in names:self.assertTrue(torch.equal(model.get_submodule(n).weight,original[n]))
            with patch.object(teacher,'load',side_effect=AssertionError('Resume must not forward completed windows')):
                evaluate(root,config,ident,Resources(),teacher)


if __name__=='__main__':unittest.main(verbosity=2)
