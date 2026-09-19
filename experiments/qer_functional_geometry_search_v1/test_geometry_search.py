"""CPU acceptance, including the actual controller engine with an independent tiny teacher.

No external model, parent dataset, network, CUDA allocation or scheduler is used.
"""
import contextlib
import copy
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
import torch.nn.functional as F
from common import PLAN, save_json, save_tensors, sha_file, read, mo, slug, seed
from geometry import spectrum, construct, candidate_key, choose, relative, scalar_check
from data_split import select_articles, spans, articles
from analysis_math import ratio_summary
from records import commit, load_record
from engine import Experiment
from report import report
from resources import Resources, BudgetReached


class MathTests(unittest.TestCase):
    def setUp(self): torch.manual_seed(772); torch.set_num_threads(2)

    def test_nine_geometries_direct_roots_scale_and_endpoints(self):
        e=torch.randn(9,11,dtype=torch.float64); aa=torch.randn(11,11,dtype=torch.float64); gg=torch.randn(9,9,dtype=torch.float64)
        a=aa@aa.T+torch.eye(11); g=gg@gg.T+torch.eye(9)
        av,au,_=spectrum(a); gv,gu,_=spectrum(g)
        for alpha in (0.,.5,1.):
            for beta in (0.,.5,1.):
                p,q,audit=construct(e,av,au,gv,gu,alpha,beta,3)
                ar=(au*av.pow(alpha/2))@au.T; gr=(gu*gv.pow(beta/2))@gu.T
                u,s,v=torch.linalg.svd(gr@e@ar,full_matrices=False)
                expected=torch.linalg.solve(gr,(u[:,:3]*s[:3])@v[:3])@torch.linalg.inv(ar)
                self.assertLess(relative(p@q,expected),1e-11)
                self.assertLessEqual(audit['condition_A'],float(av[-1]/av[0])*(1+1e-14))
                self.assertLessEqual(audit['condition_G'],float(gv[-1]/gv[0])*(1+1e-14))
        av2,au2,_=spectrum(712*a); gv2,gu2,_=spectrum(.007*g)
        p,q,_=construct(e,av2,au2,gv2,gu2,1,1,3)
        p1,q1,_=construct(e,av,au,gv,gu,1,1,3)
        self.assertLess(relative(p@q,p1@q1),1e-11)
        u,s,v=torch.linalg.svd(e,full_matrices=False)
        p,q,_=construct(e,av,au,gv,gu,0,0,3)
        self.assertLess(relative(p@q,(u[:,:3]*s[:3])@v[:3]),1e-11)

    def test_nonpositive_asymmetric_and_nonfinite_rejected(self):
        for bad in (torch.diag(torch.tensor([1.,0.],dtype=torch.float64)),
                    torch.tensor([[1.,.5],[0.,1.]],dtype=torch.float64),
                    torch.tensor([[math.nan,0.],[0.,1.]],dtype=torch.float64)):
            with self.assertRaises(AssertionError): spectrum(bad)

    def test_degenerate_boundary_and_nearzero_projection(self):
        e=torch.eye(5,dtype=torch.float64); v,u,_=spectrum(e)
        _,_,a=construct(e,v,u,v,u,0,0,2)
        self.assertTrue(a['rank_boundary_near_degenerate']); self.assertFalse(a['compensation_unique'])
        scalar_check(1e-16,0.,1.,1e-10)
        with self.assertRaises(ArithmeticError): scalar_check(1e-4,0.,1.,1e-10)

    def test_selection_exact_tie_denominator_and_no_rounded_tie(self):
        grid={candidate_key(a,b):[a,b] for a in (0.,.5,1.) for b in (0.,.5,1.)}
        scores={k:1. for k in grid}
        self.assertEqual(choose(scores,grid,2.)['selected'],'a1_b1')
        scores['a0_b0']=np.nextafter(1.,0.).item()
        self.assertEqual(choose(scores,grid,2.)['selected'],'a0_b0')
        self.assertIsNone(choose(scores,grid,1e-12)['selected'])

    def test_paired_bootstrap_ratio_of_means_aliases_and_independent_intervals(self):
        rng=np.random.default_rng(56); none=np.arange(1,17)[:,None]*rng.uniform(.5,2,(16,8))
        selected=none*.7; marginal=none*.9
        values=np.stack([none,none*.8,marginal,selected],axis=-1)
        result=ratio_summary(values,['None','A-only','Marginal-AG','Selected'],12)
        self.assertAlmostEqual(result['comparisons'][0]['d'],.2)
        self.assertEqual(result['comparisons'][0]['article']['status'],'IMPROVED')
        self.assertIn('conditional_labels',result['comparisons'][0])
        values[:,:,3]=values[:,:,2]
        alias=ratio_summary(values,['None','A-only','Marginal-AG','Selected'],12)
        self.assertEqual(alias['comparisons'][0]['article']['ci_low'],0.)
        self.assertEqual(alias['comparisons'][0]['conditional_labels']['ci_high'],0.)
        kl=ratio_summary(values.mean(axis=1,keepdims=True),['None','A-only','Marginal-AG','Selected'],12,kl=True)
        self.assertNotIn('conditional_labels',kl['comparisons'][0])
        values[:,:,0]=0
        self.assertEqual(ratio_summary(values,['None','A-only','Marginal-AG','Selected'],12)['comparisons'][0]['article']['status'],'DENOMINATOR_UNRESOLVED')

    def test_preserve_cross_position_terms(self):
        x=torch.tensor([[1.,0.],[1.,0.]],dtype=torch.float64)
        g=torch.tensor([[1.],[-1.]],dtype=torch.float64); r=torch.tensor([[1.,0.]],dtype=torch.float64)
        self.assertEqual(float((g*(x@r.T)).sum().square()),0.)
        self.assertGreater(float((g*(x@r.T)).square().sum()),0.)
        self.assertEqual(float(((g.T@x)*r).sum()),0.)


class DataTests(unittest.TestCase):
    def test_article_parser_and_history_exclusion_order_and_shortfall(self):
        parsed=list(articles([{'text':'= First Article ='},{'text':'a'},{'text':'== Subheading =='},{'text':'= Second Article ='},{'text':'b'}]))
        self.assertEqual(len(parsed),2); self.assertIn('== Subheading ==',parsed[0][3])
        candidates=[]
        for i in range(70):
            row=dict(article_id=str(i),article_title=str(i),article_text_sha256='text'+str(i))
            candidates.append((row,list(range(i*100,i*100+80))))
        forbidden=spans(candidates[2][1])
        a,rejected=select_articles(candidates,forbidden,{'1'},{'text0'},32,16,64)
        b,_=select_articles(list(reversed(candidates)),forbidden,{'1'},{'text0'},32,16,64)
        self.assertEqual([r['article_id'] for r,t in a],[r['article_id'] for r,t in b])
        self.assertEqual(len(a),48)
        self.assertFalse({'0','1','2'} & {r['article_id'] for r,t in a})
        self.assertTrue(rejected)
        short,_=select_articles(candidates[:2],set(),set(),set(),32,16,64)
        self.assertEqual(len(short),2)
        self.assertNotEqual(seed('search','article',1),seed('test','article',1))


class TinyTeacher:
    """Independent causal nonlinear toy; autograd of real sum CE, not fabricated scores."""
    def __init__(self,weights):
        self.weights=weights; generator=torch.Generator().manual_seed(891)
        self.embed=torch.randn(97,12,generator=generator)*.3
        self.head=torch.randn(9,23,generator=generator)*.2
        self.gradient_count=0; self.kl_count=0; self.label_count=0
    def unload(self): pass
    def reference(self,name,ids):
        x=self.embed[ids]; h=x@self.weights[name].T
        return h.cumsum(1).tanh(),x,h
    def labels(self,reference,seeds):
        prob=(reference[0,:-1]@self.head).double().softmax(-1)
        self.label_count+=len(seeds)
        return [torch.multinomial(prob,1,generator=torch.Generator().manual_seed(s)).flatten() for s in seeds]
    def gradient(self,name,ids,reference,x,labels,audit=False):
        w=self.weights[name].clone().requires_grad_(True); h=x@w.T
        hidden=h.cumsum(1).tanh(); logits=hidden[0,:-1]@self.head
        loss=F.cross_entropy(logits,labels,reduction='sum')
        gh,gw=torch.autograd.grad(loss,(h,w)); g=gh[0].double()
        checks=dict(sum_NLL=True)
        if audit:
            delta=relative(g.T@x[0].double(),gw.double())
            if delta>1e-5: raise RuntimeError('toy autograd')
            checks['S_autograd_relative_error']=delta
        self.gradient_count+=1
        return g,checks
    def kl(self,reference,actual):
        return float(mo.stable_kl(reference[0,:-1]@self.head,actual[0,:-1]@self.head).mean())
    def intervention_kl(self,name,ids,reference,x,h,weight,residual):
        self.kl_count+=1
        return self.kl(reference,(x@weight.T).cumsum(1).tanh()),dict(toy=True)


class QuietResources:
    def boundary(self): pass
    def estimate(self, modules): pass
    @contextlib.contextmanager
    def io(self): yield
    @contextlib.contextmanager
    def timed(self,*args,**kwargs): yield


class PipelineTests(unittest.TestCase):
    def test_missing_marginal_rebuild_exact_normalization_and_midwindow_resume(self):
        with tempfile.TemporaryDirectory() as td, patch.dict(PLAN,dict(L=12,T=11,rank=4)), contextlib.redirect_stdout(io.StringIO()):
            root=Path(td)/'run'; parents=Path(td)/'parents'; root.mkdir(); identity='rebuild-fixture'
            gen=torch.Generator().manual_seed(24); name=PLAN['modules'][0]
            weight=torch.randn(9,12,generator=gen)*.2; teacher=TinyTeacher({name:weight})
            fit=torch.randint(0,97,(8,12),generator=gen)
            save_tensors(parents/'exp03/data/fit_windows.safetensors',dict(input_ids=fit),dict(identity=PLAN['parent_identity']))
            for c in range(8):
                for k in range(4):
                    label=torch.randint(0,23,(11,),generator=gen)
                    save_tensors(parents/'exp03/data/fit_samples'/f'w{c:02d}_k{k:03d}.safetensors',dict(labels=label),
                        dict(identity=PLAN['parent_identity'],label_hash=mo.digest_tensor(label),input_hash=mo.digest_tensor(fit[c])))
            wp=root/'data/search_windows.safetensors'; save_tensors(wp,dict(input_ids=fit),{})
            tp=root/'data/test_windows.safetensors'; save_tensors(tp,dict(input_ids=fit),{})
            save_json(root/'data/article_manifest.json',dict(identity=identity,windows=[],files=dict(search=sha_file(wp),test=sha_file(tp))))
            config=dict(assets=str(parents)); assets=dict(factors={name:dict(path=None,rebuilt=True)})
            class Interrupted(QuietResources):
                @contextlib.contextmanager
                def timed(self,stage,**kw):
                    if stage=='historical_factor_rebuild' and teacher.gradient_count==5: raise BudgetReached('fixture interruption')
                    yield
            e=Experiment(root,config,identity,assets,Interrupted(),teacher=teacher,device='cpu')
            with self.assertRaises(BudgetReached): e.factors(name)
            e.resources=QuietResources(); factors,provenance=e.factors(name)
            self.assertEqual(teacher.gradient_count,32)
            self.assertTrue(provenance['rebuilt'])
            a=torch.zeros(12,12,dtype=torch.float64); gsum=torch.zeros(9,9,dtype=torch.float64)
            for c in range(8):
                ref,x,_=teacher.reference(name,fit[c:c+1]); xd=x[0].double(); a+=xd.T@xd
                for k in range(4):
                    g,_=teacher.gradient(name,fit[c:c+1],ref,x,e.old_label(c,k,fit[c:c+1])); gsum+=g.T@g
            a/=8*12; gsum/=32*11; norm=a.norm(); a/=norm; gsum*=norm
            expected_a=a+.001*a.diag().mean()*torch.eye(12,dtype=torch.float64)
            expected_g=gsum+.001*gsum.diag().mean()*torch.eye(9,dtype=torch.float64)
            self.assertLess(relative(factors['A_solve'],expected_a),1e-12)
            self.assertLess(relative(factors['G_solve'],expected_g),1e-12)
            count=teacher.gradient_count; e.factors(name); self.assertEqual(count,teacher.gradient_count)

    def test_full_two_module_budget_freeze_resume_report_and_tamper_rejection(self):
        # Keep exact article/label budgets (512 new gradients), shrink only matrix/token sizes.
        plan=copy.deepcopy(PLAN)
        with tempfile.TemporaryDirectory() as td, patch.dict(PLAN,dict(L=12,T=11,rank=4)), contextlib.redirect_stdout(io.StringIO()):
            root=Path(td)/'run'; parents=Path(td)/'parents'; root.mkdir(); identity='tiny-pipeline-identity'
            generator=torch.Generator().manual_seed(912)
            weights={}; factors={}
            for name in PLAN['modules']:
                w=torch.randn(9,12,generator=generator)*.2; wq=(w*5).round()/5; weights[name]=w
                save_tensors(parents/'exp01/quantized'/(slug(name)+'.safetensors'),dict(W0=w,Wq=wq),
                    dict(W0_hash=mo.digest_tensor(w),Wq_hash=mo.digest_tensor(wq)))
                aa=torch.randn(12,12,generator=generator,dtype=torch.float64); gg=torch.randn(9,9,generator=generator,dtype=torch.float64)
                fp=parents/'exp03/factors'/slug(name)/'marginal_solve.safetensors'
                af=aa@aa.T+torch.eye(12); gf=gg@gg.T+torch.eye(9)
                if name==PLAN['modules'][1]: af=torch.eye(12,dtype=torch.float64); gf=torch.eye(9,dtype=torch.float64)
                save_tensors(fp,dict(A_solve=af,G_solve=gf),dict(identity=PLAN['parent_identity']))
                factors[name]=dict(path=str(fp),sha256=sha_file(fp),rebuilt=False)
            files={}; meta=[]
            for role in ('search','test'):
                ids=torch.randint(0,97,(PLAN[role+'_windows'],PLAN['L']),generator=generator)
                wp=root/'data'/f'{role}_windows.safetensors'; save_tensors(wp,dict(input_ids=ids,attention_mask=torch.ones_like(ids)),dict(identity=identity))
                files[role]=sha_file(wp)
                meta += [dict(role=role,window=i,article_id=role+str(i)) for i in range(len(ids))]
            save_json(root/'data/article_manifest.json',dict(identity=identity,files=files,windows=meta))
            config=dict(assets=str(parents),modules=PLAN['modules'],profile='dual4090')
            save_json(root/'manifest.json',dict(identity=identity,config=config))
            fit=torch.randint(0,97,(8,PLAN['L']),generator=generator)
            save_tensors(parents/'exp03/data/fit_windows.safetensors',dict(input_ids=fit),dict(identity=PLAN['parent_identity']))
            teacher=TinyTeacher(weights)
            for k in range(4):
                labels=torch.randint(0,23,(PLAN['T'],),generator=generator)
                save_tensors(parents/'exp03/data/fit_samples'/f'w00_k{k:03d}.safetensors',dict(labels=labels),
                    dict(identity=PLAN['parent_identity'],input_hash=mo.digest_tensor(fit[0]),label_hash=mo.digest_tensor(labels)))
            e=Experiment(root,config,identity,dict(factors=factors),QuietResources(),teacher=teacher,device='cpu')
            with self.assertRaises(FileNotFoundError): e.verify_selection(PLAN['modules'][0])
            for name in PLAN['modules']: e.run_module(name)
            self.assertEqual(teacher.gradient_count,512+2) # exactly one historical audit per module
            self.assertEqual(teacher.label_count,256) # search/test shared across both modules
            self.assertTrue(report(root))
            summary=read(root/'summary.json'); self.assertEqual(len(summary['modules']),2)
            self.assertEqual(summary['modules'][1]['selection']['baseline_selected'],'Marginal-AG')
            self.assertEqual(summary['modules'][1]['functional']['comparisons'][0]['article']['ci_high'],0.)
            self.assertEqual(summary['modules'][1]['KL']['comparisons'][0]['article']['ci_low'],0.)
            main_kl=sum(m['counts']['KL_main'] for m in summary['modules']); repeats=sum(m['counts']['KL_replays'] for m in summary['modules'])
            self.assertEqual(teacher.kl_count,main_kl+repeats)
            self.assertEqual(sum(m['counts']['search_logical_scores']+m['counts']['test_logical_scores'] for m in summary['modules']),3584)
            counters=teacher.gradient_count,teacher.kl_count,teacher.label_count
            for name in PLAN['modules']: e.run_module(name)
            self.assertEqual(counters,(teacher.gradient_count,teacher.kl_count,teacher.label_count))
            target=e.folder(PLAN['modules'][0])/'atomic/search/w00_k000.json'
            original=target.read_bytes(); row=read(target); row['physical']['None']['score']+=1; save_json(target,row)
            with self.assertRaises(RuntimeError): e.verify_selection(PLAN['modules'][0])
            target.write_bytes(original)
            with self.assertRaises(RuntimeError): commit(target,identity,unexpected='replacement')
            # The immutable manifest binds source/config even if result checksums are still valid.
            save_json(root/'manifest.json',dict(identity=identity,config=dict(config,profile='a6000')))
            with self.assertRaises(RuntimeError): e.verify_selection(PLAN['modules'][0])
        self.assertEqual(PLAN,plan)

    def test_resources_budget_survives_resume(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); r=Resources(root,'test',1,10)
            r.started-=3; r.close('STOPPED'); first=read(root/'resource_usage.json')['active_seconds']
            r=Resources(root,'test',1e-8,10)
            try:
                with self.assertRaises(BudgetReached): r.boundary()
                self.assertGreaterEqual(r.previous['active_seconds'],first)
            finally: r.close('INCOMPLETE_BUDGET')

    def test_atomic_checksum_tampering(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'sample.json'; commit(path,'x',score=1.)
            self.assertEqual(load_record(path,'x')['score'],1.)
            row=json.loads(path.read_text()); row['score']=2.; save_json(path,row)
            with self.assertRaises(RuntimeError): load_record(path,'x')


class RealTeacherPrimitiveTests(unittest.TestCase):
    def test_tiny_llama_shared_autograd_samples_and_stable_KL(self):
        from transformers import LlamaConfig, LlamaForCausalLM
        from teacher import Teacher
        torch.manual_seed(113)
        config=LlamaConfig(vocab_size=41,hidden_size=24,intermediate_size=40,num_hidden_layers=3,
            num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=64,attention_dropout=0.)
        config._attn_implementation='eager'
        model=LlamaForCausalLM(config).float().eval().requires_grad_(False)
        with tempfile.TemporaryDirectory() as td:
            teacher=Teacher(dict(vocab_chunk=4),Path(td),'tiny',QuietResources().timed)
            teacher.model=model
            name='model.layers.1.self_attn.v_proj'; ids=torch.randint(0,41,(1,12))
            ref,x,h=teacher.reference(name,ids); labels=teacher.labels(ref,[1,2])
            self.assertEqual(labels[0].shape,(11,)); self.assertFalse(torch.equal(labels[0],labels[1]))
            g,audit=teacher.gradient(name,ids,ref,x,labels[0],audit=True)
            self.assertLess(audit['S_autograd_relative_error'],1e-5)
            g2,_=teacher.gradient(name,ids,ref,x,labels[0],audit=False)
            self.assertLess(relative(g,g2),1e-6)
            self.assertLess(abs(teacher.kl(ref,ref)),1e-10)
            target=model.get_submodule(name); original=target.weight.detach().clone()
            weight=(original*50).round()/50; residual=original.double()-weight.double()
            value,_=teacher.intervention_kl(name,ids,ref,x,h,weight,residual)
            self.assertGreaterEqual(value,0); self.assertTrue(torch.equal(original,target.weight))


class TransferTests(unittest.TestCase):
    def test_auxiliary_bundle_checksums_and_no_overwrite(self):
        from inputs_bundle import pack, extract
        with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
            root=Path(td); one=root/'one'; three=root/'three'; one.mkdir(); three.mkdir()
            data=root/'dataset'; files={}
            for role in ('train','validation'):
                for name in ('dataset_info.json','state.json','data-00000-of-00001.arrow'):
                    p=data/role/name; p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes((role+name).encode())
                    files[role+'/'+name]=sha_file(p)
            cal=root/'calibration.safetensors'; cal.write_bytes(b'fixture-only')
            save_json(three/'data/fit_windows.json',dict(dataset_path=str(data),source_sha256=files,identity='fixture'))
            save_json(one/'data_manifest.json',dict(calibration_file_hash=sha_file(cal)))
            save_json(one/'identity.json',dict(config=dict(calibration=str(cal))))
            out=root/'bundle'; pack(one,three,out)
            dst=root/'extracted'; extract(out/'inputs.tar',out/'manifest.json',sha_file(out/'manifest.json'),dst)
            self.assertEqual(sha_file(dst/'calibration.safetensors'),sha_file(cal))
            with self.assertRaises(FileExistsError): extract(out/'inputs.tar',out/'manifest.json',sha_file(out/'manifest.json'),dst)
            with self.assertRaises(ValueError): extract(out/'inputs.tar',out/'manifest.json','0'*64,root/'bad')


if __name__=='__main__': unittest.main(verbosity=2)
