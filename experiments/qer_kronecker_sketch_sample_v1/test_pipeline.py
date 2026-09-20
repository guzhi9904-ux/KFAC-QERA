"""CPU integration: real summed-loss autograd, all registered sample/candidate counts."""
import contextlib
import gc
import io
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import torch
import torch.nn.functional as F
from common import PLAN,mo,slug,save_json,save_tensors,read,sha_file
from data import sample_views
from experiment import Experiment
from diagnostics import run as diagnose
from report import report


class TinyResources:
    def __init__(self,root):
        self.root=root;self.data=dict(active_seconds=0.,timings=[],GPU_peaks={})
    def boundary(self):pass
    def flush(self):save_json(self.root/'resource_usage.json',self.data)
    @contextlib.contextmanager
    def timed(self,stage,**extra):
        start=time.monotonic();yield
        seconds=time.monotonic()-start
        self.data['active_seconds']+=seconds
        self.data['timings'].append(dict(stage=stage,seconds=seconds,completed=True,**extra));self.flush()


class TinyTeacher:
    """Nonlinear causal toy network; g is differentiated from actual summed CE."""
    def __init__(self,w):
        self.w=w.clone();self.embedding=torch.randn(23,w.shape[1])*.4
        self.head=torch.randn(w.shape[0],23)*.3;self.gradients=0;self.interventions=0
    def unload(self):pass
    def forward(self,ids,weight=None,delta=None):
        x=self.embedding[ids];h=x@(self.w if weight is None else weight).T
        if delta is not None:h=h-delta
        # Prefix mixing creates genuine cross-position coupling.
        hidden=torch.tanh(h.cumsum(1)/torch.arange(1,h.shape[1]+1).view(1,-1,1).sqrt())
        return hidden@self.head,x,h
    def reference(self,name,ids):return self.forward(ids)
    def labels(self,reference,seeds):
        probabilities=reference[0,:-1].double().softmax(-1)
        return [torch.multinomial(probabilities,1,generator=torch.Generator().manual_seed(s)).flatten() for s in seeds]
    def gradient(self,name,ids,reference,x,labels,audit=False):
        weight=self.w.clone().requires_grad_(True);logits,xx,h=self.forward(ids,weight)
        loss=F.cross_entropy(logits[0,:-1],labels,reduction='sum')
        g,gw=torch.autograd.grad(loss,(h,weight));g=g[0].double();self.gradients+=1
        checks=dict(sum_NLL=True)
        if audit:
            err=float((g.T@xx[0].double()-gw.double()).norm()/gw.double().norm())
            if err>PLAN['tolerances']['autograd']:raise RuntimeError('Toy autograd mismatch')
            checks['S_autograd_relative_error']=err
        return g,checks
    def kl(self,reference,actual):return float(mo.stable_kl(reference[0,:-1],actual[0,:-1]).mean())
    def intervention_kl(self,name,ids,reference,x,h,weight,residual):
        self.interventions+=1;actual,xx,hh=self.forward(ids,weight)
        if not torch.equal(xx,x):raise RuntimeError('Input changed')
        return self.kl(reference,actual),dict(relative_l2=0.,cosine=1.)
    def output_perturbation_kl(self,name,ids,reference,x,residual):
        logits,_,_=self.forward(ids,delta=(x.double()@residual.T).float())
        return self.kl(reference,logits)


def fixture(folder):
    root=folder/'run';root.mkdir();assets=folder/'assets';identity='tiny-integration-v1'
    config=dict(assets=str(assets),cache_GiB=.02,budget_hours=10,gram_block=8)
    w0=torch.randn(9,12)*.4;wq=(w0/.08).round()*.08
    save_tensors(assets/'exp01/quantized'/(slug(PLAN['module'])+'.safetensors'),dict(W0=w0,Wq=wq),{})
    teacher=TinyTeacher(w0);rows={};ids={}
    for role,count in [('fit',32),('eval',16)]:
        ids[role]=torch.randint(0,23,(count,PLAN['L']))
        rows[role]=[dict(window=c,role=role,token_hash=mo.digest_tensor(t),article_title=f'{role}-{c}',article_text_sha256=f'{role}-{c}') for c,t in enumerate(ids[role])]
        save_tensors(root/'data'/f'{role}_windows.safetensors',dict(input_ids=ids[role]),dict(identity=identity))
        save_json(root/'data'/f'{role}_windows.json',dict(identity=identity,windows=rows[role]))
    index=dict(identity=identity,**sample_views(rows['fit'],rows['eval']))
    save_json(root/'data/sample_index.json',index)
    files={p.relative_to(root).as_posix():sha_file(p) for p in (root/'data').iterdir()}
    save_json(root/'data/data_freeze.json',dict(identity=identity,files=files))
    save_json(root/'manifest.json',dict(identity=identity))
    for c in range(8):
        reference,_,_=teacher.reference(PLAN['module'],ids['fit'][c:c+1])
        for k,label in enumerate(teacher.labels(reference,[100+c*4+k for k in range(4)])):
            save_tensors(assets/'exp03/data/fit_samples'/f'w{c:02d}_k{k:03d}.safetensors',dict(labels=label),
                dict(identity=PLAN['parent_identity'],input_hash=mo.digest_tensor(ids['fit'][c]),label_hash=mo.digest_tensor(label)))
    resources=TinyResources(root)
    return Experiment(root,config,identity,resources,teacher=teacher,device='cpu'),teacher


class PipelineTests(unittest.TestCase):
    def test_all_samples_freeze_resume_diagnostics_and_report(self):
        torch.manual_seed(928);torch.set_num_threads(2)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp,patch.dict(PLAN,shape=[9,12],rank=4,L=12,T=11),contextlib.redirect_stdout(io.StringIO()):
            e,t=fixture(Path(tmp));root=e.root
            with self.assertRaisesRegex(RuntimeError,'before candidate freeze'):
                e.labels('eval',0,None,[e.index['eval'][0]])
            # A pilot's two real gradients are committed once and reused in the formal union.
            first=[r['id'] for r in e.index['fit'][:2]]
            e.pilot();self.assertEqual(t.gradients,2);self.assertTrue(read(root/'pilot.json')['passed'])
            t.interventions=0
            e.collect_fit(first);self.assertEqual(t.gradients,2)
            e.collect_fit();self.assertEqual(t.gradients,224)
            e.fit();self.assertEqual(len(e.check_candidates()),15)
            self.assertEqual(read(root/'statistics/S0.json')['A_hash'],read(root/'statistics/S1.json')['A_hash'])
            self.assertEqual(read(root/'statistics/S0.json')['A_asset'],read(root/'statistics/S1.json')['A_asset'])
            self.assertTrue(torch.equal(e.store.get(root/'corrections/None.safetensors')[0]['R64'],e.error))
            e.evaluate();self.assertEqual(t.gradients,480);self.assertEqual(t.interventions,255)
            e.evaluate();e.collect_fit();self.assertEqual(t.gradients,480);self.assertEqual(t.interventions,255)
            diagnose(e);diag=read(root/'summary/diagnostics.json')
            self.assertEqual(len(diag['bounds']),48);self.assertTrue(all(r['passed'] for r in diag['bounds']))
            e.resources.flush();self.assertTrue(report(root))
            v=read(root/'verification.json');self.assertEqual((v['fit_gradients'],v['eval_gradients'],v['q_rows'],v['KL_main_rows']),(224,256,3840,240))
            self.assertTrue(v['cache_checksums_verified'])
            self.assertTrue(report(root,verify=False))
            # A denominator below the registered floor has no normalized bound;
            # absolute results must still produce a report rather than min([]).
            for row in diag['bounds']:row['U_over_None']=None
            save_json(root/'summary/diagnostics.json',diag)
            self.assertTrue(report(root,verify=False))
            self.assertIn('仅报告绝对界值',(root/'RESULTS.md').read_text(encoding='utf8'))
            # Fail closed if a committed candidate changes after scores have been made.
            p=root/'manifest.json';p.write_bytes(p.read_bytes()+b'\n')
            with self.assertRaises(RuntimeError):e.check_candidates()
            e.offline();del e;gc.collect()


if __name__=='__main__':unittest.main(verbosity=2)
