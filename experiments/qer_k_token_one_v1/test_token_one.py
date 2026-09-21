import tempfile
import unittest
from pathlib import Path
import time
from types import SimpleNamespace
import torch
from j_common import Source,MODULES,BASELINES,METHOD,slug,mo,commit,sha_file,read,save_json,compare
from j_math import contract,pilot,recover_a
from j_statistics import checkpoint,restore
from j_report import report


class Tests(unittest.TestCase):
    def test_real_initialization_synchronous_parent_equivalence(self):
        torch.manual_seed(88);samples=[(torch.randn(17,13,dtype=torch.float64),torch.randn(17,7,dtype=torch.float64)) for _ in range(2)]
        am=sum(x.T@x for x,g in samples)/34
        terms=[];rows=[]
        for x,g in samples:
            v,u,row=contract(x,g,am);terms.append((v,u));rows.append(row)
        checks=pilot(samples,am,terms,16);self.assertTrue(checks['parent_first_A']['passed']);self.assertTrue(checks['parent_first_G']['passed'])
        compare(sum(r['sum_u'] for r in rows),34*am.square().sum())
        self.assertGreater(float((terms[0][0]-contract(*samples[0],torch.eye(13,dtype=torch.float64))[0]).norm()),.01)
        newa=sum(x.T@(x*g.square().sum(-1)[:,None]) for x,g in samples)/(2*16*7)
        self.assertGreater(float((terms[0][0]-contract(*samples[0],newa)[0]).norm()),.01)
        with self.assertRaises((RuntimeError,AssertionError)):contract(*samples[0],-am)

    def test_reuse_A_scaling_and_no_window_normalization(self):
        torch.manual_seed(3);x=torch.randn(17,13,dtype=torch.float64);g=torch.randn(17,7,dtype=torch.float64)
        w=g.square().sum(-1);u=x.T@(x*w[:,None]);d=w.sum().reshape(1)
        parent=dict(U=u,D=d,A_raw=u/d,G_raw=g.T@g/16)
        a1,checks=recover_a(parent,n=1,T=16,m=7);compare(a1,u/(16*7));self.assertGreater(checks['A1_over_sensitivity_A'],0)
        with self.assertRaises((RuntimeError,AssertionError)):recover_a(dict(parent,D=d*2),n=1,T=16,m=7)

    def test_checkpoint_resume_and_parent_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);v=torch.eye(7,dtype=torch.float64);rows=[]
            for i in range(1,4):
                rows.append(dict(window=i-1));checkpoint(folder,'test','K',v*i,rows,'parenthash')
            self.assertEqual(len(list((folder/'progress').glob('*.safetensors'))),2)
            actual,rr=restore(folder,'test','K','parenthash','cpu',m=7);compare(actual,v*3);self.assertEqual(len(rr),3)
            with self.assertRaises((RuntimeError,AssertionError)):restore(folder,'test','K','changed','cpu',m=7)
            path=folder/'progress/latest.json';bad=read(path);bad['count']=2;save_json(path,bad)
            with self.assertRaises((RuntimeError,AssertionError)):restore(folder,'test','K','parenthash','cpu',m=7)

    def test_report_mixed_parent_sources_and_contrast_direction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'new';source=Source.__new__(Source);source.root=Path(tmp)/'ko';source.sroot=Path(tmp)/'sens'
            source.identity='ko';source.sid='sens';source.validation=torch.arange(64).reshape(16,4)
            commit(root/'candidate_freeze.json','test',files={});freeze=sha_file(root/'candidate_freeze.json')
            commit(root/'scores/baseline_replay.json','test',passed=True);commit(root/'pilot/one_step_equivalence.json','test',passed=True)
            hashes={}
            for name in MODULES:
                for w in range(16):
                    token=mo.digest_tensor(source.validation[w]);scale=w+1
                    for method in BASELINES:
                        p,pid=source.baseline_path(name,method,w)
                        value={'None':2.,'Marginal':1.,'Sensitivity-A':.9,'Token-joint-3':.7,'Sequence':.75}[method]*scale
                        commit(p,pid,token_hash=token,scores=dict(KL=value,tokens=2047));hashes[str(p)]=sha_file(p)
                    commit(root/'scores/validation'/f'w{w:04d}'/(slug(name)+'.json'),'test',module=name,window=w,candidate=METHOD,
                           token_hash=token,freeze_hash=freeze,scores=dict(KL=.8*scale,tokens=2047))
            commit(root/'parent_assets_audit.json','test',baseline_record_hashes=hashes)
            report(root,{},'test',SimpleNamespace(base=0.,started=time.monotonic(),data={'timings':[]}),source)
            result=read(root/'summary/results.json');new=[r for r in result['rows'] if r['method']==METHOD]
            for r in new:self.assertAlmostEqual(r['recovery_percent'],60.)
            for p in result['paired_comparisons']:
                if p['left']=='Token-joint-3':self.assertAlmostEqual(p['recovery_gain_left_minus_right_pp'],5.);self.assertEqual(p['wins'],16)
                if p['right']=='Sensitivity-A':self.assertAlmostEqual(p['recovery_gain_left_minus_right_pp'],5.)
            self.assertEqual(len(hashes),320)


if __name__=='__main__':unittest.main()
