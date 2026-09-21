import tempfile
import unittest
import time
from types import SimpleNamespace
from pathlib import Path
import torch
from s_math import weighted_sum,pilot,compare,scale_relation
from s_common import Source,read,save_json
from s_statistics import checkpoint,restore
from s_common import MODULES,BASELINES,METHOD,slug,commit,sha_file,mo
from s_report import report


class Tests(unittest.TestCase):
    def test_math_controls_and_global_normalization(self):
        torch.manual_seed(41);x=torch.randn(17,13,dtype=torch.float64);g=torch.randn(17,7,dtype=torch.float64)
        self.assertTrue(pilot(x,g)['first_Token_A']['passed'])
        u,d,w=weighted_sum(x,g);g2=g*10;u2,d2,_=weighted_sum(x,g2)
        compare(u/d,u2/d2)
        # Vary both input and mass: global normalization must not average per-window A.
        u2,d2,_=weighted_sum(x*3,g2)
        actual=(u+u2)/(d+d2);wrong=(u/d+u2/d2)/2
        self.assertGreater(float((actual-wrong).norm()),.1)
        canonical=g.T@g/16;compare(d/16,canonical.trace())
        with self.assertRaises((RuntimeError,AssertionError)):compare(d/17,canonical.trace())
        self.assertEqual(w.shape,(17,))

    def test_gauge_full_G_and_wrong_G_rejected(self):
        torch.manual_seed(5);a=torch.randn(8,8,dtype=torch.float64);a=a.T@a
        g=torch.randn(6,6,dtype=torch.float64);g=g.T@g
        raw=dict(A_raw=a/a.norm(),G_raw=g*a.norm());self.assertGreater(scale_relation(raw,a,g)['raw_G_over_canonical'],0)
        bad=dict(raw,G_raw=raw['G_raw'].diag().diag())
        with self.assertRaises((RuntimeError,AssertionError)):scale_relation(bad,a,g)
        self.assertTrue(compare(torch.zeros(2),torch.zeros(2))['passed'])

    def test_checkpoint_resume_retention_and_tamper(self):
        class Store:
            def __init__(self):self.evidence={};self.verified={}
        source=Source.__new__(Source);source.store=Store();source.records={}
        source.base=type('Base',(),dict(store=Store()))()
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);u=torch.eye(4,dtype=torch.float64);d=torch.tensor(3.,dtype=torch.float64)
            rows=[]
            for n in range(1,4):
                rows.append(dict(window=n-1,sum_w=3.))
                checkpoint(folder,'test','K',u*n,d*n,rows,source)
            self.assertEqual(len(list((folder/'progress').glob('*.safetensors'))),2)
            ur,dr,rr=restore(folder,'test','K',source,'cpu')
            compare(ur,u*3);compare(dr.reshape(()),d*3);self.assertEqual(len(rr),3)
            p=folder/'progress/latest.json';row=read(p);row['count']=2;save_json(p,row)
            with self.assertRaises((RuntimeError,AssertionError)):restore(folder,'test','K',source,'cpu')

    def test_report_aggregate_KL_and_paired_sign(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'new';old=Path(tmp)/'old';validation=torch.arange(16*4).reshape(16,4)
            source=SimpleNamespace(root=old,identity='parent',validation=validation)
            commit(root/'candidate_freeze.json','test',files={});freeze=sha_file(root/'candidate_freeze.json')
            commit(root/'scores/baseline_replay.json','test',passed=True)
            commit(root/'pilot/math_checks.json','test',passed=True)
            hashes={}
            for name in MODULES:
                commit(root/'statistics'/slug(name)/'weight_summary.json','test',weight_effective_count=10.,positions=524288,max_w=1.,D=20.)
                for w in range(16):
                    scale=w+1;token=mo.digest_tensor(validation[w])
                    for method in BASELINES:
                        key='None' if method=='None' else 'N256__'+method
                        p=old/'scores/validation'/f'w{w:04d}'/(slug(name)+'___'+key+'.json')
                        value={'None':2.,'Marginal':1.,'A-only':1.5,'Token-joint':.75,'Sequence-one-step':.7}[method]*scale
                        commit(p,'parent',token_hash=token,scores=dict(KL=value,tokens=2047));hashes[str(p)]=sha_file(p)
                    commit(root/'scores/validation'/f'w{w:04d}'/(slug(name)+'.json'),'test',module=name,window=w,
                           candidate=METHOD,token_hash=token,freeze_hash=freeze,scores=dict(KL=.8*scale,tokens=2047))
            commit(root/'parent_assets_audit.json','test',baseline_record_hashes=hashes)
            resources=SimpleNamespace(base=0.,started=time.monotonic(),data={'timings':[]})
            report(root,{},'test',resources,source)
            result=read(root/'summary/results.json');new=[r for r in result['rows'] if r['method']==METHOD]
            self.assertEqual(len(new),4)
            for r in new:self.assertAlmostEqual(r['recovery_percent'],60.);self.assertAlmostEqual(r['gain_vs_Marginal_pp'],10.)
            pair=[p for p in result['paired_comparisons'] if p['reference']=='Marginal'][0]
            self.assertEqual(pair['wins'],16);self.assertAlmostEqual(pair['exploratory_low_pp'],10.)


if __name__=='__main__':unittest.main()
