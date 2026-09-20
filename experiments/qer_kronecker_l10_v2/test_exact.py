"""Independent dense oracles and parallel failure propagation for the new path."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
from common import PLAN
from exact_ops import GroupedStream,contraction,metric_inner,gram_features,validate_reassociation
import sketch_math as sm

class ExactTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(923);torch.set_num_threads(2)
        self.groups=[(torch.randn(7,9,dtype=torch.float64),[torch.randn(7,6,dtype=torch.float64) for _ in range(k)]) for k in (4,16,3)]
        self.ss=[z.T@x for x,zs in self.groups for z in zs]
        self.samples=GroupedStream(lambda:iter(self.ss),lambda:iter(self.groups))
        a=torch.randn(9,9,dtype=torch.float64);g=torch.randn(6,6,dtype=torch.float64)
        self.a=a@a.T+torch.eye(9);self.g=g@g.T+torch.eye(6)
    def test_explicit_kronecker_inner_and_full_ALS(self):
        v=torch.stack([s.T.contiguous().flatten() for s in self.ss]);h=v.T@v/(len(v)*6)
        expected=float((h*torch.kron(self.a,self.g)).sum())
        actual=metric_inner(self.samples,self.a,self.g,6)
        self.assertLess(abs(actual-expected)/expected,1e-12)
        self.assertTrue(validate_reassociation(self.samples,self.a,self.g,6)['passed'])
    def test_all_groups_not_just_first_or_equal_labels(self):
        for side,f in [('A',self.g),('G',self.a)]:
            fast,n=contraction(self.samples,f,side,6);slow,m=sm.parent.contraction(lambda:iter(self.ss),f,side,6)
            self.assertEqual((n,m),(23,23));self.assertLess(sm.relative(fast,slow),1e-12)
    def test_gram_panels_tail_and_original(self):
        keys=list(range(len(self.ss)))
        for panel in (1,13,54,100):
            fast,hn=gram_features(keys,lambda i:self.ss[i],6,'cpu',features=panel)
            slow,sn=sm.gram_original(keys,lambda i:self.ss[i],6,'cpu',block=4)
            self.assertLess(sm.relative(fast,slow),1e-12);self.assertAlmostEqual(hn/sn,1.,places=12)
    def test_full_candidate_deployment_dense_vs_grouped(self):
        a,g=self.a,self.g;b,h=a,g
        for _ in range(6):
            a,g,_=sm.full_step(self.samples,a,g,6)
            b,h,_=sm.full_step(lambda:iter(self.ss),b,h,6)
        w0=torch.randn(6,9);wq=(w0/.1).round()*.1;e=w0.double()-wq.double()
        with patch.dict(PLAN,rank=3):
            _,fast,_=sm.solve(e,a,g,3,wq,w0);_,slow,_=sm.solve(e,b,h,3,wq,w0)
        self.assertLess(sm.relative(fast['R64'],slow['R64']),1e-10)
        self.assertTrue(torch.equal(fast['W_deploy'],slow['W_deploy']))
    def test_empty_or_invalid_gram_rejected(self):
        with self.assertRaises(RuntimeError):gram_features([],lambda i:None,6,'cpu')
        with self.assertRaises(RuntimeError):gram_features([0],lambda i:self.ss[i].float(),6,'cpu')

    def test_paired_text_label_identity_and_tamper_rejected(self):
        from test_pipeline import fixture
        from paired_data import prepare_paired,copy_label
        from common import save_json,sha_file,mo
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder,patch.dict(PLAN,shape=[9,12],rank=4,L=12,T=11,paired_L31_identity='tiny-integration-v1'):
            source,teacher=fixture(Path(folder));target=Path(folder)/'paired'
            config=dict(paired_source=str(source.root));index=prepare_paired(target,config,'new-layer')
            self.assertEqual(index['fit'],source.index['fit']);self.assertEqual(index['eval'],source.index['eval'])
            row=source.index['fit'][0];label=torch.arange(11,dtype=torch.int64)
            lp=source.root/'data/labels/fit'/(row['id']+'.safetensors')
            source.store.put(lp,dict(labels=label),**row)
            source.config.update(config)
            save_json(source.root/'asset_identity.json',dict(files={str(lp):sha_file(lp)}))
            out=target/'copied.safetensors';copy_label(source,'fit',row,out)
            self.assertEqual(source.store.get(out)[1]['tensor_hashes']['labels'],mo.digest_tensor(label))
            with lp.open('ab') as f:f.write(b'changed')
            with self.assertRaisesRegex(RuntimeError,'changed after inventory'):copy_label(source,'fit',row,out)

    def test_parallel_worker_failure_propagates(self):
        from test_pipeline import fixture
        from parallel import offline_map
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder,patch.dict(PLAN,shape=[9,12],rank=4,L=12,T=11):
            e,_=fixture(Path(folder))
            def fail(worker,item):raise RuntimeError('intentional worker error')
            with self.assertRaisesRegex(RuntimeError,'intentional worker error'):offline_map(e,[0,1],fail)

if __name__=='__main__':unittest.main(verbosity=2)
