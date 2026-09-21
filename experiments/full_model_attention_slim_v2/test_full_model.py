import tempfile
import unittest
from pathlib import Path
import torch
from fm_common import *
from pilot import mathematical_checks

class Fixture:
    identity='test-only'
    def __init__(self,path):self.root=Path(path)
    cleanup_temporary=Context.cleanup_temporary

class FullModelTests(unittest.TestCase):
    def test_offline_boundary_rejects_retained_teacher_alias(self):
        from types import SimpleNamespace
        model=torch.nn.Linear(3,3)
        teacher=SimpleNamespace(model=model)
        teacher.unload=lambda:setattr(teacher,'model',None)
        with self.assertRaisesRegex(RuntimeError,'Teacher still referenced'):
            release_for_solve(teacher)

    def test_offline_boundary_collects_scoped_teacher(self):
        from types import SimpleNamespace
        teacher=SimpleNamespace(model=torch.nn.Linear(3,3))
        teacher.unload=lambda:setattr(teacher,'model',None)
        reference=weakref.ref(teacher.model)
        def phase():
            model=teacher.model
            return model.weight.detach().cpu().clone()
        result=phase()
        self.assertTrue(release_for_solve(teacher)['teacher_collected'])
        self.assertIsNone(reference())
        self.assertEqual(tuple(result.shape),(3,3))

    def test_recovery_rejects_modified_payload_and_path_escape(self):
        from recovery import checked_parent_file
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);p=root/'frozen.json';write(p,{'tokens':'original'})
            expected=sha(p)
            self.assertEqual(checked_parent_file(root,'frozen.json',expected),p.resolve())
            write(p,{'tokens':'modified'})
            with self.assertRaisesRegex(RuntimeError,'file changed'):
                checked_parent_file(root,'frozen.json',expected)
            with self.assertRaisesRegex(RuntimeError,'escapes'):
                checked_parent_file(root,'../outside',expected)

    def test_scalar_solver_matches_parent(self):
        self.assertTrue(mathematical_checks()['A_only_scalar_parent']['passed'])

    def test_general_weighted_solver_matches_parent(self):
        from solve import solve_one,roots
        torch.manual_seed(73)
        x=torch.randn(21,13,dtype=torch.float64);z=torch.randn(21,10,dtype=torch.float64)
        a=x.T@x;g=z.T@z;w0=torch.randn(10,13);wq=w0+.02*torch.randn(10,13)
        error=w0.double()-wq.double()
        result,audit=solve_one(error,a,g,w0,wq,rank=4,prepared=roots(a))
        reference,_=sm.parent.weighted_svd(error,a,g,4,.001,.001)
        self.assertTrue(compare(result['P64']@result['Q64'],reference['C64'],1e-10)['passed'])

    def test_frozen_candidate_budgets(self):
        count={s:sum(method(s,k) is not None for i in range(32) for k in KINDS) for s in STATES}
        self.assertEqual(count,dict(Teacher=0,C0=0,C1=96,C2=224,C3=224,C4=224))
        unique={(i,k,method(s,k)) for s in STATES for i in range(32) for k in KINDS if method(s,k)}
        self.assertEqual(len(unique),416)
        self.assertEqual({method('C4','k') for i in range(32)},{'Token-joint-one'})

    def test_checkpoint_transaction_and_tampering(self):
        from pilot import resume_check
        with tempfile.TemporaryDirectory() as folder:
            ctx=Fixture(folder);self.assertTrue(resume_check(ctx)['passed'])
            acc=Accumulator(ctx,'fixture',{'A':(2,2)},{'data':'original'})
            acc.add({'A':torch.eye(2,dtype=torch.float64)},0);acc.save()
            with self.assertRaises(RuntimeError):Accumulator(ctx,'fixture',{'A':(2,2)},{'data':'changed'})
            with self.assertRaises(RuntimeError):acc.add({'A':torch.eye(2,dtype=torch.float64)},0)
            with self.assertRaises(RuntimeError):ctx.cleanup_temporary(Path(folder))

    def test_first_step_sync_and_v_gqa(self):
        torch.manual_seed(17)
        samples=[(torch.randn(7,9,dtype=torch.float64),torch.randn(7,4,dtype=torch.float64)) for _ in range(2)]
        a=sum(x.T@x for x,g in samples)/14
        expected_a,expected_g=sm.token_step(lambda:iter(samples),a,torch.eye(4,dtype=torch.float64),6)
        aa=sum(x.T@(x*g.square().sum(-1)[:,None]) for x,g in samples)/(2*6*4)
        gg=sum(g.T@(g*((x@a)*x).sum(-1)[:,None]) for x,g in samples)/(2*6*a.square().sum())
        self.assertTrue(compare(aa,expected_a,1e-10)['passed']);self.assertTrue(compare(gg,expected_g,1e-10)['passed'])
        p=torch.randn(4,7,7).softmax(-1);x=torch.randn(7,9);delta=torch.randn(7,8)
        out=effective(p,x,delta,[0,0,1,1],2,'cpu',explicit=True)
        self.assertTrue(compare(out['A_sum'],out['explicit_A_sum'],1e-10)['passed'])
        av,gv=normalize(out['A_sum'],out['G_blocks_sum'],1,7,4)
        self.assertEqual(tuple(gv.shape),(4,4));self.assertTrue(torch.equal(gv[:2,2:],torch.zeros_like(gv[:2,2:])))

if __name__=='__main__':unittest.main()
