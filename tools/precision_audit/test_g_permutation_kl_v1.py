"""CPU numerical, intervention and checkpoint tests; full server pilot remains required."""
import contextlib
import copy
import gc
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
import g_permutation_kl_v1 as exp


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding=torch.nn.Embedding(7,64,dtype=torch.bfloat16)
        self.proj=torch.nn.Linear(64,64,bias=False,dtype=torch.bfloat16)
        self.head=torch.nn.Linear(64,7,bias=False,dtype=torch.bfloat16)
    def forward(self,input_ids,**kwargs):
        return SimpleNamespace(logits=self.head(self.proj(self.embedding(input_ids))))


class NumericalTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(27)

    def test_fixed_coverage_and_paired_seed_policy(self):
        self.assertEqual(len(exp.MODULES),28);self.assertEqual(len(exp.ARMS),9)
        for i in range(3):
            p=exp.permutation('module',80,i)
            self.assertTrue(torch.equal(p,exp.permutation('module',80,i)))
            self.assertFalse(torch.equal(p,exp.permutation('another-module',80,i)))
            self.assertEqual(sorted(p.tolist()),list(range(80)))
        self.assertFalse(torch.equal(exp.permutation('module',80,0),exp.permutation('module',80,1)))

    def test_permutation_preserves_effective_metric_spectrum(self):
        x=torch.randn(13,13,dtype=torch.float64)
        root=x@x.T+torch.eye(13)
        p=exp.permutation('test',13,0)
        changed=exp.permute_root(root,p)
        g=root@root.T; actual=changed@changed.T
        self.assertTrue(torch.allclose(actual,g[p][:,p],atol=1e-10,rtol=1e-12))
        self.assertTrue(torch.allclose(torch.linalg.eigvalsh(actual),torch.linalg.eigvalsh(g),atol=1e-10,rtol=1e-12))
        self.assertAlmostEqual(float(g.trace()),float(actual.trace()),places=9)
        with self.assertRaisesRegex(RuntimeError,'bijective'):exp.permute_root(root,torch.zeros(13,dtype=torch.int64))
        self.assertTrue(torch.equal(exp.permute_root(root,torch.arange(13)),root))

    def test_identity_and_permuted_closed_form_keep_error_and_a_fixed(self):
        e=torch.randn(23,19,dtype=torch.float64)
        a=torch.diag(torch.linspace(.4,2.,23,dtype=torch.float64))
        x=torch.randn(19,19,dtype=torch.float64);g=x@x.T+torch.eye(19)
        before=(e.clone(),a.clone(),g.clone())
        l,b,s,_=exp.single.solve_fp64(e,a,g,5)
        li,bi,_,_=exp.single.solve_fp64(e,a,exp.permute_root(g,torch.arange(19)),5)
        self.assertTrue(torch.allclose(l@b,li@bi,atol=1e-12))
        gp=exp.permute_root(g,exp.permutation('test',19,1))
        lp,bp,sp,_=exp.single.solve_fp64(e,a,gp,5)
        self.assertTrue(torch.allclose((a@(e-lp@bp)@gp).square().sum(),sp[5:].square().sum(),rtol=1e-11))
        self.assertFalse(torch.allclose(l@b,lp@bp))
        self.assertTrue(all(torch.equal(v,w) for v,w in zip((e,a,g),before)))

    def factor_context(self,tmp):
        root=Path(tmp);out=root/'out';out.mkdir()
        a=torch.diag(torch.linspace(.5,2,80));g=torch.diag(torch.linspace(.3,3,72))
        source=exp.single.atomic_tensors(root/'a.safetensors',{'full':a})
        binding={'shape':[72,80],'root':source}
        ctx=SimpleNamespace(output=out,identity='test',device='cpu',tasks={'proj':binding},
            inputs=exp.core.Inputs(),dg={'proj':torch.ones(72)},
            factors={m:{'proj':{'root_bits':{'a':exp.single.tensor_record(a),'g':exp.single.tensor_record(g)}}} for m in exp.REAL[1:]})
        return ctx,g,torch.randn(80,72)*.01

    def test_factor_transaction_resume_direct_cast_and_tamper(self):
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            ctx,g,e=self.factor_context(tmp)
            with patch.object(exp.parent,'make_g_root',return_value=(g,{})),patch.object(exp.frozen,'residual',return_value=e):
                r=exp.prepare_factor(ctx,'proj','full_gd_p0')
            with patch.object(exp.single,'solve_fp64',side_effect=AssertionError('completed solve repeated')):
                self.assertEqual(exp.prepare_factor(ctx,'proj','full_gd_p0'),r)
            self.assertLess(abs(r['tail_relative_error']),1e-9)
            ctx.factor_cache.clear();ctx.root_cache.clear();ctx.inputs=exp.core.Inputs()
            self.assertEqual(exp.read_factor(ctx,'proj','full_gd_p0'),r)
            path=exp.factor_path(ctx,'proj','full_gd_p0')
            data=exp.core.read_json(path);data['seed']=123;exp.core.atomic_json(path,data)
            ctx.factor_cache.clear()
            with self.assertRaisesRegex(RuntimeError,'metadata changed'):exp.read_factor(ctx,'proj','full_gd_p0')

    def test_failed_solve_and_changed_original_root_do_not_commit(self):
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            ctx,g,e=self.factor_context(tmp)
            with patch.object(exp.parent,'make_g_root',return_value=(g+1,{})):
                with self.assertRaisesRegex(RuntimeError,'bits'):exp.prepare_root(ctx,'proj','full_gd')
            self.assertFalse(exp.root_path(ctx,'proj','full_gd').exists())
            with patch.object(exp.parent,'make_g_root',return_value=(g,{})),patch.object(exp.frozen,'residual',return_value=e),patch.object(exp.single,'solve_fp64',side_effect=RuntimeError('injected')):
                with self.assertRaisesRegex(RuntimeError,'injected'):exp.prepare_factor(ctx,'proj','full_gd_p0')
            self.assertFalse(exp.factor_path(ctx,'proj','full_gd_p0').exists())
            gc.collect()  # Release mock/exception-held safetensors mappings on Windows.

    def test_seed_mean_uses_same_windows_and_direction(self):
        rows=[{'tokens':i+1,'real_kl_sum':2*(i+1),'s0_kl_sum':3*(i+1),'s1_kl_sum':4*(i+1),'s2_kl_sum':5*(i+1)} for i in range(16)]
        self.assertEqual(exp.signed_stats(rows,['s0','s1','s2'],['real'],'kl',draws=100),{'delta':2.,'ci95_low':2.,'ci95_high':2.})

    def test_isotropic_g_is_kept_as_valid_null_control(self):
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            ctx,_,e=self.factor_context(tmp);g=torch.eye(72)
            ctx.factors['full_gd']['proj']['root_bits']['g']=exp.single.tensor_record(g)
            with patch.object(exp.parent,'make_g_root',return_value=(g,{})),patch.object(exp.frozen,'residual',return_value=e):
                r=exp.prepare_factor(ctx,'proj','full_gd_p0')
            self.assertTrue(r['unchanged_metric_root']);self.assertEqual(r['root_relative_change'],0.)


class WindowTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(902)

    @contextlib.contextmanager
    def fixture(self):
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()),patch.object(exp,'WINDOWS',3),patch.object(exp,'LENGTH',9),patch.object(exp.local,'LENGTH',9):
            folder=Path(tmp);out=folder/'out';out.mkdir()
            teacher,student=Tiny().eval(),Tiny().eval();student.load_state_dict(teacher.state_dict())
            q=(teacher.proj.weight.float()+.01).bfloat16()
            raw,files,values={},{},{}
            for arm in exp.ARMS:
                a=(torch.randn(64,64)*.01).bfloat16();b=(torch.randn(64,64)*.01).bfloat16()
                values[arm]=(a,b)
                files[arm]={'file':exp.single.atomic_tensors(folder/(arm+'.safetensors'),{'A':a,'B':b})}
                if arm in exp.REAL:
                    raw[arm]={'proj':{'file':exp.single.atomic_tensors(folder/(arm+'_raw.safetensors'),{'A_fp64':a.double(),'B_fp64':b.double()})}}
            ids=torch.arange(9).reshape(1,-1)%7;mask=torch.ones_like(ids)
            old,_=exp.local.compare_window(teacher,student,'proj',values,q,ids,mask,True)
            baseline=exp.local.tensor_rows(old,0)
            qfile=exp.single.atomic_tensors(folder/'q.safetensors',{'weight_q':q})
            ctx=SimpleNamespace(output=out,identity='test',modules=('proj',),inputs=exp.core.Inputs(),factors=raw,
                tasks={'proj':{'shape':[64,64],'quant':qfile}},old_records={'proj':{'sha256':'frozen'}},
                old_rows={'proj':[{**baseline,'window':i} for i in range(3)]})
            data={'input_ids':ids.repeat(3,1),'attention_mask':mask.repeat(3,1)}
            with patch.object(exp,'read_factor',side_effect=lambda ctx,name,arm:files[arm]),patch.object(exp.frozen,'windows',return_value=data):
                yield ctx,teacher,student,files
            gc.collect()

    def test_real_nine_arm_resume_restore_and_pack_completeness(self):
        with self.fixture() as (ctx,t,s,files):
            before=exp.local.deployment(s)
            with self.assertRaises(exp.prev.Paused):exp.evaluate_module(ctx,t,s,'proj',exp.prev.Budget(10,1))
            self.assertEqual(len(exp.read_state(ctx,'proj')['records']),1)
            exp.evaluate_module(ctx,t,s,'proj',exp.prev.Budget(10,None))
            self.assertTrue(exp.read_state(ctx,'proj')['complete'])
            with patch.object(exp.local,'compare_window',side_effect=AssertionError('reran completed')):
                exp.evaluate_module(ctx,t,s,'proj',exp.prev.Budget(10,None))
            self.assertEqual(before,exp.local.deployment(s));self.assertFalse(s.proj._forward_hooks)
            self.assertEqual(exp.local.METHODS,exp.REAL)
            with patch.object(exp,'summarize',return_value={'complete':False}):
                with self.assertRaisesRegex(RuntimeError,'Incomplete'):exp.pack(ctx)

    def test_replay_failure_does_not_commit_paired_window(self):
        with self.fixture() as (ctx,t,s,files):
            ctx.old_rows['proj'][0]['full_gf_kl_sum']+=1
            before=exp.local.deployment(s)
            with self.assertRaisesRegex(RuntimeError,'replay failed'):exp.evaluate_module(ctx,t,s,'proj',exp.prev.Budget(10,None))
            self.assertFalse(exp.state_path(ctx,'proj').exists())
            self.assertEqual(before,exp.local.deployment(s));self.assertEqual(exp.local.METHODS,exp.REAL)

    def test_middle_arm_failure_restores_globals_and_weights(self):
        with self.fixture() as (ctx,t,s,files):
            original=exp.local.local_tokens;count=0
            def fail(*args):
                nonlocal count
                count+=1
                if count==5:raise RuntimeError('middle arm failed')
                return original(*args)
            before=exp.local.deployment(s)
            with patch.object(exp.local,'local_tokens',side_effect=fail):
                with self.assertRaisesRegex(RuntimeError,'middle arm'):exp.evaluate_module(ctx,t,s,'proj',exp.prev.Budget(10,None))
            self.assertEqual(before,exp.local.deployment(s));self.assertFalse(s.proj._forward_hooks)
            self.assertEqual(exp.local.METHODS,exp.REAL);self.assertFalse(exp.state_path(ctx,'proj').exists())

    def test_pilot_separate_and_checkpoint_tamper_rejected(self):
        with self.fixture() as (ctx,t,s,files):
            exp.evaluate_module(ctx,t,s,'proj',exp.prev.Budget(10,None),pilot=True)
            self.assertTrue(exp.read_state(ctx,'proj',pilot=True)['complete'])
            self.assertFalse(exp.read_state(ctx,'proj')['complete'])
            path=exp.state_path(ctx,'proj',True);value=exp.core.read_json(path)
            value['records'][0]['full_gd_p0_kl_sum']+=1;exp.core.atomic_json(path,value)
            with self.assertRaisesRegex(RuntimeError,'metadata changed'):exp.read_state(ctx,'proj',True)

    def test_replay_checks_all_real_methods_not_only_teacher(self):
        row={k:1. for k in ['teacher_nll_sum','reference_energy_sum']+[a+'_'+k+'_sum' for a in exp.REAL for k in ('sse','kl','nll')]}
        self.assertEqual(exp.replay(row,row),0)
        other=copy.deepcopy(row);other['full_gd_nll_sum']+=.001
        with self.assertRaisesRegex(RuntimeError,'replay failed'):exp.replay(row,other)

    def test_summary_contrasts_and_complete_package(self):
        with self.fixture() as (ctx,t,s,files):
            exp.evaluate_module(ctx,t,s,'proj',exp.prev.Budget(10,None))
            for arm in exp.PERMUTED:
                files[arm].update(seed=exp.SEEDS[int(arm[-1])],fixed_channels=1,root_relative_change=.2,unchanged_metric_root=False,tail_relative_error=0.)
                exp.factor_path(ctx,'proj',arm).parent.mkdir(parents=True,exist_ok=True)
                exp.core.atomic_json(exp.factor_path(ctx,'proj',arm),files[arm])
            for method in exp.REAL[1:]:
                exp.root_path(ctx,'proj',method).parent.mkdir(parents=True,exist_ok=True)
                exp.core.atomic_json(exp.root_path(ctx,'proj',method),{'test':True})
            status=exp.summarize(ctx)
            self.assertTrue(status['complete']);self.assertEqual(status['completed_arms'],9)
            import csv,tarfile
            with (ctx.output/'paired.csv').open() as stream: paired=list(csv.DictReader(stream))
            self.assertEqual(len(paired),8)
            for method in exp.REAL[1:]:
                group=[r for r in paired if r['contrast'].endswith('_minus_'+method)]
                individual=[float(r['kl_delta']) for r in group if r['seed_count']=='1']
                average=[float(r['kl_delta']) for r in group if r['seed_count']=='3'][0]
                self.assertAlmostEqual(average,sum(individual)/3,places=12)
            for name in ('experiment.json','source_baselines.json','pilot.json','background.json'):
                exp.core.atomic_json(ctx.output/name,{'test':True})
            exp.pack(ctx)
            with tarfile.open(ctx.output/(exp.VERSION+'_summary.tar.gz')) as archive:
                self.assertEqual(len(archive.getmembers()),17)
                self.assertTrue(all(not m.name.endswith('.safetensors') for m in archive.getmembers()))


if __name__=='__main__':unittest.main()
