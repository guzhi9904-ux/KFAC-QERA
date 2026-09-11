import copy
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
import qwen_gi_fp64_v1 as new
import qwen_gi_word_v1 as word
import qwen_a_fp64_target_v1 as target
import full_g_precision_r8_v1 as single
import qwen_full_a_audit_v1 as h


class MathTests(unittest.TestCase):
    def test_da_matches_dense_fp64_and_all_prefixes(self):
        torch.manual_seed(7)
        s, e = torch.rand(12)+.2, torch.randn(12, 9)
        before = s.clone()
        with patch.object(torch.linalg, "svd", wraps=torch.linalg.svd) as svd:
            vals, rows, _ = new.solve_diag(s, e, (2, 4, 8))
        self.assertIs(svd.call_args.kwargs["full_matrices"], True)
        self.assertEqual(svd.call_args.args[0].dtype, torch.float64)
        dense, _, _ = target.solve(torch.diag(s), e, (2, 4, 8))
        for r in (2, 4, 8):
            torch.testing.assert_close(vals["A_fp64"][:, :r] @ vals["B_fp64"][:r], dense["A_fp64"][:, :r] @ dense["B_fp64"][:r], atol=1e-10, rtol=1e-10)
        self.assertTrue(torch.equal(before, s))
        self.assertTrue(all(x["numerical_gate_passed"] for x in rows))

    def test_zero_negative_nan_root_no_floor(self):
        for value in (0., -1., float("nan")):
            s = torch.ones(4); s[0] = value
            with self.assertRaises(RuntimeError):
                new.solve_diag(s, torch.ones(4, 3), (2,))

    def test_bad_dtype_rank(self):
        with self.assertRaises(RuntimeError):
            new.solve_diag(torch.ones(4).double(), torch.ones(4,3), (2,))
        with self.assertRaises(ValueError):
            new.solve_diag(torch.ones(4), torch.ones(4,3), (4,))

    def test_all_10_configs(self):
        self.assertEqual(len(new.configurations()), 10)
        self.assertEqual({r for _,m,r in new.configurations() if m in new.METHODS}, {8,16,32,64})
        self.assertEqual(new.solve_all.__globals__["target"].solve, target.solve)


class StorageTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(8)
        self.values, self.rows, self.checks = new.solve_diag(torch.ones(72), torch.randn(72,68))

    def commit_factor(self, output, method, entry):
        path = output/"factors"/method/"projection.safetensors"
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {"identity":"id", "source":entry, "method":method, "status":"PASS", "rank_metrics":self.rows,
                  "checks":self.checks, "file":single.atomic_tensors(path,self.values),
                  "factor_bits":{k:single.tensor_record(v) for k,v in self.values.items()}}
        single.core.atomic_json(path.with_suffix(".json"),record)
        return record

    def test_resume_binding_and_tensor_tamper(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)
            entry={"layer":{"name":"projection", "shape":[68,72]}}
            rec=self.commit_factor(out,"diag_gi",entry)
            self.assertEqual(new.factor_record(out,"diag_gi",entry,"id",single,h),rec)
            with self.assertRaises(RuntimeError):
                new.factor_record(out,"diag_gi",entry,"other",single,h)
            bad=copy.deepcopy(entry); bad["quant"]="changed"
            with self.assertRaises(RuntimeError):
                new.factor_record(out,"diag_gi",bad,"id",single,h)
            vals=copy.deepcopy(self.values); vals["A_fp64"][0,0]+=1
            single.atomic_tensors(Path(rec["file"]["path"]),vals)
            with self.assertRaises(RuntimeError):
                new.factor_record(out,"diag_gi",entry,"id",single,h)

    def test_proxy_and_inverse_fail_closed(self):
        rec={"checks":self.checks, "rank_metrics":self.rows}
        new.require_gates(rec)
        for key, value in (("numerical_gate_passed",False),("bf16_factors_finite",False),("bf16_proxy_flags",1)):
            bad=copy.deepcopy(rec); bad["checks"][key]=value
            with self.assertRaises(RuntimeError):
                new.require_gates(bad)

    def test_rows_crc_order_identity(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"rows.json"
            rows=[{"window":0,"tokens":2047,"nll_sum":5.}]
            obj={"identity":"id","records":rows,"records_sha256":h.fingerprint(rows)}
            single.core.atomic_json(path,obj)
            self.assertEqual(new.read_rows(path,"id",2,h),rows)
            with self.assertRaises(RuntimeError):
                new.read_rows(path,"bad",2,h)
            obj["records"][0]["nll_sum"]=6.
            single.core.atomic_json(path,obj)
            with self.assertRaises(RuntimeError):
                new.read_rows(path,"id",2,h)

    def test_install_exact_bf16_two_gemms_and_bias(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)
            model=torch.nn.Module(); model.projection=torch.nn.Linear(72,68,bias=True).bfloat16()
            q=torch.randn(68,72).bfloat16()
            qfile=single.atomic_tensors(out/"q.safetensors",{"weight_q":q})
            factor=single.atomic_tensors(out/"f.safetensors",self.values)
            entry={"layer":{"name":"projection","shape":[68,72],"bias":True},"quant":{"file":qfile}}
            bias=model.projection.bias.detach().clone(); handles=[]
            new.install(model,[entry],{"projection":factor},"diag_gi",8,single,h,handles)
            x=torch.randn(2,72).bfloat16()
            expected=torch.nn.functional.linear(x,q,bias)+(x@self.values["A_bf16"][:,:8])@self.values["B_bf16"][:8]
            self.assertTrue(torch.equal(model.projection(x),expected))
            self.assertTrue(torch.equal(model.projection.bias,bias))
            for handle in handles: handle.remove()

    def test_evaluation_pause_resume_and_10_outputs(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/"new"; out.mkdir()
            source=Path(d)/"old"; source.mkdir()
            canary=source/"canary"; canary.write_bytes(b"unchanged")
            data=single.atomic_tensors(source/"data.safetensors",{"input_ids":torch.ones(3,4,dtype=torch.int64), "attention_mask":torch.ones(3,4,dtype=torch.int64)})
            qfile=single.atomic_tensors(source/"q.safetensors",{"weight_q":torch.zeros(68,72,dtype=torch.bfloat16)})
            entry={"layer":{"name":"projection","shape":[68,72],"bias":True},"quant":{"file":qfile}}
            for method in new.METHODS: self.commit_factor(out,method,entry)
            manifest={"sha256":"source", "payload":{"config":{"run_dir":str(source),"eval_batch_size":2,"eval_ce_chunk_tokens":256},"data":{"wikitext2":data}}}
            loads=[]
            class Model(torch.nn.Module):
                def __init__(self):
                    super().__init__(); self.projection=torch.nn.Linear(72,68).bfloat16(); self.hf_device_map={"projection":0}
                def forward(self,**kwargs):
                    n=kwargs["input_ids"].shape[0]
                    return SimpleNamespace(logits=torch.zeros(n,4,68))
            def teacher(c,m,dtype):
                self.assertEqual(c["run_dir"],str(out)); torch.manual_seed(1); loads.append(1); return Model()
            stage=SimpleNamespace(teacher=teacher,_input_device=lambda m:"cpu",_chunked_window_nll=lambda l,i,m,c:[(5.,2047)]*len(i))
            class Stop:
                def __init__(self): self.calls=0
                def check(self):
                    self.calls+=1
                    if self.calls==3: raise single.previous.Paused()
            with self.assertRaises(single.previous.Paused):
                new.evaluate([entry],manifest,out,"id",single,h,Stop(),stage)
            saved=h.read_json(out/"evaluation/configurations/BF16.json")
            self.assertEqual(len(saved["records"]),2)
            stop=SimpleNamespace(check=lambda:None)
            new.evaluate([entry],manifest,out,"id",single,h,stop,stage)
            self.assertEqual(h.read_json(out/"evaluation/status.json")["configurations"],10)
            count=len(loads)
            new.evaluate([entry],manifest,out,"id",single,h,stop,stage)
            self.assertEqual(len(loads),count)
            self.assertEqual(canary.read_bytes(),b"unchanged")


class WordTests(unittest.TestCase):
    def test_release_lf_and_all_payload_hashes(self):
        directory=Path(__file__).parent
        raw=(directory/"SHA256SUMS.qwen_gi_fp64").read_bytes()
        self.assertNotIn(b"\r",raw)
        names=[]
        for line in raw.decode("ascii").splitlines():
            digest,name=line.split("  ",1)
            names.append(name)
            self.assertEqual(hashlib.sha256((directory/name).read_bytes()).hexdigest(),digest)
        self.assertIn("qwen_gi_fp64_v1.py",names)
        self.assertIn("qwen_gi_word_v1.py",names)
        self.assertTrue(set(target.HELPERS)<=set(names))
        self.assertTrue(set(word.PINS)<=set(names))

    def test_helpers_still_frozen(self):
        directory=Path(__file__).parent
        for name,digest in word.PINS.items():
            self.assertEqual(h.sha256(directory/name),digest)

    def test_word_resume_result_hash_and_document_checks(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)
            summary={"ppl":2.,"documents":62}; docs=[{"document":0,"words":3,"nll_sum":2.}]
            ctx=SimpleNamespace(dual=SimpleNamespace(word_documents=lambda r,t,h:(summary,docs)),task=None,helper=None)
            for name,value in (("results.json",{}),("documents.json",docs),("deployment.json",{})):
                single.core.atomic_json(folder/name,value)
            state={"identity":"id","factors":{},"status":"PASS","summary":summary,
                   "results_file":single.file_record(folder/"results.json"),"documents_file":single.file_record(folder/"documents.json"),
                   "deployment_file":single.file_record(folder/"deployment.json")}
            single.core.atomic_json(folder/"complete.json",state)
            self.assertEqual(word.existing(folder,"id",{},ctx,single,h),state)
            with self.assertRaises(RuntimeError): word.existing(folder,"bad",{},ctx,single,h)
            single.core.atomic_json(folder/"documents.json",[])
            with self.assertRaises(RuntimeError): word.existing(folder,"id",{},ctx,single,h)


if __name__ == "__main__":
    unittest.main()
