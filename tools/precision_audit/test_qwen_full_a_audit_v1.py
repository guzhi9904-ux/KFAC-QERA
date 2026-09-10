"""Small CPU tests only; no server data or CUDA execution is available locally."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file

import qwen_full_a_audit_v1 as audit


class AuditTests(unittest.TestCase):
    def test_root_inspection_zero_row(self):
        s = torch.diag(torch.tensor([1., 0., 2.]))
        result = audit.inspect_root(s)
        self.assertEqual(result["exact_zero_rows_first_32"], [1])
        self.assertEqual(result["relative_asymmetry"], 0.)

    def test_bad_root_rejected(self):
        with self.assertRaises(RuntimeError):
            audit.inspect_root(torch.eye(3).double())
        with self.assertRaises(RuntimeError):
            audit.inspect_root(torch.full((3, 3), float("nan")))

    def test_metrics_match_direct_formula(self):
        torch.manual_seed(2)
        s = torch.diag(torch.tensor([1., 2., 3., 4.]))
        error = torch.randn(4, 3)
        u, sig, vh = torch.linalg.svd(s @ error, full_matrices=True)
        u, right = u[:, :2], sig[:2, None]*vh[:2]
        left = torch.linalg.solve(s, u)
        actual = audit.tensor_metrics(s, error, u, right, left)
        expected = float((s @ (error-left@right)).double().square().sum())
        self.assertAlmostEqual(actual["weighted_sse_after"], expected, places=10)
        self.assertLess(actual["inverse_relative_residual_fp64"], 1e-6)

    def test_replay_all_precision_branches(self):
        torch.manual_seed(3)
        s = torch.diag(torch.tensor([1., .001, .2, 2.]))
        results = {}
        with contextlib.redirect_stdout(io.StringIO()):
            audit.replay(s, torch.randn(4, 3), 2, "cpu", results.__setitem__)
        for key in ("fp32_inverse", "fp64_inverse", "fp64_inverse_cast_fp32"):
            self.assertLessEqual(results[key]["after_over_before"], 1.00001)
        self.assertLess(results["fp64_inverse"]["inverse_relative_residual_fp64"], 1e-12)

    def test_hash_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"input"
            path.write_bytes(b"abc")
            record = {"path": str(path), "bytes": 3, "sha256": audit.sha256(path)}
            path.write_bytes(b"def")
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
                audit.verify(record)

    def test_cli_fixture_read_only(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            module = "model.layers.1.mlp.down_proj"
            config = {"experiment_variant": "qwen25_base_mxint3_v1", "ranks": [1, 2]}
            def record(path):
                return {"path": str(path), "bytes": path.stat().st_size, "sha256": audit.sha256(path)}
            (root/"roots").mkdir()
            (root/"quantized").mkdir()
            root_path = root/"roots"/(audit.safe_name(module)+".safetensors")
            raw_path, model_path = root/"raw.safetensors", root/"model.safetensors"
            s = torch.diag(torch.tensor([1., 2., 3., 4.]))
            save_file({"full": s, "diag": s.diagonal().contiguous()}, str(root_path))
            save_file({"full": (s.double() @ s.double())*524288,
                       "diag": s.diagonal().square()*524288}, str(raw_path))
            weight = torch.randn(3, 4).bfloat16()
            save_file({module+".weight": weight}, str(model_path))
            payload = {"config": config, "groups": [{"target": module, "layers": [
                {"name": module, "shape": [3, 4], "weight_file": model_path.name}]}],
                "model_files": {model_path.name: record(model_path)}}
            digest = audit.fingerprint(payload)
            (root/"manifest.json").write_text(json.dumps({"payload": payload, "sha256": digest}))
            (root/"config.json").write_text(json.dumps(config))
            root_path.with_suffix(".json").write_text(json.dumps({"status": "PASS",
                "manifest_sha256": digest, "file": record(root_path), "raw_reference": record(raw_path)}))
            quant_path = root/"quantized"/(audit.safe_name(module)+".safetensors")
            save_file({"weight_q": (weight.float()*.9).bfloat16()}, str(quant_path))
            quant_path.with_suffix(".json").write_text(json.dumps({"status": "PASS",
                "manifest_sha256": digest, "file": record(quant_path), "fp32_bf16_equal": True,
                "quantization": {"name": "mxint", "width": 3, "block_size": 32, "block_axis": -1}}))
            before = {str(p): audit.sha256(p) for p in root.rglob("*") if p.is_file()}
            for options in ([], ["--replay", "--device", "cpu"]):
                with patch.object(sys, "argv", ["audit", "--run-dir", str(root), *options]), \
                        contextlib.redirect_stdout(io.StringIO()) as output:
                    audit.main()
                self.assertIn("AUDIT_COMPLETE_NOT_AN_EXPERIMENT_PASS", output.getvalue())
            after = {str(p): audit.sha256(p) for p in root.rglob("*") if p.is_file()}
            self.assertEqual(before, after)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
