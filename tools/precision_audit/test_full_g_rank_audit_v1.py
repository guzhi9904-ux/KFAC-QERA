"""CPU mathematical/provenance/resume tests; no real server GPU artifacts used."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import sys

import torch
from safetensors.torch import save_file
import full_g_rank_audit_v1 as audit


def factors(error, a, g, rank):
    u, s, vh = torch.linalg.svd(audit.apply_a(a, error) @ g, full_matrices=True)
    left = u[:, :rank]/a[:, None] if a.ndim == 1 else torch.linalg.solve(a, u[:, :rank])
    right = torch.linalg.solve(g.T, (s[:rank, None]*vh[:rank]).T).T
    return left.contiguous(), right.contiguous()


class RankAuditTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        torch.set_num_threads(2)

    def test_effective_root(self):
        gram = torch.diag(torch.tensor([0., 1., 2.], dtype=torch.float64))
        root, diagnostics = audit.full_root(gram, 7)
        self.assertEqual(diagnostics["floored_eigenvalues"], 1)
        target = torch.diag(torch.tensor([1e-6, 1., 2.]))
        torch.testing.assert_close(root @ root, target)

    def test_root_matches_frozen_implementation(self):
        experiments = Path(__file__).parent/"qera_mxint4_full_ag"/"experiments"
        sys.path.insert(0, str(experiments))
        try:
            from qera_diag_g_isolation.full_g_v1.numerics import full_root as frozen_root
            z = torch.randn(12, 9, dtype=torch.float64)
            gram = z @ z.T
            actual, info = audit.full_root(gram, 256*2047)
            expected, saved = frozen_root(gram, 256*2047)
            self.assertTrue(torch.equal(actual, expected))
            for key in ("floored_eigenvalues", "negative_eigenvalues", "normalized_min_eigenvalue",
                        "normalized_max_eigenvalue", "raw_mean_diagonal", "relative_floor"):
                self.assertEqual(info[key], saved[key])
        finally:
            sys.path.pop(0)

    def test_non_psd_rejected(self):
        with self.assertRaises(RuntimeError):
            audit.full_root(torch.diag(torch.tensor([-1., 2.], dtype=torch.float64)), 1)

    def test_rank_metrics_against_direct_objective(self):
        error = torch.randn(9, 7)
        a = torch.diag(torch.linspace(.5, 2., 9))
        z = torch.randn(7, 7)
        g, _ = audit.full_root(z.double() @ z.double().T + torch.eye(7), 1)
        left, right = factors(error, a, g, 5)
        with contextlib.redirect_stdout(io.StringIO()):
            rows = audit.measure(error, a, g, left, right, ranks=(1, 3, 5), reference=True)
        for row in rows:
            r = row["rank"]
            direct = float((a @ (error-left[:, :r] @ right[:r]) @ g).double().square().sum())
            self.assertAlmostEqual(row["sse_after"], direct, places=9)
            self.assertLess(row["left_orthogonality"], 1e-5)
            self.assertLess(abs(row["excess_over_svd_tail_over_before"]), 1e-5)
            self.assertNotIn("NONMONOTONIC_RANK", row["flags"])

    def test_flags_capture_growth_and_nonmonotonicity(self):
        rows = [{"rank": r, "sse_before": 10., "sse_after": loss,
                 "weight_mse": 1., "correction_norm": 1.} for r, loss in [(1, 8.), (2, 9.), (3, 12.)]]
        audit.flag_rows(rows)
        self.assertIn("NONMONOTONIC_RANK", rows[1]["flags"])
        self.assertIn("WORSE_THAN_NO_CORRECTION", rows[2]["flags"])

    def test_nonfinite_flag_and_json(self):
        rows = [{"sse_before": 1., "sse_after": float("nan"), "weight_mse": 1., "correction_norm": 1.}]
        audit.flag_rows(rows)
        self.assertIn("NONFINITE", rows[0]["flags"])
        json.dumps(audit.clean(rows), allow_nan=False)

    def test_output_isolation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for output in (root, root/"source", root/"source"/"audit"):
                with self.assertRaises(RuntimeError):
                    audit.disjoint(output, [root/"source"])
            audit.disjoint(root/"audit", [root/"source"])

    def test_hash_tamper(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"input"
            path.write_bytes(b"abc")
            record = {"path": str(path), "sha256": audit.sha(path), "bytes": 3}
            path.write_bytes(b"xyz")
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
                audit.Inputs().verify(record)

    def test_module_end_to_end_and_resume_summary(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, output = root/"source", root/"audit"
            source.mkdir()
            output.mkdir()
            (output/"modules").mkdir()
            def save(name, tensors):
                path = source/(name+".safetensors")
                save_file(tensors, str(path))
                return {"path": str(path), "bytes": path.stat().st_size, "sha256": audit.sha(path)}
            name = "model.layers.0.self_attn.q_proj"
            weight = torch.randn(80, 72).bfloat16()
            q = (weight.float()*.9).bfloat16()
            error = (weight.float()-q.float()).T
            a_diag = torch.linspace(.5, 2., 72)
            a_full = torch.diag(a_diag)
            gram = torch.diag(torch.linspace(.1, 1., 80).double())
            g, info = audit.full_root(gram, 256*2047)
            binding = {"root": save("root", {"diag": a_diag, "full": a_full}),
                       "gram": save("gram", {"gram": gram, "diagonal": gram.diagonal().contiguous()}),
                       "model": save("model", {name+".weight": weight}),
                       "quant": save("quant", {"weight_q": q}), "shape": [80, 72], "corrections": {}}
            for method, a in (("diag_gf", a_diag), ("full_gf", a_full)):
                left, right = factors(error, a, g, 64)
                binding["corrections"][method] = {"file": save(method, {"A": left, "B": right}),
                    "g_diagnostics": info, "weighted_sse_before": audit.norm2(audit.apply_a(a, error) @ g),
                    "weighted_sse_after": audit.norm2(audit.apply_a(a, error-left @ right) @ g), "g_inverse_residual": 0.}
            before = {p.name: audit.sha(p) for p in source.iterdir()}
            with contextlib.redirect_stdout(io.StringIO()):
                report = audit.audit_module(name, binding, audit.Inputs(), "cpu", True, True)
            self.assertEqual(len(report["rows"]), 16)
            self.assertEqual(before, {p.name: audit.sha(p) for p in source.iterdir()})
            identity = "fixture"
            report.update(status="MODULE_AUDITED", audit_identity=identity, binding_sha256=audit.fingerprint(binding))
            path = output/"modules"/(audit.safe_name(name)+".json")
            audit.atomic_json(path, report)
            result = audit.summarize(output, identity, [(name, binding)])
            self.assertEqual(result["audited_modules"], 1)
            self.assertEqual(result["status"], "AUDIT_COMPLETE")
            self.assertTrue((output/"rank_metrics.csv").exists())
            self.assertEqual(audit.summarize(output, identity, [(name, binding)]), result)
            with self.assertRaises(RuntimeError):
                audit.summarize(output, "changed_identity", [(name, binding)])


if __name__ == "__main__":
    unittest.main()
