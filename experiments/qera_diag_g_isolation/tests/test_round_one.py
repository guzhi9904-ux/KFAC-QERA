import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from qera_diag_g_isolation.math_ops import (
    ce_hidden_gradient, correction_drift, diagonal_increment, diagonal_scale, solve_weighted,
)
from qera_diag_g_isolation.pipeline import (
    _load_g_checkpoint, _save_g_checkpoint, check_control, configurations, evaluation_records,
)
from qera_diag_g_isolation.storage import atomic_json, disjoint_roots, read_json


@pytest.mark.parametrize("batch,chunk", [(1, 1), (1, 4), (2, 3)])
def test_ce_sum_seed_matches_autograd(batch, chunk):
    torch.manual_seed(9)
    hidden = torch.randn(batch, 7, 5, requires_grad=True)
    weight = torch.randn(11, 5)
    ids = torch.randint(0, 11, (batch, 7))
    mask = torch.ones(batch, 7, dtype=torch.bool)
    mask[0, -2:] = False
    valid = mask[:, :-1] & mask[:, 1:]
    loss = F.cross_entropy(F.linear(hidden[:, :-1], weight)[valid], ids[:, 1:][valid], reduction="sum")
    reference = torch.autograd.grad(loss, hidden)[0]
    gradient, nll, count = ce_hidden_gradient(hidden, weight, ids, mask, chunk)
    torch.testing.assert_close(gradient, reference, atol=2e-6, rtol=2e-6)
    assert nll == pytest.approx(float(loss.detach()), rel=2e-6)
    assert count == int(valid.sum())
    assert (gradient[:, -1] == 0).all()


def test_ce_requires_fp32_and_nonempty_tokens():
    x, w = torch.ones(1, 3, 2), torch.ones(4, 2)
    ids = torch.zeros(1, 3, dtype=torch.long)
    with pytest.raises(ValueError):
        ce_hidden_gradient(x.double(), w, ids, torch.ones_like(ids))
    with pytest.raises(ValueError):
        ce_hidden_gradient(x, w, ids, torch.zeros_like(ids))


def test_batch_independent_sum_gradient_and_g():
    torch.manual_seed(0)
    x, w = torch.randn(2, 6, 4), torch.randn(7, 4)
    ids, mask = torch.randint(7, (2, 6)), torch.ones(2, 6, dtype=torch.bool)
    combined, _, _ = ce_hidden_gradient(x, w, ids, mask)
    g = diagonal_increment(combined, mask)
    separate = sum(diagonal_increment(ce_hidden_gradient(x[i:i+1], w, ids[i:i+1], mask[i:i+1])[0], mask[i:i+1]) for i in range(2))
    torch.testing.assert_close(g, separate, atol=1e-6, rtol=1e-6)


def test_g_full_diagonal_consistency_and_channel_independence():
    torch.manual_seed(1)
    gradient = torch.randn(1, 13, 7)
    mask = torch.ones(1, 13, dtype=torch.bool)
    mask[:, -1] = False
    flat = gradient[mask].double()
    expected = (flat.T @ flat).diag()
    torch.testing.assert_close(diagonal_increment(gradient, mask, 3), expected)
    assert not torch.equal(diagonal_increment(gradient, mask), diagonal_increment(gradient * 2, mask))


def test_g_floor_and_global_scale_invariance():
    g = torch.tensor([0., 1., 2., 1e-12], dtype=torch.float64)
    a, diag = diagonal_scale(g, 16)
    b, _ = diagonal_scale(g * 100, 1600)
    torch.testing.assert_close(a, b)
    assert diag["floored_channels"] == 2
    assert (a > 0).all()
    with pytest.raises(ValueError):
        diagonal_scale(torch.zeros(5), 2)
    with pytest.raises(ValueError):
        diagonal_scale(torch.tensor([float("nan")]), 2)


@pytest.mark.parametrize("full", [False, True])
def test_weighted_solution_and_identity_reduction(full):
    torch.manual_seed(11)
    error = torch.randn(9, 7, dtype=torch.float64)
    scale = torch.rand(9, dtype=torch.float64) + .5
    if full:
        t = torch.randn(9, 9, dtype=torch.float64)
        scale = t @ t.T + torch.eye(9, dtype=torch.float64)
    g = torch.rand(7, dtype=torch.float64) + .3
    left, right, metrics = solve_weighted(error, scale, g, 3)
    weighted = (scale @ error if full else scale[:, None] * error) * g
    optimum = torch.linalg.svdvals(weighted)[3:].square().sum()
    assert metrics["weighted_sse_after"] == pytest.approx(float(optimum), rel=1e-10)
    # Reconstruct the exact official unweighted formula (full SVD).
    u, s, vh = torch.linalg.svd(scale @ error if full else torch.diag(scale) @ error)
    reference_left = torch.linalg.solve(scale if full else torch.diag(scale), u[:, :5])
    reference_right = torch.diag(s[:5]) @ vh[:5]
    a, b, _ = solve_weighted(error, scale, torch.ones_like(g), 5)
    assert max(correction_drift(a, b, reference_left, reference_right, [1, 2, 3, 5]).values()) < 1e-10
    assert left.shape == (9, 3) and right.shape == (3, 7)


def test_invalid_weighted_inputs():
    with pytest.raises(ValueError):
        solve_weighted(torch.ones(4, 3), torch.ones(4), torch.zeros(3), 2)
    with pytest.raises(ValueError):
        solve_weighted(torch.ones(4, 3), torch.ones(4), torch.ones(3), 5)


def test_sequence_gradient_includes_future_attention_paths():
    torch.manual_seed(18)
    z = torch.randn(1, 5, 3, requires_grad=True)
    # Causal prefix mixing models cross-token dependence explicitly.
    hidden = z.cumsum(dim=1)
    weight, ids, mask = torch.randn(9, 3), torch.randint(9, (1, 5)), torch.ones(1, 5, dtype=torch.bool)
    seed, _, _ = ce_hidden_gradient(hidden, weight, ids, mask)
    expected = torch.autograd.grad(F.cross_entropy(F.linear(hidden[:, :-1], weight).reshape(-1, 9), ids[:, 1:].reshape(-1), reduction="sum"), z, retain_graph=True)[0]
    actual = torch.autograd.grad(hidden, z, grad_outputs=seed)[0]
    torch.testing.assert_close(actual, expected)
    assert not torch.allclose(actual[:, 0], seed[:, 0])


@pytest.mark.parametrize("suffix", ["same", "child", "parent"])
def test_protect_source_tree(tmp_path, suffix):
    source = tmp_path / "old"
    destination = {"same": source, "child": source / "new", "parent": tmp_path}[suffix]
    with pytest.raises(ValueError):
        disjoint_roots(source, destination)
    disjoint_roots(source, tmp_path / "new")


def toy_manifest():
    return {"sha256": "test", "payload": {"groups": [{"layers": [{"name": "q", "shape": [3, 2]}, {"name": "k", "shape": [2, 2]}]}]}}


def test_checkpoint_resume_and_retention(tmp_path):
    config, manifest = {"run_dir": str(tmp_path)}, toy_manifest()
    initial, count, _ = _load_g_checkpoint(config, manifest)
    assert count == 0
    for n in (8, 64, 128, 256):
        sums = {name: torch.ones_like(value) * n for name, value in initial.items()}
        _save_g_checkpoint(config, manifest, sums, n, n * 5.)
        actual, count, nll = _load_g_checkpoint(config, manifest)
        assert count == n and nll == n * 5
        for name in sums:
            torch.testing.assert_close(actual[name], sums[name])
    assert len(list((tmp_path / "statistics/checkpoints").glob("*.safetensors"))) == 4
    # Unpublished newer/orphan files never advance the resume pointer.
    atomic_json(tmp_path / "statistics/checkpoints/orphan.json", {"windows_completed": 999})
    assert _load_g_checkpoint(config, manifest)[1] == 256


def test_checkpoint_corruption_fails_closed(tmp_path):
    config, manifest = {"run_dir": str(tmp_path)}, toy_manifest()
    sums, _, _ = _load_g_checkpoint(config, manifest)
    _save_g_checkpoint(config, manifest, sums, 8, 2.)
    path = tmp_path / "statistics/checkpoints/window_0008.safetensors"
    path.write_bytes(b"corruption")
    with pytest.raises(RuntimeError):
        _load_g_checkpoint(config, manifest)


def test_evaluation_checkpoint_validation(tmp_path):
    path = tmp_path / "evaluation.json"
    atomic_json(path, {"protocol_sha256": "p", "records": [{"window": 0, "tokens": 2047, "nll_sum": 3.}]})
    assert len(evaluation_records(path, "p", 138)) == 1
    with pytest.raises(RuntimeError):
        evaluation_records(path, "changed", 138)
    bad = read_json(path)
    bad["records"][0]["window"] = 1
    atomic_json(path, bad)
    with pytest.raises(RuntimeError):
        evaluation_records(path, "p", 138)


def test_configuration_matrix():
    configs = configurations({"ranks": [8, 16, 32, 64]})
    assert len(configs) == 18 and len({x[0] for x in configs}) == 18
    assert sum("GD" in x[0] for x in configs) == 8


def test_control_gate_does_not_silently_accept_drift(tmp_path):
    import math
    manifest = {"payload": {"control_reference_ppl": {"BF16": 6.24}}}
    config = {"run_dir": str(tmp_path), "control_ppl_tolerance": .01}
    rows = [{"window": i, "tokens": 2047, "nll_sum": math.log(6.24) * 2047} for i in range(138)]
    check_control(config, manifest, "BF16", rows)
    assert read_json(tmp_path / "evaluation/control_checks/BF16.json")["status"] == "PASS"
    for row in rows:
        row["nll_sum"] = math.log(7.0) * 2047
    with pytest.raises(RuntimeError):
        check_control(config, manifest, "BF16", rows)
    assert read_json(tmp_path / "evaluation/control_checks/BF16.json")["status"] == "FAIL"


def test_evaluation_releases_model_before_next_configuration(tmp_path, monkeypatch):
    import weakref
    from qera_diag_g_isolation import pipeline as p
    config = {"run_dir": str(tmp_path), "eval_max_memory": {}, "eval_batch_size": 1,
              "eval_ce_chunk_tokens": 256, "ranks": [8, 16, 32, 64]}
    manifest = {"sha256": "m", "payload": {"source_config": {}, "data": {"wikitext2": {}}}}
    data = {"input_ids": torch.zeros(2, 2048, dtype=torch.long), "attention_mask": torch.ones(2, 2048)}
    previous = []
    class Toy(torch.nn.Module):
        hf_device_map = {"": 0}
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(2, 2)
        def get_input_embeddings(self):
            return self.embed
        def forward(self, input_ids, **kwargs):
            from types import SimpleNamespace
            return SimpleNamespace(logits=torch.zeros(*input_ids.shape, 2))
    def load(*args):
        if previous:
            assert previous[-1]() is None, "Previous model still referenced while loading next"
        value = Toy()
        previous.append(weakref.ref(value))
        return value
    monkeypatch.setattr(p, "load_manifest", lambda c: manifest)
    monkeypatch.setattr(p, "gpu_check", lambda: None)
    monkeypatch.setattr(p, "verify_model", lambda m: None)
    monkeypatch.setattr(p, "checked_tensors", lambda r: data)
    monkeypatch.setattr(p, "load_model", load)
    monkeypatch.setattr(p, "_reset_evaluation_memory_peaks", lambda x: None)
    monkeypatch.setattr(p, "_evaluation_inputs", lambda *a, **k: {})
    monkeypatch.setattr(p, "configurations", lambda c: [("BF16", None, None), ("W4_MXINT", "wq", None)])
    monkeypatch.setattr(p, "summarize", lambda c: {"status": "PASS"})
    p.evaluate(config)
    assert len(previous) == 2
    assert all(ref() is None for ref in previous)
