"""CPU numerical and fault-injection contracts; GPU gates remain mandatory."""
from contextlib import nullcontext
from copy import deepcopy
import importlib
import math
from pathlib import Path
import re
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qera_diag_g_isolation import pipeline as legacy
from qera_diag_g_isolation.full_svd_v1.solver import solve_weighted
from qera_diag_g_isolation.math_ops import correction_drift, diagonal_scale, ce_hidden_gradient
from qera_diag_g_isolation.storage import atomic_json, atomic_tensors, file_record, fingerprint, load_manifest, read_json
from qera_diag_g_isolation.full_g_v1 import checkpoint, numerics, pipeline, run
from qera_diag_g_isolation.mxint3_v1.test_mxint3 import fixture_run
collector = importlib.import_module("qera_diag_g_isolation.full_g_v1.collect")


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(1)
    torch.manual_seed(121)


def values(d=5, multiplier=1.):
    x = torch.arange(3*d, dtype=torch.float64).reshape(3, d) / 17
    gram = x.T @ x * multiplier
    return {"layer": {"gram": gram, "diagonal": gram.diagonal().clone()}}


def test_fp64_tiled_gram_matches_direct_and_chunking():
    gradient = torch.randn(2, 7, 9)
    mask = torch.rand(2, 7) > .3
    exact = gradient[mask].double().T @ gradient[mask].double()
    for tile in (1, 4, 32):
        gram, diagonal = torch.zeros(9, 9, dtype=torch.float64), torch.zeros(9, dtype=torch.float64)
        numerics.accumulate_gram(gradient, mask, gram, diagonal, tile)
        assert torch.allclose(gram, exact, atol=1e-12, rtol=1e-12)
        assert numerics.check_diagonal(gram, diagonal) < 1e-12
        numerics.accumulate_gram(gradient, mask, gram, diagonal, tile)
        assert torch.allclose(gram, exact * 2, atol=1e-12, rtol=1e-12)
    with pytest.raises(ValueError, match="FP32"):
        numerics.accumulate_gram(gradient.double(), mask, gram, diagonal)


def test_root_diagonal_extension_and_psd_floor():
    raw = torch.tensor([0., 1e-12, .2, 4., 20.], dtype=torch.float64)
    vector, _ = diagonal_scale(raw, 17, 1e-6)
    dense, report = numerics.full_root(torch.diag(raw), 17, 1e-6)
    assert torch.allclose(dense, torch.diag(vector), atol=1e-7, rtol=1e-7)
    assert report["floored_eigenvalues"] == 2
    identity, _ = numerics.full_root(torch.eye(5, dtype=torch.float64), 100)
    assert torch.equal(identity, torch.eye(5))
    bad = torch.diag(torch.tensor([-1., 2., 3.], dtype=torch.float64))
    with pytest.raises(RuntimeError, match="non-PSD"):
        numerics.full_root(bad, 1)
    bad[0, 1] = 1
    with pytest.raises(RuntimeError, match="asymmetry"):
        numerics.full_root(bad, 1)


@pytest.mark.parametrize("full_a", [False, True])
@pytest.mark.parametrize("identity", [False, True])
def test_dense_solver_reproduces_frozen_vector_solver_all_prefixes(full_a, identity):
    error = torch.randn(96, 80)
    a = torch.rand(96) + .5
    if full_a:
        a = torch.diag(a) + .01
    g = torch.ones(80) if identity else torch.rand(80) + .5
    left, right, _ = numerics.solve_full(error, a, torch.diag(g), 64)
    ref_a, ref_b, _ = solve_weighted(error, a, g, 64)
    assert max(correction_drift(left, right, ref_a, ref_b, [8, 16, 32, 64]).values()) < .001


def test_full_solver_weighted_optimum_and_g_orientation():
    error = torch.randn(12, 9)
    x = torch.randn(20, 9, dtype=torch.float64)
    g, _ = numerics.full_root(x.T @ x, 20)
    a = torch.rand(12) + .5
    left, right, metrics = numerics.solve_full(error, a, g, 4)
    singular = torch.linalg.svdvals((a[:, None] * error) @ g)
    assert metrics["weighted_sse_after"] == pytest.approx(float(singular[4:].double().square().sum()), rel=2e-5)
    assert metrics["weighted_sse_after"] < metrics["weighted_sse_before"]
    assert metrics["g_inverse_residual"] < 1e-5
    assert left.shape == (12, 4) and right.shape == (4, 9)


def test_checkpoint_roundtrip_and_rolling_cleanup(tmp_path):
    store = checkpoint.Checkpoints(tmp_path, "manifest", 0, {"layer": 5})
    v, n, nll = store.load()
    assert n == nll == 0 and v["layer"]["gram"].count_nonzero() == 0
    first = store.save(values(), 8, 100.)
    restored, n, nll = store.load()
    assert n == 8 and nll == 100 and torch.equal(restored["layer"]["gram"], values()["layer"]["gram"])
    store.save(values(multiplier=2), 16, 200.)
    assert not (store.root / first["generation"]).exists()
    assert len(list(store.root.glob("gen_*"))) == 1
    with pytest.raises(RuntimeError, match="advance"):
        store.save(values(), 16, 200.)


@pytest.mark.parametrize("failure_point", ["tensor", "metadata", "pointer"])
def test_crash_during_checkpoint_never_advances_pointer(tmp_path, monkeypatch, failure_point):
    store = checkpoint.Checkpoints(tmp_path, "manifest", 0, {"layer": 5})
    old = store.save(values(), 8, 100.)
    original_json, original_tensor = checkpoint.atomic_json, checkpoint.atomic_tensors
    def bad_json(path, value):
        if (failure_point == "metadata" and Path(path).name == "STATE.json") or (
            failure_point == "pointer" and Path(path).name == "CURRENT.json"):
            raise RuntimeError("simulated power loss")
        original_json(path, value)
    def bad_tensor(*args, **kwargs):
        if failure_point == "tensor":
            raise RuntimeError("simulated power loss")
        original_tensor(*args, **kwargs)
    monkeypatch.setattr(checkpoint, "atomic_json", bad_json)
    monkeypatch.setattr(checkpoint, "atomic_tensors", bad_tensor)
    with pytest.raises(RuntimeError, match="power loss"):
        store.save(values(multiplier=2), 16, 200.)
    assert store.load(tensors=False) == old
    assert store.load()[1] == 8
    monkeypatch.setattr(checkpoint, "atomic_json", original_json)
    monkeypatch.setattr(checkpoint, "atomic_tensors", original_tensor)
    store.cleanup()
    store.save(values(multiplier=2), 16, 200.)
    assert store.load()[1] == 16
    assert len(list(store.root.glob("gen_*"))) == 1


def test_crash_after_publish_resumes_new_generation(tmp_path, monkeypatch):
    store = checkpoint.Checkpoints(tmp_path, "manifest", 0, {"layer": 5})
    store.save(values(), 8, 100.)
    original = store.cleanup
    monkeypatch.setattr(store, "cleanup", lambda: (_ for _ in ()).throw(RuntimeError("crash after commit")))
    with pytest.raises(RuntimeError, match="after commit"):
        store.save(values(multiplier=2), 16, 200.)
    assert store.load()[1] == 16
    monkeypatch.setattr(store, "cleanup", original)
    store.cleanup()
    assert len(list(store.root.glob("gen_*"))) == 1


def test_corrupt_and_foreign_checkpoints_fail_closed(tmp_path):
    store = checkpoint.Checkpoints(tmp_path, "manifest", 0, {"layer": 5})
    state = store.save(values(), 8, 100.)
    other = checkpoint.Checkpoints(tmp_path, "other", 0, {"layer": 5})
    with pytest.raises(RuntimeError, match="protocol"):
        other.load()
    Path(state["files"]["layer"]["path"]).write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="Immutable"):
        store.load()


def test_checkpoint_rejects_partial_window_coverage_and_foreign_files(tmp_path):
    store = checkpoint.Checkpoints(tmp_path, "manifest", 0, {"layer": 5})
    with pytest.raises(ValueError, match="whole-window"):
        store.save({}, 8, 100.)
    first = store.save(values(), 8, 100.)
    important = store.root / first["generation"] / "user_notes.txt"
    important.write_text("preserve")
    with pytest.raises(RuntimeError, match="Unexpected files"):
        store.save(values(multiplier=2), 16, 200.)
    assert important.read_text() == "preserve"
    assert store.load()[1] == 16  # Commit succeeded; only cleanup was refused.


def test_stop_budget_and_evaluation_commit_before_pause(tmp_path):
    stop = checkpoint.Stop(0)
    with pytest.raises(checkpoint.Paused):
        stop.check()
    originals = legacy.configurations, legacy._evaluation_inputs, legacy.summarize, legacy.atomic_json
    path = tmp_path / "evaluation/configurations/X.json"
    with pytest.raises(checkpoint.Paused):
        with pipeline.evaluation_binding(stop):
            legacy.atomic_json(path, {"records": [1, 2]})
    assert read_json(path) == {"records": [1, 2]}
    assert (legacy.configurations, legacy._evaluation_inputs, legacy.summarize, legacy.atomic_json) == originals


def test_shard_plan_and_configuration_matrix():
    items = []
    dims = {"self_attn.q_proj": 4096, "self_attn.k_proj": 1024, "self_attn.v_proj": 1024,
            "self_attn.o_proj": 4096, "mlp.gate_proj": 14336, "mlp.up_proj": 14336, "mlp.down_proj": 4096}
    for block in range(32):
        items.extend({"name": f"model.layers.{block}.{suffix}", "shape": [d, 4096]} for suffix, d in dims.items())
    shards = run.shard_layers(items)
    assert [len(s) for s in shards] == [56] * 4
    assert sum(l["shape"][0]**2 * 8 for l in shards[0]) / 2**30 == 27.625
    assert {x["name"] for shard in shards for x in shard} == {x["name"] for x in items}
    with pytest.raises(RuntimeError):
        run.shard_layers(items[:-1])
    config = {"ranks": [8, 16, 32, 64]}
    choices = pipeline.configurations(config)
    assert len(choices) == len({x[0] for x in choices}) == 26
    assert choices[-1] == ("FULL_GF_R64", "full_gf", 64)
    assert choices[18] == ("DIAG_GF_R8", "diag_gf", 8)


@pytest.fixture
def small_run(tmp_path, monkeypatch):
    config = {"run_dir": str(tmp_path / "full"), "ranks": [1, 2, 3, 4], "identity_product_tolerance": .001,
              "control_ppl_tolerance": .01, "solve_device": "cpu", "g_relative_floor": 1e-6}
    name = "model.layers.0.self_attn.k_proj"
    root = tmp_path / "source"
    root.mkdir()
    weight = torch.randn(8, 12).bfloat16()
    q = (weight.float() * 2).round().div(2).bfloat16()
    roots = {"diag": torch.rand(12) + .5, "full": torch.eye(12) + .01}
    atomic_tensors(root / "roots.safetensors", roots)
    atomic_tensors(root / "q.safetensors", {"weight_q": q})
    x = torch.randn(40, 8, dtype=torch.float64)
    gram = x.T @ x
    diagonal = gram.diagonal().clone()
    atomic_tensors(root / "g.safetensors", {name: diagonal})
    g, _ = diagonal_scale(diagonal, 256 * 2047)
    baseline = {"teacher": {}, "wq": {name: {"quant": file_record(root / "q.safetensors")}}}
    error = (weight.float() - q.float()).T
    for kind in ("diag", "full"):
        for suffix, vector in (("gi", torch.ones(8)), ("gd", g)):
            method = kind + "_" + suffix
            a, b, _ = solve_weighted(error, roots[kind], vector, 4)
            path = root / f"{method}.safetensors"
            atomic_tensors(path, {"A": a, "B": b})
            baseline[method] = {name: {"quant": baseline["wq"][name]["quant"], "correction": file_record(path)}}
    layer = {"name": name, "shape": [8, 12]}
    protocol = {"shards": [[layer]], "baseline_inputs": baseline,
                "diagonal_g_state": {"file": file_record(root / "g.safetensors"), "teacher_nll_sum": 100.},
                "parent_diagonal_tolerance": 1e-6, "teacher_nll_relative_tolerance": 1e-6,
                "baseline_evaluations": {}}
    payload = {"config": config, "code": {}, "source_config": {}, "groups": [{
        "roots": file_record(root / "roots.safetensors"), "layers": [layer]}],
        "full_g_protocol": protocol, "control_reference_ppl": {}}
    for title, method, _ in pipeline.mx3.configurations(config):
        path = root / f"{title}.json"
        atomic_json(path, {"records": [{"window": i, "tokens": 2047, "nll_sum": 2047 * math.log(6.)} for i in range(138)]})
        protocol["baseline_evaluations"][title] = file_record(path)
    manifest = {"sha256": fingerprint(payload), "payload": payload}
    atomic_json(Path(config["run_dir"]) / "manifest.json", manifest)
    store = collector.checkpoints(config, manifest, 0)
    store.save({name: {"gram": gram, "diagonal": diagonal}}, 256, 100.)
    monkeypatch.setattr(pipeline, "verify_model", lambda *_: None)
    monkeypatch.setattr(pipeline, "weight_tensor", lambda *_: weight.clone())
    monkeypatch.setattr(pipeline, "import_official_qera", lambda *_: None)
    module = ModuleType("qera.approximate")
    module._compute_scale_inv_dot_U = lambda a, u: u / a[:, None] if a.ndim == 1 else torch.linalg.solve(a, u)
    monkeypatch.setitem(sys.modules, "qera.approximate", module)
    return config, manifest, name, store


def test_solve_resume_and_import_26_configuration_summary(small_run):
    config, manifest, name, _ = small_run
    source_before = {p: p.read_bytes() for p in Path(config["run_dir"]).parent.joinpath("source").iterdir()}
    pipeline.import_baseline_evaluations(config, manifest)
    assert pipeline.summarize(config)["configurations"] == 18
    pipeline.solve(config, checkpoint.Stop())
    first = pipeline.owned_correction(config, manifest, "full_gf", name)
    pipeline.solve(config, checkpoint.Stop())
    assert first == pipeline.owned_correction(config, manifest, "full_gf", name)
    for title, method, _ in pipeline.configurations(config)[18:]:
        inputs = pipeline.evaluation_inputs(config, manifest, method)
        protocol = fingerprint({"manifest": manifest["sha256"], "name": title, "artifacts": inputs})
        atomic_json(Path(config["run_dir"]) / "evaluation/configurations" / f"{title}.json", {
            "protocol_sha256": protocol,
            "records": [{"window": i, "tokens": 2047, "nll_sum": 2047 * math.log(5.9)} for i in range(138)]})
    result = pipeline.summarize(config)
    assert result["status"] == "PASS" and result["configurations"] == result["expected_configurations"] == 26
    assert source_before == {p: p.read_bytes() for p in source_before}
    path = Path(config["run_dir"]) / "evaluation/configurations/FULL_GF_R4.json"
    state = read_json(path)
    state["records"].pop()
    atomic_json(path, state)
    assert pipeline.summarize(config)["status"] == "INCOMPLETE"


def test_solve_interruption_after_first_correction_does_not_overwrite(small_run, monkeypatch):
    config, manifest, name, _ = small_run
    original = pipeline.legacy.save_artifact
    def interrupt_after_save(path, *args, **kwargs):
        result = original(path, *args, **kwargs)
        if Path(path).parent.name == "diag_gf":
            raise checkpoint.Paused("simulated server cutoff")
        return result
    monkeypatch.setattr(pipeline.legacy, "save_artifact", interrupt_after_save)
    with pytest.raises(checkpoint.Paused):
        pipeline.solve(config, checkpoint.Stop())
    saved = pipeline.owned_correction(config, manifest, "diag_gf", name)
    stamp = Path(saved["file"]["path"]).stat().st_mtime_ns
    monkeypatch.setattr(pipeline.legacy, "save_artifact", original)
    pipeline.solve(config, checkpoint.Stop())
    assert Path(saved["file"]["path"]).stat().st_mtime_ns == stamp
    assert pipeline.owned_correction(config, manifest, "full_gf", name)


def test_failed_gate_cannot_produce_full_g_correction(small_run, monkeypatch):
    config, _, _, _ = small_run
    monkeypatch.setattr(pipeline, "correction_drift", lambda *_: {str(r): .003 for r in config["ranks"]})
    with pytest.raises(RuntimeError, match="regression failed"):
        pipeline.solve(config, checkpoint.Stop())
    assert not list((Path(config["run_dir"]) / "corrections").rglob("*.safetensors"))


def test_collect_parent_mismatch_stops_before_solving(small_run):
    config, manifest, _, store = small_run
    state = store.load(tensors=False)
    state["teacher_nll_sum"] = 300.
    with pytest.raises(RuntimeError, match="Teacher NLL"):
        collector.compare_parent(config, manifest, 0, store, state)


def test_actual_collect_loop_resumes_whole_windows(tmp_path, monkeypatch):
    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(16, 8)
            self.proj = torch.nn.Linear(8, 8, bias=False)
        def forward(self, input_ids, **_):
            return SimpleNamespace(last_hidden_state=self.proj(self.embed(input_ids)))
    class Teacher(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Decoder()
            self.head = torch.nn.Linear(8, 16, bias=False)
            self.hf_device_map = {"model": 0, "head": 1}
        def get_input_embeddings(self):
            return self.model.embed
        def get_output_embeddings(self):
            return self.head
    teacher = Teacher()
    # Distinct windows catch repeats/skips and off-by-one resume indexing.
    ids = torch.randint(0, 16, (256, 2048))
    data_path = tmp_path / "data.safetensors"
    atomic_tensors(data_path, {"input_ids": ids, "attention_mask": torch.ones_like(ids)})
    config = {"run_dir": str(tmp_path / "run"), "collect_max_memory": {},
              "ce_gradient_chunk_tokens": 128, "checkpoint_every_windows": 8}
    layer = {"name": "model.proj", "shape": [8, 8]}
    payload = {"config": config, "code": {}, "source_config": {}, "data": {"calibration": file_record(data_path)},
               "full_g_protocol": {"shards": [[layer]], "gram_row_tile": 3,
                                   "teacher_device_map": {"model": "0", "head": "1"}}}
    manifest = {"sha256": fingerprint(payload), "payload": payload}
    atomic_json(Path(config["run_dir"]) / "manifest.json", manifest)
    monkeypatch.setattr(collector, "load_model", lambda *_: deepcopy(teacher))
    monkeypatch.setattr(collector, "verify_model", lambda *_: None)
    monkeypatch.setattr(collector, "_input_device", lambda _: "cpu")
    monkeypatch.setattr(collector, "_reset_evaluation_memory_peaks", lambda *_: None)
    monkeypatch.setattr(torch.autograd.graph, "save_on_cpu", lambda **_: nullcontext())
    with pytest.raises(checkpoint.Paused):
        collector.collect(config, checkpoint.Stop(), max_new_windows=4)
    store = collector.checkpoints(config, manifest, 0)
    v4, n4, nll4 = store.load()
    assert n4 == 4
    with pytest.raises(checkpoint.Paused):
        collector.collect(config, checkpoint.Stop(), max_new_windows=3)
    v7, n7, nll7 = store.load()
    assert n7 == 7 and nll7 > nll4
    other_config = dict(config, run_dir=str(tmp_path / "uninterrupted"))
    other_payload = deepcopy(payload)
    other_payload["config"] = other_config
    other_manifest = {"sha256": fingerprint(other_payload), "payload": other_payload}
    atomic_json(Path(other_config["run_dir"]) / "manifest.json", other_manifest)
    with pytest.raises(checkpoint.Paused):
        collector.collect(other_config, checkpoint.Stop(), max_new_windows=7)
    uninterrupted, n_full, nll_full = collector.checkpoints(other_config, other_manifest, 0).load()
    assert n_full == n7 and nll_full == nll7
    assert torch.equal(v7["model.proj"]["gram"], uninterrupted["model.proj"]["gram"])
    assert torch.equal(v7["model.proj"]["diagonal"], uninterrupted["model.proj"]["diagonal"])
    # Failure AFTER a hook mutates in-memory sums must leave durable count at 7.
    original = collector.accumulate_gram
    def bad(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("simulated partial backward")
    monkeypatch.setattr(collector, "accumulate_gram", bad)
    with pytest.raises(RuntimeError, match="partial backward"):
        collector.collect(config, checkpoint.Stop(), max_new_windows=1)
    assert store.load()[1] == 7
    assert torch.equal(store.load()[0]["model.proj"]["gram"], v7["model.proj"]["gram"])


def test_prepare_isolated_and_restartable_after_manifest_commit(small_run, monkeypatch, tmp_path):
    small_config, small_manifest, _, _ = small_run
    parent = {"run_dir": str(tmp_path / "parent"), "teacher_dtype": "float32", "collect_batch_size": 1,
              "ce_gradient_chunk_tokens": 128, "num_calibration_windows": 256, "sequence_length": 2048,
              "ranks": [8, 16, 32, 64], "g_relative_floor": 1e-6, "identity_product_tolerance": 1e-3,
              "eval_batch_size": 8, "eval_ce_chunk_tokens": 256, "save_activations_on_cpu": True,
              "control_ppl_tolerance": .01}
    child = run.derived_config(parent)
    source = deepcopy(small_manifest)
    source["payload"]["data"] = {}
    protocol = deepcopy(source["payload"]["full_g_protocol"])
    protocol["baseline_ppl"] = {}
    new_evaluations = {}
    for title, _, _ in pipeline.mx3.configurations(child):
        old = re.sub(r"_R(8|16|32|64)$", lambda m: "_R" + str([8, 16, 32, 64].index(int(m[1])) + 1), title)
        new_evaluations[title] = protocol["baseline_evaluations"][old]
        protocol["baseline_ppl"][title] = 6.
    protocol["baseline_evaluations"] = new_evaluations
    monkeypatch.setattr(run, "audit_source", lambda _: (small_config, source, deepcopy(protocol)))
    monkeypatch.setattr(run, "shard_layers", lambda _: protocol["shards"])
    original = pipeline.import_baseline_evaluations
    monkeypatch.setattr(pipeline, "import_baseline_evaluations", lambda *_: (_ for _ in ()).throw(RuntimeError("crash after manifest")))
    with pytest.raises(RuntimeError, match="after manifest"):
        run.prepare(parent, child)
    manifest_bytes = (Path(child["run_dir"]) / "manifest.json").read_bytes()
    monkeypatch.setattr(pipeline, "import_baseline_evaluations", original)
    manifest = run.prepare(parent, child)
    assert load_manifest(child) == manifest
    assert (Path(child["run_dir"]) / "manifest.json").read_bytes() == manifest_bytes
    assert pipeline.summarize(child)["configurations"] == 18
    assert not list(Path(child["run_dir"]).rglob("*.safetensors"))
    assert run.prepare(parent, child) == manifest
    changed = dict(child, g_relative_floor=.001)
    with pytest.raises(RuntimeError, match="preserve"):
        run.prepare(parent, changed)


def test_summary_rejects_tampered_new_evaluation_protocol(small_run):
    config, manifest, _, _ = small_run
    pipeline.import_baseline_evaluations(config, manifest)
    path = Path(config["run_dir"]) / "evaluation/configurations/W3_MXINT.json"
    state = read_json(path)
    state["protocol_sha256"] = "wrong"
    atomic_json(path, state)
    with pytest.raises(RuntimeError, match="protocol"):
        pipeline.summarize(config)


def test_audit_requires_complete_matching_mxint3_without_source_writes(fixture_run):
    parent_config, parent, child, _, _ = fixture_run
    source = run.mx3_run.prepare(parent_config, parent, child)
    pipeline.mx3.quantize(child)
    pipeline.mx3.solve(child)
    runtime = Path(parent_config["run_dir"]) / "statistics/runtime.json"
    atomic_json(runtime, {"dtype": "float32", "batch_size": 1, "save_on_cpu": True,
                          "hf_device_map": {"model": "0"}})
    for title, method, _ in pipeline.mx3.configurations(child):
        artifacts = pipeline.mx3.evaluation_inputs(child, source, method)
        digest = fingerprint({"manifest": source["sha256"], "name": title, "artifacts": artifacts})
        atomic_json(Path(child["run_dir"]) / "evaluation/configurations" / f"{title}.json", {
            "protocol_sha256": digest,
            "records": [{"window": i, "tokens": 2047, "nll_sum": 2047 * math.log(6.)} for i in range(138)]})
    before = {p: p.read_bytes() for p in Path(parent_config["run_dir"]).rglob("*") if p.is_file()}
    _, audited, result = run.audit_source(parent_config)
    assert audited == source and len(result["baseline_evaluations"]) == 18
    assert before == {p: p.read_bytes() for p in before}
    last = Path(child["run_dir"]) / "evaluation/configurations/FULL_GD_R4.json"
    state = read_json(last)
    state["records"].pop()
    atomic_json(last, state)
    with pytest.raises(RuntimeError, match="completed MXINT3"):
        run.audit_source(parent_config)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA hardware; server gates remain mandatory")
def test_cuda_fp64_gram_and_dense_diagonal_gate():
    device = "cuda:0"
    torch.backends.cuda.matmul.allow_tf32 = False
    gradient = torch.randn(1, 129, 96, device=device)
    mask = torch.ones(1, 129, dtype=torch.bool)
    gram, diagonal = torch.zeros(96, 96, dtype=torch.float64), torch.zeros(96, dtype=torch.float64)
    numerics.accumulate_gram(gradient, mask, gram, diagonal, 32)
    assert numerics.check_diagonal(gram, diagonal) < 1e-10
    error = torch.randn(128, 96, device=device)
    scale_a = torch.rand(128, device=device) + .5
    scale_g, _ = diagonal_scale(diagonal, 129)
    scale_g = scale_g.to(device)
    a, b, _ = numerics.solve_full(error, scale_a, torch.diag(scale_g), 64)
    ref_a, ref_b, _ = solve_weighted(error, scale_a, scale_g, 64)
    assert max(correction_drift(a, b, ref_a, ref_b, [8, 16, 32, 64]).values()) < .001
