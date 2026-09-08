"""CPU contract/integration tests; server numerical gates are never bypassed."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qera_diag_g_isolation import pipeline as legacy
from qera_diag_g_isolation.full_svd_v1 import solver
from qera_diag_g_isolation.mxint3_v1 import pipeline, run
from qera_diag_g_isolation.mxint3_v1.run import audit_mxint4 as actual_audit_mxint4
from qera_diag_g_isolation.storage import atomic_json, file_record, fingerprint, load_manifest, read_json


def inverse(scale, u):
    return torch.linalg.solve(torch.diag(scale) if scale.ndim == 1 else scale, u)


def mock_official():
    """Independent reference algebra for orchestration tests, not a server reference."""
    def quantizer(weight, **kwargs):
        assert kwargs == {"width": 3, "block_size": 32, "block_axis": -1}
        return (weight.float() * 2).round().div(2).to(weight.dtype)

    def compute_ab(name, layer, scale, config):
        assert config["w_quantizer"] == pipeline.QUANTIZATION
        quantized = quantizer(layer.weight, **{k: v for k, v in config["w_quantizer"].items() if k != "name"})
        error = (layer.weight - quantized).T
        matrix = (torch.diag(scale) if scale.ndim == 1 else scale) @ error
        u, s, vh = torch.linalg.svd(matrix, full_matrices=True)
        rank = config["rank"]
        a, b = inverse(scale, u[:, :rank]), torch.diag(s[:rank]) @ vh[:rank]
        config["w_quantizer"].pop("name")
        return {name + ".A": a, name + ".B": b}, float((error - a @ b).square().mean())
    return {"mxint_quantizer": quantizer, "compute_ab": compute_ab}


@pytest.fixture
def fixture_run(tmp_path, monkeypatch):
    torch.manual_seed(41)
    torch.set_num_threads(1)
    root = tmp_path / "g256"
    root.mkdir()
    config = {"run_dir": str(root), "ranks": [1, 2, 3, 4], "identity_product_tolerance": .001,
              "control_ppl_tolerance": .01, "solve_device": "cpu", "g_relative_floor": 1e-6}
    name = "model.layers.0.self_attn.k_proj"
    roots_path = tmp_path / "a_roots.safetensors"
    from qera_diag_g_isolation.storage import atomic_tensors
    atomic_tensors(roots_path, {"diag": torch.linspace(.5, 1.5, 32), "full": torch.eye(32) + .01})
    frozen_data = tmp_path / "frozen_data.txt"
    frozen_data.write_text("frozen calibration and evaluation fixture")
    code = tmp_path / "frozen_code.py"
    code.write_text("# never edit this parent source")
    old_gi = file_record(roots_path)  # Deliberately unusable W4 factors; must never be consumed.
    layer = {"name": name, "shape": [8, 32], "weight_file": "fixture", "gi": {"diag": old_gi, "full": old_gi}}
    payload = {"config": config, "source_config": {"quantization": dict(pipeline.QUANTIZATION, width=4)},
               "data": {"calibration": file_record(frozen_data), "wikitext2": file_record(frozen_data)},
               "groups": [{"roots": file_record(roots_path), "raw_a": file_record(roots_path), "layers": [layer]}],
               "code": {str(code): file_record(code)}, "model_files": {},
               "control_reference_ppl": {"BF16": 6., "W4_MXINT": 7., "DIAG_GI_R1": 6.5}}
    parent = {"sha256": fingerprint(payload), "payload": payload}
    atomic_json(root / "manifest.json", parent)
    legacy._save_g_checkpoint(config, parent, {name: torch.arange(1., 9., dtype=torch.float64)}, 256, 10.)
    monkeypatch.setattr(run, "audit_mxint4", lambda *_: {"ppl": {"BF16": 6.}, "fixture": True})
    child_config = run.derived_config(config)
    Path(child_config["run_dir"]).mkdir()
    weight = torch.randn(8, 32).bfloat16()
    monkeypatch.setattr(pipeline, "verify_model", lambda *_: None)
    monkeypatch.setattr(pipeline, "weight_tensor", lambda *_: weight.clone())
    official = mock_official()
    monkeypatch.setattr(pipeline, "import_official_qera", lambda *_: official)
    approximate = ModuleType("qera.approximate")
    approximate._compute_scale_inv_dot_U = inverse
    monkeypatch.setitem(sys.modules, "qera.approximate", approximate)
    return config, parent, child_config, layer, official


def snapshot(root, exclude="mxint3_v1"):
    return {p: p.read_bytes() for p in Path(root).rglob("*") if p.is_file() and exclude not in p.parts}


def test_configuration_matrix_and_width_are_isolated():
    config = {"ranks": [8, 16, 32, 64]}
    choices = pipeline.configurations(config)
    assert len(choices) == 18 and len({x[0] for x in choices}) == 18
    assert choices[1] == ("W3_MXINT", "wq", None)
    assert legacy.configurations(config)[1][0] == "W4_MXINT"
    for r in config["ranks"]:
        value = pipeline.layer_config(r)
        assert value["rank"] == r and value["w_quantizer"]["width"] == 3
        assert value["x_quantizer"] == {"name": "bypass"}
        value["w_quantizer"].pop("name")
    assert pipeline.layer_config(64)["w_quantizer"] == pipeline.QUANTIZATION


def test_prepare_preserves_parent_pins_code_and_reuses_only_statistics(fixture_run):
    config, parent, child, layer, _ = fixture_run
    before = snapshot(config["run_dir"])
    manifest = run.prepare(config, parent, child)
    assert load_manifest(config) == parent
    assert manifest["payload"]["groups"][0]["layers"][0]["gi"] == {}
    assert manifest["payload"]["control_reference_ppl"] == {"BF16": 6.}
    assert manifest["payload"]["source_config"]["quantization"]["width"] == 3
    assert parent["payload"]["source_config"]["quantization"]["width"] == 4
    tensors, count, _ = legacy._load_g_checkpoint(child, manifest)
    assert count == 256 and tensors[layer["name"]].numel() == 8
    assert not list(Path(child["run_dir"]).rglob("*.safetensors"))
    child_before = snapshot(child["run_dir"], exclude="nothing")
    assert run.prepare(config, parent, child) == manifest
    assert child_before == snapshot(child["run_dir"], exclude="nothing")
    assert before == snapshot(config["run_dir"])


def test_rejects_changed_settings_and_unknown_output(fixture_run):
    config, parent, child, _, _ = fixture_run
    changed = deepcopy(child)
    changed["g_relative_floor"] = 1e-4
    with pytest.raises(RuntimeError, match="Only weight bit"):
        run.prepare(config, parent, changed)
    note = Path(child["run_dir"]) / "keep.txt"
    note.write_text("do not overwrite")
    with pytest.raises(RuntimeError, match="Unrecognized"):
        run.prepare(config, parent, child)
    assert note.read_text() == "do not overwrite"


def test_rejects_corrupt_reused_g(fixture_run):
    config, parent, child, _, _ = fixture_run
    state = read_json(Path(config["run_dir"]) / "statistics/collect_state.json")
    Path(state["file"]["path"]).write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="Immutable input"):
        run.prepare(config, parent, child)


def test_rejects_code_change_on_resume(fixture_run, monkeypatch):
    config, parent, child, _, _ = fixture_run
    run.prepare(config, parent, child)
    original = run.file_record
    def changed(path):
        record = original(path)
        if Path(path) == Path(pipeline.__file__):
            record["sha256"] = "changed"
        return record
    monkeypatch.setattr(run, "file_record", changed)
    with pytest.raises(RuntimeError, match="differs"):
        run.prepare(config, parent, child)


def test_quantize_solve_and_resume_all_four_new_corrections(fixture_run):
    config, parent, child, layer, _ = fixture_run
    frozen = snapshot(config["run_dir"])
    manifest = run.prepare(config, parent, child)
    pipeline.quantize(child)
    pipeline.solve(child)
    for method in ("diag_gi", "full_gi", "diag_gd", "full_gd"):
        files = pipeline.evaluation_inputs(child, manifest, method)[layer["name"]]
        assert Path(files["correction"]["path"]).is_relative_to(child["run_dir"])
        assert Path(files["quant"]["path"]).is_relative_to(child["run_dir"])
    before = snapshot(child["run_dir"], exclude="nothing")
    pipeline.quantize(child)
    pipeline.solve(child)
    assert before == snapshot(child["run_dir"], exclude="nothing")
    assert frozen == snapshot(config["run_dir"])


def test_partial_gd_solve_resumes_without_rewriting_gi(fixture_run, monkeypatch):
    config, parent, child, layer, _ = fixture_run
    manifest = run.prepare(config, parent, child)
    pipeline.quantize(child)
    original = pipeline.solve_weighted
    def interrupted(error, a, g, *args):
        if not torch.equal(g, torch.ones_like(g)):
            raise RuntimeError("simulated interruption before GD")
        return original(error, a, g, *args)
    monkeypatch.setattr(pipeline, "solve_weighted", interrupted)
    with pytest.raises(RuntimeError, match="simulated"):
        pipeline.solve(child)
    gi = pipeline.owned_artifact(child, manifest, "corrections/diag_gi", layer["name"])
    before = Path(gi["file"]["path"]).read_bytes()
    monkeypatch.setattr(pipeline, "solve_weighted", original)
    pipeline.solve(child)
    assert before == Path(gi["file"]["path"]).read_bytes()
    pipeline.evaluation_inputs(child, manifest, "diag_gd")


def test_identity_gate_failure_never_commits_corrections(fixture_run, monkeypatch):
    config, parent, child, _, _ = fixture_run
    run.prepare(config, parent, child)
    pipeline.quantize(child)
    monkeypatch.setattr(pipeline, "correction_drift", lambda *args: {str(r): .003 for r in child["ranks"]})
    with pytest.raises(RuntimeError, match="identity-G regression failed"):
        pipeline.solve(child)
    assert not list((Path(child["run_dir"]) / "corrections").rglob("*.json"))
    assert not (Path(child["run_dir"]) / "solve_complete.json").exists()


def test_quantization_dtype_mismatch_is_fatal(fixture_run, monkeypatch):
    config, parent, child, _, official = fixture_run
    run.prepare(config, parent, child)
    original = official["mxint_quantizer"]
    def bad(weight, **kwargs):
        result = original(weight, **kwargs)
        return result + .125 if weight.dtype == torch.float32 else result
    monkeypatch.setitem(official, "mxint_quantizer", bad)
    with pytest.raises(RuntimeError, match="quantization mismatch"):
        pipeline.quantize(child)
    assert not list((Path(child["run_dir"]) / "quantized").rglob("*.safetensors"))


def test_rejects_parent_weight_pointer_and_failed_identity_metadata(fixture_run):
    config, parent, child, layer, _ = fixture_run
    manifest = run.prepare(config, parent, child)
    pipeline.quantize(child)
    pipeline.solve(child)
    path = legacy.artifact_path(child, "corrections/diag_gi", layer["name"]).with_suffix(".json")
    record = read_json(path)
    changed = deepcopy(record)
    changed["identity_product_drift"]["1"] = .003
    atomic_json(path, changed)
    with pytest.raises(RuntimeError, match="identity regression"):
        pipeline.evaluation_inputs(child, manifest, "diag_gi")
    atomic_json(path, record)
    qpath = legacy.artifact_path(child, "quantized", layer["name"]).with_suffix(".json")
    qrecord = read_json(qpath)
    qrecord["file"]["path"] = str(Path(config["run_dir"]) / "old_w4.safetensors")
    atomic_json(qpath, qrecord)
    with pytest.raises(RuntimeError, match="newly generated"):
        pipeline.evaluation_inputs(child, manifest, "wq", check=False)


def test_binding_restores_legacy_even_after_error():
    original = legacy.configurations, legacy._evaluation_inputs
    with pytest.raises(RuntimeError):
        with pipeline.evaluation_binding():
            assert legacy.configurations is pipeline.configurations
            assert legacy._evaluation_inputs is pipeline.evaluation_inputs
            raise RuntimeError("interrupted")
    assert (legacy.configurations, legacy._evaluation_inputs) == original


def test_summary_checks_complete_windows_and_bf16_only(fixture_run):
    config, parent, child, _, _ = fixture_run
    manifest = run.prepare(config, parent, child)
    pipeline.quantize(child)
    pipeline.solve(child)
    for name, method, _ in pipeline.configurations(child):
        artifacts = pipeline.evaluation_inputs(child, manifest, method)
        protocol = fingerprint({"manifest": manifest["sha256"], "name": name, "artifacts": artifacts})
        import math
        rows = [{"window": i, "tokens": 2047, "nll_sum": 2047 * math.log(6. if name == "BF16" else 10.)}
                for i in range(138)]
        atomic_json(Path(child["run_dir"]) / "evaluation/configurations" / f"{name}.json",
                    {"protocol_sha256": protocol, "records": rows})
    assert pipeline.summarize(child)["status"] == "PASS"
    checks = list((Path(child["run_dir"]) / "evaluation/control_checks").glob("*.json"))
    assert [x.stem for x in checks] == ["BF16"]
    summary = (Path(child["run_dir"]) / "evaluation/ppl_summary_wikitext2.csv").read_text()
    assert "W3_MXINT" in summary and "W4_MXINT" not in summary
    last_path = Path(child["run_dir"]) / "evaluation/configurations/FULL_GD_R4.json"
    last = read_json(last_path)
    last["records"].pop()
    atomic_json(last_path, last)
    partial = pipeline.summarize(child)
    assert partial["status"] == "INCOMPLETE" and partial["configurations"] == 17
    assert "FULL_GD_R4" not in (Path(child["run_dir"]) / "evaluation/ppl_summary_wikitext2.csv").read_text()


@pytest.mark.parametrize("full_a", [False, True])
def test_real_pinned_official_mxint3_identity_small_matrix(full_a, monkeypatch):
    """Optional local source test: run actual pinned quantizer + official compute_ab.

    Avoid importing the whole HF package: only unused utility imports are stubbed.
    No solver/quantizer body is rewritten or mocked in this test.
    """
    source = Path(__file__).resolve().parents[4] / "QERA/src/qera"
    if not source.is_dir():
        pytest.skip("Optional sibling official QERA source not available")
    from qera_original_a_isolation.common import OFFICIAL_FILE_SHA256, sha256_source_file
    for name in ("approximate.py", "quantize/quantizers/mxint.py"):
        assert sha256_source_file(source / name) == OFFICIAL_FILE_SHA256["src/qera/" + name]
    spec = importlib.util.spec_from_file_location("actual_mxint", source / "quantize/quantizers/mxint.py")
    mx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mx)
    utils = ModuleType("qera.utils")
    for name in ("find_matched_pattern", "get_layer_by_name", "get_full_device_map", "move_module_to_device"):
        setattr(utils, name, lambda *args: None)
    quant = ModuleType("qera.quantize")
    quant.get_quantizer = lambda name: mx.mxint_quantizer if name == "mxint" else (lambda x: x)
    monkeypatch.setitem(sys.modules, "qera.utils", utils)
    monkeypatch.setitem(sys.modules, "qera.quantize", quant)
    spec = importlib.util.spec_from_file_location("qera.actual_test_approximate", source / "approximate.py")
    approximate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(approximate)
    torch.manual_seed(71)
    weight = torch.randn(80, 96).bfloat16()
    q32 = mx.mxint_quantizer(weight.float(), width=3, block_size=32, block_axis=-1)
    q16 = mx.mxint_quantizer(weight, width=3, block_size=32, block_axis=-1)
    assert torch.equal(q32, q16.float())
    scale = torch.rand(96) + .5
    if full_a:
        scale = torch.diag(scale) + .01
    ref_a, ref_b, _ = pipeline.official_identity(
        {"compute_ab": approximate._compute_scales_and_error_for_fc}, "test", weight.float(), scale, 64)
    a, b, _ = solver.solve_weighted((weight.float() - q32).T, scale, torch.ones(80), 64,
                                   approximate._compute_scale_inv_dot_U)
    from qera_diag_g_isolation.math_ops import correction_drift
    assert max(correction_drift(a, b, ref_a, ref_b, [8, 16, 32, 64]).values()) <= .001


def test_audit_mxint4_requires_complete_matching_comparison_without_writes(fixture_run):
    import math
    config, parent, _, layer, _ = fixture_run
    full_config = run.full_run.derived_config(config)
    root = Path(full_config["run_dir"])
    report_path = Path(config["run_dir"]) / "diagnostics/report.json"
    report = {"status": "AUDIT_COMPLETE", "manifest_sha256": parent["sha256"],
              "layer": layer["name"], "method": "diag",
              "inputs": {k: {"exact_equal": True} for k in (
                  "fresh_official_q_vs_frozen_q", "new_error_vs_official_error", "new_svd_input_vs_official_svd_input")},
              "variants": {}}
    for name in ("official_replay", "old_input_full_svd", "new_input_full_svd", "current_solver",
                 "old_input_reduced_svd", "new_input_reduced_svd"):
        value = .003 if name in ("current_solver", "old_input_reduced_svd", "new_input_reduced_svd") else 0.
        report["variants"][name] = {"vs_saved_production_gate": {str(r): value for r in config["ranks"]}}
    atomic_json(report_path, report)
    payload = deepcopy(parent["payload"])
    payload["config"] = full_config
    payload["solver_transition"] = {
        "variant": "full_svd_v1", "svd_full_matrices": True, "parent_manifest_sha256": parent["sha256"],
        "parent_manifest": file_record(Path(config["run_dir"]) / "manifest.json"),
        "audit_report": file_record(report_path),
    }
    manifest = {"sha256": fingerprint(payload), "payload": payload}
    atomic_json(root / "manifest.json", manifest)
    for directory in ("quantized", "corrections/diag_gd", "corrections/full_gd"):
        legacy.save_artifact(legacy.artifact_path(full_config, directory, layer["name"]), manifest,
                             {"fixture": torch.ones(1)})
    for name, method, _ in legacy.configurations(full_config):
        artifacts = legacy._evaluation_inputs(full_config, manifest, method, check=False)
        protocol = fingerprint({"manifest": manifest["sha256"], "name": name, "artifacts": artifacts})
        ppl = payload["control_reference_ppl"].get(name, 8.)
        rows = [{"window": i, "tokens": 2047, "nll_sum": 2047 * math.log(ppl)} for i in range(138)]
        atomic_json(root / "evaluation/configurations" / f"{name}.json",
                    {"protocol_sha256": protocol, "records": rows})
    before = snapshot(config["run_dir"])
    result = actual_audit_mxint4(config, parent)
    assert len(result["evaluations"]) == 18 and result["ppl"]["BF16"] == pytest.approx(6.)
    assert before == snapshot(config["run_dir"])
    last_path = root / "evaluation/configurations/FULL_GD_R4.json"
    last = read_json(last_path)
    last["records"].pop()
    atomic_json(last_path, last)
    with pytest.raises(RuntimeError, match="Finish the 18"):
        actual_audit_mxint4(config, parent)


def test_partial_prepare_can_resume(fixture_run, monkeypatch):
    config, parent, child, _, _ = fixture_run
    original = run.full_run.write_same_or_new
    def interrupted(path, value):
        if Path(path).name == "config.json":
            raise RuntimeError("simulate initialization interruption")
        return original(path, value)
    monkeypatch.setattr(run.full_run, "write_same_or_new", interrupted)
    with pytest.raises(RuntimeError, match="initialization interruption"):
        run.prepare(config, parent, child)
    assert (Path(child["run_dir"]) / "initialization.json").exists()
    monkeypatch.setattr(run.full_run, "write_same_or_new", original)
    manifest = run.prepare(config, parent, child)
    assert load_manifest(child) == manifest
