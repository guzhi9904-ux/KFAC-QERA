"""Full-G solve and unchanged BF16 evaluation with explicit frozen W3 routing."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import math

import torch

from qera_original_a_isolation.common import import_official_qera
from qera_diag_g_isolation import pipeline as legacy
from qera_diag_g_isolation.mxint3_v1 import pipeline as mx3
from qera_diag_g_isolation.math_ops import correction_drift, diagonal_scale
from qera_diag_g_isolation.storage import (
    atomic_json, checked_tensors, fingerprint, heartbeat, layers, load_manifest, log,
    read_json, verify, verify_model, weight_tensor,
)
from .collect import completed_audit
from .numerics import full_root, solve_full

_legacy_summary = legacy.summarize


def configurations(config):
    return mx3.configurations(config) + [
        (f"{method.upper()}_R{rank}", method, rank)
        for method in ("diag_gf", "full_gf") for rank in config["ranks"]]


def passing_drift(values, config):
    return (set(values) == {str(r) for r in config["ranks"]} and all(
        math.isfinite(v) and 0 <= v <= config["identity_product_tolerance"] for v in values.values()))


def owned_correction(config, manifest, method, name, check=True):
    path = legacy.artifact_path(config, f"corrections/{method}", name)
    record = legacy.completed_artifact(path, manifest, check)
    if record is None:
        return None
    if (Path(record["file"]["path"]).resolve() != path.resolve()
            or record.get("method") != method or record.get("layer") != name
            or record.get("rank") != max(config["ranks"]) or record.get("svd_full_matrices") is not True
            or record.get("quantization") != mx3.QUANTIZATION
            or not passing_drift(record.get("identity_product_drift", {}), config)
            or not passing_drift(record.get("diagonal_product_drift", {}), config)):
        raise RuntimeError("Full-G correction lacks passing identity/diagonal regressions")
    baseline = manifest["payload"]["full_g_protocol"]["baseline_inputs"]
    a_kind = method.split("_")[0]
    if (record.get("gi_reference") != baseline[a_kind + "_gi"][name]["correction"]
            or record.get("gd_reference") != baseline[a_kind + "_gd"][name]["correction"]
            or record.get("quant_reference") != baseline["wq"][name]["quant"]):
        raise RuntimeError("Full-G correction baseline binding mismatch")
    return record


def solve(config, stop):
    manifest = load_manifest(config)
    protocol = manifest["payload"]["full_g_protocol"]
    by_name = {}
    for i in range(len(protocol["shards"])):
        store, state, _ = completed_audit(config, manifest, i)
        for name in store.dimensions:
            by_name[name] = (store, state)
    verify_model(manifest)
    import_official_qera(manifest["payload"]["source_config"])
    from qera.approximate import _compute_scale_inv_dot_U
    rank, device = max(config["ranks"]), config["solve_device"]
    count = 0
    with torch.no_grad():
        for group in manifest["payload"]["groups"]:
            stop.check()
            roots = checked_tensors(group["roots"])
            for layer in group["layers"]:
                stop.check()
                count += 1
                name = layer["name"]
                store, state = by_name[name]
                pending = []
                for kind in ("diag", "full"):
                    record = owned_correction(config, manifest, kind + "_gf", name)
                    if record is None:
                        pending.append(kind)
                    elif record.get("raw_g_reference") != state["files"][name]:
                        raise RuntimeError("Full-G correction bound to a different raw G")
                log("solve-full-g", f"module={count}/{len(layers(manifest))} {name} pending={pending}")
                if not pending:
                    continue
                values = store.read_layer(state, name)
                # Direct diagonal was already checked against parent at collect completion.
                diagonal, _ = diagonal_scale(values["diagonal"], 256 * 2047, config["g_relative_floor"])
                diagonal = diagonal.to(device)
                qrecord = protocol["baseline_inputs"]["wq"][name]["quant"]
                error_t = (weight_tensor(manifest, layer).float() -
                           checked_tensors(qrecord)["weight_q"].float()).T.to(device)
                # Persist each expensive gate independently, so restart does not repeat it.
                gates = {}
                for kind in pending:
                    stop.check()
                    path = Path(config["run_dir"]) / "diagnostics/solver_gates" / f"{name}.{kind}.json"
                    gi_record = protocol["baseline_inputs"][kind + "_gi"][name]["correction"]
                    gd_record = protocol["baseline_inputs"][kind + "_gd"][name]["correction"]
                    binding = {"manifest": manifest["sha256"], "gi": gi_record, "gd": gd_record,
                               "quant": qrecord, "roots": group["roots"], "raw_g": state["files"][name]}
                    digest = fingerprint(binding)
                    if path.exists():
                        gate = read_json(path)
                        if (gate.get("protocol_sha256") != digest or gate.get("status") != "PASS"
                                or not passing_drift(gate.get("identity", {}), config)
                                or not passing_drift(gate.get("diagonal", {}), config)):
                            raise RuntimeError("Resumed solver gate mismatch")
                    else:
                        scale_a = roots[kind].to(device)
                        results = {}
                        for label, vector, reference in (
                            ("identity", torch.ones_like(diagonal), gi_record),
                            ("diagonal", diagonal, gd_record),
                        ):
                            with heartbeat("gate-full-g", f"{name} A={kind} G={label}"):
                                # Exercise the DENSE matrix route, not the old vector solver.
                                # Use the effective root verbatim, without re-normalizing
                                # already-floored eigenvalues. Root extension is unit-tested.
                                a, b, _ = solve_full(error_t, scale_a, torch.diag(vector), rank, _compute_scale_inv_dot_U)
                                ref = checked_tensors(reference)
                                results[label] = correction_drift(a, b, ref["A"].to(device), ref["B"].to(device), config["ranks"])
                                del a, b, ref
                            if not passing_drift(results[label], config):
                                raise RuntimeError(f"Full-G {label} regression failed: {name} {kind} {results[label]}")
                        gate = {"status": "PASS", "protocol_sha256": digest, **results}
                        atomic_json(path, gate)
                        del scale_a
                    gates[kind] = gate
                    stop.check()
                with heartbeat("root-full-g", f"{name} FP64 eigh and eigenvalue floor"):
                    root_g, diagnostics = full_root(values["gram"].to(device), 256 * 2047, config["g_relative_floor"])
                del values
                stop.check()
                for kind in pending:
                    with heartbeat("solve-full-g", f"{name} A={kind} dense G full SVD"):
                        a, b, metrics = solve_full(error_t, roots[kind].to(device), root_g, rank, _compute_scale_inv_dot_U)
                    method = kind + "_gf"
                    legacy.save_artifact(legacy.artifact_path(config, f"corrections/{method}", name), manifest,
                        {"A": a, "B": b}, layer=name, method=method, rank=rank, quantization=mx3.QUANTIZATION,
                        identity_product_drift=gates[kind]["identity"], diagonal_product_drift=gates[kind]["diagonal"],
                        gi_reference=protocol["baseline_inputs"][kind + "_gi"][name]["correction"],
                        gd_reference=protocol["baseline_inputs"][kind + "_gd"][name]["correction"],
                        quant_reference=qrecord, raw_g_reference=state["files"][name], g_diagnostics=diagnostics, **metrics)
                    log("solve-full-g", f"saved {method} {name} rank{rank}_sse="
                        f"{metrics['weighted_sse_before']:.6g}->{metrics['weighted_sse_after']:.6g}")
                    del a, b
                    stop.check()
                del root_g, error_t, diagonal
                torch.cuda.empty_cache()
            del roots
    atomic_json(Path(config["run_dir"]) / "solve_complete.json", {
        "status": "PASS", "manifest_sha256": manifest["sha256"], "modules": count})


def evaluation_inputs(config, manifest, method, check=True):
    protocol = manifest["payload"]["full_g_protocol"]
    if method in (None, "wq", "diag_gi", "full_gi", "diag_gd", "full_gd"):
        result = protocol["baseline_inputs"][method or "teacher"]
    elif method in ("diag_gf", "full_gf"):
        result = {}
        states = {}
        from .collect import checkpoints
        for index in range(len(protocol["shards"])):
            store = checkpoints(config, manifest, index)
            state = store.load(tensors=False)
            if state is None or state["windows_completed"] != 256:
                raise RuntimeError("Full-G collection is incomplete")
            states.update(state["files"])
        for layer in layers(manifest):
            name = layer["name"]
            record = owned_correction(config, manifest, method, name, check)
            if record is None or record.get("raw_g_reference") != states[name]:
                raise RuntimeError(f"Missing/mismatched full-G correction: {method} {name}")
            result[name] = {"quant": protocol["baseline_inputs"]["wq"][name]["quant"], "correction": record["file"]}
    else:
        raise ValueError(f"Unknown method: {method}")
    if check:
        for files in result.values():
            for record in files.values():
                verify(record)
    return result


@contextmanager
def evaluation_binding(stop=None):
    previous = legacy.configurations, legacy._evaluation_inputs, legacy.summarize, legacy.atomic_json
    def save_and_check(path, value):
        atomic_json(path, value)
        if stop is not None and Path(path).parent.name == "configurations":
            stop.check()  # Evaluation batch is durable before a cooperative stop.
    legacy.configurations, legacy._evaluation_inputs, legacy.summarize = configurations, evaluation_inputs, summarize
    legacy.atomic_json = save_and_check
    try:
        yield
    finally:
        legacy.configurations, legacy._evaluation_inputs, legacy.summarize, legacy.atomic_json = previous


def import_baseline_evaluations(config, manifest):
    from qera_diag_g_isolation.full_svd_v1.run import write_same_or_new
    protocol = manifest["payload"]["full_g_protocol"]
    for name, method, _ in mx3.configurations(config):
        source = protocol["baseline_evaluations"][name]
        rows = read_json(verify(source))["records"]
        artifacts = evaluation_inputs(config, manifest, method, check=False)
        digest = fingerprint({"manifest": manifest["sha256"], "name": name, "artifacts": artifacts})
        write_same_or_new(Path(config["run_dir"]) / "evaluation/configurations" / f"{name}.json", {
            "protocol_sha256": digest, "manifest_sha256": manifest["sha256"], "configuration": name,
            "records": rows, "complete": True, "reused_from": source,
            "note": "Imported verified MXINT3 evaluation; not a new evaluation or independent replicate"})


def evaluate(config, stop, only=None):
    stop.check()
    manifest = load_manifest(config)
    for index in range(len(manifest["payload"]["full_g_protocol"]["shards"])):
        completed_audit(config, manifest, index)
    with evaluation_binding(stop):
        return legacy.evaluate(config, only)


def summarize(config):
    with evaluation_binding():
        result = _legacy_summary(config)
    expected = len(configurations(config))
    result.update(expected_configurations=expected,
                  status="PASS" if result["configurations"] == expected else "INCOMPLETE",
                  reused_configurations=18, new_full_g_configurations=8)
    atomic_json(Path(config["run_dir"]) / "evaluation/status.json", result)
    return result
