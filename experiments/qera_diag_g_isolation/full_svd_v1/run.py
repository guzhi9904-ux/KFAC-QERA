#!/usr/bin/env python3
"""Resume solved/evaluated outputs with full SVD, reusing immutable parent G/Wq.

The original run's manifest, tensors, metadata and numerical gates stay intact.
All new state is bound to a derived manifest in its full_svd_v1 subdirectory.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from qera_diag_g_isolation import pipeline
from qera_diag_g_isolation.full_svd_v1.solver import bind_solver
from qera_diag_g_isolation.run import environment_check
from qera_diag_g_isolation.storage import (
    atomic_json, config_from_file, file_record, fingerprint, gpu_check, heartbeat,
    layers, load_manifest, log, read_json, run_lock, verify,
)

VARIANT = "full_svd_v1"


def audit_supports_full_svd(report, manifest_sha256, tolerance, ranks):
    """Require the controlled evidence actually observed on the failing layer."""
    try:
        if (report["status"] != "AUDIT_COMPLETE" or report["manifest_sha256"] != manifest_sha256
                or report["layer"] != "model.layers.0.self_attn.k_proj" or report["method"] != "diag"):
            return False
        inputs = report["inputs"]
        for key in ("fresh_official_q_vs_frozen_q", "new_error_vs_official_error",
                    "new_svd_input_vs_official_svd_input"):
            if inputs[key]["exact_equal"] is not True:
                return False
        for name in ("official_replay", "old_input_full_svd", "new_input_full_svd"):
            values = report["variants"][name]["vs_saved_production_gate"]
            if any(not math.isfinite(values[str(r)]) or not 0 <= values[str(r)] <= tolerance for r in ranks):
                return False
        for name in ("current_solver", "old_input_reduced_svd", "new_input_reduced_svd"):
            values = report["variants"][name]["vs_saved_production_gate"]
            if any(not math.isfinite(values[str(r)]) or values[str(r)] < 0 for r in ranks):
                return False
            if max(values[str(r)] for r in ranks) <= tolerance:
                return False
        return True
    except (KeyError, TypeError, ValueError):
        return False


def choose_audit(parent_config, parent, explicit=None):
    candidates = [Path(explicit).resolve()] if explicit else sorted(
        Path(parent_config["run_dir"]).glob("diagnostics/identity_g_*/report.json"), reverse=True,
    )
    for path in candidates:
        report = read_json(path)
        if audit_supports_full_svd(report, parent["sha256"], parent_config["identity_product_tolerance"],
                                   parent_config["ranks"]):
            return file_record(path)
    raise RuntimeError("No matching completed identity-G audit supports full SVD; preserve the run and audit first")


def write_same_or_new(path, value):
    """Resume initialization without overwriting even a mismatching JSON file."""
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise RuntimeError(f"Existing derived state differs; do not overwrite: {path}")
    else:
        atomic_json(path, value)


def validate_parent(config, manifest):
    if "solver_transition" in manifest["payload"]:
        raise RuntimeError("Pass the original G256 configuration, not an already derived run")
    with heartbeat("full-svd", "verifying retained G256 checkpoint"):
        sums, count, _ = pipeline._load_g_checkpoint(config, manifest)
    del sums
    if count != 256:
        raise RuntimeError(f"Need committed G256; found {count}. This runner never collects G")
    state_path = Path(config["run_dir"]) / "statistics/collect_state.json"
    state = read_json(state_path)
    if state["status"] != "PASS":
        raise RuntimeError("Parent G checkpoint is not complete")
    snapshot_metadata = Path(state["file"]["path"]).with_suffix(".json")
    if read_json(snapshot_metadata) != state:
        raise RuntimeError("Committed G pointer and retained snapshot metadata differ")
    return state, file_record(state_path), file_record(snapshot_metadata)


def derived_config(parent_config):
    config = deepcopy(parent_config)
    root = Path(parent_config["run_dir"]).resolve()
    destination = root / VARIANT
    if destination.is_symlink() or destination.resolve().parent != root:
        raise RuntimeError("Derived output must be the fixed full_svd_v1 child of the original G run")
    config["run_dir"] = str(destination)
    return config


def prepare_derived(parent_config, parent, config, audit_path=None):
    """Bind the completed raw G and frozen Wq by verified references, not copies.

    Existing pipeline readers consume each metadata file's explicit tensor record,
    so these references do not need fake tensor files, symlinks, or hard links.
    """
    root = Path(config["run_dir"])
    parent_path = Path(parent_config["run_dir"]) / "manifest.json"
    source_state, state_record, snapshot_record = validate_parent(parent_config, parent)
    manifest_path = root / "manifest.json"
    extension_records = {str(path.resolve()): file_record(path) for path in (
        Path(__file__), Path(__file__).with_name("solver.py"),
    )}
    existing = read_json(manifest_path) if manifest_path.exists() else None
    initialization_path = root / "initialization.json"
    initialized = read_json(initialization_path) if initialization_path.exists() else None
    if existing or initialized:
        # Keep the originally chosen audit when new audit reports arrive later.
        saved_transition = (existing["payload"]["solver_transition"] if existing else initialized["transition"])
        audit_record = saved_transition["audit_report"]
        verify(audit_record)
        if audit_path and str(Path(audit_path).resolve()) != audit_record["path"]:
            raise RuntimeError("This derived run is already bound to another audit report")
    else:
        unexpected = [path.name for path in root.iterdir() if path.name != ".run.lock"]
        if unexpected:
            raise RuntimeError(f"Unrecognized files in derived output; refusing adoption: {unexpected}")
        audit_record = choose_audit(parent_config, parent, audit_path)
    if not audit_supports_full_svd(read_json(audit_record["path"]), parent["sha256"],
                                   parent_config["identity_product_tolerance"], parent_config["ranks"]):
        raise RuntimeError("Stored audit does not justify this solver transition")

    transition = {
        "variant": VARIANT, "parent_manifest": file_record(parent_path),
        "parent_manifest_sha256": parent["sha256"], "parent_collect_state": state_record,
        "parent_snapshot_metadata": snapshot_record, "audit_report": audit_record,
        "svd_full_matrices": True, "dtype": "float32", "driver": "default",
        "runtime_binding": "pipeline.solve_weighted = full_svd_v1.solver.solve_weighted for both GI/GD",
        "unchanged_identity_tolerance": parent_config["identity_product_tolerance"],
        "unchanged_control_ppl_tolerance": parent_config["control_ppl_tolerance"],
        "retention": "Parent raw G snapshots and Wq stay in place; derived metadata references their hashes",
    }
    payload = deepcopy(parent["payload"])
    payload["config"] = config
    payload["code"].update(extension_records)
    payload["solver_transition"] = transition
    derived = {"sha256": fingerprint(payload), "payload": payload}
    marker = {"manifest_sha256": derived["sha256"], "transition": transition}
    write_same_or_new(initialization_path, marker)
    write_same_or_new(root / "config.json", config)  # JSON is accepted by the existing YAML loader.
    write_same_or_new(manifest_path, derived)
    load_manifest(config)  # Check all old and new code hashes before referencing any artifact.

    checkpoint = deepcopy(source_state)
    checkpoint["manifest_sha256"] = derived["sha256"]
    write_same_or_new(root / "statistics/collect_state.json", checkpoint)
    write_same_or_new(root / "statistics/checkpoints/window_0256.json", checkpoint)
    quant_records = {}
    all_layers = list(layers(parent))
    for index, layer in enumerate(all_layers, 1):
        with heartbeat("full-svd", f"verify frozen Wq {index}/{len(all_layers)}"):
            parent_artifact = pipeline.artifact_path(parent_config, "quantized", layer["name"])
            record = pipeline.completed_artifact(parent_artifact, parent)
        if record is None or record.get("fp32_bf16_quantization_equal") is not True:
            raise RuntimeError(f"Missing validated parent Wq: {layer['name']}")
        derived_record = deepcopy(record)
        derived_record["manifest_sha256"] = derived["sha256"]
        derived_record["reused_from_parent_manifest"] = parent["sha256"]
        target = pipeline.artifact_path(config, "quantized", layer["name"]).with_suffix(".json")
        write_same_or_new(target, derived_record)
        quant_records[layer["name"]] = record["file"]
        log("full-svd", f"verified/referenced Wq {index}/{len(all_layers)} {layer['name']}")
    write_same_or_new(root / "quantized/complete.json", {
        "manifest_sha256": derived["sha256"], "files": quant_records, "status": "PASS",
    })
    write_same_or_new(root / "reuse_complete.json", {
        "manifest_sha256": derived["sha256"], "parent_manifest_sha256": parent["sha256"],
        "status": "PASS", "g_windows_reused": 256, "quantized_layers_reused": len(quant_records),
    })
    log("full-svd", f"ready output={root}; reused G256 and Wq; parent files unchanged")
    return derived


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Original G256 config, not the derived config")
    parser.add_argument("--audit-report", type=Path, help="Defaults to the latest matching completed audit")
    parser.add_argument("command", choices=("prepare", "solve", "evaluate", "summary", "run"), nargs="?", default="run")
    args = parser.parse_args(argv)
    parent_config = config_from_file(args.config)
    config = derived_config(parent_config)
    environment_check()
    if args.command != "summary":
        gpu_check()
    torch.set_num_threads(config["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(1234)
    # Keep both locks for the entire continuation, excluding the legacy writer too.
    with run_lock(parent_config):
        parent = load_manifest(parent_config)
        with run_lock(config):
            prepare_derived(parent_config, parent, config, args.audit_report)
            if args.command in ("run", "solve"):
                log("full-svd", "starting solve; full SVD for identity checks AND diagonal G; tolerances unchanged")
                with bind_solver(pipeline):
                    pipeline.solve(config)
            if args.command in ("run", "evaluate"):
                log("full-svd", "starting 18-configuration evaluation with original control gates")
                result = pipeline.evaluate(config)
            elif args.command == "summary":
                result = pipeline.summarize(config)
            else:
                result = {"status": "PASS", "stage": args.command, "run_dir": config["run_dir"]}
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
