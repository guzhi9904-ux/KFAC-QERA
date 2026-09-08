#!/usr/bin/env python3
"""Audit and run MXINT3 using frozen teacher A256/G256. No collection stage."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from qera_diag_g_isolation import pipeline as legacy
from qera_diag_g_isolation.full_svd_v1 import run as full_run
from qera_diag_g_isolation.full_svd_v1 import solver as full_solver
from qera_diag_g_isolation.mxint3_v1 import pipeline
from qera_diag_g_isolation.run import environment_check
from qera_diag_g_isolation.storage import (
    config_from_file, file_record, fingerprint, gpu_check, layers, load_manifest,
    log, read_json, run_lock, verify,
)

VARIANT = "mxint3_v1"


def derived_config(parent_config):
    config = deepcopy(parent_config)
    root = Path(parent_config["run_dir"]).resolve()
    destination = root / VARIANT
    if destination.is_symlink() or destination.resolve().parent != root:
        raise RuntimeError("MXINT3 output must be the fixed mxint3_v1 child, not a symlink")
    config.update(run_dir=str(destination), experiment_variant=VARIANT,
                  quantization=deepcopy(pipeline.QUANTIZATION))
    return config


def audit_mxint4(parent_config, parent):
    """Read-only audit: require the completed, full-SVD MXINT4 comparison run."""
    config = full_run.derived_config(parent_config)
    manifest = load_manifest(config)
    transition = manifest["payload"]["solver_transition"]
    if (transition["variant"] != "full_svd_v1" or transition["svd_full_matrices"] is not True
            or transition["parent_manifest_sha256"] != parent["sha256"]):
        raise RuntimeError("MXINT4 comparison must be full_svd_v1 from the same original G run")
    verify(transition["parent_manifest"])
    verify(transition["audit_report"])
    if not full_run.audit_supports_full_svd(read_json(transition["audit_report"]["path"]), parent["sha256"],
                                           config["identity_product_tolerance"], config["ranks"]):
        raise RuntimeError("MXINT4 solver transition audit is invalid")
    records, ppls = {}, {}
    for name, method, _ in legacy.configurations(config):
        artifacts = legacy._evaluation_inputs(config, manifest, method, check=False)
        protocol = fingerprint({"manifest": manifest["sha256"], "name": name, "artifacts": artifacts})
        path = Path(config["run_dir"]) / "evaluation/configurations" / f"{name}.json"
        rows = legacy.evaluation_records(path, protocol, 138)
        if len(rows) != 138:
            raise RuntimeError(f"Finish the 18 MXINT4 evaluations before MXINT3: {name}")
        ppl = math.exp(sum(r["nll_sum"] for r in rows) / sum(r["tokens"] for r in rows))
        reference = manifest["payload"]["control_reference_ppl"].get(name)
        if reference is not None and abs(ppl - reference) > config["control_ppl_tolerance"]:
            raise RuntimeError(f"MXINT4 control gate failed: {name}")
        records[name], ppls[name] = file_record(path), ppl
    return {"manifest": file_record(Path(config["run_dir"]) / "manifest.json"),
            "manifest_sha256": manifest["sha256"], "evaluations": records, "ppl": ppls}


def prepare(parent_config, parent, config):
    root = Path(config["run_dir"])
    if config != derived_config(parent_config):
        raise RuntimeError("Only weight bit width and output location may change")
    if parent["payload"]["source_config"]["quantization"] != {
        "name": "mxint", "width": 4, "block_size": 32, "block_axis": -1,
    }:
        raise RuntimeError("Expected the frozen MXINT4 source protocol")
    source_state, state_record, snapshot_record = full_run.validate_parent(parent_config, parent)
    if "quantization_transition" in parent["payload"]:
        raise RuntimeError("Pass the original G256 configuration")
    original_manifest = root / "manifest.json"
    initialization = root / "initialization.json"
    if not original_manifest.exists() and not initialization.exists():
        unexpected = [p.name for p in root.iterdir() if p.name != ".run.lock"]
        if unexpected:
            raise RuntimeError(f"Unrecognized MXINT3 output; refusing adoption: {unexpected}")
    comparison = audit_mxint4(parent_config, parent)
    # Validate all reused frozen data and A roots at prepare, and again on consumption.
    for record in parent["payload"]["data"].values():
        verify(record)
    for group in parent["payload"]["groups"]:
        verify(group["raw_a"])
        verify(group["roots"])
    payload = deepcopy(parent["payload"])
    payload["config"] = config
    payload["source_config"]["quantization"] = deepcopy(pipeline.QUANTIZATION)
    # Remove old W4 factors: accidental legacy GI routing must fail, not silently reuse them.
    for layer in layers({"payload": payload}):
        layer["gi"] = {}
    # W3 and W3+GI are NEW baselines. W4 PPL is not a regression target for them.
    payload["control_reference_ppl"] = {"BF16": comparison["ppl"]["BF16"]}
    payload["quantization_transition"] = {
        "variant": VARIANT, "parent_manifest": file_record(Path(parent_config["run_dir"]) / "manifest.json"),
        "parent_manifest_sha256": parent["sha256"], "parent_collect_state": state_record,
        "parent_snapshot_metadata": snapshot_record, "mxint4_comparison": comparison,
        "from": {"name": "mxint", "width": 4, "block_size": 32, "block_axis": -1},
        "to": deepcopy(pipeline.QUANTIZATION), "reuse": ["teacher A256 roots", "teacher raw diagonal G256", "frozen data"],
        "regenerate": ["Wq", "official GI", "GD", "all 18 evaluations"],
        "identity_reference": "fresh pinned official QERA at MXINT3 for every module/A/rank prefix",
        "unchanged_identity_tolerance": config["identity_product_tolerance"],
        "unchanged_bf16_ppl_tolerance": config["control_ppl_tolerance"],
        "svd_full_matrices": True, "solve_dtype": "float32", "eval_dtype": "bfloat16",
        "scope": "224 existing projection targets; no Full-G, no gradient recollection",
    }
    for path in (Path(__file__), Path(pipeline.__file__), Path(__file__).with_name("__init__.py"),
                 Path(full_solver.__file__), Path(full_run.__file__)):
        payload["code"][str(path.resolve())] = file_record(path)
    manifest = {"sha256": fingerprint(payload), "payload": payload}
    write = full_run.write_same_or_new
    write(initialization, {"manifest_sha256": manifest["sha256"]})
    write(root / "config.json", config)
    write(original_manifest, manifest)
    load_manifest(config)
    checkpoint = deepcopy(source_state)
    checkpoint["manifest_sha256"] = manifest["sha256"]
    write(root / "statistics/collect_state.json", checkpoint)
    write(root / "statistics/checkpoints/window_0256.json", checkpoint)
    write(root / "audit.json", {
        "status": "PASS", "manifest_sha256": manifest["sha256"],
        "quantization": pipeline.QUANTIZATION, "ranks": config["ranks"],
        "g_windows_reused": 256, "a_groups": len(payload["groups"]), "targets": len(layers(manifest)),
        "prediction_tokens_per_evaluation": 282486, "expected_configurations": 18,
        "baseline": "W3_MXINT", "g_representation": "diagonal",
        "old_weights_and_corrections_reused": False, "historical_ppl_gate": ["BF16"],
    })
    log("audit-mxint3", f"PASS output={root}; reuse A/G, regenerate W3/GI/GD; no collection")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Original G256 YAML (not derived config)")
    parser.add_argument("command", choices=("prepare", "quantize", "solve", "evaluate", "summary", "run"),
                        nargs="?", default="run")
    parser.add_argument("--only", help="Only for evaluate, e.g. W3_MXINT or FULL_GD_R64")
    args = parser.parse_args(argv)
    if args.only and args.command != "evaluate":
        parser.error("--only is valid only with evaluate")
    parent_config = config_from_file(args.config)
    config = derived_config(parent_config)
    if args.command != "summary":
        environment_check()
        gpu_check()
    torch.set_num_threads(config["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(1234)
    # Lock original and comparison writers as well as the new run. No legacy stage is invoked.
    with run_lock(parent_config), run_lock(full_run.derived_config(parent_config)), run_lock(config):
        parent = load_manifest(parent_config)
        if args.command in ("prepare", "run"):
            prepare(parent_config, parent, config)
        else:
            manifest = load_manifest(config)
            pipeline.require_variant(config, manifest)
            for key in ("parent_manifest", "parent_collect_state", "parent_snapshot_metadata"):
                verify(manifest["payload"]["quantization_transition"][key])
        if args.command in ("quantize", "run"):
            pipeline.quantize(config)
        if args.command in ("solve", "run"):
            pipeline.solve(config)
        if args.command in ("evaluate", "run"):
            result = pipeline.evaluate(config, args.only)
        elif args.command == "summary":
            result = pipeline.summarize(config)
        else:
            result = {"status": "PASS", "stage": args.command, "run_dir": config["run_dir"]}
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
