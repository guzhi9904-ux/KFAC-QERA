#!/usr/bin/env python3
"""Restartable Full-G on two RTX4090s, isolated from completed MXINT3 baselines."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import re
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from qera_diag_g_isolation import pipeline as legacy
from qera_diag_g_isolation.mxint3_v1 import pipeline as mx3
from qera_diag_g_isolation.mxint3_v1 import run as mx3_run
from qera_diag_g_isolation.full_svd_v1.run import write_same_or_new
from qera_diag_g_isolation.run import environment_check
from qera_diag_g_isolation.storage import (
    atomic_json, checked_tensors, config_from_file, file_record, fingerprint, gpu_check,
    layers, load_manifest, log, read_json, run_lock, verify,
)
from qera_diag_g_isolation.full_g_v1 import pipeline
from qera_diag_g_isolation.full_g_v1.collect import collect, memory_status
from qera_diag_g_isolation.full_g_v1.checkpoint import Paused, Stop

VARIANT = "mxint3_full_g_v1"


def derived_config(parent_config):
    config = deepcopy(parent_config)
    root = Path(parent_config["run_dir"]).resolve()
    destination = root / VARIANT
    if destination.is_symlink() or destination.resolve().parent != root:
        raise RuntimeError("Full-G output must be the fixed, non-symlink child")
    config.update(run_dir=str(destination), experiment_variant=VARIANT, quantization=deepcopy(mx3.QUANTIZATION))
    return config


def shard_layers(items, blocks_per_shard=8):
    if blocks_per_shard not in (8, 16, 32):
        raise ValueError("Invalid shard width")
    groups = [[] for _ in range(32 // blocks_per_shard)]
    seen = set()
    for layer in items:
        match = re.fullmatch(r"model\.layers\.(\d+)\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))", layer["name"])
        if match is None or not 0 <= int(match[1]) < 32 or layer["name"] in seen:
            raise RuntimeError("Unexpected/duplicate Full-G target")
        seen.add(layer["name"])
        groups[int(match[1]) // blocks_per_shard].append({"name": layer["name"], "shape": layer["shape"]})
    if len(seen) != 224 or any(len(g) != blocks_per_shard * 7 for g in groups):
        raise RuntimeError("Require all 224 targets in non-overlapping whole-block shards")
    return groups


def audit_source(parent_config):
    config = mx3_run.derived_config(parent_config)
    manifest = load_manifest(config)
    mx3.require_variant(config, manifest)
    transition = manifest["payload"]["quantization_transition"]
    parent = load_manifest(parent_config)
    if transition["parent_manifest_sha256"] != parent["sha256"] or transition["svd_full_matrices"] is not True:
        raise RuntimeError("MXINT3 source must use the same original G run and full SVD")
    for key in ("parent_manifest", "parent_collect_state", "parent_snapshot_metadata"):
        verify(transition[key])
    inputs = {}
    verified = set()
    for method in (None, "wq", "diag_gi", "full_gi", "diag_gd", "full_gd"):
        inputs[method or "teacher"] = mx3.evaluation_inputs(config, manifest, method, check=False)
        for files in inputs[method or "teacher"].values():
            for record in files.values():
                key = fingerprint(record)
                if key not in verified:
                    verify(record)
                    verified.add(key)
    evaluations, ppls = {}, {}
    for name, method, _ in mx3.configurations(config):
        digest = fingerprint({"manifest": manifest["sha256"], "name": name,
                              "artifacts": inputs[method or "teacher"]})
        path = Path(config["run_dir"]) / "evaluation/configurations" / f"{name}.json"
        rows = legacy.evaluation_records(path, digest, 138)
        if len(rows) != 138:
            raise RuntimeError(f"Require completed MXINT3 baseline: {name}")
        ppls[name] = math.exp(sum(r["nll_sum"] for r in rows) / (138 * 2047))
        reference = manifest["payload"]["control_reference_ppl"].get(name)
        if reference is not None and abs(ppls[name] - reference) > config["control_ppl_tolerance"]:
            raise RuntimeError(f"MXINT3 baseline control gate failed: {name}")
        evaluations[name] = file_record(path)
    values, count, _ = legacy._load_g_checkpoint(config, manifest)
    del values
    if count != 256:
        raise RuntimeError("Require complete frozen diagonal G256")
    state_path = Path(config["run_dir"]) / "statistics/collect_state.json"
    runtime_path = Path(parent_config["run_dir"]) / "statistics/runtime.json"
    runtime = read_json(runtime_path)
    if runtime.get("dtype") != "float32" or runtime.get("batch_size") != 1 or runtime.get("save_on_cpu") is not True:
        raise RuntimeError("Original teacher runtime differs from expected FP32 batch-one CPU-save protocol")
    return config, manifest, {
        "baseline_inputs": inputs, "baseline_evaluations": evaluations, "baseline_ppl": ppls,
        "baseline_manifest": file_record(Path(config["run_dir"]) / "manifest.json"),
        "baseline_manifest_sha256": manifest["sha256"],
        "diagonal_g_state": read_json(state_path), "diagonal_g_state_file": file_record(state_path),
        "teacher_runtime": file_record(runtime_path), "teacher_device_map": runtime["hf_device_map"],
    }


def prepare(parent_config, config):
    if config != derived_config(parent_config):
        raise RuntimeError("Full-G must preserve the frozen experiment configuration")
    root = Path(config["run_dir"])
    root.mkdir(parents=True, exist_ok=True)
    if not (root / "manifest.json").exists() and not (root / "initialization.json").exists():
        unexpected = [p.name for p in root.iterdir() if p.name not in (".run.lock",)]
        if unexpected:
            raise RuntimeError(f"Unrecognized Full-G output, refusing adoption: {unexpected}")
    _, source, protocol = audit_source(parent_config)
    # No dtype/chunk/BS/data changes hidden behind a new representation.
    expected = {"teacher_dtype": "float32", "collect_batch_size": 1, "ce_gradient_chunk_tokens": 128,
                "num_calibration_windows": 256, "sequence_length": 2048, "ranks": [8, 16, 32, 64],
                "g_relative_floor": 1e-6, "identity_product_tolerance": 1e-3,
                "eval_batch_size": 8, "eval_ce_chunk_tokens": 256, "save_activations_on_cpu": True}
    if any(config.get(k) != v for k, v in expected.items()):
        raise RuntimeError("Full-G protocol must match the completed MXINT3 experiment")
    payload = deepcopy(source["payload"])
    payload["config"] = config
    payload["control_reference_ppl"] = protocol["baseline_ppl"]
    for record in payload["data"].values():
        verify(record)
    for group in payload["groups"]:
        verify(group["roots"])
    protocol.update({
        "variant": VARIANT, "shards": shard_layers(layers(source)), "blocks_per_shard": 8,
        "gram_dtype": "float64", "accumulator_dtype": "float64", "gram_row_tile": 512,
        "parent_diagonal_tolerance": 1e-6, "gram_diagonal_tolerance": 1e-10,
        "teacher_nll_relative_tolerance": 1e-6,
        "definition": "G = sum(delta_t delta_t^T)/524032; delta=d(sequence next-token CE SUM)/d(projection output at valid t)",
        "not_per_loss_fisher": True, "center_gradients": False, "cross_module_covariance": False,
        "normalization": "symmetrize in FP64; divide by trace/d; floor eigenvalues at 1e-6; square root to FP32",
        "solve_dtype": "float32", "svd_full_matrices": True, "eval_dtype": "bfloat16",
        "resume": "whole-window atomic shard generations, rolling; per-method solve; per-batch evaluation",
        "new_configurations": 8, "reused_configurations": 18,
    })
    payload["full_g_protocol"] = protocol
    payload["g_definition"] = protocol["definition"]
    for path in Path(__file__).parent.glob("*.py"):
        if not path.name.startswith("test_"):
            payload["code"][str(path.resolve())] = file_record(path)
    # These dependencies were not necessarily pinned by the original parent.
    for path in (Path(mx3.__file__), Path(mx3_run.__file__)):
        payload["code"][str(path.resolve())] = file_record(path)
    manifest = {"sha256": fingerprint(payload), "payload": payload}
    write_same_or_new(root / "initialization.json", {"manifest_sha256": manifest["sha256"]})
    write_same_or_new(root / "config.json", config)
    write_same_or_new(root / "manifest.json", manifest)
    load_manifest(config)
    pipeline.import_baseline_evaluations(config, manifest)
    write_same_or_new(root / "audit.json", {
        "status": "PASS", "manifest_sha256": manifest["sha256"], "targets": 224, "a_groups": 128,
        "quantization": mx3.QUANTIZATION, "ranks": config["ranks"], "calibration_windows_per_shard": 256,
        "shards": 4, "total_window_passes": 1024, "raw_full_g_GiB": 110.5,
        "active_full_g_GiB": 27.625, "retained_baselines": 18, "new_full_g_evaluations": 8,
        "identity_tolerance_unchanged": .001,
        "note": "Protocol/source audit only. Collection/solver numerical gates remain to be run on server."})
    log("audit-full-g", f"PASS {root}; A/GI/GD/W3/data frozen; new Full-G only")
    return manifest


def verify_resume(config):
    manifest = load_manifest(config)
    protocol = manifest["payload"]["full_g_protocol"]
    if protocol["variant"] != VARIANT or config["experiment_variant"] != VARIANT:
        raise RuntimeError("Wrong Full-G variant")
    for key in ("baseline_manifest", "diagonal_g_state_file", "teacher_runtime"):
        verify(protocol[key])
    for record in protocol["baseline_evaluations"].values():
        verify(record)
    return manifest


def doctor(config):
    devices = gpu_check()
    if any("4090" not in d["name"] for d in devices):
        raise RuntimeError("This variant is pinned to two RTX4090 GPUs; do not silently switch hardware")
    memory = memory_status()
    if memory.get("cgroup_limit_GiB", float("inf")) < 180:
        raise RuntimeError("Use the previous 224GB RAM tier; cgroup memory below 180 GiB")
    return {"devices": devices, "memory": memory,
            "free_disk_GiB": shutil.disk_usage(config["run_dir"]).free / 2**30,
            "note": "Pilot must measure actual RAM/GPU peaks; capacity is not guaranteed by this check."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Original diagonal-G YAML, NOT a derived config")
    parser.add_argument("command", choices=("prepare", "pilot", "collect", "solve", "evaluate", "summary", "run"),
                        nargs="?", default="run")
    parser.add_argument("--max-hours", type=float, help="Cooperative per-invocation time budget; leave room before server cutoff")
    parser.add_argument("--max-new-windows", type=int, help="Stop after this many newly collected window passes")
    parser.add_argument("--only", help="Only with evaluate, e.g. FULL_GF_R64")
    args = parser.parse_args(argv)
    if args.only and args.command != "evaluate":
        parser.error("--only is only valid with evaluate")
    if args.max_hours is not None and (not math.isfinite(args.max_hours) or args.max_hours <= 0):
        parser.error("--max-hours must be positive and finite")
    if args.max_new_windows is not None and (args.max_new_windows <= 0 or args.command not in ("collect", "run", "pilot")):
        parser.error("--max-new-windows must be positive and used with collect/run/pilot")
    parent_config = config_from_file(args.config)
    config = derived_config(parent_config)
    stop = Stop(None if args.max_hours is None else args.max_hours * 3600)
    torch.set_num_threads(config["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(1234)
    try:
        with stop.installed(), run_lock(parent_config), run_lock(mx3_run.derived_config(parent_config)), run_lock(config):
            if args.command != "summary":
                environment_check()
                report = doctor(config)
                log("doctor-full-g", json.dumps(report))
                if not (Path(config["run_dir"]) / "manifest.json").exists() and report["free_disk_GiB"] < 160:
                    raise RuntimeError("Need at least 160 GiB free for raw G, rolling checkpoint, corrections and reserve")
            if args.command in ("prepare", "pilot", "run"):
                if (not (Path(config["run_dir"]) / "manifest.json").exists()
                        or not (Path(config["run_dir"]) / "audit.json").exists() or args.command == "prepare"):
                    prepare(parent_config, config)
                else:
                    manifest = verify_resume(config)
                    # Also finishes an initialization interrupted while importing baseline rows.
                    pipeline.import_baseline_evaluations(config, manifest)
            else:
                verify_resume(config)
            if args.command in ("pilot", "collect", "run"):
                limit = args.max_new_windows
                if args.command == "pilot" and limit is None:
                    limit = 4
                collect(config, stop, limit)
            if args.command in ("solve", "run"):
                pipeline.solve(config, stop)
            if args.command in ("evaluate", "run"):
                result = pipeline.evaluate(config, stop, args.only)
            else:
                result = pipeline.summarize(config)
            print(json.dumps(result, indent=2), flush=True)
        return 0
    except Paused as error:
        log("paused-full-g", str(error))
        return 75


if __name__ == "__main__":
    raise SystemExit(main())
