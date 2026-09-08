#!/usr/bin/env python3
"""Round one: frozen A256, identity versus diagonal G256. No old outputs written."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from qera_diag_g_isolation.storage import (
    atomic_json, config_from_file, gpu_check, load_manifest, log, prepare, read_json, run_lock,
)
from qera_diag_g_isolation.pipeline import collect, evaluate, quantize, solve, summarize


def environment_check():
    for name, expected in (("torch", "2.3.0"), ("transformers", "4.44.2")):
        found = importlib.metadata.version(name).split("+")[0]
        if found != expected:
            raise RuntimeError(f"Reuse qera-original-a environment: expected {name}={expected}, found {found}. Do not reinstall the root project.")


def doctor(config):
    environment_check()
    devices = gpu_check()
    source = Path(config["source_run_dir"])
    if not (source / "config_resolved.json").is_file():
        raise FileNotFoundError("Original A run's config_resolved.json is missing")
    old_config = read_json(source / "config_resolved.json")
    for path in (source / "data/calibration.safetensors", source / "data/wikitext2.safetensors",
                 Path(old_config["model_path"]), Path(old_config["qera_source_dir"])):
        if not path.exists():
            raise FileNotFoundError(path)
    limits = {}
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes",
                 "/sys/fs/cgroup/memory.current"):
        try:
            limits[path] = Path(path).read_text().strip()
        except OSError:
            pass
    numeric_limits = [int(v) for k, v in limits.items() if not k.endswith("memory.current") and v.isdigit() and int(v) < 2**60]
    if numeric_limits and min(numeric_limits) < 128 * 2**30:
        raise RuntimeError("FP32 backward with CPU saved activations requires the 224G tier; detected cgroup limit below 128 GiB")
    free = shutil.disk_usage(config["run_dir"]).free
    if free < 25 * 2**30:
        raise RuntimeError("Need at least 25 GiB free in the new output filesystem")
    manifest_path = Path(config["run_dir"]) / "manifest.json"
    if manifest_path.exists():
        load_manifest(config)
    result = {"status": "PASS", "source_audit": "prepared" if manifest_path.exists() else "run prepare next",
              "devices": devices, "cgroup_memory": limits,
              "host_ram_is_not_container_limit": True, "free_disk_GiB": free / 2**30,
              "cpu_affinity_count": len(os.sched_getaffinity(0)),
              "calibration_windows": 256, "sequence_length": 2048, "g_targets": 224,
              "collect_dtype": "float32", "collect_batch_size": 1,
              "eval_batch_size": config["eval_batch_size"], "expected_configurations": 18,
              "g_raw_retained": True}
    atomic_json(Path(config["run_dir"]) / "doctor.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("command", choices=["doctor", "prepare", "quantize", "collect", "solve", "evaluate", "summary", "run"])
    parser.add_argument("--only", help="Evaluate one named configuration, e.g. DIAG_GD_R32")
    args = parser.parse_args()
    config = config_from_file(args.config)
    if args.only and args.command != "evaluate":
        parser.error("--only is valid only for evaluate")
    torch.set_num_threads(config["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(1234)
    if args.command != "summary":
        environment_check()
        gpu_check()
    with run_lock(config):
        if args.command == "doctor":
            result = doctor(config)
        elif args.command == "prepare":
            doctor(config)
            saved = prepare(config)
            result = {"status": "PASS", "manifest_sha256": saved["sha256"]}
        elif args.command == "run":
            doctor(config)
            if not (Path(config["run_dir"]) / "manifest.json").exists():
                prepare(config)
            else:
                load_manifest(config)
            for stage in (quantize, collect, solve):
                log("runner", f"starting {stage.__name__}")
                stage(config)
            result = evaluate(config)
        elif args.command == "evaluate":
            result = evaluate(config, args.only)
        else:
            fn = {"quantize": quantize, "collect": collect, "solve": solve, "summary": summarize}[args.command]
            result = fn(config) or {"status": "PASS", "stage": args.command}
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
