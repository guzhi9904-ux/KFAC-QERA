#!/usr/bin/env python3
"""Qwen2.5-7B Base MXINT3 transfer: strict protocol, isolated deployment."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.dont_write_bytecode = True  # Do not create pyc files in shared official QERA.

import torch
import yaml

from qera_diag_g_isolation.storage import atomic_json, file_record, fingerprint, gpu_check, load_manifest, log, read_json, run_lock
from qera_diag_g_isolation.run import environment_check
from qera_diag_g_isolation.full_g_v1.checkpoint import Stop, Paused
from qera_diag_g_isolation.full_svd_v1.run import write_same_or_new
from qwen25_base_isolation_v1 import data, protocol, stages


def environment():
    return {name: importlib.metadata.version(name) for name in (
        "torch", "transformers", "accelerate", "datasets", "tokenizers", "safetensors", "numpy", "scipy")}


def settings_from_file(path):
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if set(value) != {"model_path", "model_id", "run_dir", "llama_g_run"} or value["model_id"] != "Qwen/Qwen2.5-7B":
        raise RuntimeError("Only the explicit Qwen2.5-7B Base transfer settings are accepted")
    for key in ("model_path", "run_dir", "llama_g_run"):
        value[key] = str(Path(value[key]).resolve())
    return value


def assert_not_shared(settings):
    # Execute BEFORE creating output, locking, importing a model or dataset write.
    fg = protocol.read_manifest(Path(settings["llama_g_run"]) / "mxint3_full_g_v1/manifest.json")
    roots = []
    for record in fg["payload"]["code"].values():
        path = Path(record["path"])
        if "experiments" in path.parts:
            roots.append(Path(*path.parts[:path.parts.index("experiments")]))
    a = fg["payload"]["source_config"]
    protocol.assert_separate(Path(__file__).resolve().parents[2], settings["run_dir"], roots,
        [settings["llama_g_run"], fg["payload"]["config"]["source_run_dir"], settings["model_path"], a["model_path"], a["qera_source_dir"]])
    protocol.verify_protected(fg["payload"]["code"].values())
    protocol.verify_private_helpers(Path(__file__).resolve().parents[2], fg["payload"]["code"].values())


def check_initial_output(root):
    """HF imports can create our private cache before initialization commits.

    Cache contents are not experiment results: raw data and token windows still
    pass the exact replay gates. Never adopt statistics or unknown output files.
    """
    root = Path(root)
    if not (root / "initialization.json").exists():
        unexpected = [p.name for p in root.iterdir() if p.name not in {".run.lock", "dataset_cache"}]
        if unexpected:
            raise RuntimeError(f"Unknown output contents; refusing adoption: {unexpected}")
    cache = root / "dataset_cache"
    if cache.is_symlink() or (cache.exists() and not cache.is_dir()):
        raise RuntimeError("Private dataset_cache must be a real directory, not a file/symlink")
    if cache.exists() and cache.resolve().parent != root.resolve():
        raise RuntimeError("Private dataset_cache escapes the Qwen output")


def prepare(settings, allow_download=False, offline_raw_dir=None, wikitext_cache_dir=None):
    root = Path(settings["run_dir"])
    # Check before importing official QERA/Transformers, and again afterwards:
    # their import-time cache creation must not fail the empty-output guard.
    check_initial_output(root)
    reference = protocol.audit_reference(settings)
    config = protocol.resolved_config(settings, reference)
    check_initial_output(root)
    write_same_or_new(root / "initialization.json", {"settings": settings, "config": config, "environment": environment()})
    model_config, model_files, groups = protocol.model_inventory(config["model_path"])
    data_details = data.prepare_data(config, reference, allow_download, offline_raw_dir, wikitext_cache_dir)
    source_config = dict(reference["a_config"])
    source_config.update(model_path=config["model_path"], run_dir=config["run_dir"], quantization=protocol.QUANT)
    payload = {"config": config, "source_config": source_config, "model_config": model_config,
               "model_files": model_files, "groups": groups, "code": {},
               "data": {k: v["file"] for k, v in data_details.items()}, "data_details": data_details,
               "protected_llama_files": reference["protected"], "environment": environment(),
               "reference": {"llama_a_config": reference["a_config"], "llama_dg_config": reference["dg_config"],
                             "llama_mx3_manifest": reference["llama_mx3_manifest"]["sha256"]},
               "transfer": {"model_id": settings["model_id"], "quantization": protocol.QUANT,
                 "retokenize": True, "chat_template": False, "eval_count_dynamic": True,
                 "a_arithmetic": "FP32 GEMM; CPU FP64 full sum, CPU FP32 diag sum; pinned scipy sqrtm",
                 "g_arithmetic": "FP32 frozen teacher and CE-sum seed; FP64 square/reduce/accumulate",
                 "a_batch_size": config["a_batch_size"], "g_batch_size": 1, "eval_batch_size": 8,
                 "placement_change": "Qwen balanced on two RTX4090; not bitwise-identical device placement to Llama",
                 "a_sharding": "8 input groups per shard, only memory scheduling changes",
                 "bias": "QKV bias unchanged; only seven projection weights quantized/compensated",
                 "not_paper_qwen_reproduction": True, "no_reuse_llama_tokens_or_statistics": True}}
    # All imported experiment code is local to the isolated checkout, except the
    # unchanged official QERA checkout explicitly verified by import_official_qera.
    experiments = Path(__file__).resolve().parents[1]
    for directory in ("qwen25_base_isolation_v1", "qera_original_a_isolation", "qera_diag_g_isolation"):
        for path in (experiments / directory).rglob("*.py"):
            if not path.name.startswith("test"):
                payload["code"][str(path.resolve())] = file_record(path)
    official_root = Path(source_config["qera_source_dir"])
    from qera_original_a_isolation.common import OFFICIAL_FILE_SHA256
    for relative in (*OFFICIAL_FILE_SHA256, "src/qera/datasets/__init__.py"):
        path = official_root / relative
        payload["code"][str(path.resolve())] = file_record(path)
    manifest = {"sha256": fingerprint(payload), "payload": payload}
    write_same_or_new(root / "config.json", config)
    write_same_or_new(root / "manifest.json", manifest)
    protocol.verify_protected(reference["protected"])
    write_same_or_new(root / "audit.json", {"status": "PASS", "manifest_sha256": manifest["sha256"],
        "targets": 196, "a_groups": 112, "a_calibration_windows": 256,
        "evaluation_windows": data_details["wikitext2"]["windows"],
        "llama_data_replay": "exact for calibration and WikiText2",
        "protected_llama_unchanged": True, "scope": "input/protocol audit; GPU numerical gates not yet run"})
    log("audit-qwen", f"PASS: 196 targets; 112 A groups; output={root}")
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("command", choices=("prepare", "pilot-a", "collect-a", "roots", "quantize", "solve-gi",
                       "evaluate-gi", "pilot-dg", "collect-dg", "solve-gd", "evaluate-gd", "summary", "run"))
    parser.add_argument("--allow-download", action="store_true", help="Allow raw dataset acquisition ONLY during prepare")
    parser.add_argument("--offline-raw-dir", type=Path, help="Verified calibration.jsonl.gz and calibration.source.json")
    parser.add_argument("--wikitext-cache-dir", type=Path, help="Read-only directory of original wikitext-{split}.arrow")
    parser.add_argument("--max-hours", type=float)
    args = parser.parse_args(argv)
    if args.allow_download and args.command != "prepare":
        parser.error("--allow-download only applies to explicit prepare")
    if bool(args.offline_raw_dir) != bool(args.wikitext_cache_dir):
        parser.error("Provide both --offline-raw-dir and --wikitext-cache-dir")
    if args.offline_raw_dir and (args.command != "prepare" or args.allow_download):
        parser.error("Offline directories apply only to prepare, without --allow-download")
    if args.max_hours is not None and (not math.isfinite(args.max_hours) or args.max_hours <= 0):
        parser.error("--max-hours must be positive and finite")
    settings = settings_from_file(args.config)
    # HF reads offline flags at import time; set them before audit imports.
    for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if args.allow_download:
            os.environ.pop(key, None)
        else:
            os.environ[key] = "1"
    assert_not_shared(settings)
    cache = Path(settings["run_dir"]) / "dataset_cache"
    for key, suffix in (("HF_HOME", "hf"), ("HF_HUB_CACHE", "hub"),
                        ("HF_DATASETS_CACHE", "datasets"), ("HF_MODULES_CACHE", "modules"),
                        ("TRANSFORMERS_CACHE", "transformers")):
        os.environ[key] = str(cache / suffix)
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    environment_check()
    stop = Stop(None if args.max_hours is None else args.max_hours*3600)
    try:
        with stop.installed(), run_lock(settings):
            if args.command == "prepare":
                if shutil.disk_usage(settings["run_dir"]).free < 180*2**30:
                    raise RuntimeError("Need at least 180 GiB free for Qwen A/roots/W3/checkpoints and reserve")
                config = prepare(settings, args.allow_download, args.offline_raw_dir, args.wikitext_cache_dir)
            else:
                config = read_json(Path(settings["run_dir"]) / "config.json")
                initialized = read_json(Path(settings["run_dir"]) / "initialization.json")
                if initialized["settings"] != settings or initialized["config"] != config or initialized["environment"] != environment():
                    raise RuntimeError("Settings/environment changed on resume")
            manifest = load_manifest(config)
            protocol.verify_protected(manifest["payload"]["protected_llama_files"])
            torch.set_num_threads(config["cpu_threads"])
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.manual_seed(1234)
            if args.command not in ("prepare", "summary"):
                devices = gpu_check()
                if any("4090" not in x["name"] for x in devices):
                    raise RuntimeError("This deployment requires two RTX4090s")
            actions = {
                "pilot-a": lambda: stages.collect(config, stop, "a", max_new_windows=max(4, config["a_batch_size"])),
                "collect-a": lambda: stages.collect(config, stop, "a"),
                "roots": lambda: stages.roots(config, stop),
                "quantize": lambda: stages.quantize(config, stop),
                "solve-gi": lambda: stages.solve(config, stop, "gi"),
                "evaluate-gi": lambda: stages.evaluate(config, stop, "gi"),
                "pilot-dg": lambda: stages.collect(config, stop, "dg", max_new_windows=4),
                "collect-dg": lambda: stages.collect(config, stop, "dg"),
                "solve-gd": lambda: stages.solve(config, stop, "gd"),
                "evaluate-gd": lambda: stages.evaluate(config, stop, "gd"),
            }
            order = ("collect-a", "roots", "quantize", "solve-gi", "evaluate-gi", "collect-dg", "solve-gd", "evaluate-gd")
            for command in order if args.command == "run" else [args.command]:
                stop.check()
                if command in actions:
                    protocol.verify_protected(manifest["payload"]["protected_llama_files"])
                    log("qwen-runner", f"starting {command}")
                    actions[command]()
            protocol.verify_protected(manifest["payload"]["protected_llama_files"])
            print(json.dumps(stages.summarize(config), indent=2), flush=True)
        return 0
    except Paused as error:
        log("paused-qwen", str(error))
        return 75


if __name__ == "__main__":
    raise SystemExit(main())
