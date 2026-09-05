from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

import psutil
import torch
import yaml

from .config import load_config, output_root
from .modeling import inspect_model_config
from .utils import canonical_sha256, ensure_layout, save_json, utc_now


def doctor(config: Mapping[str, Any]) -> dict[str, Any]:
    root = output_root(config)
    ensure_layout(root)
    packages = {}
    for name in ("torch", "transformers", "accelerate", "datasets", "safetensors", "pandas", "matplotlib"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    model = inspect_model_config(config)
    device = str(config["model"]["device"])
    cuda_requested = device.startswith("cuda")
    if cuda_requested and not torch.cuda.is_available():
        raise RuntimeError("CUDA is requested by the config but torch.cuda.is_available() is false")
    if str(config["model"]["dtype"]) == "bfloat16" and cuda_requested and not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected CUDA device does not report native BF16 support")
    config_for_hash = {key: value for key, value in config.items() if key != "_config_path"}
    resolved_config_path = root / "config_resolved.yaml"
    if resolved_config_path.is_file():
        existing = yaml.safe_load(resolved_config_path.read_text(encoding="utf-8"))
        if canonical_sha256(existing) != canonical_sha256(config_for_hash):
            raise RuntimeError(f"RUN_DIR is locked to a different resolved config: {resolved_config_path}; use a new RUN_DIR")
    else:
        resolved_config_path.write_text(yaml.safe_dump(config_for_hash, sort_keys=False, allow_unicode=True), encoding="utf-8")
    result = {
        "status": "PASS",
        "captured_at_utc": utc_now(),
        "config_path": config["_config_path"],
        "resolved_config_path": str(resolved_config_path),
        "config_sha256": canonical_sha256(config_for_hash),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        "cuda_runtime": torch.version.cuda,
        "bf16_supported": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        "ram_total_bytes": psutil.virtual_memory().total,
        "ram_available_bytes": psutil.virtual_memory().available,
        "disk_free_bytes": shutil.disk_usage(root).free,
        "model": model,
    }
    save_json(root / "state" / "doctor.json", result)
    return result


def status(config: Mapping[str, Any]) -> dict[str, Any]:
    root = output_root(config)
    files = sorted(root.glob("state/*.json")) if root.is_dir() else []
    rows = []
    for path in files:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            rows.append({"file": path.name, "status": value.get("status"), "completed_at_utc": value.get("completed_at_utc")})
        except Exception as error:
            rows.append({"file": path.name, "status": "UNREADABLE", "error": str(error)})
    return {"output_dir": str(root), "state_files": rows}


def _assert_config_lock(config: Mapping[str, Any]) -> None:
    path = output_root(config) / "config_resolved.yaml"
    if not path.is_file():
        raise RuntimeError("Resolved config lock is absent; run doctor first")
    existing = yaml.safe_load(path.read_text(encoding="utf-8"))
    current = {key: value for key, value in config.items() if key != "_config_path"}
    if canonical_sha256(existing) != canonical_sha256(current):
        raise RuntimeError(f"RUN_DIR is locked to a different config: {path}; use a new RUN_DIR")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MXINT4 full-A/full-G validation experiments")
    parser.add_argument("--config", required=True, help="YAML config path")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor", help="Validate environment and resolve config")
    data_parser = subparsers.add_parser("prepare-data", help="Tokenize and freeze calibration/evaluation windows")
    data_parser.add_argument("--role", choices=("all", "calibration", "wikitext2", "c4"), default="all")
    subparsers.add_parser("quantize", help="Discover target modules and freeze MXINT4 weights")
    subparsers.add_parser("plan", help="Create RAM-bounded full-A/full-G collection shards")
    collect_parser = subparsers.add_parser("collect", help="Collect full A/G for one shard")
    collect_parser.add_argument("--shard", type=int, required=True)
    collect_parser.add_argument("--solve", action="store_true", help="Solve all six methods immediately after collection")
    solve_parser = subparsers.add_parser("solve", help="Solve all six methods for one collected shard")
    solve_parser.add_argument("--shard", type=int, required=True)
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate PPL on frozen windows")
    evaluate_parser.add_argument("--dataset", choices=("all", "wikitext2", "c4"), default="all")
    subparsers.add_parser("analyze", help="Create PPL tables, focused plots, energy plots, and contrasts")
    subparsers.add_parser("status", help="Show checkpoint files")
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = load_config(Path(args.config))
    if args.command not in {"doctor", "status"}:
        _assert_config_lock(config)
    if args.command == "doctor":
        result = doctor(config)
    elif args.command == "prepare-data":
        from .data import prepare_all, prepare_role

        result = prepare_all(config) if args.role == "all" else prepare_role(config, args.role)
    elif args.command == "quantize":
        from .quantization import quantize_model

        result = quantize_model(config)
    elif args.command == "plan":
        from .statistics import make_shard_plan

        result = make_shard_plan(config)
    elif args.command == "collect":
        from .solver import solve_shard
        from .statistics import collect_shard

        collected = collect_shard(config, args.shard)
        result = {"collection": collected, "solve": solve_shard(config, args.shard)} if args.solve else collected
    elif args.command == "solve":
        from .solver import solve_shard

        result = solve_shard(config, args.shard)
    elif args.command == "evaluate":
        from .evaluate import evaluate_dataset

        result = (
            {role: evaluate_dataset(config, role) for role in ("wikitext2", "c4")}
            if args.dataset == "all"
            else evaluate_dataset(config, args.dataset)
        )
    elif args.command == "analyze":
        from .analysis import analyze_all

        result = analyze_all(config)
    elif args.command == "status":
        result = status(config)
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
