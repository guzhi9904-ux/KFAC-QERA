#!/usr/bin/env python3
"""Evaluate completed MXINT3/4 artifacts with QERA's pinned WikiText harness.

The official QERA checkout and the KFAC-QERA artifact runs are immutable inputs.
Only the separately configured output directory is written.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import importlib.metadata
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Any, Mapping

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))

import torch
import yaml
from safetensors.torch import load_file

from qera_diag_g_isolation import pipeline as mxint4_pipeline
from qera_diag_g_isolation.mxint3_v1 import pipeline as mxint3_pipeline
from qera_diag_g_isolation.storage import (
    atomic_json,
    disjoint_roots,
    fingerprint,
    layers,
    load_manifest,
    read_json,
    verify,
)
from qera_original_a_isolation.common import OFFICIAL_QERA_COMMIT, import_official_qera, sha256_file
from qera_original_a_isolation.harness_word_ppl import (
    HARNESS_COMMIT,
    HARNESS_HASHES,
    core_versions,
    dataset_identity,
    extract_word_ppl,
    verify_harness,
)


FIXED_PROTOCOL = {
    "task": "wikitext",
    "context_length": 4096,
    "max_position_embeddings": 4096,
    "dtype": "bfloat16",
    "attn_implementation": "eager",
    "device_map": "auto-balanced",
    "lm_eval_batch_size": "auto",
    "num_fewshot": None,
    "methods": ["wq", "diag_gd", "full_gd"],
}
SUPPORTED_QUANTIZATION = {
    3: {"name": "mxint", "width": 3, "block_size": 32, "block_axis": -1},
    4: {"name": "mxint", "width": 4, "block_size": 32, "block_axis": -1},
}


def log(message: str) -> None:
    print(f"[official-word-ppl] {time.strftime('%Y-%m-%dT%H:%M:%S%z')} {message}", flush=True)


def _write_harness_json(path: str | Path, value: Any) -> None:
    """Atomically preserve the harness payload, including its environment-specific scalar types."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")

    def convert(item):
        scalar = getattr(item, "item", None)
        return scalar() if callable(scalar) else str(item)

    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, default=convert)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _path(value: str | Path) -> str:
    return str(Path(os.path.expandvars(str(value))).expanduser().resolve())


def load_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Evaluation config must be a YAML mapping")
    for key, expected in FIXED_PROTOCOL.items():
        if config.get(key) != expected:
            raise ValueError(f"Official table protocol fixes {key}={expected!r}")
    if config.get("official_qera_commit") != OFFICIAL_QERA_COMMIT:
        raise ValueError(f"official_qera_commit must be {OFFICIAL_QERA_COMMIT}")
    if config.get("harness_commit") != HARNESS_COMMIT:
        raise ValueError(f"harness_commit must be {HARNESS_COMMIT}")
    ranks = config.get("ranks")
    if not isinstance(ranks, list) or not ranks or ranks != sorted(set(ranks)) or any(
        not isinstance(rank, int) or rank <= 0 for rank in ranks
    ):
        raise ValueError("ranks must be a nonempty sorted list of unique positive integers")
    for key in ("bf16_reference", "bf16_tolerance"):
        if not isinstance(config.get(key), (int, float)) or not math.isfinite(float(config[key])):
            raise ValueError(f"{key} must be finite")
    if config["bf16_reference"] <= 0 or not 0 <= config["bf16_tolerance"] < 1:
        raise ValueError("Invalid BF16 reference/tolerance")
    if not isinstance(config.get("expected_visible_gpus"), int) or config["expected_visible_gpus"] <= 0:
        raise ValueError("expected_visible_gpus must be a positive integer")

    for key in ("model_path", "official_qera_root", "harness_source", "output_dir"):
        if key not in config:
            raise ValueError(f"Missing required path: {key}")
        config[key] = _path(config[key])
    runs = config.get("artifact_runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("artifact_runs must be a nonempty list")
    names: set[str] = set()
    widths: set[int] = set()
    for item in runs:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str) or not item["name"]:
            raise ValueError("Each artifact run needs a nonempty name")
        if item["name"] in names:
            raise ValueError(f"Duplicate artifact run name: {item['name']}")
        width = item.get("width")
        if width not in SUPPORTED_QUANTIZATION or width in widths:
            raise ValueError("Each of MXINT3 and MXINT4 may be configured at most once")
        item["run_dir"] = _path(item["run_dir"])
        names.add(item["name"])
        widths.add(width)

    output = Path(config["output_dir"])
    for immutable in (config["model_path"], config["official_qera_root"], config["harness_source"]):
        disjoint_roots(immutable, output)
    for item in runs:
        disjoint_roots(item["run_dir"], output)
    config["_config_path"] = str(source)
    return config


@dataclass(frozen=True)
class EvaluationConfiguration:
    name: str
    method: str
    rank: int | None
    artifacts: Mapping[str, Mapping[str, Mapping[str, Any]]]


@dataclass(frozen=True)
class ArtifactRun:
    name: str
    width: int
    run_dir: Path
    config: Mapping[str, Any]
    manifest: Mapping[str, Any]
    configurations: tuple[EvaluationConfiguration, ...]

    def qualified(self, name: str) -> str:
        return f"{self.name}:{name}"


def _configuration_name(width: int, method: str, rank: int | None) -> str:
    if method == "wq":
        return f"W{width}_MXINT"
    if rank is None:
        raise ValueError("Corrected configurations require a rank")
    return f"{method.upper()}_R{rank}"


def _validate_variant(width: int, manifest: Mapping[str, Any]) -> None:
    payload = manifest["payload"]
    if payload["source_config"].get("quantization") != SUPPORTED_QUANTIZATION[width]:
        raise RuntimeError(f"Manifest is not the expected MXINT{width} block-32 protocol")
    if width == 4:
        transition = payload.get("solver_transition", {})
        if transition.get("variant") != "full_svd_v1" or transition.get("svd_full_matrices") is not True:
            raise RuntimeError("MXINT4 input must be the completed full_svd_v1 comparison run")
    else:
        transition = payload.get("quantization_transition", {})
        if (
            payload["config"].get("experiment_variant") != "mxint3_v1"
            or transition.get("variant") != "mxint3_v1"
            or transition.get("svd_full_matrices") is not True
        ):
            raise RuntimeError("MXINT3 input must be the completed mxint3_v1 comparison run")


def load_artifact_run(spec: Mapping[str, Any], requested_ranks: list[int]) -> ArtifactRun:
    root = Path(spec["run_dir"])
    raw = read_json(root / "manifest.json")
    run_config = raw["payload"]["config"]
    if Path(run_config["run_dir"]).resolve() != root.resolve():
        raise RuntimeError(f"Manifest run_dir does not match configured artifact root: {root}")
    manifest = load_manifest(run_config)
    width = int(spec["width"])
    _validate_variant(width, manifest)
    available_ranks = manifest["payload"]["config"].get("ranks", [])
    if any(rank not in available_ranks for rank in requested_ranks):
        raise RuntimeError(f"Requested ranks {requested_ranks} are not all present in {root}: {available_ranks}")

    resolver = mxint3_pipeline.evaluation_inputs if width == 3 else mxint4_pipeline._evaluation_inputs
    configurations: list[EvaluationConfiguration] = []
    for method in FIXED_PROTOCOL["methods"]:
        ranks: list[int | None] = [None] if method == "wq" else requested_ranks
        for rank in ranks:
            artifacts = resolver(run_config, manifest, method, check=False)
            configurations.append(
                EvaluationConfiguration(_configuration_name(width, method, rank), method, rank, artifacts)
            )
    return ArtifactRun(spec["name"], width, root, run_config, manifest, tuple(configurations))


def _record_identity(record: Mapping[str, Any]) -> dict[str, Any]:
    return {key: record[key] for key in ("path", "sha256", "bytes")}


def _configuration_identity(configuration: EvaluationConfiguration) -> dict[str, Any]:
    return {
        "configuration": configuration.name,
        "method": configuration.method,
        "rank": configuration.rank,
        "artifacts": {
            module: {kind: _record_identity(record) for kind, record in sorted(records.items())}
            for module, records in sorted(configuration.artifacts.items())
        },
    }


def _check_record_stat(record: Mapping[str, Any]) -> None:
    path = Path(record["path"])
    if not path.is_file() or path.stat().st_size != record["bytes"]:
        raise RuntimeError(f"Missing or size-changed immutable artifact: {path}")


def _artifact_records(plan: ArtifactRun) -> dict[tuple[str, str], Mapping[str, Any]]:
    unique: dict[tuple[str, str], Mapping[str, Any]] = {}
    for configuration in plan.configurations:
        for records in configuration.artifacts.values():
            for record in records.values():
                key = (str(Path(record["path"]).resolve()), record["sha256"])
                unique[key] = record
    return unique


def _validate_shared_model(config: Mapping[str, Any], plans: list[ArtifactRun], hash_files: bool) -> dict[str, Any]:
    expected_path = Path(config["model_path"]).resolve()
    reference: dict[str, Any] | None = None
    verified: set[tuple[str, str]] = set()
    for plan in plans:
        source = plan.manifest["payload"]["source_config"]
        if Path(source["model_path"]).resolve() != expected_path:
            raise RuntimeError(f"Artifact run {plan.name} was produced from a different model")
        model_files = plan.manifest["payload"]["model_files"]
        identity = {name: _record_identity(record) for name, record in sorted(model_files.items())}
        comparable = {name: {"sha256": value["sha256"], "bytes": value["bytes"]} for name, value in identity.items()}
        if reference is None:
            reference = comparable
        elif comparable != reference:
            raise RuntimeError("MXINT3 and MXINT4 manifests do not pin the same model files")
        for record in model_files.values():
            if Path(record["path"]).resolve().parent != expected_path:
                raise RuntimeError(f"Pinned model file is outside model_path: {record['path']}")
            key = (str(Path(record["path"]).resolve()), record["sha256"])
            if key not in verified:
                verify(record) if hash_files else _check_record_stat(record)
                verified.add(key)
    if reference is None:
        raise RuntimeError("No model identity was found")
    return {"path": str(expected_path), "files": reference}


def _cuda_summary(expected: int) -> list[dict[str, Any]]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != expected:
        raise RuntimeError(
            f"Expected exactly {expected} visible CUDA devices; set CUDA_VISIBLE_DEVICES to idle GPUs before running"
        )
    return [
        {
            "index": index,
            "name": torch.cuda.get_device_name(index),
            "total_GiB": torch.cuda.get_device_properties(index).total_memory / 2**30,
        }
        for index in range(expected)
    ]


def doctor(config: Mapping[str, Any], plans: list[ArtifactRun]) -> dict[str, Any]:
    verify_harness(config["harness_source"], check_import=True)
    official = import_official_qera(
        {"qera_source_dir": config["official_qera_root"], "official_qera_commit": config["official_qera_commit"]}
    )
    model = _validate_shared_model(config, plans, hash_files=False)
    artifacts = {}
    for plan in plans:
        records = _artifact_records(plan)
        for record in records.values():
            _check_record_stat(record)
        artifacts[plan.name] = {
            "width": plan.width,
            "manifest_sha256": plan.manifest["sha256"],
            "unique_tensor_files": len(records),
            "configurations": [configuration.name for configuration in plan.configurations],
        }
    result = {
        "status": "PASS",
        "writes_only": config["output_dir"],
        "immutable_inputs": [
            config["official_qera_root"],
            config["harness_source"],
            *(str(plan.run_dir) for plan in plans),
        ],
        "official_qera_commit": official["commit"],
        "harness_commit": HARNESS_COMMIT,
        "model": model,
        "artifacts": artifacts,
        "visible_gpus": _cuda_summary(config["expected_visible_gpus"]),
        "fixed_protocol": FIXED_PROTOCOL,
    }
    log(json.dumps(result, ensure_ascii=False, default=str))
    return result


def _prepare_protocol(config: Mapping[str, Any], plans: list[ArtifactRun]) -> tuple[dict[str, Any], str]:
    verify_harness(config["harness_source"], check_import=True)
    official = import_official_qera(
        {"qera_source_dir": config["official_qera_root"], "official_qera_commit": config["official_qera_commit"]}
    )
    from lm_eval.tasks import TaskManager, get_task_dict

    log("loading pinned harness WikiText document-level test set")
    task = get_task_dict(["wikitext"], TaskManager(verbosity="INFO"))["wikitext"]
    data = dataset_identity(task)
    log(f"dataset documents={data['documents']} pages_sha256={data['raw_pages_sha256']}")
    model = _validate_shared_model(config, plans, hash_files=True)
    protocol = {
        "protocol_version": 1,
        "purpose": "QERA-paper-compatible document-level WikiText word perplexity for immutable artifacts",
        "fixed_protocol": FIXED_PROTOCOL,
        "official_qera_commit": official["commit"],
        "harness_commit": HARNESS_COMMIT,
        "harness_hashes": HARNESS_HASHES,
        "dataset": data,
        "model": model,
        "artifact_runs": [
            {
                "name": plan.name,
                "width": plan.width,
                "run_dir": str(plan.run_dir),
                "manifest_sha256": plan.manifest["sha256"],
                "configurations": [configuration.name for configuration in plan.configurations],
            }
            for plan in plans
        ],
        "packages": core_versions(),
        "lm_eval_distribution": importlib.metadata.version("lm_eval"),
        "runner_sha256": sha256_file(__file__),
        "bf16_gate": {
            "reference": float(config["bf16_reference"]),
            "tolerance": float(config["bf16_tolerance"]),
        },
        "expected_visible_gpus": config["expected_visible_gpus"],
        "seeds": {"random": 0, "numpy": 1234, "torch": 1234, "fewshot": 1234},
        "artifact_semantics": "BF16-emulated MXINT Wq plus runtime output += (input @ A[:, :r]) @ B[:r]",
    }
    protocol_hash = fingerprint(protocol)
    path = Path(config["output_dir"]) / "protocol.json"
    if path.exists():
        existing = read_json(path)
        if fingerprint(existing) != protocol_hash:
            raise RuntimeError(
                f"Output directory belongs to a different protocol; choose a new output_dir: {path.parent}"
            )
    else:
        atomic_json(path, protocol)
    return protocol, protocol_hash


@contextlib.contextmanager
def output_lock(root: str | Path):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if os.name != "posix":
        yield
        return
    import fcntl

    with (root / ".evaluation.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another evaluator owns {root}") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _load_model(config: Mapping[str, Any]):
    import transformers
    from accelerate import dispatch_model
    from qera.utils import create_device_map

    _cuda_summary(config["expected_visible_gpus"])
    log(
        f"loading model={config['model_path']} dtype=bfloat16 context=4096 "
        f"attention=eager device_map=auto-balanced"
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        config["model_path"],
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        _attn_implementation="eager",
        max_position_embeddings=4096,
    )
    model.eval()
    model.config.use_cache = False
    if model.config.max_position_embeddings != 4096:
        raise RuntimeError("Model did not retain max_position_embeddings=4096")
    device_map = create_device_map(model, "auto-balanced")
    model = dispatch_model(model, device_map=device_map)
    return model, device_map


def _result_path(root: Path, qualified: str) -> Path:
    if qualified == "BF16":
        return root / "BF16"
    variant, configuration = qualified.split(":", 1)
    return root / variant / configuration


def _existing_result(root: Path, qualified: str, protocol_hash: str) -> dict[str, Any] | None:
    directory = _result_path(root, qualified)
    state_path = directory / "complete.json"
    if not state_path.exists():
        return None
    state = read_json(state_path)
    if state.get("status") != "PASS" or state.get("protocol_sha256") != protocol_hash:
        raise RuntimeError(f"Completed result uses a different protocol: {state_path}")
    result_path = directory / "results.json"
    if state.get("results_sha256") != sha256_file(result_path):
        raise RuntimeError(f"Saved harness result changed: {result_path}")
    if extract_word_ppl(read_json(result_path)) != state["word_ppl"]:
        raise RuntimeError(f"Saved word-PPL does not match its harness result: {result_path}")
    return state


def _run_harness(model: torch.nn.Module) -> tuple[dict[str, Any], float]:
    from qera.evaluate import evaluate_harness_downstream

    with torch.no_grad():
        result = evaluate_harness_downstream(
            model,
            tasks=["wikitext"],
            num_fewshot=None,
            use_cache=None,
            batch_size="auto",
        )
    return result, extract_word_ppl(result)


def _save_result(
    root: Path,
    qualified: str,
    protocol_hash: str,
    model: torch.nn.Module,
    device_map: Mapping[str, Any],
    details: Mapping[str, Any],
) -> dict[str, Any]:
    previous = _existing_result(root, qualified, protocol_hash)
    if previous is not None:
        log(f"configuration={qualified} already complete word_ppl={previous['word_ppl']:.6f}")
        return previous
    started = time.monotonic()
    log(f"configuration={qualified} harness evaluation started")
    result, ppl = _run_harness(model)
    directory = _result_path(root, qualified)
    result_path = directory / "results.json"
    _write_harness_json(result_path, result)
    record = {
        "status": "PASS",
        "configuration": qualified,
        "metric": "word_perplexity",
        "word_ppl": ppl,
        "context_length": 4096,
        "elapsed_seconds": time.monotonic() - started,
        "protocol_sha256": protocol_hash,
        "results_sha256": sha256_file(result_path),
        "device_map": dict(device_map),
        **details,
    }
    atomic_json(directory / "complete.json", record)
    log(f"configuration={qualified} PASS word_ppl={ppl:.6f}")
    return record


def _baseline_gate(config: Mapping[str, Any], record: Mapping[str, Any]) -> dict[str, Any]:
    difference = abs(float(record["word_ppl"]) - float(config["bf16_reference"]))
    return {
        "status": "PASS" if difference <= config["bf16_tolerance"] else "FAIL",
        "observed_word_ppl": record["word_ppl"],
        "reference_word_ppl": config["bf16_reference"],
        "absolute_difference": difference,
        "tolerance": config["bf16_tolerance"],
        "note": "Engineering gate anchored to the reproduced official bd7fc86 result, not a paper-specified tolerance",
    }


def _load_tensor_record(record: Mapping[str, Any], tensor_names: set[str]) -> dict[str, torch.Tensor]:
    verify(record)
    tensors = load_file(str(record["path"]), device="cpu")
    if set(tensors) != tensor_names:
        raise RuntimeError(f"Unexpected tensors in {record['path']}: {sorted(tensors)}")
    return tensors


def _load_weight_cache(configuration: EvaluationConfiguration) -> dict[str, torch.Tensor]:
    cache = {}
    started = time.monotonic()
    for index, (module, records) in enumerate(configuration.artifacts.items(), 1):
        tensors = _load_tensor_record(records["quant"], {"weight_q"})
        cache[module] = tensors["weight_q"]
        if index % 16 == 0 or index == len(configuration.artifacts):
            log(f"verified/loaded Wq {index}/{len(configuration.artifacts)} elapsed={time.monotonic()-started:.0f}s")
    return cache


def _install_weights(model: torch.nn.Module, cache: Mapping[str, torch.Tensor]) -> None:
    modules = dict(model.named_modules())
    with torch.no_grad():
        for name, weight in cache.items():
            if name not in modules or not hasattr(modules[name], "weight"):
                raise RuntimeError(f"Target module is missing from model: {name}")
            module = modules[name]
            if tuple(module.weight.shape) != tuple(weight.shape):
                raise RuntimeError(
                    f"Wq shape mismatch for {name}: {tuple(weight.shape)} != {tuple(module.weight.shape)}"
                )
            module.weight.copy_(weight.to(device=module.weight.device, dtype=module.weight.dtype))


def _load_correction_cache(configuration: EvaluationConfiguration) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    cache = {}
    started = time.monotonic()
    for index, (module, records) in enumerate(configuration.artifacts.items(), 1):
        tensors = _load_tensor_record(records["correction"], {"A", "B"})
        a, b = tensors["A"], tensors["B"]
        if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
            raise RuntimeError(f"Invalid correction shapes for {module}: A={tuple(a.shape)} B={tuple(b.shape)}")
        cache[module] = (a, b)
        if index % 16 == 0 or index == len(configuration.artifacts):
            log(
                f"verified/loaded corrections {index}/{len(configuration.artifacts)} "
                f"elapsed={time.monotonic()-started:.0f}s"
            )
    return cache


@contextlib.contextmanager
def correction_hooks(
    model: torch.nn.Module,
    cache: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
    rank: int,
):
    modules = dict(model.named_modules())
    handles = []
    try:
        for name, (a_cpu, b_cpu) in cache.items():
            module = modules.get(name)
            if module is None or not hasattr(module, "weight"):
                raise RuntimeError(f"Correction target is missing from model: {name}")
            if rank > a_cpu.shape[1] or rank > b_cpu.shape[0]:
                raise RuntimeError(f"Rank {rank} exceeds saved correction rank for {name}")
            if a_cpu.shape[0] != module.weight.shape[1] or b_cpu.shape[1] != module.weight.shape[0]:
                raise RuntimeError(f"Correction/model dimensions differ for {name}")
            a = a_cpu[:, :rank].to(device=module.weight.device, dtype=module.weight.dtype)
            b = b_cpu[:rank].to(device=module.weight.device, dtype=module.weight.dtype)

            def hook(_module, args, output, left=a, right=b):
                return output + (args[0] @ left) @ right

            handles.append(module.register_forward_hook(hook))
        yield
    finally:
        for handle in handles:
            handle.remove()
        handles.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _artifact_protocol_hash(base_hash: str, plan: ArtifactRun, configuration: EvaluationConfiguration) -> str:
    return fingerprint(
        {
            "base_protocol_sha256": base_hash,
            "artifact_run": plan.name,
            "width": plan.width,
            "manifest_sha256": plan.manifest["sha256"],
            **_configuration_identity(configuration),
        }
    )


def _selected(qualified: str, only: set[str]) -> bool:
    return not only or qualified in only


def _evaluate_artifact_run(
    config: Mapping[str, Any],
    plan: ArtifactRun,
    base_hash: str,
    only: set[str],
) -> list[dict[str, Any]]:
    selected = [item for item in plan.configurations if _selected(plan.qualified(item.name), only)]
    if not selected:
        return []
    root = Path(config["output_dir"])
    records: list[dict[str, Any]] = []
    pending: list[EvaluationConfiguration] = []
    for item in selected:
        item_protocol = _artifact_protocol_hash(base_hash, plan, item)
        previous = _existing_result(root, plan.qualified(item.name), item_protocol)
        if previous is None:
            pending.append(item)
        else:
            records.append(previous)
            log(f"configuration={plan.qualified(item.name)} already complete word_ppl={previous['word_ppl']:.6f}")
    if not pending:
        return records
    model, device_map = _load_model(config)
    weight_configuration = next(item for item in plan.configurations if item.method == "wq")
    weight_cache = _load_weight_cache(weight_configuration)
    try:
        _install_weights(model, weight_cache)
        correction_cache: dict[str, dict[str, tuple[torch.Tensor, torch.Tensor]]] = {}
        for item in pending:
            qualified = plan.qualified(item.name)
            item_protocol = _artifact_protocol_hash(base_hash, plan, item)
            details = {
                "artifact_run": plan.name,
                "artifact_width": plan.width,
                "artifact_manifest_sha256": plan.manifest["sha256"],
                "method": item.method,
                "rank": item.rank,
                "artifact_identity": _configuration_identity(item),
            }
            if item.method == "wq":
                records.append(
                    _save_result(Path(config["output_dir"]), qualified, item_protocol, model, device_map, details)
                )
                continue
            if item.method not in correction_cache:
                correction_cache[item.method] = _load_correction_cache(item)
            with correction_hooks(model, correction_cache[item.method], int(item.rank)):
                records.append(
                    _save_result(Path(config["output_dir"]), qualified, item_protocol, model, device_map, details)
                )
        correction_cache.clear()
    finally:
        weight_cache.clear()
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return records


def _known_configurations(plans: list[ArtifactRun]) -> set[str]:
    return {"BF16"} | {plan.qualified(item.name) for plan in plans for item in plan.configurations}


def _all_completed(config: Mapping[str, Any], plans: list[ArtifactRun], base_hash: str) -> list[dict[str, Any]]:
    root = Path(config["output_dir"])
    records = []
    baseline = _existing_result(root, "BF16", base_hash)
    if baseline is not None:
        records.append(baseline)
    for plan in plans:
        for item in plan.configurations:
            item_hash = _artifact_protocol_hash(base_hash, plan, item)
            record = _existing_result(root, plan.qualified(item.name), item_hash)
            if record is not None:
                records.append(record)
    return records


def write_summary(config: Mapping[str, Any], records: list[Mapping[str, Any]]) -> None:
    root = Path(config["output_dir"])
    rows = sorted(records, key=lambda record: record["configuration"])
    atomic_json(root / "summary.json", {"status": "PASS", "results": rows})
    fields = [
        "configuration",
        "metric",
        "word_ppl",
        "context_length",
        "artifact_width",
        "method",
        "rank",
        "elapsed_seconds",
    ]
    temporary = root / ".summary.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, root / "summary.csv")


def evaluate(config: Mapping[str, Any], plans: list[ArtifactRun], stage: str, only: set[str]) -> dict[str, Any]:
    unknown = only - _known_configurations(plans)
    if unknown:
        raise ValueError(f"Unknown --only configurations: {sorted(unknown)}")
    _cuda_summary(config["expected_visible_gpus"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(1234)
    root = Path(config["output_dir"])
    with output_lock(root):
        _, base_hash = _prepare_protocol(config, plans)
        baseline = _existing_result(root, "BF16", base_hash)
        if stage in ("bf16", "all") and (not only or "BF16" in only):
            if baseline is None:
                model, device_map = _load_model(config)
                try:
                    baseline = _save_result(root, "BF16", base_hash, model, device_map, {"method": "bf16"})
                finally:
                    del model
                    gc.collect()
                    torch.cuda.empty_cache()
        else:
            if baseline is None:
                raise RuntimeError("Run the BF16 stage first with this exact protocol")
        gate = _baseline_gate(config, baseline)
        atomic_json(root / "bf16_reference_check.json", gate)
        if gate["status"] != "PASS":
            raise RuntimeError(f"BF16 official reference gate failed; refusing artifact evaluation: {gate}")

        if stage in ("artifacts", "all"):
            for plan in plans:
                _evaluate_artifact_run(config, plan, base_hash, only)
        completed = _all_completed(config, plans, base_hash)
        write_summary(config, completed)
        result = {
            "status": "PASS",
            "stage": stage,
            "bf16_gate": gate,
            "completed_configurations": len(completed),
            "output_dir": str(root),
        }
        atomic_json(root / "status.json", result)
        return result


def main(argv: list[str] | None = None) -> int:
    for variable in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ[variable] = "1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Check immutable sources, manifests, artifact presence, and GPUs")
    run_parser = commands.add_parser("evaluate", help="Run or resume official word-PPL evaluation")
    run_parser.add_argument("--stage", choices=("bf16", "artifacts", "all"), default="bf16")
    run_parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Repeatable: BF16 or <artifact-name>:<configuration>, e.g. mxint4:DIAG_GD_R32",
    )
    commands.add_parser("summary", help="Validate and rewrite summaries from completed results")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    plans = [load_artifact_run(spec, config["ranks"]) for spec in config["artifact_runs"]]
    if args.command == "doctor":
        result = doctor(config, plans)
    elif args.command == "evaluate":
        result = evaluate(config, plans, args.stage, set(args.only))
    else:
        protocol_path = Path(config["output_dir"]) / "protocol.json"
        if not protocol_path.is_file():
            raise RuntimeError("No protocol.json exists; run an evaluation stage first")
        protocol = read_json(protocol_path)
        base_hash = fingerprint(protocol)
        with output_lock(config["output_dir"]):
            records = _all_completed(config, plans, base_hash)
            write_summary(config, records)
        result = {"status": "PASS", "completed_configurations": len(records), "output_dir": config["output_dir"]}
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
