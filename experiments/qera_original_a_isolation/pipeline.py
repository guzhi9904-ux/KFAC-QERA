from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from .common import (
    InputGroup,
    get_module,
    import_official_qera,
    load_json,
    qera_layer_config,
    run_dir,
    safe_name,
    save_json,
    sha256_file,
    torch_dtype,
    utc_now,
)


def log(stage: str, message: str) -> None:
    print(f"[{stage}] {utc_now()} {message}", flush=True)


def atomic_safetensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    prepared = {name: tensor.detach().cpu().contiguous() for name, tensor in tensors.items()}
    save_file(prepared, str(temporary))
    os.replace(temporary, path)


def _real_sqrtm_root(
    root: np.ndarray, sample_count: int
) -> tuple[np.ndarray, float, float]:
    """Match official QERA's complex-to-real cast and retain diagnostics.

    SciPy may return a complex square root for a numerically near-PSD matrix.
    Official QERA converts that result to a real floating tensor, which drops
    the imaginary component, before applying the sample-count normalization.
    """
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    max_imaginary_raw = float(np.max(np.abs(root.imag))) if np.iscomplexobj(root) else 0.0
    normalizer = math.sqrt(sample_count)
    max_imaginary_normalized = max_imaginary_raw / normalizer
    real_root = np.asarray(np.real(root), dtype=np.float64) / normalizer
    return real_root, max_imaginary_raw, max_imaginary_normalized


def frozen_path(config: dict[str, Any], role: str) -> Path:
    return run_dir(config) / "data" / f"{role}.safetensors"


def _take_streaming_rows(stream, count: int) -> tuple[list[dict[str, Any]], str]:
    rows: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    for row in stream:
        text = row["text"]
        payload = text.encode("utf-8")
        digest.update(len(payload).to_bytes(8, byteorder="little"))
        digest.update(payload)
        rows.append(row)
        if len(rows) % 128 == 0 or len(rows) == count:
            log("data", f"streamed SlimPajama raw rows={len(rows)}/{count}")
        if len(rows) == count:
            break
    if len(rows) != count:
        raise RuntimeError(f"SlimPajama stream ended after {len(rows)}/{count} rows")
    return rows, digest.hexdigest()


def _load_streaming_slimpajama_prefix(config: dict[str, Any], count: int):
    import datasets as hf_datasets
    from huggingface_hub import HfApi

    dataset_id = config.get("slimpajama_dataset", "DKYoon/SlimPajama-6B")
    requested_revision = config.get("slimpajama_revision", "main")
    resolved_revision = HfApi().dataset_info(dataset_id, revision=requested_revision).sha
    log(
        "data",
        f"streaming dataset={dataset_id} revision={resolved_revision} raw_prefix_rows={count}",
    )
    stream = hf_datasets.load_dataset(
        dataset_id,
        split="train",
        streaming=True,
        revision=resolved_revision,
    )
    rows, text_sha256 = _take_streaming_rows(stream, count)
    raw = hf_datasets.DatasetDict({"train": hf_datasets.Dataset.from_list(rows)})
    provenance = {
        "acquisition_mode": "streaming_prefix",
        "dataset_id": dataset_id,
        "requested_revision": requested_revision,
        "resolved_revision": resolved_revision,
        "raw_prefix_rows": count,
        "raw_text_sha256": text_sha256,
    }
    return raw, provenance


def prepare_data(config: dict[str, Any], roles: Iterable[str], allow_download: bool) -> dict[str, Any]:
    offline_variables = ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE")
    if allow_download:
        for variable in offline_variables:
            os.environ.pop(variable, None)
    else:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    qera = import_official_qera(config)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config["model_path"], local_files_only=True)
    outputs: dict[str, Any] = {}
    for role in roles:
        path = frozen_path(config, role)
        acquisition_mode = (
            config.get("calibration_acquisition", "official_full")
            if role == "calibration"
            else "official_cached_or_downloaded"
        )
        if path.exists():
            manifest = load_json(path.with_suffix(".json"))
            expected = {
                "model_path": config["model_path"],
                "sequence_length": int(config["sequence_length"]),
                "qera_commit": qera["commit"],
                "acquisition_mode": acquisition_mode,
            }
            actual = {key: manifest.get(key) for key in expected}
            if actual != expected or manifest.get("sha256") != sha256_file(path):
                raise RuntimeError(f"Frozen {role} data provenance mismatch: expected={expected}, actual={actual}")
            log("data", f"{role} already frozen: {path}")
            outputs[role] = manifest
            continue
        if role == "calibration":
            dataset_name = "slim_pajama_6b"
            split = "train"
            limit = int(config["num_calibration_samples"])
            raw_limit = 20 * limit
        elif role == "wikitext2":
            dataset_name = "wikitext2"
            split = "test"
            limit = None
            raw_limit = None
        else:
            raise ValueError(f"Unknown data role: {role}")
        log("data", f"loading QERA dataset={dataset_name} acquisition={acquisition_mode}")
        data_provenance: dict[str, Any] = {"acquisition_mode": acquisition_mode}
        if role == "calibration" and acquisition_mode == "streaming_prefix":
            raw_module, stream_provenance = _load_streaming_slimpajama_prefix(config, raw_limit)
            data_provenance.update(stream_provenance)
            module = qera["preprocess_data_module"](
                raw_module,
                dataset_name,
                tokenizer=tokenizer,
                padding="max_length",
                max_length=int(config["sequence_length"]),
                num_proc=int(config["num_workers"]),
            )
        else:
            module = qera["get_data_module"](
                name=dataset_name,
                tokenizer=tokenizer,
                padding="max_length",
                max_length=int(config["sequence_length"]),
                num_workers=int(config["num_workers"]),
                num_raw_samples=raw_limit,
            )
        dataset = module[split]
        count = len(dataset) if limit is None else min(limit, len(dataset))
        if limit is not None and count != limit:
            raise RuntimeError(f"Official preprocessing produced only {count}/{limit} calibration windows")
        rows = dataset.select(range(count))[:]
        input_ids = torch.tensor(rows["input_ids"], dtype=torch.int64)
        attention_mask = torch.tensor(rows.get("attention_mask", torch.ones_like(input_ids)), dtype=torch.int64)
        atomic_safetensors(path, {"input_ids": input_ids, "attention_mask": attention_mask})
        manifest = {
            "status": "PASS",
            "created_at_utc": utc_now(),
            "role": role,
            "official_dataset_name": dataset_name,
            "split": split,
            "windows": count,
            "sequence_length": int(input_ids.shape[1]),
            "model_path": config["model_path"],
            "token_count": int(attention_mask.sum().item()),
            "qera_commit": qera["commit"],
            "tensor_file": str(path),
            "sha256": sha256_file(path),
            **data_provenance,
        }
        save_json(path.with_suffix(".json"), manifest)
        outputs[role] = manifest
        log("data", f"froze {role} windows={count} sha256={manifest['sha256']}")
    return outputs


def _groups_from_config(model_config: Any) -> list[InputGroup]:
    layers = int(model_config.num_hidden_layers)
    hidden = int(model_config.hidden_size)
    intermediate = int(model_config.intermediate_size)
    groups: list[InputGroup] = []
    for index in range(layers):
        prefix = f"model.layers.{index}"
        groups.extend(
            [
                InputGroup(
                    f"{prefix}.self_attn.k_proj",
                    (f"{prefix}.self_attn.q_proj", f"{prefix}.self_attn.v_proj"),
                    hidden,
                ),
                InputGroup(f"{prefix}.self_attn.o_proj", (), hidden),
                InputGroup(f"{prefix}.mlp.gate_proj", (f"{prefix}.mlp.up_proj",), hidden),
                InputGroup(f"{prefix}.mlp.down_proj", (), intermediate),
            ]
        )
    return groups


def create_plan(config: dict[str, Any]) -> dict[str, Any]:
    path = run_dir(config) / "plan.json"
    fingerprint_fields = {key: value for key, value in config.items() if key != "_config_path"}
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_fields, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if path.exists():
        existing = load_json(path)
        if existing.get("fingerprint") != fingerprint:
            raise RuntimeError("Existing plan belongs to a different experiment configuration/run_dir")
        return existing
    from transformers import AutoConfig

    model_config = AutoConfig.from_pretrained(config["model_path"], local_files_only=True)
    groups = _groups_from_config(model_config)
    per_shard = int(config["groups_per_shard"])
    shards = [groups[index : index + per_shard] for index in range(0, len(groups), per_shard)]
    raw_bytes = sum(group.in_features**2 * 8 + group.in_features * 8 for group in groups)
    root_bytes = sum(group.in_features**2 * 4 + group.in_features * 4 for group in groups)
    value = {
        "created_at_utc": utc_now(),
        "fingerprint": fingerprint,
        "model_type": getattr(model_config, "model_type", None),
        "num_hidden_layers": int(model_config.num_hidden_layers),
        "hidden_size": int(model_config.hidden_size),
        "intermediate_size": int(model_config.intermediate_size),
        "group_count": len(groups),
        "target_module_count": sum(len(group.all_layers) for group in groups),
        "shard_count": len(shards),
        "estimated_raw_bytes": raw_bytes,
        "estimated_root_bytes": root_bytes,
        "shards": [
            [
                {
                    "target": group.target,
                    "shares": list(group.shares),
                    "in_features": group.in_features,
                }
                for group in shard
            ]
            for shard in shards
        ],
    }
    save_json(path, value)
    save_json(run_dir(config) / "config_resolved.json", config)
    log(
        "plan",
        f"groups={len(groups)} targets={value['target_module_count']} shards={len(shards)} "
        f"raw_GiB={raw_bytes / 2**30:.2f} roots_GiB={root_bytes / 2**30:.2f}",
    )
    return value


def groups_for_shard(config: dict[str, Any], shard: int) -> list[InputGroup]:
    plan = create_plan(config)
    if shard < 0 or shard >= plan["shard_count"]:
        raise ValueError(f"shard must be in [0, {plan['shard_count'] - 1}]")
    return [InputGroup(item["target"], tuple(item["shares"]), item["in_features"]) for item in plan["shards"][shard]]


def _model_kwargs(config: dict[str, Any], dtype_name: str, device_map: str | None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype(dtype_name),
        "local_files_only": True,
        "low_cpu_mem_usage": True,
        "_attn_implementation": "eager",
    }
    if device_map:
        kwargs["device_map"] = device_map
        maximum = config.get("max_memory")
        if maximum:
            kwargs["max_memory"] = {int(key) if str(key).isdigit() else key: value for key, value in maximum.items()}
    return kwargs


def load_model(config: dict[str, Any], dtype_name: str, device_map: str | None):
    from transformers import AutoModelForCausalLM

    log("model", f"loading path={config['model_path']} dtype={dtype_name} device_map={device_map or 'none'}")
    model = AutoModelForCausalLM.from_pretrained(
        config["model_path"], **_model_kwargs(config, dtype_name, device_map)
    )
    model.eval()
    model.config.use_cache = False
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    return model


def _input_device(model: torch.nn.Module) -> torch.device:
    return model.get_input_embeddings().weight.device


def _checkpoint_dir(config: dict[str, Any], shard: int, windows: int) -> Path:
    return run_dir(config) / "statistics" / "checkpoints" / f"shard_{shard:03d}" / f"window_{windows:04d}"


def _save_collect_checkpoint(
    config: dict[str, Any],
    shard: int,
    groups: list[InputGroup],
    sums: dict[str, torch.Tensor],
    diagonals: dict[str, torch.Tensor],
    sample_count: int,
    windows: int,
    previous: Path | None,
) -> Path:
    final = _checkpoint_dir(config, shard, windows)
    temporary = final.with_name(f".{final.name}.{uuid.uuid4().hex}.tmp")
    temporary.mkdir(parents=True, exist_ok=False)
    for group in groups:
        save_file(
            {
                "rxx_sum": sums[group.target].contiguous(),
                "diag_sum": diagonals[group.target].contiguous(),
                "sample_count": torch.tensor([sample_count], dtype=torch.int64),
                "windows_completed": torch.tensor([windows], dtype=torch.int64),
            },
            str(temporary / f"{safe_name(group.target)}.safetensors"),
        )
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists():
        shutil.rmtree(final)
    os.replace(temporary, final)
    state = {
        "status": "IN_PROGRESS",
        "updated_at_utc": utc_now(),
        "shard": shard,
        "windows_completed": windows,
        "sample_count": sample_count,
        "checkpoint_dir": str(final),
    }
    save_json(run_dir(config) / "state" / f"collect_shard_{shard:03d}.json", state)
    if previous and previous.exists() and previous != final:
        shutil.rmtree(previous)
    return final


def collect_shard(config: dict[str, Any], shard: int) -> dict[str, Any]:
    groups = groups_for_shard(config, shard)
    frozen = load_file(str(frozen_path(config, "calibration")))
    inputs = frozen["input_ids"]
    masks = frozen["attention_mask"]
    total_windows = int(config["num_calibration_samples"])
    if inputs.shape[0] != total_windows:
        raise RuntimeError(f"Frozen calibration window mismatch: {inputs.shape[0]} != {total_windows}")
    state_path = run_dir(config) / "state" / f"collect_shard_{shard:03d}.json"
    raw_dir = run_dir(config) / "statistics" / "raw"
    if state_path.exists() and load_json(state_path).get("status") == "PASS":
        log("collect", f"shard={shard} already complete")
        return load_json(state_path)
    model = load_model(config, config["profiling_dtype"], config.get("profiling_device_map", "auto"))
    modules = dict(model.named_modules())
    for group in groups:
        module = modules.get(group.target)
        if not isinstance(module, torch.nn.Linear) or int(module.in_features) != group.in_features:
            raise RuntimeError(f"Planned module does not match loaded model: {group.target}")
    sums: dict[str, torch.Tensor] = {}
    diagonals: dict[str, torch.Tensor] = {}
    start_window = 0
    sample_count = 0
    previous: Path | None = None
    if state_path.exists():
        state = load_json(state_path)
        previous = Path(state["checkpoint_dir"])
        start_window = int(state["windows_completed"])
        sample_count = int(state["sample_count"])
        for group in groups:
            tensors = load_file(str(previous / f"{safe_name(group.target)}.safetensors"))
            tensor_window = int(tensors["windows_completed"].item())
            tensor_samples = int(tensors["sample_count"].item())
            if tensor_window != start_window or tensor_samples != sample_count:
                raise RuntimeError(
                    f"Inconsistent collect checkpoint for {group.target}: "
                    f"windows={tensor_window}/{start_window}, samples={tensor_samples}/{sample_count}"
                )
            sums[group.target] = tensors["rxx_sum"]
            diagonals[group.target] = tensors["diag_sum"]
        log("collect", f"resuming shard={shard} from window={start_window}/{total_windows}")
    else:
        for group in groups:
            sums[group.target] = torch.zeros((group.in_features, group.in_features), dtype=torch.float64)
            diagonals[group.target] = torch.zeros(group.in_features, dtype=torch.float32)

    handles = []

    def make_hook(name: str):
        @torch.no_grad()
        def hook(_module, args, _output):
            x = args[0].reshape(-1, args[0].shape[-1]).to(torch.float32)
            delta = (x.transpose(0, 1) @ x).to(device="cpu", dtype=torch.float64)
            sums[name].add_(delta)
            diagonals[name].add_((x.square().sum(dim=0)).to(device="cpu", dtype=torch.float64))

        return hook

    for group in groups:
        handles.append(modules[group.target].register_forward_hook(make_hook(group.target)))
    interval = int(config["checkpoint_every_windows"])
    started = time.monotonic()
    input_device = _input_device(model)
    batch_size = int(config["calibration_batch_size"])
    try:
        with torch.inference_mode():
            for index in range(start_window, total_windows, batch_size):
                completed = min(index + batch_size, total_windows)
                ids = inputs[index:completed].to(input_device)
                mask = masks[index:completed].to(input_device)
                model(input_ids=ids, attention_mask=mask, use_cache=False)
                sample_count += int(mask.sum().item())
                elapsed = time.monotonic() - started
                rate = elapsed / max(1, completed - start_window)
                eta = rate * (total_windows - completed)
                log(
                    "collect",
                    f"shard={shard} window={completed}/{total_windows} elapsed={elapsed:.0f}s eta={eta:.0f}s",
                )
                if completed % interval == 0 and completed < total_windows:
                    previous = _save_collect_checkpoint(
                        config, shard, groups, sums, diagonals, sample_count, completed, previous
                    )
    finally:
        for handle in handles:
            handle.remove()
    raw_dir.mkdir(parents=True, exist_ok=True)
    for group in groups:
        atomic_safetensors(
            raw_dir / f"{safe_name(group.target)}.safetensors",
            {
                "rxx_sum": sums[group.target],
                "diag_sum": diagonals[group.target],
                "sample_count": torch.tensor([sample_count], dtype=torch.int64),
                "windows_completed": torch.tensor([total_windows], dtype=torch.int64),
            },
        )
    result = {
        "status": "PASS",
        "completed_at_utc": utc_now(),
        "shard": shard,
        "groups": [group.target for group in groups],
        "windows_completed": total_windows,
        "sample_count": sample_count,
        "profiling_dtype": config["profiling_dtype"],
    }
    save_json(state_path, result)
    if previous and previous.exists():
        shutil.rmtree(previous)
    log("collect", f"shard={shard} complete groups={len(groups)} samples={sample_count}")
    return result


def compute_roots(config: dict[str, Any], shard: int) -> dict[str, Any]:
    qera = import_official_qera(config)
    groups = groups_for_shard(config, shard)
    raw_dir = run_dir(config) / "statistics" / "raw"
    root_dir = run_dir(config) / "statistics" / "roots"
    root_dir.mkdir(parents=True, exist_ok=True)
    completed = []
    for index, group in enumerate(groups, 1):
        output = root_dir / f"{safe_name(group.target)}.safetensors"
        metadata = output.with_suffix(".json")
        if output.exists() and metadata.exists():
            root_metadata = load_json(metadata)
            raw_path = raw_dir / f"{safe_name(group.target)}.safetensors"
            if root_metadata.get("root_sha256") != sha256_file(output):
                raise RuntimeError(f"Stored root checksum mismatch: {output}")
            if root_metadata.get("raw_sha256") != sha256_file(raw_path):
                raise RuntimeError(f"Raw statistics changed after root computation: {raw_path}")
            completed.append(group.target)
            log("roots", f"shard={shard} group={group.target} already complete")
            continue
        raw_path = raw_dir / f"{safe_name(group.target)}.safetensors"
        tensors = load_file(str(raw_path))
        n = int(tensors["sample_count"].item())
        rxx_sum = tensors["rxx_sum"].numpy()
        rxx_sum = (rxx_sum + rxx_sum.T) * 0.5
        log("roots", f"shard={shard} group={group.target} sqrtm dimension={group.in_features} ({index}/{len(groups)})")
        root = qera["sqrtm_scipy"](rxx_sum)
        root, max_imaginary_raw, max_imaginary_normalized = _real_sqrtm_root(root, n)
        warning_threshold = float(config["sqrtm_max_imaginary"])
        if max_imaginary_raw > warning_threshold:
            log(
                "roots",
                f"shard={shard} group={group.target} sqrtm complex diagnostic "
                f"raw_max_imaginary={max_imaginary_raw:.6g} "
                f"normalized_max_imaginary={max_imaginary_normalized:.6g}; "
                "discarding imaginary component to match official QERA float cast",
            )
        covariance = rxx_sum / n
        denominator = max(float(np.linalg.norm(covariance)), np.finfo(np.float64).eps)
        residual = float(np.linalg.norm(root @ root - covariance) / denominator)
        if not math.isfinite(residual):
            raise RuntimeError(f"Non-finite sqrtm residual for {group.target}: {residual}")
        diag_scale = torch.sqrt(torch.clamp(tensors["diag_sum"] / n, min=0)).to(torch.float32)
        full_scale = torch.from_numpy(root).to(torch.float32)
        atomic_safetensors(output, {"full": full_scale, "diag": diag_scale})
        save_json(
            metadata,
            {
                "status": "PASS",
                "completed_at_utc": utc_now(),
                "target": group.target,
                "shares": list(group.shares),
                "sample_count": n,
                "dimension": group.in_features,
                "full_root_dtype": "float32",
                "diag_root_dtype": "float32",
                "sqrtm_implementation": "official_qera_scipy_blocked",
                "sqrtm_complex_policy": "discard_imaginary_like_official_float_cast",
                "sqrtm_relative_residual": residual,
                "sqrtm_max_imaginary": max_imaginary_raw,
                "sqrtm_max_imaginary_raw": max_imaginary_raw,
                "sqrtm_max_imaginary_normalized": max_imaginary_normalized,
                "sqrtm_imaginary_warning_threshold_raw": warning_threshold,
                "raw_sha256": sha256_file(raw_path),
                "root_sha256": sha256_file(output),
            },
        )
        completed.append(group.target)
        del tensors, rxx_sum, root, covariance, full_scale, diag_scale
        gc.collect()
    result = {"status": "PASS", "completed_at_utc": utc_now(), "shard": shard, "groups": completed}
    save_json(run_dir(config) / "state" / f"roots_shard_{shard:03d}.json", result)
    return result


def _correction_path(config: dict[str, Any], method: str, layer_name: str) -> Path:
    return run_dir(config) / "corrections" / method / f"{safe_name(layer_name)}.safetensors"


def solve_shard(config: dict[str, Any], shard: int) -> dict[str, Any]:
    qera = import_official_qera(config)
    groups = groups_for_shard(config, shard)
    model = load_model(config, config["solve_dtype"], None)
    modules = dict(model.named_modules())
    solve_device = torch.device(config["solve_device"])
    maximum_rank = max(config["ranks"])
    metrics_path = run_dir(config) / "solve_metrics.jsonl"
    solved = []
    for group in groups:
        roots = load_file(str(run_dir(config) / "statistics" / "roots" / f"{safe_name(group.target)}.safetensors"))
        for layer_name in group.all_layers:
            layer = modules.get(layer_name)
            if not isinstance(layer, torch.nn.Linear):
                raise RuntimeError(f"Expected Linear target: {layer_name}")
            layer.to(solve_device)
            for method, root_key in (("diag", "diag"), ("full", "full")):
                output = _correction_path(config, method, layer_name)
                metadata = output.with_suffix(".json")
                if output.exists() and metadata.exists():
                    correction_metadata = load_json(metadata)
                    if correction_metadata.get("sha256") != sha256_file(output):
                        raise RuntimeError(f"Stored correction checksum mismatch: {output}")
                    log("solve", f"{method} {layer_name} already complete")
                    continue
                started = time.monotonic()
                log("solve", f"{method} {layer_name} rank={maximum_rank} starting")
                ab, mse = qera["compute_ab"](
                    layer_name,
                    layer,
                    roots[root_key],
                    qera_layer_config(maximum_rank),
                )
                a = ab[layer_name + ".A"].detach().cpu()
                b = ab[layer_name + ".B"].detach().cpu()
                atomic_safetensors(output, {"A": a, "B": b})
                record = {
                    "status": "PASS",
                    "completed_at_utc": utc_now(),
                    "method": method,
                    "layer": layer_name,
                    "rank": maximum_rank,
                    "mse": mse,
                    "elapsed_seconds": time.monotonic() - started,
                    "official_compute_function": "qera.approximate._compute_scales_and_error_for_fc",
                    "sha256": sha256_file(output),
                }
                save_json(metadata, record)
                metrics_path.parent.mkdir(parents=True, exist_ok=True)
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                log("solve", f"{method} {layer_name} complete mse={mse:.6g}")
            layer.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            solved.append(layer_name)
        del roots
    result = {"status": "PASS", "completed_at_utc": utc_now(), "shard": shard, "layers": solved}
    save_json(run_dir(config) / "state" / f"solve_shard_{shard:03d}.json", result)
    return result


def _configuration_names(config: dict[str, Any]) -> list[tuple[str, str | None, int | None]]:
    values: list[tuple[str, str | None, int | None]] = [("BF16", None, None), ("W4_MXINT", "wq", None)]
    for method in ("diag", "full"):
        for rank in config["ranks"]:
            values.append((f"QERA_{method.upper()}_R{rank}", method, int(rank)))
    return values


def _quantize_model(model: torch.nn.Module, quantizer) -> list[str]:
    names = []
    suffixes = (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    )
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and name.endswith(suffixes):
                module.weight.data = quantizer(module.weight.data, width=4, block_size=32, block_axis=-1)
                names.append(name)
    return names


def _attach_corrections(model: torch.nn.Module, config: dict[str, Any], method: str, rank: int):
    handles = []
    modules = dict(model.named_modules())
    plan = create_plan(config)
    layer_names = [name for shard in plan["shards"] for group in shard for name in (group["target"], *group["shares"])]
    for name in layer_names:
        module = modules[name]
        tensors = load_file(str(_correction_path(config, method, name)))
        a = tensors["A"][:, :rank].to(device=module.weight.device, dtype=module.weight.dtype)
        b = tensors["B"][:rank, :].to(device=module.weight.device, dtype=module.weight.dtype)

        def make_hook(a_local: torch.Tensor, b_local: torch.Tensor):
            @torch.no_grad()
            def hook(_module, args, output):
                return output + (args[0] @ a_local) @ b_local

            return hook

        handles.append(module.register_forward_hook(make_hook(a, b)))
    return handles


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    valid_lines = []
    for index, line in enumerate(lines):
        if line.strip():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                if index != len(lines) - 1:
                    raise
                log("resume", f"discarding incomplete final JSONL record: {path}")
                path.write_text("".join(valid_lines), encoding="utf-8")
                break
            records.append(record)
            valid_lines.append(line)
    return records


def _chunked_window_nll(logits, ids, mask, chunk_tokens: int):
    """Keep full-vocabulary CE and the original per-window reduction order.

    Slice token rows before CE, avoiding the full shifted-logits copy and
    full-batch log-softmax workspace. The vocabulary is never split.
    """
    if chunk_tokens <= 0:
        raise ValueError("ce_chunk_tokens must be positive")
    labels = ids[:, 1:].to(logits.device)
    valid = mask[:, 1:].to(logits.device).bool()
    losses = torch.empty(labels.shape, dtype=logits.dtype, device=logits.device)
    for row in range(labels.shape[0]):
        for start in range(0, labels.shape[1], chunk_tokens):
            end = min(start + chunk_tokens, labels.shape[1])
            losses[row, start:end] = F.cross_entropy(
                logits[row, start:end, :], labels[row, start:end], reduction="none"
            )
    return [
        (float(losses[row][valid[row]].sum().item()), int(valid[row].sum().item()))
        for row in range(labels.shape[0])
    ]


def _evaluation_load_config(config: dict[str, Any], dual_gpu: bool) -> dict[str, Any]:
    # Do not change fields participating in the original experiment fingerprint.
    loading = dict(config)
    if dual_gpu:
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            raise RuntimeError("--dual-gpu requires two visible CUDA GPUs")
        loading["max_memory"] = {0: "10GiB", 1: "10GiB", "cpu": "180GiB"}
        loading["eval_device_map"] = "balanced"
    return loading


def _reset_evaluation_memory_peaks(gpu_indices: list[int]) -> None:
    for gpu_index in gpu_indices:
        # torch 2.3 can reject resetPeakMemoryStats before the CUDA caching
        # allocator is initialized. A real allocation also initializes a
        # previously unused secondary device. Keep model loading in the peak.
        probe = torch.empty(1, dtype=torch.uint8, device=f"cuda:{gpu_index}")
        del probe
        torch.cuda.reset_peak_memory_stats(gpu_index)


def _evaluate_one(
    config: dict[str, Any], name: str, method: str | None, rank: int | None, windows,
    *, dual_gpu: bool = False, batch_size: int | None = None, ce_chunk_tokens: int = 2048,
):
    output = run_dir(config) / "evaluation" / "per_window" / f"{safe_name(name)}.jsonl"
    existing = _read_jsonl(output)
    for expected_window, record in enumerate(existing):
        if record.get("configuration") != name or int(record.get("window", -1)) != expected_window:
            raise RuntimeError(f"Non-contiguous evaluation checkpoint: {output}")
    start = len(existing)
    if start >= windows["input_ids"].shape[0]:
        log("evaluate", f"configuration={name} already complete windows={start}")
        return existing
    loading = _evaluation_load_config(config, dual_gpu)
    gpu_indices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    _reset_evaluation_memory_peaks(gpu_indices)
    model = load_model(loading, config["eval_dtype"], loading.get("eval_device_map", "auto"))
    device_map = getattr(model, "hf_device_map", {})
    log("evaluate", f"configuration={name} device_map={device_map}")
    if dual_gpu:
        placements = {str(value) for value in device_map.values()}
        if not placements or not placements <= {"0", "1", "cuda:0", "cuda:1"}:
            raise RuntimeError(f"Dual-GPU evaluation requires GPU-resident weights; got {device_map}")
    save_json(
        run_dir(config) / "evaluation" / "runtime" / f"{safe_name(name)}_{uuid.uuid4().hex}.json",
        {
            "created_at_utc": utc_now(), "configuration": name, "start_window": start,
            "batch_size": batch_size if batch_size is not None else int(config["eval_batch_size"]),
            "ce_chunk_tokens": ce_chunk_tokens, "dual_gpu": dual_gpu,
            "hf_device_map": {key: str(value) for key, value in device_map.items()},
            "max_memory": loading.get("max_memory"), "torch": torch.__version__,
        },
    )
    handles = []
    if method is not None:
        qera = import_official_qera(config)
        count = _quantize_model(model, qera["mxint_quantizer"])
        expected_count = int(create_plan(config)["target_module_count"])
        if len(count) != expected_count:
            raise RuntimeError(f"Quantized target count mismatch: {len(count)} != {expected_count}")
        log("evaluate", f"configuration={name} quantized_modules={len(count)}")
        if method in ("diag", "full"):
            handles = _attach_corrections(model, config, method, int(rank))
    input_device = _input_device(model)
    total = int(windows["input_ids"].shape[0])
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    batch_size = batch_size if batch_size is not None else int(config["eval_batch_size"])
    log("evaluate", f"configuration={name} starting window={start}/{total} batch_size={batch_size} ce_chunk_tokens={ce_chunk_tokens}")
    try:
        with output.open("a", encoding="utf-8") as handle, torch.inference_mode():
            for index in range(start, total, batch_size):
                completed = min(index + batch_size, total)
                ids = windows["input_ids"][index:completed].to(input_device)
                mask = windows["attention_mask"][index:completed].to(input_device)
                logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
                window_metrics = _chunked_window_nll(logits, ids, mask, ce_chunk_tokens)
                # Release the previous logits before starting the next forward.
                del logits, ids, mask
                for offset, window_index in enumerate(range(index, completed)):
                    nll_sum, tokens = window_metrics[offset]
                    record = {
                        "configuration": name,
                        "window": window_index,
                        "nll_sum": nll_sum,
                        "tokens": tokens,
                    }
                    handle.write(json.dumps(record) + "\n")
                    existing.append(record)
                handle.flush()
                os.fsync(handle.fileno())
                elapsed = time.monotonic() - started
                eta = elapsed / max(1, completed - start) * (total - completed)
                memory_log = " ".join(
                    f"gpu{gpu_index}_peak_allocated_GiB={torch.cuda.max_memory_allocated(gpu_index) / 2**30:.2f} "
                    f"gpu{gpu_index}_peak_reserved_GiB={torch.cuda.max_memory_reserved(gpu_index) / 2**30:.2f}"
                    for gpu_index in gpu_indices
                )
                log(
                    "evaluate",
                    f"configuration={name} window={completed}/{total} elapsed={elapsed:.0f}s eta={eta:.0f}s {memory_log}",
                )
    finally:
        for registered in handles:
            registered.remove()
        handles.clear()
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return existing


def write_evaluation_tables(config: dict[str, Any]) -> dict[str, Any]:
    directory = run_dir(config) / "evaluation"
    per_window_path = directory / "wikitext2_per_window.csv"
    summary_path = directory / "ppl_summary_wikitext2.csv"
    all_records = []
    summaries = []
    for name, method, rank in _configuration_names(config):
        records = _read_jsonl(directory / "per_window" / f"{safe_name(name)}.jsonl")
        if not records:
            continue
        all_records.extend(records)
        nll = sum(float(record["nll_sum"]) for record in records)
        tokens = sum(int(record["tokens"]) for record in records)
        summaries.append(
            {
                "configuration": name,
                "method": method or "teacher",
                "rank": "" if rank is None else rank,
                "windows": len(records),
                "prediction_tokens": tokens,
                "nll_sum": nll,
                "ppl": math.exp(nll / tokens),
            }
        )
    directory.mkdir(parents=True, exist_ok=True)
    if all_records:
        with per_window_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["configuration", "window", "nll_sum", "tokens"])
            writer.writeheader()
            writer.writerows(all_records)
    if summaries:
        with summary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
            writer.writeheader()
            writer.writerows(summaries)
    result = {
        "status": "PASS" if len(summaries) == len(_configuration_names(config)) else "INCOMPLETE",
        "completed_at_utc": utc_now(),
        "configurations": len(summaries),
        "summary_path": str(summary_path),
        "per_window_path": str(per_window_path),
    }
    save_json(directory / "summary.json", result)
    return result


def evaluate(
    config: dict[str, Any], only: str | None = None, *, dual_gpu: bool = False,
    batch_size: int | None = None, ce_chunk_tokens: int = 2048,
) -> dict[str, Any]:
    if batch_size is not None and batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if ce_chunk_tokens <= 0:
        raise ValueError("ce_chunk_tokens must be positive")
    create_plan(config)  # Validate the saved plan without changing the YAML.
    _evaluation_load_config(config, dual_gpu)  # Fail before model loading if a GPU is missing.
    windows = load_file(str(frozen_path(config, "wikitext2")))
    matched = False
    for name, method, rank in _configuration_names(config):
        if only and name != only:
            continue
        matched = True
        _evaluate_one(
            config, name, method, rank, windows, dual_gpu=dual_gpu,
            batch_size=batch_size, ce_chunk_tokens=ce_chunk_tokens,
        )
        write_evaluation_tables(config)
    if only and not matched:
        choices = [item[0] for item in _configuration_names(config)]
        raise ValueError(f"Unknown configuration: {only}; choose from {choices}")
    return write_evaluation_tables(config)


def doctor(config: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    try:
        qera = import_official_qera(config)
        checks["official_qera"] = {"status": "PASS", "commit": qera["commit"]}
    except Exception as error:
        checks["official_qera"] = {"status": "FAIL", "error": str(error)}
    try:
        from transformers import AutoConfig

        model_config = AutoConfig.from_pretrained(config["model_path"], local_files_only=True)
        checks["model"] = {
            "status": "PASS",
            "path": config["model_path"],
            "model_type": model_config.model_type,
        }
    except Exception as error:
        checks["model"] = {"status": "FAIL", "path": config["model_path"], "error": str(error)}
    for role in ("calibration", "wikitext2"):
        path = frozen_path(config, role)
        checks[role] = {"status": "PASS" if path.is_file() else "MISSING", "path": str(path)}
    try:
        plan = create_plan(config)
        checks["storage_plan"] = {
            "status": "PASS",
            "raw_GiB": plan["estimated_raw_bytes"] / 2**30,
            "roots_GiB": plan["estimated_root_bytes"] / 2**30,
            "shards": plan["shard_count"],
        }
    except Exception as error:
        checks["storage_plan"] = {"status": "FAIL", "error": str(error)}
    required_failures = [
        checks["official_qera"]["status"],
        checks["model"]["status"],
        checks["storage_plan"]["status"],
    ]
    status = "FAIL" if "FAIL" in required_failures else "PASS"
    if status == "PASS" and any(checks[role]["status"] == "MISSING" for role in ("calibration", "wikitext2")):
        status = "NEEDS_DATA"
    result = {
        "status": status,
        "captured_at_utc": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "checks": checks,
    }
    save_json(run_dir(config) / "doctor.json", result)
    return result


def run_all(config: dict[str, Any]) -> dict[str, Any]:
    for role in ("calibration", "wikitext2"):
        if not frozen_path(config, role).is_file():
            raise RuntimeError(f"Frozen {role} data missing; run prepare-data explicitly first")
    plan = create_plan(config)
    for shard in range(plan["shard_count"]):
        collect_shard(config, shard)
        compute_roots(config, shard)
        solve_shard(config, shard)
    return evaluate(config)
