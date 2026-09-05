from __future__ import annotations

import contextlib
import gc
import json
import math
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import psutil
import torch
import torch.nn.functional as F
from torch import nn

from .config import output_root
from .data import load_windows
from .modeling import base_model, load_model, output_embedding
from .quantization import quant_paths, reconstruct
from .utils import (
    atomic_safetensors,
    canonical_sha256,
    deterministic_runtime,
    ensure_layout,
    get_module,
    heartbeat,
    load_safetensors,
    log,
    progress_status,
    safe_name,
    save_json,
    sha256_file,
    tensor_sha256,
    utc_now,
)


@dataclass
class FullAccumulator:
    name: str
    d_in: int
    d_out: int
    a_sum: torch.Tensor = field(init=False)
    g_sum: torch.Tensor = field(init=False)
    a_direct_sum: torch.Tensor = field(init=False)
    g_direct_sum: torch.Tensor = field(init=False)
    a_count: int = 0
    g_count: int = 0

    def __post_init__(self) -> None:
        self.a_sum = torch.zeros((self.d_in, self.d_in), dtype=torch.float64)
        self.g_sum = torch.zeros((self.d_out, self.d_out), dtype=torch.float64)
        self.a_direct_sum = torch.zeros(self.d_in, dtype=torch.float64)
        self.g_direct_sum = torch.zeros(self.d_out, dtype=torch.float64)

    @property
    def dense_bytes(self) -> int:
        return sum(value.numel() * value.element_size() for value in (self.a_sum, self.g_sum, self.a_direct_sum, self.g_direct_sum))


class FullCollector:
    def __init__(self, model: nn.Module, names: list[str], *, chunk_size: int, expected_dtype: torch.dtype) -> None:
        self.names = names
        self.chunk_size = chunk_size
        self.expected_dtype = expected_dtype
        self.modules: dict[str, nn.Linear] = {}
        self.acc: dict[str, FullAccumulator] = {}
        for name in names:
            module = get_module(model, name)
            if not isinstance(module, nn.Linear):
                raise TypeError(name)
            self.modules[name] = module
            self.acc[name] = FullAccumulator(name, module.in_features, module.out_features)
        self.active_mask: torch.Tensor | None = None
        self.forward_seen: set[str] = set()
        self.gradient_seen: set[str] = set()
        self.handles: list[Any] = []

    def _accumulate(self, item: FullAccumulator, kind: str, rows: torch.Tensor) -> None:
        selected = rows.detach().float().contiguous()
        gram = torch.zeros((selected.shape[1], selected.shape[1]), dtype=torch.float32, device=selected.device)
        direct = torch.zeros(selected.shape[1], dtype=torch.float32, device=selected.device)
        for start in range(0, selected.shape[0], self.chunk_size):
            block = selected[start : start + self.chunk_size]
            gram.add_(block.T @ block)
            direct.add_(torch.sum(block * block, dim=0, dtype=torch.float32))
        gram64 = gram.cpu().double()
        direct64 = direct.cpu().double()
        count = int(selected.shape[0])
        if kind == "A":
            item.a_sum.add_(gram64)
            item.a_direct_sum.add_(direct64)
            item.a_count += count
        elif kind == "G":
            item.g_sum.add_(gram64)
            item.g_direct_sum.add_(direct64)
            item.g_count += count
        else:
            raise ValueError(kind)
        del selected, gram, direct, gram64, direct64

    def _hook(self, name: str) -> Callable[..., None]:
        item = self.acc[name]

        def hook(_module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            if self.active_mask is None or name in self.forward_seen:
                raise RuntimeError(f"Invalid or repeated forward hook: {name}")
            if not inputs or not isinstance(inputs[0], torch.Tensor) or not isinstance(output, torch.Tensor):
                raise TypeError(f"Unexpected Linear hook signature: {name}")
            x = inputs[0]
            if x.dtype != self.expected_dtype or output.dtype != self.expected_dtype:
                raise RuntimeError(f"Activation dtype mismatch: {name}; input={x.dtype}, output={output.dtype}")
            if x.shape[:-1] != self.active_mask.shape or output.shape[:-1] != self.active_mask.shape:
                raise RuntimeError(f"Activation shape mismatch: {name}")
            self._accumulate(item, "A", x[self.active_mask].reshape(-1, item.d_in))
            self.forward_seen.add(name)

            def gradient_hook(gradient: torch.Tensor) -> None:
                if self.active_mask is None or name in self.gradient_seen:
                    raise RuntimeError(f"Invalid or repeated gradient hook: {name}")
                if gradient.dtype != self.expected_dtype:
                    raise RuntimeError(f"Gradient dtype mismatch: {name}; gradient={gradient.dtype}")
                self._accumulate(item, "G", gradient[self.active_mask].reshape(-1, item.d_out))
                self.gradient_seen.add(name)

            output.register_hook(gradient_hook)
            return None

        return hook

    def __enter__(self) -> "FullCollector":
        self.handles = [self.modules[name].register_forward_hook(self._hook(name)) for name in self.names]
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def exact_ce_hidden_gradient(
    hidden: torch.Tensor,
    output_weight: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    chunk_size: int,
) -> tuple[torch.Tensor, float, int]:
    if hidden.ndim != 3 or hidden.shape[0] != 1:
        raise ValueError("Expected hidden state [1, sequence, width]")
    valid = attention_mask[:, :-1].bool() & attention_mask[:, 1:].bool()
    selected = torch.nonzero(valid.reshape(-1), as_tuple=False).flatten()
    source = hidden[:, :-1].reshape(-1, hidden.shape[-1])
    targets = input_ids[:, 1:].reshape(-1)
    grad_source = torch.zeros_like(source)
    loss_sum = 0.0
    with torch.no_grad():
        for start in range(0, selected.numel(), chunk_size):
            positions = selected[start : start + chunk_size]
            logits = F.linear(source.index_select(0, positions), output_weight)
            log_probs = F.log_softmax(logits.float(), dim=-1)
            labels = targets.index_select(0, positions)
            loss_sum += float((-log_probs.gather(1, labels[:, None])).double().sum().item())
            grad_logits = log_probs.exp()
            grad_logits[torch.arange(labels.numel(), device=hidden.device), labels] -= 1
            grad_source.index_copy_(0, positions, (grad_logits.to(output_weight.dtype) @ output_weight).to(hidden.dtype))
    gradient = torch.zeros_like(hidden)
    gradient[:, :-1] = grad_source.reshape_as(hidden[:, :-1])
    if not bool(torch.isfinite(gradient).all().item()) or not math.isfinite(loss_sum):
        raise FloatingPointError("Nonfinite CE hidden gradient")
    return gradient, loss_sum, int(selected.numel())


def make_shard_plan(config: Mapping[str, Any]) -> dict[str, Any]:
    root = output_root(config)
    manifest_path = root / "module_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("module_manifest.json is absent; run quantize first")
    modules = json.loads(manifest_path.read_text(encoding="utf-8"))["modules"]
    limit = int(float(config["runtime"]["max_dense_ram_gib_per_shard"]) * 1024**3)
    if limit <= 0:
        raise ValueError("runtime.max_dense_ram_gib_per_shard must be positive")
    # First-fit decreasing keeps dense A/G RAM under the configured cap while
    # reducing repeated model forward/backward passes.
    ordered = sorted(modules, key=lambda row: int(row["estimated_dense_accumulator_bytes"]), reverse=True)
    shards: list[dict[str, Any]] = []
    for row in ordered:
        size = int(row["estimated_dense_accumulator_bytes"])
        target = next((shard for shard in shards if int(shard["estimated_dense_accumulator_bytes"]) + size <= limit), None)
        if target is None:
            target = {"modules": [], "estimated_dense_accumulator_bytes": 0, "estimated_raw_statistics_bytes": 0}
            shards.append(target)
        target["modules"].append(row["module"])
        target["estimated_dense_accumulator_bytes"] += size
        target["estimated_raw_statistics_bytes"] += int(
            row.get("estimated_raw_statistics_bytes", size + 4 * int(row["weight_parameters"]))
        )
    for index, shard in enumerate(shards):
        shard["index"] = index
        shard["module_count"] = len(shard["modules"])
        shard["over_configured_limit"] = int(shard["estimated_dense_accumulator_bytes"]) > limit
    retained_raw_bytes = sum(int(shard["estimated_raw_statistics_bytes"]) for shard in shards)
    disk_free_bytes = shutil.disk_usage(root).free
    retain_raw = not bool(config["runtime"].get("cleanup_raw_after_solve", False))
    warnings = []
    if retain_raw and retained_raw_bytes > disk_free_bytes:
        warnings.append(
            "Estimated retained raw A/G statistics exceed the currently free space on the RUN_DIR filesystem"
        )
    result = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "max_dense_ram_bytes_per_shard": limit,
        "raw_statistics_retained_after_solve": retain_raw,
        "estimated_retained_raw_statistics_bytes": retained_raw_bytes if retain_raw else 0,
        "disk_free_bytes_at_plan": disk_free_bytes,
        "warnings": warnings,
        "shard_count": len(shards),
        "shards": shards,
    }
    save_json(root / "state" / "shard_plan.json", result)
    return result


def _plan_shard(root: Path, shard_index: int) -> dict[str, Any]:
    path = root / "state" / "shard_plan.json"
    if not path.is_file():
        raise FileNotFoundError("shard_plan.json is absent; run plan first")
    plan = json.loads(path.read_text(encoding="utf-8"))
    try:
        return plan["shards"][shard_index]
    except IndexError as error:
        raise ValueError(f"Invalid shard index: {shard_index}; available=0..{plan['shard_count'] - 1}") from error


def collect_shard(config: Mapping[str, Any], shard_index: int) -> dict[str, Any]:
    root = output_root(config)
    ensure_layout(root)
    state_path = root / "state" / f"collect_shard_{shard_index:04d}.json"
    if state_path.is_file():
        existing = json.loads(state_path.read_text(encoding="utf-8"))
        if existing.get("status") == "PASS":
            return existing
    shard = _plan_shard(root, shard_index)
    names = [str(name) for name in shard["modules"]]
    log(root, f"shard={shard_index} loading frozen calibration windows", "collect")
    windows = load_windows(config, "calibration")
    deterministic_runtime(int(config["runtime"]["deterministic_seed"]), bool(config["runtime"].get("allow_tf32", False)))
    log(root, f"shard={shard_index} loading model={config['model']['name_or_path']}", "collect")
    with heartbeat(root, f"shard={shard_index} loading model", "collect"):
        model, _, device = load_model(config, require_input_grads=True)
    log(root, f"shard={shard_index} model ready device={device}", "collect")
    decoder = base_model(model, config)
    lm_weight = output_embedding(model).weight
    expected_dtype = lm_weight.dtype
    collector = FullCollector(
        model,
        names,
        chunk_size=int(config["statistics"]["gram_token_chunk_size"]),
        expected_dtype=expected_dtype,
    )
    total_tokens = 0
    teacher_nll = 0.0
    started = time.time()
    log(root, f"collecting shard={shard_index} modules={len(names)} windows={len(windows)}", "collect")
    with collector:
        for index, window in enumerate(windows):
            log(
                root,
                f"shard={shard_index} starting window={index + 1}/{len(windows)} progress={progress_status(index, len(windows), started)}",
                "collect",
            )
            ids = window["input_ids"].to(device)
            mask = window["attention_mask"].to(device).bool()
            valid = mask[:, :-1] & mask[:, 1:]
            active_mask = torch.zeros_like(mask)
            active_mask[:, :-1] = valid
            collector.active_mask = active_mask
            collector.forward_seen.clear()
            collector.gradient_seen.clear()
            save_context = (
                torch.autograd.graph.save_on_cpu(pin_memory=False, device_type="cuda")
                if device.type == "cuda"
                else contextlib.nullcontext()
            )
            with heartbeat(root, f"shard={shard_index} window={index + 1}/{len(windows)} forward-backward", "collect"):
                with save_context:
                    outputs = decoder(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True)
                    hidden = outputs.last_hidden_state
                    if collector.forward_seen != set(names):
                        raise RuntimeError(f"Forward coverage mismatch: missing={set(names) - collector.forward_seen}")
                    gradient, nll, token_count = exact_ce_hidden_gradient(
                        hidden,
                        lm_weight,
                        ids,
                        mask,
                        chunk_size=int(config["runtime"]["lm_head_chunk_size"]),
                    )
                    hidden.backward(gradient)
            if collector.gradient_seen != set(names):
                raise RuntimeError(f"Gradient coverage mismatch: missing={set(names) - collector.gradient_seen}")
            if any(parameter.grad is not None for parameter in model.parameters()):
                raise RuntimeError("Frozen model parameter accumulated a gradient")
            total_tokens += token_count
            teacher_nll += nll
            collector.active_mask = None
            log(
                root,
                f"shard={shard_index} completed window tokens={token_count} progress={progress_status(index + 1, len(windows), started)}",
                "collect",
            )
            del ids, mask, valid, active_mask, outputs, hidden, gradient
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    ordered_window_hash = canonical_sha256([window["window_hash"] for window in windows])
    block_size = int(config["quantization"]["block_size"])
    raw_rows = []
    for name in names:
        item = collector.acc[name]
        if item.a_count != total_tokens or item.g_count != total_tokens:
            raise RuntimeError(f"Statistic token count mismatch: {name}")
        module = get_module(model, name)
        reference_bf16 = module.weight.detach().cpu().to(torch.bfloat16).contiguous()
        reference_fp32 = reference_bf16.float().contiguous()
        quant_artifact, quant_metadata_path = quant_paths(root, name)
        quant = load_safetensors(quant_artifact)
        quant_meta = json.loads(quant_metadata_path.read_text(encoding="utf-8"))
        if tensor_sha256(reference_bf16) != quant_meta["reference_weight_bf16_sha256"]:
            raise RuntimeError(f"Checkpoint changed after quantization: {name}")
        wq_fp32 = reconstruct(quant["codes"], quant["exponents"], reference_fp32.shape[1], block_size)
        error = (reference_fp32 - wq_fp32).contiguous()
        artifact = root / "statistics" / "raw" / f"{safe_name(name)}.safetensors"
        atomic_safetensors(
            artifact,
            {
                "a_sum": item.a_sum,
                "g_sum": item.g_sum,
                "a_direct_sum": item.a_direct_sum,
                "g_direct_sum": item.g_direct_sum,
                "error_fp32": error,
            },
        )
        metadata = {
            "schema_version": 1,
            "module": name,
            "shape": [item.d_out, item.d_in],
            "valid_prediction_token_count": total_tokens,
            "ordered_window_set_sha256": ordered_window_hash,
            "a_sum_sha256": tensor_sha256(item.a_sum),
            "g_sum_sha256": tensor_sha256(item.g_sum),
            "a_direct_sum_sha256": tensor_sha256(item.a_direct_sum),
            "g_direct_sum_sha256": tensor_sha256(item.g_direct_sum),
            "error_fp32_sha256": tensor_sha256(error),
            "artifact": str(artifact),
            "artifact_sha256": sha256_file(artifact),
        }
        save_json(artifact.with_suffix(".json"), metadata)
        raw_rows.append(metadata)
        log(root, f"saved raw A/G {name}", "collect")
    result = {
        "status": "PASS",
        "completed_at_utc": utc_now(),
        "shard_index": shard_index,
        "modules": names,
        "module_count": len(names),
        "windows": len(windows),
        "valid_prediction_token_count": total_tokens,
        "teacher_nll_sum": teacher_nll,
        "dense_accumulator_bytes": sum(item.dense_bytes for item in collector.acc.values()),
        "peak_process_rss_bytes": psutil.Process().memory_info().rss,
        "elapsed_seconds": time.time() - started,
        "raw_rows_sha256": canonical_sha256(raw_rows),
    }
    save_json(state_path, result)
    del model, collector
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result
