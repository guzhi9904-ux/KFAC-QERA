from __future__ import annotations

import gc
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import torch
import torch.nn.functional as F

from .config import output_root
from .data import load_windows
from .modeling import base_model, load_model, output_embedding
from .quantization import quant_paths
from .utils import (
    canonical_sha256,
    deterministic_runtime,
    get_module,
    load_safetensors,
    log,
    read_csv,
    safe_name,
    save_csv,
    save_json,
    sha256_file,
    tensor_sha256,
    upsert_csv,
    utc_now,
)


def configurations(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = [
        {"configuration": "BF16_TEACHER", "method": "BF16_TEACHER", "a_level": "-", "g_level": "-", "rank": 0},
        {"configuration": "MXINT4_WQ", "method": "MXINT4_WQ", "a_level": "-", "g_level": "-", "rank": 0},
    ]
    for rank in config["statistics"]["ranks"]:
        for method in config["statistics"]["methods"]:
            method = str(method)
            rows.append(
                {
                    "configuration": f"{method}_R{rank}",
                    "method": method,
                    "a_level": "diag" if method.startswith("AD") else "full",
                    "g_level": {"GI": "I", "GD": "diag", "GF": "full"}[method.split("_")[1]],
                    "rank": int(rank),
                }
            )
    return rows


def _module_names(root: Path) -> list[str]:
    path = root / "module_manifest.json"
    if not path.is_file():
        raise FileNotFoundError("module_manifest.json is absent; run quantize first")
    return [str(row["module"]) for row in json.loads(path.read_text(encoding="utf-8"))["modules"]]


def install_quantized_weights(model: torch.nn.Module, root: Path, names: list[str]) -> str:
    rows = []
    with torch.no_grad():
        for name in names:
            module = get_module(model, name)
            artifact, metadata_path = quant_paths(root, name)
            tensors = load_safetensors(artifact)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if sha256_file(artifact) != metadata["artifact_sha256"]:
                raise RuntimeError(f"Quantization artifact hash mismatch: {name}")
            if tensor_sha256(module.weight.detach().cpu().to(torch.bfloat16)) != metadata["reference_weight_bf16_sha256"]:
                raise RuntimeError(f"Checkpoint differs from quantization reference: {name}")
            module.weight.copy_(tensors["wq_bf16"].to(device=module.weight.device, dtype=module.weight.dtype))
            rows.append({"module": name, "artifact_sha256": metadata["artifact_sha256"]})
    return canonical_sha256(rows)


class LowRankBranches:
    def __init__(self, model: torch.nn.Module, root: Path, names: list[str], method: str, rank: int, maximum_rank: int) -> None:
        self.model = model
        self.root = root
        self.names = names
        self.method = method
        self.rank = rank
        self.maximum_rank = maximum_rank
        self.handles: list[Any] = []
        self.rows: list[dict[str, Any]] = []

    def __enter__(self) -> "LowRankBranches":
        for name in self.names:
            module = get_module(self.model, name)
            artifact = self.root / "corrections" / f"{safe_name(name)}__{self.method}__r{self.maximum_rank}.safetensors"
            metadata_path = artifact.with_suffix(".json")
            tensors = load_safetensors(artifact)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if sha256_file(artifact) != metadata["artifact_sha256"]:
                raise RuntimeError(f"Correction artifact hash mismatch: {name}/{self.method}")
            left = tensors["left"][:, : self.rank].to(module.weight.device, dtype=module.weight.dtype)
            right_t = tensors["right"][:, : self.rank].T.contiguous().to(module.weight.device, dtype=module.weight.dtype)

            def hook(_module: torch.nn.Module, inputs: tuple[Any, ...], output: torch.Tensor, left=left, right_t=right_t) -> torch.Tensor:
                return output + F.linear(F.linear(inputs[0], right_t), left)

            self.handles.append(module.register_forward_hook(hook))
            self.rows.append({"module": name, "artifact_sha256": metadata["artifact_sha256"], "rank": self.rank})
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    @property
    def joint_hash(self) -> str:
        return canonical_sha256(self.rows)


@torch.no_grad()
def nll_sum(model: torch.nn.Module, config: Mapping[str, Any], window: Mapping[str, Any], device: torch.device) -> tuple[float, int]:
    ids = window["input_ids"].to(device)
    mask = window["attention_mask"].to(device).bool()
    hidden = base_model(model, config)(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True).last_hidden_state
    output_weight = output_embedding(model).weight
    valid = mask[:, :-1] & mask[:, 1:]
    selected = torch.nonzero(valid.reshape(-1), as_tuple=False).flatten()
    source = hidden[:, :-1].reshape(-1, hidden.shape[-1])
    targets = ids[:, 1:].reshape(-1)
    total = 0.0
    chunk_size = int(config["runtime"]["lm_head_chunk_size"])
    for start in range(0, selected.numel(), chunk_size):
        positions = selected[start : start + chunk_size]
        logits = F.linear(source.index_select(0, positions), output_weight)
        log_probs = F.log_softmax(logits.float(), dim=-1)
        labels = targets.index_select(0, positions)
        total += float((-log_probs.gather(1, labels[:, None])).double().sum().item())
    count = int(selected.numel())
    if count <= 0 or not math.isfinite(total):
        raise FloatingPointError("Invalid evaluation NLL")
    return total, count


def _evaluate_configuration(
    model: torch.nn.Module,
    config: Mapping[str, Any],
    role: str,
    row: Mapping[str, Any],
    windows: list[dict[str, Any]],
    device: torch.device,
    metrics_path: Path,
    model_hash: str,
) -> None:
    root = output_root(config)
    existing = read_csv(metrics_path)
    for window in windows:
        matched = [
            item
            for item in existing
            if item.get("configuration") == row["configuration"] and int(item.get("window_id", -1)) == window["window_id"]
        ]
        if len(matched) == 1 and matched[0].get("window_hash") == window["window_hash"] and matched[0].get("model_hash") == model_hash:
            continue
        started = time.time()
        nll, token_count = nll_sum(model, config, window, device)
        result = {
            "dataset": role,
            "configuration": row["configuration"],
            "method": row["method"],
            "a_level": row["a_level"],
            "g_level": row["g_level"],
            "rank": row["rank"],
            "window_id": window["window_id"],
            "window_hash": window["window_hash"],
            "token_count": token_count,
            "nll_sum": nll,
            "nll_mean": nll / token_count,
            "model_hash": model_hash,
            "elapsed_seconds": time.time() - started,
            "complete": True,
        }
        upsert_csv(metrics_path, [result], ("dataset", "configuration", "window_id"))
        existing = read_csv(metrics_path)
        log(root, f"{role} {row['configuration']} window={window['window_id'] + 1}/{len(windows)} nll={nll/token_count:.8f}", "evaluate")


def summarize_ppl(metrics_path: Path, destination: Path) -> list[dict[str, Any]]:
    frame = pd.read_csv(metrics_path)
    rows = []
    for keys, group in frame.groupby(["dataset", "configuration", "method", "a_level", "g_level", "rank"], dropna=False):
        dataset, configuration, method, a_level, g_level, rank = keys
        nll = float(group["nll_sum"].sum())
        tokens = int(group["token_count"].sum())
        rows.append(
            {
                "dataset": dataset,
                "configuration": configuration,
                "method": method,
                "a_level": a_level,
                "g_level": g_level,
                "rank": int(rank),
                "windows": int(group["window_id"].nunique()),
                "token_count": tokens,
                "aggregate_nll_sum": nll,
                "aggregate_mean_nll": nll / tokens,
                "perplexity": math.exp(nll / tokens),
                "aggregation": "exp(sum_window_nll_sum/sum_window_token_count)",
            }
        )
    save_csv(destination, rows)
    return rows


def evaluate_dataset(config: Mapping[str, Any], role: str) -> dict[str, Any]:
    if role not in {"wikitext2", "c4"}:
        raise ValueError("role must be wikitext2 or c4")
    root = output_root(config)
    names = _module_names(root)
    windows = load_windows(config, role)
    metrics_path = root / "evaluation" / f"{role}_per_window.csv"
    deterministic_runtime(int(config["runtime"]["deterministic_seed"]), bool(config["runtime"].get("allow_tf32", False)))
    sequence = configurations(config)
    teacher_row = sequence[0]
    teacher, _, device = load_model(config, require_input_grads=False)
    teacher_hash = canonical_sha256({"kind": "teacher", "model": config["model"]["name_or_path"]})
    _evaluate_configuration(teacher, config, role, teacher_row, windows, device, metrics_path, teacher_hash)
    del teacher
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model, _, device = load_model(config, require_input_grads=False)
    quant_hash = install_quantized_weights(model, root, names)
    _evaluate_configuration(model, config, role, sequence[1], windows, device, metrics_path, quant_hash)
    maximum_rank = int(config["statistics"]["maximum_rank"])
    for row in sequence[2:]:
        with LowRankBranches(model, root, names, str(row["method"]), int(row["rank"]), maximum_rank) as branches:
            model_hash = canonical_sha256({"quantization": quant_hash, "corrections": branches.joint_hash})
            _evaluate_configuration(model, config, role, row, windows, device, metrics_path, model_hash)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary_path = root / "evaluation" / f"ppl_summary_{role}.csv"
    rows = summarize_ppl(metrics_path, summary_path)
    result = {
        "status": "PASS",
        "completed_at_utc": utc_now(),
        "dataset": role,
        "windows": len(windows),
        "configurations": len(sequence),
        "summary_rows": len(rows),
        "summary_path": str(summary_path),
    }
    save_json(root / "state" / f"evaluation_{role}_complete.json", result)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result
