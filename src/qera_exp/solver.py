from __future__ import annotations

import gc
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import torch

from .config import output_root
from .modeling import layer_index, projection_name
from .utils import (
    atomic_safetensors,
    heartbeat,
    load_safetensors,
    log,
    progress_status,
    safe_name,
    save_csv,
    save_json,
    sha256_file,
    tensor_sha256,
    upsert_csv,
)


@dataclass
class Metric:
    level: Literal["I", "diag", "full"]
    dimension: int
    left_sqrt: torch.Tensor | None = None
    left_inverse: torch.Tensor | None = None
    right_sqrt: torch.Tensor | None = None
    right_inverse: torch.Tensor | None = None


def _finite(value: torch.Tensor, label: str) -> None:
    if not bool(torch.isfinite(value).all().item()):
        raise FloatingPointError(f"Nonfinite {label}")


def _relative(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = max(float(torch.linalg.vector_norm(right).item()), torch.finfo(torch.float64).tiny)
    return float(torch.linalg.vector_norm(left - right).item() / denominator)


def analyze_metric(
    raw: torch.Tensor,
    direct_diag: torch.Tensor,
    *,
    label: str,
    damping_lambda: float,
    negative_relative_tolerance: float,
    root_method: str,
) -> tuple[dict[str, Any], Metric, Metric]:
    started = time.time()
    raw = raw.detach().cpu().double().contiguous()
    direct_diag = direct_diag.detach().cpu().double().contiguous()
    if raw.ndim != 2 or raw.shape[0] != raw.shape[1] or direct_diag.shape != (raw.shape[0],):
        raise ValueError(f"Malformed metric: {label}")
    _finite(raw, label)
    symmetric = ((raw + raw.T) * 0.5).contiguous()
    symmetry_error = _relative(raw, symmetric)
    diagonal = symmetric.diag().clone()
    diagonal_error = _relative(diagonal, direct_diag)
    dimension = raw.shape[0]
    trace = float(diagonal.sum().item())
    if not math.isfinite(trace) or trace <= 0:
        raise RuntimeError(f"Nonpositive trace: {label}")
    shift = damping_lambda * trace / dimension
    diag_damped = direct_diag.clamp_min(0) + shift
    diag_root = diag_damped.sqrt()
    diag = Metric(
        "diag",
        dimension,
        left_sqrt=diag_root,
        left_inverse=diag_root.reciprocal(),
        right_sqrt=diag_root,
        right_inverse=diag_root.reciprocal(),
    )
    summary: dict[str, Any] = {
        "label": label,
        "dimension": dimension,
        "root_method": root_method,
        "trace": trace,
        "damping_shift": shift,
        "symmetry_relative_error": symmetry_error,
        "diagonal_vs_direct_relative_error": diagonal_error,
        "raw_sha256": tensor_sha256(raw),
        "direct_diag_sha256": tensor_sha256(direct_diag),
    }
    if symmetry_error > 1e-6 or diagonal_error > 1e-5:
        raise RuntimeError(f"Metric audit failed for {label}: symmetry={symmetry_error}, diagonal={diagonal_error}")
    if root_method == "eigh":
        eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
        _finite(eigenvalues, f"{label} eigenvalues")
        maximum = max(float(eigenvalues.abs().max().item()), torch.finfo(torch.float64).tiny)
        minimum = float(eigenvalues.min().item())
        if minimum < -negative_relative_tolerance * maximum:
            raise RuntimeError(f"Substantive negative spectrum in {label}: min={minimum}, max_abs={maximum}")
        projected = eigenvalues.clamp_min(0)
        damped = projected + shift
        sqrt = ((eigenvectors * damped.sqrt().unsqueeze(0)) @ eigenvectors.T).contiguous()
        inverse = ((eigenvectors * damped.rsqrt().unsqueeze(0)) @ eigenvectors.T).contiguous()
        full = Metric("full", dimension, sqrt, inverse, sqrt, inverse)
        probability = projected / projected.sum()
        positive_probability = probability[probability > 0]
        summary.update(
            {
                "minimum_eigenvalue": minimum,
                "maximum_eigenvalue": float(eigenvalues.max().item()),
                "negative_eigenvalue_count": int((eigenvalues < 0).sum().item()),
                "damped_condition_number": float(damped.max().item() / damped.min().item()),
                "entropy_effective_rank": float(torch.exp(-(positive_probability * positive_probability.log()).sum()).item()),
            }
        )
    elif root_method == "cholesky":
        damped_matrix = symmetric + shift * torch.eye(dimension, dtype=torch.float64)
        chol, info = torch.linalg.cholesky_ex(damped_matrix, check_errors=False)
        if int(info.max().item()) != 0:
            raise RuntimeError(f"Damped metric is not positive definite under Cholesky: {label}; info={int(info.max().item())}")
        inverse = torch.linalg.solve_triangular(chol, torch.eye(dimension, dtype=torch.float64), upper=False)
        # A right factor is C; a G left factor is C^T when M=C_G^T E C_A.
        full = Metric(
            "full",
            dimension,
            left_sqrt=chol.T.contiguous(),
            left_inverse=inverse.T.contiguous(),
            right_sqrt=chol,
            right_inverse=inverse,
        )
        summary.update(
            {
                "minimum_eigenvalue": None,
                "maximum_eigenvalue": None,
                "negative_eigenvalue_count": None,
                "damped_condition_number": None,
                "entropy_effective_rank": None,
            }
        )
    else:
        raise ValueError("statistics.root_method must be 'eigh' or 'cholesky'")
    if full.left_sqrt is None or full.right_sqrt is None:
        raise RuntimeError("Full metric construction failed")
    reconstructed = full.left_sqrt.T @ full.left_sqrt
    damped_reference = symmetric + shift * torch.eye(dimension, dtype=torch.float64)
    root_residual = _relative(reconstructed, damped_reference)
    summary["root_reconstruction_relative_residual"] = root_residual
    summary["elapsed_seconds"] = time.time() - started
    if root_residual > 1e-7:
        raise RuntimeError(f"Metric root residual failed for {label}: {root_residual}")
    return summary, diag, full


def _left(metric: Metric, value: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    if metric.level == "I":
        return value
    operator = metric.left_inverse if inverse else metric.left_sqrt
    if operator is None:
        raise RuntimeError("Missing left metric operator")
    return operator.unsqueeze(1) * value if operator.ndim == 1 else operator @ value


def _right(metric: Metric, value: torch.Tensor) -> torch.Tensor:
    if metric.level == "I":
        return value
    if metric.right_sqrt is None:
        raise RuntimeError("Missing right metric operator")
    return value * metric.right_sqrt.unsqueeze(0) if metric.right_sqrt.ndim == 1 else value @ metric.right_sqrt


def _unmap_right_factor(metric: Metric, value: torch.Tensor) -> torch.Tensor:
    if metric.level == "I":
        return value
    if metric.right_inverse is None:
        raise RuntimeError("Missing inverse right metric operator")
    return metric.right_inverse.unsqueeze(1) * value if metric.right_inverse.ndim == 1 else metric.right_inverse.T @ value


def solve_weighted(error: torch.Tensor, a: Metric, g: Metric, rank: int, device: str = "auto") -> dict[str, Any]:
    error = error.detach().cpu().double().contiguous()
    if error.shape != (g.dimension, a.dimension) or rank > min(error.shape):
        raise ValueError("Error/metric/rank shape mismatch")
    transformed = _right(a, _left(g, error))
    _finite(transformed, "transformed quantization error")
    if device == "auto":
        target = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        target = torch.device(device)
    transformed_svd = transformed.to(device=target, dtype=torch.float32 if target.type == "cuda" else torch.float64)
    u_device, singular_device, vh_device = torch.linalg.svd(transformed_svd, full_matrices=False)
    singular = singular_device.detach().cpu().double()
    u = u_device[:, :rank].detach().cpu().double()
    v = vh_device[:rank, :].T.detach().cpu().double()
    root_singular = singular[:rank].sqrt()
    left = _left(g, u * root_singular.unsqueeze(0), inverse=True)
    right = _unmap_right_factor(a, v * root_singular.unsqueeze(0))
    correction = left @ right.T
    mapped = _right(a, _left(g, correction))
    target_rank = u @ torch.diag(singular[:rank]) @ v.T
    mapped_residual = _relative(mapped, target_rank)
    left_bf16 = left.to(torch.bfloat16).contiguous()
    right_bf16 = right.to(torch.bfloat16).contiguous()
    storage_drift = _relative(left_bf16.double() @ right_bf16.double().T, correction)
    before = float(transformed.square().sum().item())
    after = float((transformed - target_rank).square().sum().item())
    if mapped_residual > 1e-6 or storage_drift > 5e-3 or after > before * (1 + 1e-6):
        raise RuntimeError(
            f"Weighted solve gate failed: mapped={mapped_residual}, storage={storage_drift}, before={before}, after={after}"
        )
    return {
        "left": left_bf16,
        "right": right_bf16,
        "singular_values": singular,
        "objective_before": before,
        "objective_after": after,
        "mapped_back_relative_residual": mapped_residual,
        "stored_bf16_relative_drift": storage_drift,
    }


def _method_metrics(method: str, ad: Metric, af: Metric, gi: Metric, gd: Metric, gf: Metric) -> tuple[Metric, Metric]:
    mapping = {
        "AD_GI": (ad, gi),
        "AD_GD": (ad, gd),
        "AD_GF": (ad, gf),
        "AF_GI": (af, gi),
        "AF_GD": (af, gd),
        "AF_GF": (af, gf),
    }
    return mapping[method]


def solve_raw_module(config: Mapping[str, Any], module: str) -> dict[str, Any]:
    root = output_root(config)
    log(root, f"module={module} loading raw A/G", "solve")
    raw_artifact = root / "statistics" / "raw" / f"{safe_name(module)}.safetensors"
    raw_metadata = raw_artifact.with_suffix(".json")
    if not raw_artifact.is_file() or not raw_metadata.is_file():
        raise FileNotFoundError(f"Raw statistics absent: {module}")
    raw = load_safetensors(raw_artifact)
    meta = __import__("json").loads(raw_metadata.read_text(encoding="utf-8"))
    if sha256_file(raw_artifact) != meta["artifact_sha256"]:
        raise RuntimeError(f"Raw statistics hash mismatch: {module}")
    count = int(meta["valid_prediction_token_count"])
    a_raw, g_raw = raw["a_sum"].double() / count, raw["g_sum"].double() / count
    a_direct, g_direct = raw["a_direct_sum"].double() / count, raw["g_direct_sum"].double() / count
    statistics = config["statistics"]
    log(root, f"module={module} factorizing A dimension={a_raw.shape[0]} method={statistics['root_method']}", "solve")
    with heartbeat(root, f"module={module} factorizing A", "solve"):
        a_summary, ad, af = analyze_metric(
            a_raw,
            a_direct,
            label=f"{module}:A",
            damping_lambda=float(statistics["lambda_a"]),
            negative_relative_tolerance=float(statistics["negative_eigen_relative_tolerance"]),
            root_method=str(statistics["root_method"]),
        )
    log(root, f"module={module} A factor ready elapsed={a_summary['elapsed_seconds']:.1f}s", "solve")
    log(root, f"module={module} factorizing G dimension={g_raw.shape[0]} method={statistics['root_method']}", "solve")
    with heartbeat(root, f"module={module} factorizing G", "solve"):
        g_summary, gd, gf = analyze_metric(
            g_raw,
            g_direct,
            label=f"{module}:G",
            damping_lambda=float(statistics["lambda_g"]),
            negative_relative_tolerance=float(statistics["negative_eigen_relative_tolerance"]),
            root_method=str(statistics["root_method"]),
        )
    log(root, f"module={module} G factor ready elapsed={g_summary['elapsed_seconds']:.1f}s", "solve")
    error = raw["error_fp32"].double()
    gi = Metric("I", g_raw.shape[0])
    maximum_rank = int(statistics["maximum_rank"])
    if maximum_rank > min(error.shape):
        raise ValueError(f"statistics.maximum_rank={maximum_rank} exceeds the minimum weight dimension for {module}: {min(error.shape)}")
    rows = []
    energy_rows = []
    methods = list(statistics["methods"])
    methods_started = time.time()
    for method_index, method in enumerate(methods, 1):
        a_metric, g_metric = _method_metrics(str(method), ad, af, gi, gd, gf)
        method_started = time.time()
        log(
            root,
            f"module={module} starting method={method} SVD "
            f"progress={progress_status(method_index - 1, len(methods), methods_started)}",
            "solve",
        )
        with heartbeat(root, f"module={module} method={method} SVD", "solve"):
            solved = solve_weighted(
                error,
                a_metric,
                g_metric,
                maximum_rank,
                str(statistics.get("solve_device", "auto")),
            )
        artifact = root / "corrections" / f"{safe_name(module)}__{method}__r{maximum_rank}.safetensors"
        atomic_safetensors(artifact, {"left": solved["left"], "right": solved["right"]})
        singular = solved["singular_values"]
        total_energy = max(float(singular.square().sum().item()), torch.finfo(torch.float64).tiny)
        captures = {
            str(rank): float(singular[:rank].square().sum().item() / total_energy)
            for rank in range(1, maximum_rank + 1)
        }
        row = {
            "module": module,
            "layer": layer_index(module),
            "projection": projection_name(module),
            "method": method,
            "a_level": "diag" if str(method).startswith("AD") else "full",
            "g_level": {"GI": "I", "GD": "diag", "GF": "full"}[str(method).split("_")[1]],
            "maximum_rank": maximum_rank,
            "left_sha256": tensor_sha256(solved["left"]),
            "right_sha256": tensor_sha256(solved["right"]),
            "artifact": str(artifact),
            "artifact_sha256": sha256_file(artifact),
            "objective_before": solved["objective_before"],
            "objective_after": solved["objective_after"],
            "mapped_back_relative_residual": solved["mapped_back_relative_residual"],
            "stored_bf16_relative_drift": solved["stored_bf16_relative_drift"],
            "rank_energy_capture": captures,
        }
        save_json(artifact.with_suffix(".json"), row)
        rows.append(row)
        for rank, capture in captures.items():
            energy_rows.append(
                {
                    "module": module,
                    "layer": layer_index(module),
                    "projection": projection_name(module),
                    "method": method,
                    "rank": int(rank),
                    "captured_energy_fraction": capture,
                }
            )
        log(
            root,
            f"module={module} completed method={method} method_elapsed={time.time() - method_started:.1f}s "
            f"progress={progress_status(method_index, len(methods), methods_started)}",
            "solve",
        )
        del solved, singular
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    upsert_csv(root / "correction_manifest.csv", rows, ("module", "method"))
    upsert_csv(root / "rank_energy.csv", energy_rows, ("module", "method", "rank"))
    metric_summary = {"module": module, "valid_prediction_token_count": count, "a": a_summary, "g": g_summary}
    save_json(root / "statistics" / "metrics" / f"{safe_name(module)}.json", metric_summary)
    if bool(config["runtime"].get("cleanup_raw_after_solve", False)):
        raw_artifact.unlink()
        raw_metadata.unlink()
    return {"status": "PASS", "module": module, "methods": len(rows), "maximum_rank": maximum_rank}


def solve_shard(config: Mapping[str, Any], shard_index: int) -> dict[str, Any]:
    import json

    root = output_root(config)
    plan = json.loads((root / "state" / "shard_plan.json").read_text(encoding="utf-8"))
    try:
        shard = plan["shards"][shard_index]
    except IndexError as error:
        raise ValueError(f"Invalid shard index: {shard_index}") from error
    modules = [str(module) for module in shard["modules"]]
    started = time.time()
    log(root, f"shard={shard_index} solving modules={len(modules)}", "solve")
    completed = []
    for index, module in enumerate(modules, 1):
        log(
            root,
            f"shard={shard_index} starting module={module} progress={progress_status(index - 1, len(modules), started)}",
            "solve",
        )
        completed.append(solve_raw_module(config, module))
        log(
            root,
            f"shard={shard_index} completed module={module} progress={progress_status(index, len(modules), started)}",
            "solve",
        )
    result = {"status": "PASS", "shard_index": shard_index, "modules": [row["module"] for row in completed]}
    save_json(root / "state" / f"solve_shard_{shard_index:04d}.json", result)
    return result
