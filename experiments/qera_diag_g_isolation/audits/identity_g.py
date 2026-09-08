#!/usr/bin/env python3
"""Read existing round-one artifacts and diagnose one identity-G regression.

Only a unique diagnostics directory is written. No manifest, checkpoints,
thresholds, or corrections are migrated or replaced by this command.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import math
from pathlib import Path
import sys
import time
import traceback
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from qera_original_a_isolation.common import import_official_qera, qera_layer_config
from qera_diag_g_isolation.math_ops import apply_a, correction_drift, solve_weighted
from qera_diag_g_isolation.pipeline import _load_g_checkpoint, artifact_path, completed_artifact
from qera_diag_g_isolation.storage import (
    atomic_json, checked_tensors, config_from_file, file_record, heartbeat,
    load_manifest, log, run_lock, verify, weight_tensor,
)


def relative_difference(actual, expected):
    """Measure after conversion, so FP32 subtraction does not hide discrepancies."""
    actual, expected = actual.detach().double(), expected.detach().double()
    if actual.shape != expected.shape:
        raise ValueError(f"Shape mismatch: {actual.shape} versus {expected.shape}")
    delta = actual - expected
    denominator = float(torch.linalg.vector_norm(expected))
    return {
        "exact_equal": bool(torch.equal(actual, expected)),
        "relative_frobenius": float(torch.linalg.vector_norm(delta)) / max(denominator, 1e-30),
        "max_absolute": float(delta.abs().max()),
    }


def tensor_description(value):
    return {
        "shape": list(value.shape), "dtype": str(value.dtype),
        "stride": list(value.stride()), "contiguous": value.is_contiguous(),
        "finite": bool(torch.isfinite(value).all()),
    }


def cpu_factors(left, right):
    return left.detach().cpu(), right.detach().cpu()


def product_difference(actual, expected, ranks, row_chunk=256):
    """Compare products formed in FP64, independently of the production gate."""
    left, right = (x.double().cpu() for x in actual)
    ref_left, ref_right = (x.double().cpu() for x in expected)
    result = {}
    for rank in ranks:
        numerator, denominator = 0.0, 0.0
        for begin in range(0, left.shape[0], row_chunk):
            reference = ref_left[begin:begin + row_chunk, :rank] @ ref_right[:rank]
            delta = left[begin:begin + row_chunk, :rank] @ right[:rank] - reference
            numerator += float(delta.square().sum())
            denominator += float(reference.square().sum())
        result[str(rank)] = math.sqrt(numerator) / max(math.sqrt(denominator), 1e-30)
    return result


def weighted_objectives(factors, scale, error, ranks, row_chunk=256):
    """FP64 objectives in the original A-weighted metric, not a PPL estimate."""
    left, right = (x.double().cpu() for x in factors)
    scale, error = scale.double().cpu(), error.double().cpu()
    weighted_error, weighted_left = apply_a(scale, error), apply_a(scale, left)
    before = float(weighted_error.square().sum())
    result = {}
    for rank in ranks:
        after = 0.0
        for begin in range(0, error.shape[0], row_chunk):
            residual = weighted_error[begin:begin + row_chunk] - (
                weighted_left[begin:begin + row_chunk, :rank] @ right[:rank]
            )
            after += float(residual.square().sum())
        result[str(rank)] = {"sse": after, "relative_to_uncorrected": after / max(before, 1e-30)}
    return result


def spectral_description(singular, ranks):
    singular = singular.detach().double().cpu()
    boundaries = {}
    for rank in ranks:
        kept = float(singular[rank - 1])
        omitted = float(singular[rank]) if rank < singular.numel() else None
        boundaries[str(rank)] = {
            "sigma_r": kept, "sigma_r_plus_1": omitted,
            "relative_gap": None if omitted is None else (kept - omitted) / max(abs(kept), 1e-30),
            "optimal_tail_sse_for_this_matrix": float(singular[rank:].square().sum()),
        }
    return {"boundaries": boundaries, "leading_values": singular[:max(ranks) + 1].tolist()}


def controlled_svd(matrix, scale, ranks, inverse, full_matrices, driver=None):
    """Use the same right-factor construction for all controlled comparisons."""
    kwargs = {} if driver is None else {"driver": driver}
    u, singular, vh = torch.linalg.svd(matrix, full_matrices=full_matrices, **kwargs)
    rank = max(ranks)
    left = inverse(scale, u[:, :rank])
    right = torch.diag(singular[:rank]) @ vh[:rank]
    alternate_right = singular[:rank, None] * vh[:rank]
    diagnostics = spectral_description(singular, ranks)
    diagnostics["right_matmul_vs_elementwise"] = relative_difference(right, alternate_right)
    return cpu_factors(left, right), diagnostics


def record_variant(report, name, factors, saved, scale, error, ranks, device, **details):
    record = {
        **details,
        "factor_dtypes": [str(x.dtype) for x in factors],
        "vs_saved_fp64_products": product_difference(factors, saved, ranks),
        "weighted_objectives_fp64": weighted_objectives(factors, scale, error, ranks),
    }
    # The actual gate first multiplies in the stored factor dtype (usually FP32).
    if all(x.dtype == torch.float32 for x in factors):
        record["vs_saved_production_gate"] = correction_drift(
            *(x.to(device) for x in (*factors, *saved)), ranks,
        )
    report["variants"][name] = record
    log("identity-audit", f"{name} vs_saved={record.get('vs_saved_production_gate', record['vs_saved_fp64_products'])}")


def select_layer(manifest, name):
    matches = [(group, layer) for group in manifest["payload"]["groups"]
               for layer in group["layers"] if layer["name"] == name]
    if len(matches) != 1:
        raise ValueError(f"Expected one manifest entry for {name}, found {len(matches)}")
    return matches[0]


def audit(config, manifest, args, report, report_path):
    ranks, device = config["ranks"], torch.device(args.device)
    group, layer = select_layer(manifest, args.layer)
    # Verify the committed G snapshot before diagnosing solve; do not recollect it.
    sums, windows, _ = _load_g_checkpoint(config, manifest)
    if windows != config["num_calibration_windows"]:
        raise RuntimeError(f"Expected completed G; found {windows} windows")
    del sums
    report["g_checkpoint"] = {"windows": windows, "prediction_tokens": windows * 2047,
                              "collect_state": file_record(Path(config["run_dir"]) / "statistics/collect_state.json")}
    model_records = manifest["payload"]["model_files"]
    verify(model_records[layer["weight_file"]])
    scale = checked_tensors(group["roots"])[args.method].float().to(device)
    saved_tensors = checked_tensors(layer["gi"][args.method])
    saved = cpu_factors(saved_tensors["A"], saved_tensors["B"])
    quant_state = completed_artifact(artifact_path(config, "quantized", args.layer), manifest)
    if quant_state is None:
        raise RuntimeError("Frozen quantized artifact is missing")
    stored_q = checked_tensors(quant_state["file"])["weight_q"]
    original_weight = weight_tensor(manifest, layer)
    if original_weight.dtype != torch.bfloat16:
        raise RuntimeError("Expected the original BF16 checkpoint")
    official = import_official_qera(manifest["payload"]["source_config"])
    from qera.approximate import _compute_scale_inv_dot_U

    weight = original_weight.float().to(device)
    fresh_q = official["mxint_quantizer"](weight, width=4, block_size=32, block_axis=-1)
    error = (original_weight.float() - stored_q.float()).T.to(device)
    old_error = (weight - fresh_q).T
    old_matrix = torch.diag(scale) @ old_error if scale.ndim == 1 else scale @ old_error
    new_matrix = apply_a(scale, error) * torch.ones(error.shape[1], device=device)[None, :]
    report["inputs"] = {
        "weight_file": model_records[layer["weight_file"]], "roots": group["roots"],
        "saved_gi": layer["gi"][args.method], "frozen_wq": quant_state["file"],
        "fresh_official_q_vs_frozen_q": relative_difference(fresh_q, stored_q.to(device)),
        "new_error_vs_official_error": relative_difference(error, old_error),
        "new_svd_input_vs_official_svd_input": relative_difference(new_matrix, old_matrix),
        "old_svd_input": tensor_description(old_matrix), "new_svd_input": tensor_description(new_matrix),
        "scale": tensor_description(scale), "saved_factors": [tensor_description(x) for x in saved],
    }
    if scale.ndim == 1:
        positive = scale[scale > 0]
        report["inputs"]["diagonal_scale_range"] = {
            "min": float(scale.min()), "max": float(scale.max()),
            "nonpositive_count": int((scale <= 0).sum()),
            "max_over_min_positive": None if not positive.numel() else float(scale.max() / positive.min()),
        }
    atomic_json(report_path, report)
    log("identity-audit", f"input comparison={report['inputs']['new_svd_input_vs_official_svd_input']}")
    factors = {"saved": saved}

    # Execute the actual official function, not just a reimplementation of its formula.
    linear = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False, device="meta")
    linear.weight = torch.nn.Parameter(weight, requires_grad=False)
    with heartbeat("identity-audit", "actual official replay"):
        ab, mse = official["compute_ab"](args.layer, linear, scale.clone(), qera_layer_config(max(ranks)))
    factors["official_replay"] = cpu_factors(ab[args.layer + ".A"], ab[args.layer + ".B"])
    record_variant(report, "official_replay", factors["official_replay"], saved, scale, error,
                   ranks, device, official_mse=float(mse))
    atomic_json(report_path, report)
    del ab, linear

    with heartbeat("identity-audit", "actual current solver with identity G"):
        left, right, metrics = solve_weighted(error, scale, torch.ones(error.shape[1], device=device),
                                             max(ranks), _compute_scale_inv_dot_U)
    factors["current_solver"] = cpu_factors(left, right)
    record_variant(report, "current_solver", factors["current_solver"], saved, scale, error,
                   ranks, device, production_metrics=metrics)
    atomic_json(report_path, report)
    del left, right

    for name, matrix, full in (
        ("old_input_full_svd", old_matrix, True),
        ("old_input_reduced_svd", old_matrix, False),
        ("new_input_full_svd", new_matrix, True),
        ("new_input_reduced_svd", new_matrix, False),
    ):
        with heartbeat("identity-audit", name):
            factors[name], spectrum = controlled_svd(matrix, scale, ranks, _compute_scale_inv_dot_U, full)
        record_variant(report, name, factors[name], saved, scale, error, ranks, device,
                       full_matrices=full, driver="default", spectrum=spectrum)
        atomic_json(report_path, report)

    if not args.skip_fp64_reference:
        # Rebuild weighting in FP64, rather than only upcasting a rounded FP32 M.
        with heartbeat("identity-audit", "FP64 reference with QR-based gesvd"):
            scale64 = scale.double()
            matrix64 = apply_a(scale64, error.double())
            factors["fp64_reference"], spectrum = controlled_svd(
                matrix64, scale64, ranks, _compute_scale_inv_dot_U, False, driver="gesvd",
            )
        record_variant(report, "fp64_reference", factors["fp64_reference"], saved, scale, error,
                       ranks, device, driver="gesvd", spectrum=spectrum,
                       matrix_vs_fp32=relative_difference(matrix64, old_matrix))
        del scale64, matrix64
        atomic_json(report_path, report)

    pairs = {
        "official_replay_vs_old_input_full": ("official_replay", "old_input_full_svd"),
        "full_vs_reduced_same_old_input": ("old_input_full_svd", "old_input_reduced_svd"),
        "old_vs_new_input_same_full_svd": ("new_input_full_svd", "old_input_full_svd"),
        "full_vs_reduced_same_new_input": ("new_input_full_svd", "new_input_reduced_svd"),
        "current_vs_controlled_new_reduced": ("current_solver", "new_input_reduced_svd"),
    }
    report["controlled_comparisons_fp64_products"] = {
        name: {"actual": actual, "reference": reference,
               "relative_frobenius": product_difference(factors[actual], factors[reference], ranks)}
        for name, (actual, reference) in pairs.items()
    }
    if "fp64_reference" in factors:
        report["vs_fp64_reference"] = {
            name: product_difference(value, factors["fp64_reference"], ranks)
            for name, value in factors.items() if name != "fp64_reference"
        }
    tolerance = config["identity_product_tolerance"]
    report["gate_observations"] = {
        "unchanged_tolerance": tolerance,
        "official_replay_matches_saved": max(report["variants"]["official_replay"]["vs_saved_production_gate"].values()) <= tolerance,
        "current_solver_matches_saved": max(report["variants"]["current_solver"]["vs_saved_production_gate"].values()) <= tolerance,
        "note": "These observations do not establish a root cause or authorize a protocol migration.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--layer", default="model.layers.0.self_attn.k_proj")
    parser.add_argument("--method", choices=("diag", "full"), default="diag")
    parser.add_argument("--device", choices=("cuda:0", "cuda:1"), default="cuda:0")
    parser.add_argument("--skip-fp64-reference", action="store_true", help="Omit the slower FP64 reference SVD")
    args = parser.parse_args(argv)
    config = config_from_file(args.config)
    for package, version in (("torch", "2.3.0"), ("transformers", "4.44.2")):
        actual = importlib.metadata.version(package).split("+")[0]
        if actual != version:
            raise RuntimeError(f"Use existing qera-original-a environment: {package}={version}, found {actual}")
    if not torch.cuda.is_available():
        raise RuntimeError("Run this audit on the server GPU; local CPU tests cannot reproduce its SVD")
    torch.cuda.set_device(args.device)
    torch.set_num_threads(config["cpu_threads"])
    inherited_tf32 = {"matmul": torch.backends.cuda.matmul.allow_tf32, "cudnn": torch.backends.cudnn.allow_tf32}
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(1234)
    with run_lock(config), torch.no_grad():
        manifest = load_manifest(config)
        report_path = Path(config["run_dir"]) / "diagnostics" / (
            time.strftime("identity_g_%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
        ) / "report.json"
        report = {
            "status": "RUNNING", "layer": args.layer, "method": args.method,
            "manifest_sha256": manifest["sha256"], "audit_script": file_record(__file__),
            "environment": {"torch": torch.__version__, "cuda": torch.version.cuda,
                            "gpu": torch.cuda.get_device_name(args.device),
                            "device": args.device, "cpu_threads": torch.get_num_threads(),
                            "inherited_tf32": inherited_tf32, "effective_tf32": False,
                            "historical_tf32": "unknown; inherited settings are not evidence of the old run"},
            "variants": {}, "scope": "diagnostics only; no production artifacts or gates changed",
        }
        log("identity-audit", f"report={report_path}")
        atomic_json(report_path, report)
        try:
            audit(config, manifest, args, report, report_path)
            report["status"] = "AUDIT_COMPLETE"
        except Exception:
            report["status"] = "AUDIT_ERROR"
            report["traceback"] = traceback.format_exc()
            atomic_json(report_path, report)
            raise
        atomic_json(report_path, report)
        log("identity-audit", f"AUDIT_COMPLETE observations={report['gate_observations']}")
        log("identity-audit", f"report={report_path}; audit completion does NOT mean the identity gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
