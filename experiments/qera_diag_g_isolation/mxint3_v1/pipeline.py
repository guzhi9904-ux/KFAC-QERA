"""MXINT3-only stages. Never use the legacy W4/GI artifact routing here."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import math
from pathlib import Path
import shutil

import torch

from qera_original_a_isolation.common import import_official_qera, qera_layer_config
from qera_diag_g_isolation import pipeline as legacy
from qera_diag_g_isolation.full_svd_v1.solver import solve_weighted
from qera_diag_g_isolation.math_ops import correction_drift, diagonal_scale
from qera_diag_g_isolation.storage import (
    atomic_json, checked_tensors, heartbeat, layers, load_manifest, log,
    verify_model, weight_tensor,
)

QUANTIZATION = {"name": "mxint", "width": 3, "block_size": 32, "block_axis": -1}


def layer_config(rank):
    config = qera_layer_config(rank)
    config["w_quantizer"] = deepcopy(QUANTIZATION)
    return config


def configurations(config):
    result = [("BF16", None, None), ("W3_MXINT", "wq", None)]
    for method in ("diag_gi", "full_gi", "diag_gd", "full_gd"):
        result.extend((f"{method.upper()}_R{rank}", method, rank) for rank in config["ranks"])
    return result


def require_variant(config, manifest):
    if (config.get("experiment_variant") != "mxint3_v1"
            or config.get("quantization") != QUANTIZATION
            or manifest["payload"]["source_config"]["quantization"] != QUANTIZATION):
        raise RuntimeError("This runner requires the isolated MXINT3 manifest")


def owned_artifact(config, manifest, directory, name, check=True):
    path = legacy.artifact_path(config, directory, name)
    if not path.resolve().is_relative_to(Path(config["run_dir"]).resolve()):
        raise RuntimeError("MXINT3 artifact path escapes the new output directory")
    record = legacy.completed_artifact(path, manifest, check=check)
    if record is not None:
        # No parent W4 weights/corrections, even through an explicit metadata pointer.
        if Path(record["file"]["path"]).resolve() != path.resolve():
            raise RuntimeError("MXINT3 weights/corrections must be newly generated in this run")
        if record.get("quantization") != QUANTIZATION or record.get("layer") != name:
            raise RuntimeError("MXINT3 artifact quantization/layer mismatch")
        if directory.startswith("corrections/"):
            if record.get("method") != directory.split("/")[1] or record.get("rank") != max(config["ranks"]):
                raise RuntimeError("Correction method/rank mismatch")
            drift = record.get("identity_product_drift", {})
            if set(drift) != {str(r) for r in config["ranks"]} or any(
                not math.isfinite(v) or not 0 <= v <= config["identity_product_tolerance"] for v in drift.values()
            ):
                raise RuntimeError("Correction has no passing MXINT3 identity regression")
        elif record.get("fp32_bf16_quantization_equal") is not True:
            raise RuntimeError("Missing FP32/BF16 quantization equivalence gate")
    return record


def quantize(config):
    manifest = load_manifest(config)
    require_variant(config, manifest)
    verify_model(manifest)
    quantizer = import_official_qera(manifest["payload"]["source_config"])["mxint_quantizer"]
    kwargs = {k: v for k, v in QUANTIZATION.items() if k != "name"}
    pending = [x for x in layers(manifest) if owned_artifact(config, manifest, "quantized", x["name"]) is None]
    required_bytes = sum(math.prod(x["shape"]) * 2 for x in pending) + 2 * 2**30
    if shutil.disk_usage(config["run_dir"]).free < required_bytes:
        raise RuntimeError("Insufficient disk space for remaining BF16-emulated MXINT3 weights plus 2 GiB reserve")
    records = {}
    with torch.no_grad():
        for index, layer in enumerate(layers(manifest), 1):
            name = layer["name"]
            record = owned_artifact(config, manifest, "quantized", name)
            if record is None:
                weight = weight_tensor(manifest, layer).to(config["solve_device"])
                if weight.dtype != torch.bfloat16:
                    raise RuntimeError("Expected original BF16 checkpoint weights")
                q32, q16 = quantizer(weight.float(), **kwargs), quantizer(weight, **kwargs)
                if not torch.isfinite(q32).all() or not torch.equal(q32, q16.float()):
                    raise RuntimeError(f"MXINT3 FP32/BF16 quantization mismatch: {name}")
                record = legacy.save_artifact(legacy.artifact_path(config, "quantized", name), manifest,
                    {"weight_q": q16}, layer=name, quantization=QUANTIZATION,
                    fp32_bf16_quantization_equal=True)
                del weight, q32, q16
            records[name] = record["file"]
            log("quantize-mxint3", f"module={index}/{len(layers(manifest))} {name}")
    atomic_json(Path(config["run_dir"]) / "quantized/complete.json", {
        "status": "PASS", "manifest_sha256": manifest["sha256"], "files": records,
        "quantization": QUANTIZATION,
    })


def official_identity(official, name, weight, scale, rank):
    linear = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False, device="meta")
    linear.weight = torch.nn.Parameter(weight, requires_grad=False)
    # Fresh config: the pinned official implementation pops the quantizer name.
    ab, mse = official["compute_ab"](name, linear, scale.clone(), layer_config(rank))
    left, right = ab[name + ".A"], ab[name + ".B"]
    if not math.isfinite(float(mse)) or not torch.isfinite(left).all() or not torch.isfinite(right).all():
        raise RuntimeError("Nonfinite official MXINT3 identity baseline")
    return left, right, float(mse)


def solve(config):
    manifest = load_manifest(config)
    require_variant(config, manifest)
    sums, count, _ = legacy._load_g_checkpoint(config, manifest)
    if count != 256:
        raise RuntimeError("Require retained teacher G256; this runner never collects G")
    verify_model(manifest)
    official = import_official_qera(manifest["payload"]["source_config"])
    from qera.approximate import _compute_scale_inv_dot_U
    rank, device = max(config["ranks"]), config["solve_device"]
    completed = 0
    with torch.no_grad():
        for group in manifest["payload"]["groups"]:
            roots = checked_tensors(group["roots"])
            for layer in group["layers"]:
                completed += 1
                name = layer["name"]
                log("solve-mxint3", f"module={completed}/{len(layers(manifest))} {name}")
                quant = owned_artifact(config, manifest, "quantized", name)
                if quant is None:
                    raise RuntimeError("Run MXINT3 quantize first")
                weight = weight_tensor(manifest, layer).float().to(device)
                weight_q = checked_tensors(quant["file"])["weight_q"].float().to(device)
                # Verify the official baseline will use this exact frozen Wq.
                fresh = official["mxint_quantizer"](weight, width=3, block_size=32, block_axis=-1)
                if not torch.equal(fresh, weight_q):
                    raise RuntimeError("Official replay and frozen MXINT3 Wq differ")
                del fresh
                error_t = (weight.cpu() - weight_q.cpu()).T.to(device)
                del weight_q
                scale_g, diagnostics = diagonal_scale(sums[name], count * 2047, config["g_relative_floor"])
                effective = legacy.artifact_path(config, "statistics/effective_g", name)
                if legacy.completed_artifact(effective, manifest) is None:
                    legacy.save_artifact(effective, manifest, {"g_sum": sums[name], "sqrt_g_effective": scale_g},
                                         count=count * 2047, **diagnostics)
                scale_g = scale_g.to(device)
                for method in ("diag", "full"):
                    scale_a = roots[method].to(device)
                    gi = owned_artifact(config, manifest, f"corrections/{method}_gi", name)
                    gd = owned_artifact(config, manifest, f"corrections/{method}_gd", name)
                    if gi is None:
                        with heartbeat("solve-mxint3", f"{name} {method} official GI + identity regression"):
                            ref_a, ref_b, mse = official_identity(official, name, weight, scale_a, rank)
                            a, b, _ = solve_weighted(error_t, scale_a, torch.ones_like(scale_g), rank,
                                                     _compute_scale_inv_dot_U)
                            drift = correction_drift(a, b, ref_a, ref_b, config["ranks"])
                        if any(not math.isfinite(v) or not 0 <= v <= config["identity_product_tolerance"]
                               for v in drift.values()):
                            raise RuntimeError(f"MXINT3 identity-G regression failed: {name} {method} {drift}")
                        gi = legacy.save_artifact(legacy.artifact_path(config, f"corrections/{method}_gi", name),
                            manifest, {"A": ref_a, "B": ref_b}, layer=name, method=method + "_gi", rank=rank,
                            quantization=QUANTIZATION, identity_product_drift=drift, official_mse=mse,
                            reference="pinned official QERA compute_ab with width=3; NOT old MXINT4 factors",
                            svd_full_matrices=True)
                        del a, b, ref_a, ref_b
                    if gd is None:
                        with heartbeat("solve-mxint3", f"{name} {method} diagonal-G full SVD"):
                            a, b, metrics = solve_weighted(error_t, scale_a, scale_g, rank, _compute_scale_inv_dot_U)
                        legacy.save_artifact(legacy.artifact_path(config, f"corrections/{method}_gd", name),
                            manifest, {"A": a, "B": b}, layer=name, method=method + "_gd", rank=rank,
                            quantization=QUANTIZATION, identity_product_drift=gi["identity_product_drift"],
                            gi_reference=gi["file"], g_diagnostics=diagnostics, **metrics)
                        log("solve-mxint3", f"saved {method}_gd {name} rank64_sse="
                            f"{metrics['weighted_sse_before']:.6g}->{metrics['weighted_sse_after']:.6g}")
                        del a, b
                    elif gd.get("gi_reference") != gi["file"]:
                        raise RuntimeError("Resumed GD is bound to a different GI artifact")
                    del scale_a
                del weight, error_t, scale_g
                torch.cuda.empty_cache()
            del roots
    atomic_json(Path(config["run_dir"]) / "solve_complete.json", {
        "status": "PASS", "manifest_sha256": manifest["sha256"], "modules": completed,
        "quantization": QUANTIZATION,
    })


def evaluation_inputs(config, manifest, method, check=True):
    require_variant(config, manifest)
    if method not in (None, "wq", "diag_gi", "full_gi", "diag_gd", "full_gd"):
        raise ValueError(f"Unknown MXINT3 method: {method}")
    result = {}
    for layer in layers(manifest) if method is not None else []:
        name = layer["name"]
        quant = owned_artifact(config, manifest, "quantized", name, check)
        if quant is None:
            raise RuntimeError("Missing MXINT3 weight")
        files = {"quant": quant["file"]}
        if method != "wq":
            correction = owned_artifact(config, manifest, f"corrections/{method}", name, check)
            if correction is None:
                raise RuntimeError(f"Missing MXINT3 {method} correction")
            if method.endswith("_gd"):
                gi = owned_artifact(config, manifest, f"corrections/{method.removesuffix('_gd')}_gi", name, check)
                if gi is None or correction.get("gi_reference") != gi["file"]:
                    raise RuntimeError("GD is not bound to the current MXINT3 GI baseline")
            files["correction"] = correction["file"]
        result[name] = files
    return result


@contextmanager
def evaluation_binding():
    """Reuse unchanged BF16 evaluation and CSV/resume logic, with explicit W3 routing."""
    previous = legacy.configurations, legacy._evaluation_inputs
    legacy.configurations, legacy._evaluation_inputs = configurations, evaluation_inputs
    try:
        yield
    finally:
        legacy.configurations, legacy._evaluation_inputs = previous


def evaluate(config, only=None):
    with evaluation_binding():
        return legacy.evaluate(config, only)


def summarize(config):
    with evaluation_binding():
        return legacy.summarize(config)
