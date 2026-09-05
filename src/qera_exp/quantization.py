from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from .config import output_root
from .modeling import discover_target_modules, load_model, module_manifest
from .utils import (
    atomic_safetensors,
    canonical_sha256,
    deterministic_runtime,
    ensure_layout,
    get_module,
    load_safetensors,
    log,
    safe_name,
    save_csv,
    save_json,
    sha256_file,
    tensor_sha256,
    utc_now,
)


@dataclass(frozen=True)
class MXINTResult:
    codes: torch.Tensor
    exponents: torch.Tensor
    reconstructed: torch.Tensor
    padding: int


def quantize_mxint4(weight: torch.Tensor, block_size: int = 32) -> MXINTResult:
    if weight.ndim != 2:
        raise ValueError("Expected Linear weight [d_out, d_in]")
    source = weight.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if not bool(torch.isfinite(source).all().item()):
        raise FloatingPointError("Nonfinite source weight")
    rows, width = source.shape
    padding = (-int(width)) % int(block_size)
    padded = F.pad(source, (0, padding)) if padding else source
    grouped = padded.reshape(rows, -1, block_size)
    sign = grouped < 0
    magnitude = grouped.abs()
    magnitude = torch.where(magnitude >= torch.finfo(torch.bfloat16).smallest_normal, magnitude, 0.0)
    zero = torch.all(magnitude == 0.0, dim=-1, keepdim=True)
    exponent = ((magnitude.view(torch.int32) >> 23) & 0xFF).max(dim=-1, keepdim=True).values
    exponent = torch.where(zero, torch.ones_like(exponent), exponent).to(torch.uint8)
    scale = (exponent.to(torch.int32) << 23).view(torch.float32)
    code_magnitude = torch.round(magnitude / scale * 4.0).clamp(0, 7).to(torch.int8)
    codes = torch.where(sign, -code_magnitude, code_magnitude)
    reconstructed = codes.float() * scale / 4.0
    return MXINTResult(
        codes=codes.reshape(rows, -1)[:, :width].contiguous(),
        exponents=exponent.squeeze(-1).contiguous(),
        reconstructed=reconstructed.reshape(rows, -1)[:, :width].contiguous(),
        padding=padding,
    )


def reconstruct(codes: torch.Tensor, exponents: torch.Tensor, width: int, block_size: int = 32) -> torch.Tensor:
    if codes.dtype != torch.int8 or exponents.dtype != torch.uint8 or codes.ndim != 2 or exponents.ndim != 2:
        raise TypeError("Malformed MXINT code/exponent tensors")
    rows = codes.shape[0]
    padded_width = exponents.shape[1] * int(block_size)
    padding = padded_width - int(width)
    if padding < 0:
        raise ValueError("Exponent tensor does not cover requested width")
    padded_codes = F.pad(codes, (0, padding)) if padding else codes
    scale = (exponents.to(torch.int32) << 23).view(torch.float32).unsqueeze(-1)
    return (padded_codes.reshape(rows, -1, block_size).float() * scale / 4.0).reshape(rows, -1)[:, :width].contiguous()


def quant_paths(root: Path, module: str) -> tuple[Path, Path]:
    artifact = root / "quantization" / f"{safe_name(module)}__mxint4.safetensors"
    return artifact, artifact.with_suffix(".json")


def quantize_model(config: Mapping[str, Any]) -> dict[str, Any]:
    root = output_root(config)
    ensure_layout(root)
    deterministic_runtime(int(config["runtime"]["deterministic_seed"]), bool(config["runtime"].get("allow_tf32", False)))
    model, _, _ = load_model(config, require_input_grads=False)
    names = discover_target_modules(model, config)
    manifest_rows = module_manifest(model, names)
    save_csv(root / "module_manifest.csv", manifest_rows)
    save_json(root / "module_manifest.json", {"modules": manifest_rows, "module_count": len(manifest_rows)})
    block_size = int(config["quantization"]["block_size"])
    quant_rows: list[dict[str, Any]] = []
    for index, name in enumerate(names, 1):
        module = get_module(model, name)
        reference_bf16 = module.weight.detach().cpu().to(torch.bfloat16).contiguous()
        reference_fp32 = reference_bf16.float().contiguous()
        artifact, metadata_path = quant_paths(root, name)
        reference_hash = tensor_sha256(reference_bf16)
        if artifact.exists() != metadata_path.exists():
            raise RuntimeError(f"Partial quantization checkpoint: {name}")
        if artifact.is_file() and metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            tensors = load_safetensors(artifact)
            if metadata.get("reference_weight_bf16_sha256") != reference_hash:
                raise RuntimeError(f"Existing quantization belongs to a different checkpoint: {name}")
            if metadata.get("artifact_sha256") != sha256_file(artifact):
                raise RuntimeError(f"Existing quantization artifact hash mismatch: {name}")
            rebuilt = reconstruct(tensors["codes"], tensors["exponents"], reference_fp32.shape[1], block_size)
            if tensor_sha256(rebuilt) != metadata.get("wq_fp32_sha256"):
                raise RuntimeError(f"Existing quantization reconstruction mismatch: {name}")
        else:
            result = quantize_mxint4(reference_fp32, block_size)
            rebuilt = reconstruct(result.codes, result.exponents, reference_fp32.shape[1], block_size)
            if not torch.equal(result.reconstructed, rebuilt):
                raise RuntimeError(f"MXINT reconstruction is not bit-identical: {name}")
            wq_bf16 = rebuilt.to(torch.bfloat16).contiguous()
            error = reference_fp32 - rebuilt
            atomic_safetensors(artifact, {"codes": result.codes, "exponents": result.exponents, "wq_bf16": wq_bf16})
            metadata = {
                "schema_version": 1,
                "module": name,
                "shape": list(reference_fp32.shape),
                "width": 4,
                "block_size": block_size,
                "block_axis": -1,
                "rounding": "nearest_even",
                "signed_levels": [-7, 7],
                "padding_elements_per_row": result.padding,
                "reference_weight_bf16_sha256": reference_hash,
                "reference_weight_fp32_sha256": tensor_sha256(reference_fp32),
                "codes_sha256": tensor_sha256(result.codes),
                "exponents_sha256": tensor_sha256(result.exponents),
                "wq_fp32_sha256": tensor_sha256(rebuilt),
                "wq_bf16_sha256": tensor_sha256(wq_bf16),
                "quantization_mse": float(error.square().mean().item()),
                "quantization_mae": float(error.abs().mean().item()),
                "endpoint_rate": float((result.codes.abs() == 7).float().mean().item()),
                "zero_rate": float((result.codes == 0).float().mean().item()),
            }
            metadata["artifact_sha256"] = sha256_file(artifact)
            save_json(metadata_path, metadata)
        quant_rows.append({**metadata, "artifact": str(artifact), "metadata": str(metadata_path)})
        log(root, f"quantized {index}/{len(names)} {name}", "quantize")
        del reference_bf16, reference_fp32, rebuilt
        if index % 7 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    result = {
        "status": "PASS",
        "completed_at_utc": utc_now(),
        "module_count": len(names),
        "covered_weight_parameters": sum(int(row["weight_parameters"]) for row in manifest_rows),
        "module_manifest_sha256": canonical_sha256(manifest_rows),
        "quantization_rows_sha256": canonical_sha256(quant_rows),
    }
    save_csv(root / "quantization_manifest.csv", quant_rows)
    save_json(root / "state" / "quantization_complete.json", result)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result
