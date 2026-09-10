#!/usr/bin/env python3
"""Read-only Qwen Full-A audit. No experiment imports, writes, or repairs.

Default: verify referenced input hashes and inspect CPU tensors.
--replay: additionally repeat full SVD and compare FP32/FP64 inverse solves.
Reports are printed to stdout; redirect to a NEW log outside frozen runs.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import threading
import time

sys.dont_write_bytecode = True

import torch
from safetensors import safe_open
from safetensors.torch import load_file


def log(message):
    print(f"[audit-qwen-full-a] {time.strftime('%Y-%m-%dT%H:%M:%S%z')} {message}", flush=True)


@contextmanager
def progress(message):
    log(message)
    started, event = time.monotonic(), threading.Event()
    def tick():
        while not event.wait(30):
            log(f"{message} working elapsed={time.monotonic()-started:.0f}s")
    thread = threading.Thread(target=tick, daemon=True)
    thread.start()
    try:
        yield
    finally:
        event.set()
        thread.join()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(record):
    path = Path(record["path"])
    with progress(f"verify {path}"):
        if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise RuntimeError(f"Input hash/size mismatch: {path}")
    return path


def safe_name(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).replace(".", "__")


def owned_record(root, category, name, digest):
    path = root / category / (safe_name(name) + ".safetensors")
    record = read_json(path.with_suffix(".json"))
    if record["status"] != "PASS" or record["manifest_sha256"] != digest:
        raise RuntimeError(f"Artifact manifest/status mismatch: {path}")
    if Path(record["file"]["path"]).resolve() != path.resolve():
        raise RuntimeError(f"Artifact points elsewhere: {path}")
    verify(record["file"])
    return record


def vector_stats(x):
    x = x.detach().cpu().double()
    positive = x[x > 0]
    return {"min": float(x.min()), "max": float(x.max()),
            "zero_count": int((x == 0).sum()), "negative_count": int((x < 0).sum()),
            "min_positive": float(positive.min()) if positive.numel() else None}


def norm2(x):
    # Accumulate squared norm in FP64 without allocating a full FP64 matrix.
    return sum(float(chunk.double().square().sum()) for chunk in x.split(256, dim=0))


def inspect_root(root):
    if root.dtype != torch.float32 or root.ndim != 2 or root.shape[0] != root.shape[1]:
        raise RuntimeError("Expected saved square FP32 Full-A root")
    if not torch.isfinite(root).all():
        raise RuntimeError("Nonfinite Full-A root")
    row_max = root.abs().amax(dim=1)
    asym = 0.
    for start in range(0, len(root), 256):
        end = min(start+256, len(root))
        asym += norm2(root[start:end].double() - root[:, start:end].T.double())
    return {"shape": list(root.shape), "dtype": str(root.dtype),
            "diagonal": vector_stats(root.diagonal()),
            "row_abs_max": vector_stats(row_max),
            "exact_zero_rows_first_32": (row_max == 0).nonzero().flatten()[:32].tolist(),
            "relative_asymmetry": math.sqrt(asym / max(norm2(root), 1e-300)),
            "warning": "Row/diagonal scales are NOT singular values or a condition-number estimate."}


def tensor_metrics(s, error, u, right, left):
    """Native execution metric plus FP64 check of S L = U. No damping."""
    dtype = left.dtype
    sd, ed, rd = s.to(dtype), error.to(dtype), right.to(dtype)
    correction = left @ rd
    remainder = ed - correction
    before = norm2(sd @ ed)
    after = norm2(sd @ remainder)
    result = {"arithmetic": str(dtype), "weighted_sse_before": before,
              "weighted_sse_after": after,
              "after_over_before": after / before if before else None,
              "weight_mse_after": norm2(remainder) / remainder.numel(),
              "correction_frobenius": math.sqrt(norm2(correction)),
              "left_abs_max": float(left.abs().max()),
              "left_frobenius": math.sqrt(norm2(left))}
    del sd, ed, rd, correction, remainder
    # Small RHS (rank64). Evaluate inverse residual in FP64, block by block.
    left64, u64 = left.double(), u.double()
    inverse_error = 0.
    for start in range(0, len(s), 256):
        end = min(start+256, len(s))
        inverse_error += norm2(s[start:end].double() @ left64 - u64[start:end])
    result["inverse_relative_residual_fp64"] = math.sqrt(inverse_error / max(norm2(u64), 1e-300))
    return result


def replay(root, error, rank, device, emit):
    """Hold FP32 root, error and SVD triplets fixed; vary inverse precision only."""
    if not 0 < rank <= min(error.shape) or error.shape[0] != root.shape[0]:
        raise ValueError("Rank/shape mismatch")
    s, e = root.to(device), error.to(device)
    with progress("FP32 weighted matrix and full_matrices=True SVD"):
        weighted = s @ e
        u_all, singular, vh_all = torch.linalg.svd(weighted, full_matrices=True)
        u = u_all[:, :rank].clone()
        right = singular[:rank, None] * vh_all[:rank]
        del u_all, vh_all
        truncated = norm2(weighted - u @ right)
        baseline = norm2(weighted)
        emit("svd", {"rank": rank, "weighted_sse_before": baseline,
                     "truncated_sse_before_inverse": truncated,
                     "sse_ratio": truncated / baseline if baseline else None,
                     "sigma_max": float(singular[0]), "sigma_at_rank": float(singular[rank-1]),
                     "note": "These singular values belong to S@E, NOT to S."})
        del weighted, singular
    left32 = None
    with progress("FP32 direct solve and reconstruction diagnostics"):
        try:
            left32 = torch.linalg.solve(s, u)
        except torch.linalg.LinAlgError as exc:
            emit("fp32_inverse", {"error": str(exc),
                 "note": "Direct solve failed; official turbulence fallback is deliberately NOT applied."})
        if left32 is not None:
            emit("fp32_inverse", tensor_metrics(s, e, u, right, left32))
    with progress("FP64 direct solve using SAME saved FP32 root and SAME FP32 SVD triplets"):
        try:
            left64 = torch.linalg.solve(s.double(), u.double())
        except torch.linalg.LinAlgError as exc:
            emit("fp64_inverse", {"error": str(exc), "note": "No regularization/pseudoinverse applied."})
            return
        emit("fp64_inverse", tensor_metrics(s, e, u, right, left64))
        if left32 is not None:
            emit("inverse_comparison", {"relative_left_difference":
                 math.sqrt(norm2(left32.double()-left64) / max(norm2(left64), 1e-300))})
    with progress("FP64 inverse result cast back to FP32; diagnostic ONLY"):
        emit("fp64_inverse_cast_fp32", tensor_metrics(s, e, u, right, left64.float()))


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--module", default="model.layers.1.mlp.down_proj")
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(14)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(1234)
    report = {"audit_version": 1, "module": args.module, "torch": torch.__version__,
              "script_sha256": sha256(__file__), "mode": "replay" if args.replay else "inspect",
              "scope": "Read-only diagnostic, NOT a protocol migration or gate pass."}
    def emit(key, value):
        report[key] = clean(value)
        log(key + "=" + json.dumps(report[key], ensure_ascii=False, allow_nan=False))
    with torch.no_grad():
        root_dir = args.run_dir.resolve()
        manifest = read_json(root_dir / "manifest.json")
        payload, digest = manifest["payload"], manifest["sha256"]
        if fingerprint(payload) != digest or payload["config"] != read_json(root_dir / "config.json"):
            raise RuntimeError("Manifest/config identity mismatch")
        if payload["config"].get("experiment_variant") != "qwen25_base_mxint3_v1":
            raise RuntimeError("Expected isolated Qwen2.5-7B Base MXINT3 run")
        for record in payload.get("code", {}).values():
            verify(record)
        emit("environment", {"saved": payload.get("environment"),
                             "current_torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                             "tf32": False, "cpu_threads": 14})
        saved_torch = payload.get("environment", {}).get("torch")
        if args.replay and saved_torch is not None and saved_torch != torch.__version__:
            raise RuntimeError("Use the original Qwen torch environment for numerical replay")
        group = next(g for g in payload["groups"] if any(l["name"] == args.module for l in g["layers"]))
        layer = next(l for l in group["layers"] if l["name"] == args.module)
        root_record = owned_record(root_dir, "roots", group["target"], digest)
        verify(root_record["raw_reference"])
        emit("provenance", {"manifest_sha256": digest, "root_record": root_record})
        values = load_file(root_record["file"]["path"])
        s = values["full"]
        if tuple(s.shape) != (layer["shape"][1],)*2:
            raise RuntimeError("Module/root shape mismatch")
        with progress("CPU root inspection (no eigendecomposition)"):
            emit("saved_root", inspect_root(s))
            emit("saved_diagonal_root", vector_stats(values["diag"]))
            raw = load_file(root_record["raw_reference"]["path"])
            if raw["full"].dtype != torch.float64 or raw["diag"].dtype != torch.float32:
                raise RuntimeError("Unexpected raw A dtypes")
            emit("raw_gram_diagonal", vector_stats(raw["full"].diagonal()/524288))
            emit("raw_diag_accumulator", vector_stats(raw["diag"]/524288))
            del raw
        if args.replay:
            device = torch.device(args.device)
            if device.type == "cuda":
                free, total = torch.cuda.mem_get_info(device)
                emit("gpu", {"name": torch.cuda.get_device_name(device), "free_GiB": free/2**30,
                             "total_GiB": total/2**30})
                if free < 16*2**30:
                    raise RuntimeError("Need >=16 GiB free on selected GPU; do not contend with another experiment")
            quant = owned_record(root_dir, "quantized", args.module, digest)
            expected_quant = {"name": "mxint", "width": 3, "block_size": 32, "block_axis": -1}
            if quant.get("quantization") != expected_quant or quant.get("fp32_bf16_equal") is not True:
                raise RuntimeError("Expected audited W3 MXINT artifact")
            model_record = payload["model_files"][layer["weight_file"]]
            model_path = verify(model_record)
            with safe_open(str(model_path), framework="pt", device="cpu") as handle:
                weight = handle.get_tensor(args.module + ".weight")
            q = load_file(quant["file"]["path"])["weight_q"]
            if weight.dtype != torch.bfloat16 or q.dtype != torch.bfloat16 or weight.shape != q.shape:
                raise RuntimeError("Expected matching BF16 model and frozen Wq tensors")
            error = (weight.float()-q.float()).T
            emit("replay_inputs", {"quant_record": quant, "model_record": model_record,
                                   "error_shape": list(error.shape), "error_sse": norm2(error)})
            del weight, q
            gc.collect()
            replay(s, error, max(payload["config"]["ranks"]), device, emit)
    emit("status", "AUDIT_COMPLETE_NOT_AN_EXPERIMENT_PASS")
    print(json.dumps(clean(report), indent=2, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
