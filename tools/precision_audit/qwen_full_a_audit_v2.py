#!/usr/bin/env python3
"""Read-only Qwen audit v2: distribution-to-distribution environment checks.

Reuse hash-pinned v1 numerical and input-verification functions unchanged.
Never change torch.__version__, the manifest, or any old source file.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import torch

HELPER_SHA = "a3adec7792edb696130039d92869235d9e06d8fd8001ee8d41ba6a53c1659b7f"
PACKAGES = ("torch", "transformers", "accelerate", "datasets", "tokenizers", "safetensors", "numpy", "scipy")
EXPECTED_RUNTIME = "2.3.0+cu121"
EXPECTED_CUDA = "12.1"


def environment_report(saved, current, runtime, cuda):
    """Version sources stay separate; CUDA suffixes are not blindly stripped."""
    if set(saved) != set(PACKAGES) or set(current) != set(PACKAGES):
        raise RuntimeError("Expected all eight original distribution versions")
    mismatches = {k: {"saved": saved[k], "current": current[k]} for k in PACKAGES if saved[k] != current[k]}
    return {"saved_distributions": saved, "current_distributions": current,
            "distribution_mismatches": mismatches, "torch_runtime": str(runtime), "cuda_runtime": cuda,
            "expected_torch_runtime": EXPECTED_RUNTIME, "expected_cuda_runtime": EXPECTED_CUDA,
            "status": "PASS" if not mismatches and str(runtime) == EXPECTED_RUNTIME and cuda == EXPECTED_CUDA else "FAIL",
            "runtime_reference": "Original Qwen runtime log and user environment verification, 2026-09-11",
            "tf32": False, "cpu_threads": 14}


def load_helper(path):
    path = Path(path).resolve()
    if hashlib.sha256(path.read_bytes()).hexdigest() != HELPER_SHA:
        raise RuntimeError("Frozen v1 helper hash mismatch; do not edit or bypass")
    spec = importlib.util.spec_from_file_location("qwen_audit_frozen_v1", path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--helper-file", type=Path, default=Path(__file__).with_name("qwen_full_a_audit_v1.py"))
    p.add_argument("--module", default="model.layers.1.mlp.down_proj")
    p.add_argument("--replay", action="store_true")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args(argv)
    h = load_helper(args.helper_file)
    torch.set_num_threads(14)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(1234)
    report = {"audit_version": 2, "module": args.module, "mode": "replay" if args.replay else "inspect",
              "script_sha256": h.sha256(__file__), "helper_sha256": HELPER_SHA,
              "scope": "Read-only; environment comparison fixed, v1 numerical functions unchanged. Not a repair or protocol pass."}
    def emit(key, value):
        report[key] = h.clean(value)
        h.log(key + "=" + json.dumps(report[key], ensure_ascii=False, allow_nan=False))
    with torch.no_grad():
        root = args.run_dir.resolve()
        manifest = h.read_json(root/"manifest.json")
        payload, digest = manifest["payload"], manifest["sha256"]
        if h.fingerprint(payload) != digest or payload["config"] != h.read_json(root/"config.json"):
            raise RuntimeError("Manifest/config identity mismatch")
        if payload["config"].get("experiment_variant") != "qwen25_base_mxint3_v1":
            raise RuntimeError("Expected isolated Qwen2.5-7B Base MXINT3 run")
        current = {name: importlib.metadata.version(name) for name in PACKAGES}
        environment = environment_report(payload["environment"], current, torch.__version__, torch.version.cuda)
        emit("environment", environment)
        if args.replay and environment["status"] != "PASS":
            raise RuntimeError("Distribution or recorded CUDA build mismatch; replay blocked")
        for record in payload.get("code", {}).values():
            h.verify(record)
        group = next(g for g in payload["groups"] if any(l["name"] == args.module for l in g["layers"]))
        layer = next(l for l in group["layers"] if l["name"] == args.module)
        root_record = h.owned_record(root, "roots", group["target"], digest)
        h.verify(root_record["raw_reference"])
        emit("provenance", {"manifest_sha256": digest, "root_record": root_record})
        values = h.load_file(root_record["file"]["path"])
        s = values["full"]
        if tuple(s.shape) != (layer["shape"][1],)*2:
            raise RuntimeError("Module/root shape mismatch")
        with h.progress("CPU root inspection (no eigendecomposition)"):
            emit("saved_root", h.inspect_root(s))
            emit("saved_diagonal_root", h.vector_stats(values["diag"]))
            raw = h.load_file(root_record["raw_reference"]["path"])
            if raw["full"].dtype != torch.float64 or raw["diag"].dtype != torch.float32:
                raise RuntimeError("Unexpected raw A dtypes")
            emit("raw_gram_diagonal", h.vector_stats(raw["full"].diagonal()/524288))
            emit("raw_diag_accumulator", h.vector_stats(raw["diag"]/524288))
            del raw
        if args.replay:
            device = torch.device(args.device)
            if device.type == "cuda":
                free, total = torch.cuda.mem_get_info(device)
                emit("gpu", {"name": torch.cuda.get_device_name(device), "free_GiB": free/2**30, "total_GiB": total/2**30})
                if free < 16*2**30:
                    raise RuntimeError("Need >=16 GiB free on selected GPU; no contention with another experiment")
            quant = h.owned_record(root, "quantized", args.module, digest)
            if quant.get("quantization") != {"name": "mxint", "width": 3, "block_size": 32, "block_axis": -1} or quant.get("fp32_bf16_equal") is not True:
                raise RuntimeError("Expected audited W3 MXINT artifact")
            model_record = payload["model_files"][layer["weight_file"]]
            with h.safe_open(str(h.verify(model_record)), framework="pt", device="cpu") as f:
                weight = f.get_tensor(args.module+".weight")
            q = h.load_file(quant["file"]["path"])["weight_q"]
            if weight.dtype != torch.bfloat16 or q.dtype != torch.bfloat16 or weight.shape != q.shape:
                raise RuntimeError("Expected matching BF16 W and Wq")
            error = (weight.float()-q.float()).T
            emit("replay_inputs", {"quant_record": quant, "model_record": model_record,
                                   "error_shape": list(error.shape), "error_sse": h.norm2(error)})
            del weight, q
            gc.collect()
            h.replay(s, error, max(payload["config"]["ranks"]), device, emit)
        emit("status", "AUDIT_COMPLETE_NOT_AN_EXPERIMENT_PASS")
        print(json.dumps(h.clean(report), indent=2, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
