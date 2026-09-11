#!/usr/bin/env python3
"""One Qwen module, stored FP32 Full-A root, FP64 product/SVD/solve, G=I.

Diagnostic only. Candidate factors and reports live in a new locked directory.
No model deployment, PPL, recollection, new root, regularization or fallback.
"""
from __future__ import annotations
import argparse
import importlib
import importlib.metadata
import importlib.util
import math
from pathlib import Path
import shutil
import sys
import time

sys.dont_write_bytecode = True
import torch

TARGET = "model.layers.1.mlp.down_proj"
RANKS = (8, 16, 32, 64)
VERSION = "qwen_a_fp64_target_v1"
INVERSE_TOL = 1e-9
TAIL_TOL = 1e-9
HELPERS = {
    "full_g_rank_audit_v1.py": "1c39e84f7f2c1ac9dfdcbb4ee7291760cd047c8222cae0cd9404e0dd5276d3e9",
    "full_g_target_probe_v1.py": "eb2db6b5d544132fa16fa9f463edf3ee7039eefea261212e13b7f4b4983bf827",
    "full_g_precision_r8_v1.py": "793f1bc5ef0c27362af9acae4989cb089aedd6ddc9597213ab64ca15b47f4e7e",
    "qwen_full_a_audit_v1.py": "a3adec7792edb696130039d92869235d9e06d8fd8001ee8d41ba6a53c1659b7f",
}
V2_SHA = "669696c95d529f71bdd1a5ef1e02b2d37a176b65cbd6d506cbef44aee5b8bb45"


def norm2(x):
    return sum(float(chunk.double().square().sum()) for chunk in x.split(256, dim=0))


def relative(x, reference):
    return math.sqrt(norm2(x.double()-reference.double()) / max(norm2(reference), 1e-30))


def solve(root, error, ranks=RANKS):
    """Same algebra/order as Llama stored-root solve, including explicit G=I.

    Return failed numerical gates as diagnostics, not silently relaxed solves.
    FP64 and rounded-BF16 proxies are evaluated in common FP64 arithmetic.
    """
    if root.dtype != torch.float32 or error.dtype != torch.float32:
        raise RuntimeError("Require exact stored FP32 root and original FP32 quantization error")
    if root.ndim != 2 or error.ndim != 2 or root.shape != (error.shape[0],)*2:
        raise RuntimeError("Root/error shape mismatch")
    if not torch.isfinite(root).all() or not torch.isfinite(error).all():
        raise RuntimeError("Nonfinite inputs")
    if tuple(sorted(set(ranks))) != tuple(ranks) or not ranks or not 0 < ranks[0] <= ranks[-1] <= min(error.shape):
        raise ValueError("Invalid ranks")
    sa, e = root.double(), error.double()
    g = torch.eye(e.shape[1], dtype=torch.float64, device=e.device)
    weighted = sa @ e @ g
    u, singular, vh = torch.linalg.svd(weighted, full_matrices=True)
    k = max(ranks)
    u, vh = u[:, :k].clone(), vh[:k].clone()
    right_target = singular[:k, None]*vh
    left = torch.linalg.solve(sa, u)
    right = torch.linalg.solve(g.T, right_target.T).T
    if not torch.isfinite(left).all() or not torch.isfinite(right).all():
        raise RuntimeError("Nonfinite FP64 factors; no artifact accepted")
    inverse_a = relative(sa @ left, u)
    inverse_g = relative(right @ g, right_target)
    before, error_norm = norm2(weighted), math.sqrt(norm2(e))
    del weighted
    rows = []
    previous_after = before
    for rank in ranks:
        l, b = left[:, :rank], right[:rank]
        correction = l @ b
        remainder = e-correction
        after = norm2(sa @ remainder @ g)
        tail = norm2(singular[rank:])
        excess = (after-tail)/max(before, 1e-30)
        inverse_r = relative(sa @ l, u[:, :rank])
        finite = all(math.isfinite(v) for v in (before, after, tail, excess, inverse_r, inverse_g))
        numerical_pass = (finite and inverse_r <= INVERSE_TOL and inverse_g <= INVERSE_TOL
                          and abs(excess) <= TAIL_TOL and after <= previous_after + max(before, 1e-30)*TAIL_TOL)
        lr, br = l.bfloat16().double(), b.bfloat16().double()
        rounded_finite = bool(torch.isfinite(lr).all() and torch.isfinite(br).all())
        rounded_sse = rounding_ratio = rounded_mse = None
        if rounded_finite:
            rounded = lr @ br
            rounded_sse = norm2(sa @ (e-rounded) @ g)
            rounded_mse = norm2(e-rounded)/e.numel()
            rounding_ratio = math.sqrt(norm2(rounded-correction))/max(error_norm, 1e-30)
            rounded_finite = all(math.isfinite(v) for v in (rounded_sse, rounded_mse, rounding_ratio))
        rows.append({"rank": rank, "weighted_sse_before": before, "weighted_sse_after_fp64": after,
                     "svd_tail_sse_fp64": tail, "relative_tail_difference": excess,
                     "a_inverse_relative_residual_fp64": inverse_r, "g_inverse_relative_residual_fp64": inverse_g,
                     "numerical_gate_passed": numerical_pass,
                     "weight_mse_before": norm2(e)/e.numel(), "weight_mse_after_fp64": norm2(remainder)/e.numel(),
                     "correction_frobenius": math.sqrt(norm2(correction)), "error_frobenius": error_norm,
                     "correction_over_error_norm": math.sqrt(norm2(correction))/max(error_norm, 1e-30),
                     "left_abs_max": float(l.abs().max()), "right_abs_max": float(b.abs().max()),
                     "bf16_factors_finite": rounded_finite, "bf16_rounded_sse_fp64_proxy": rounded_sse,
                     "bf16_rounded_weight_mse_proxy": rounded_mse,
                     "bf16_rounding_product_over_error_norm": rounding_ratio,
                     "bf16_proxy_objective_increased_flag": not rounded_finite or rounded_sse > before*(1+1e-5)+1e-20})
        previous_after = after
    info = {"a_inverse_relative_residual_fp64": inverse_a, "g_inverse_relative_residual_fp64": inverse_g,
            "inverse_tolerance": INVERSE_TOL, "tail_tolerance": TAIL_TOL,
            "numerical_gate_passed": all(r["numerical_gate_passed"] for r in rows),
            "bf16_factors_finite": all(r["bf16_factors_finite"] for r in rows),
            "bf16_proxy_flags": sum(r["bf16_proxy_objective_increased_flag"] for r in rows),
            "note": "BF16-rounded factors multiplied in FP64: not actual BF16 activations, not endpoint PPL."}
    values = {"A_fp64": left.cpu(), "B_fp64": right.cpu(),
              "A_bf16": left.bfloat16().cpu(), "B_bf16": right.bfloat16().cpu(),
              "singular_values": singular.cpu()}
    return values, rows, info


def load_libraries(tools_dir, v2_path):
    import hashlib
    for name, digest in HELPERS.items():
        if hashlib.sha256((tools_dir/name).read_bytes()).hexdigest() != digest:
            raise RuntimeError("Frozen helper mismatch: "+name)
    if hashlib.sha256(v2_path.read_bytes()).hexdigest() != V2_SHA:
        raise RuntimeError("Version-audit helper mismatch")
    sys.path.insert(0, str(tools_dir))
    single = importlib.import_module("full_g_precision_r8_v1")
    for module in (single, single.core, single.previous):
        if single.core.sha(module.__file__) != HELPERS[Path(module.__file__).name]:
            raise RuntimeError("Wrong imported helper")
    spec = importlib.util.spec_from_file_location("qwen_audit_versions_v2", v2_path)
    v2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v2)
    h = v2.load_helper(tools_dir/"qwen_full_a_audit_v1.py")
    return single, v2, h


def validate_candidate(values, shape):
    d_out, d_in = shape
    specs = {"A_fp64": ((d_in,64), torch.float64), "B_fp64": ((64,d_out), torch.float64),
             "A_bf16": ((d_in,64), torch.bfloat16), "B_bf16": ((64,d_out), torch.bfloat16),
             "singular_values": ((min(shape),), torch.float64)}
    if set(values) != set(specs):
        raise RuntimeError("Candidate schema mismatch")
    for key, (dims, dtype) in specs.items():
        if tuple(values[key].shape) != dims or values[key].dtype != dtype:
            raise RuntimeError("Candidate shape/dtype mismatch")
        if key.endswith("fp64") or key == "singular_values":
            if not torch.isfinite(values[key]).all():
                raise RuntimeError("Nonfinite FP64 candidate")
    for key in ("A", "B"):
        if not torch.equal(values[key+"_fp64"].bfloat16(), values[key+"_bf16"]):
            raise RuntimeError("Candidate BF16 factors not direct FP64 casts")


def resume(output, identity, shape, single, h):
    path = output/"report.json"
    if not path.exists():
        return None
    record = h.read_json(path)
    if (record.get("experiment_identity") != identity or record.get("status") != "DIAGNOSTIC_COMPLETE"
            or [r["rank"] for r in record["rank_metrics"]] != list(RANKS)):
        raise RuntimeError("Existing report identity/status mismatch")
    for field, name in (("candidate_file", "candidate_factors.safetensors"), ("metrics_file", "rank_metrics.csv")):
        if Path(record[field]["path"]).resolve() != (output/name).resolve():
            raise RuntimeError("Existing report path mismatch")
        h.verify(record[field])
    values = h.load_file(record["candidate_file"]["path"])
    validate_candidate(values, shape)
    if record["factor_bits"] != {k: single.tensor_record(v) for k,v in values.items()}:
        raise RuntimeError("Candidate tensor bits changed")
    return record


def outcome_code(record):
    return 0 if record["checks"]["numerical_gate_passed"] and record["checks"]["bf16_factors_finite"] else 2


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("run-dir", "output-dir", "tools-dir", "v2-helper"):
        p.add_argument("--"+key, required=True, type=Path)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args(argv)
    run, output = args.run_dir.resolve(), args.output_dir.resolve()
    single, v2, h = load_libraries(args.tools_dir.resolve(), args.v2_helper.resolve())
    core = single.core
    manifest = h.read_json(run/"manifest.json")
    payload, digest = manifest["payload"], manifest["sha256"]
    if h.fingerprint(payload) != digest or payload["config"] != h.read_json(run/"config.json"):
        raise RuntimeError("Manifest/config identity mismatch")
    if payload["config"].get("experiment_variant") != "qwen25_base_mxint3_v1" or payload["config"].get("ranks") != list(RANKS):
        raise RuntimeError("Require frozen Qwen base W3 four-rank protocol")
    protected = [run.parent, args.tools_dir, args.v2_helper.parent, Path(__file__).parent]
    for settings in (payload["config"], payload.get("source_config", {})):
        protected.extend(settings[k] for k in ("model_path", "run_dir", "source_run_dir", "qera_source_dir") if k in settings)
    for record in payload.get("code", {}).values():
        source = Path(record["path"])
        if "experiments" in source.parts:
            protected.append(Path(*source.parts[:source.parts.index("experiments")]))
    core.disjoint(output, protected)
    if args.output_dir.is_symlink() or (output.exists() and any(x.is_symlink() for x in output.rglob("*"))):
        raise RuntimeError("No symlinks in new output")
    current = {name: importlib.metadata.version(name) for name in v2.PACKAGES}
    environment = v2.environment_report(payload["environment"], current, torch.__version__, torch.version.cuda)
    h.log("environment="+str(environment))
    if environment["status"] != "PASS":
        raise RuntimeError("Original Qwen environment required")
    torch.set_num_threads(14); torch.manual_seed(1234)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    for record in payload.get("code", {}).values():
        h.verify(record)
    group = next(g for g in payload["groups"] if any(l["name"] == TARGET for l in g["layers"]))
    layer = next(l for l in group["layers"] if l["name"] == TARGET)
    if list(layer["shape"]) != [3584, 18944]:
        raise RuntimeError("Unexpected Qwen target shape")
    root_rec = h.owned_record(run, "roots", group["target"], digest)
    h.verify(root_rec["raw_reference"])
    quant = h.owned_record(run, "quantized", TARGET, digest)
    if quant.get("quantization") != {"name":"mxint", "width":3, "block_size":32, "block_axis":-1} or quant.get("fp32_bf16_equal") is not True:
        raise RuntimeError("Frozen W3 quantization mismatch")
    model_rec = payload["model_files"][layer["weight_file"]]
    h.verify(model_rec)
    device = torch.device(args.device)
    if device.type != "cuda" or "4090" not in torch.cuda.get_device_name(device):
        raise RuntimeError("Server experiment requires a 4090 CUDA device")
    experiment = {"version": VERSION, "module": TARGET, "method": "full_gi", "ranks": list(RANKS), "solve_rank": 64,
                  "script_sha256": h.sha256(__file__), "helpers": HELPERS, "v2_sha256": V2_SHA,
                  "manifest": single.file_record(run/"manifest.json"), "root": root_rec, "quant": quant,
                  "model": model_rec, "shape": layer["shape"], "environment": environment,
                  "device": str(device), "gpu_name": torch.cuda.get_device_name(device),
                  "policy": "Stored FP32 A root and FP32 error promoted to FP64; G=I; FP64 product/full SVD/two solves; direct BF16 casts",
                  "inverse_tolerance": INVERSE_TOL, "tail_tolerance": TAIL_TOL,
                  "new_statistics": False, "new_root": False, "regularization": False,
                  "ppl": False, "bf16_metrics": "Rounded-factor proxy measured in FP64, not activation execution"}
    identity = h.fingerprint(experiment)
    if output.exists() and not (output/"experiment.json").exists() and any(output.iterdir()):
        raise RuntimeError("Refusing unowned nonempty output")
    output.mkdir(parents=True, exist_ok=True)
    with core.audit_lock(output):
        exp_path = output/"experiment.json"
        if exp_path.exists() and h.read_json(exp_path) != experiment:
            raise RuntimeError("Experiment changed: use a new output; do not overwrite")
        if not exp_path.exists():
            core.atomic_json(exp_path, experiment)
        record = resume(output, identity, layer["shape"], single, h)
        if record is not None:
            h.log("ALREADY COMPLETE: "+str(record["checks"]))
            return outcome_code(record)
        try:
            if torch.cuda.mem_get_info(device)[0] < 18*2**30:
                raise RuntimeError("Need >=18 GiB free on selected GPU; no other GPU experiment")
            if shutil.disk_usage(output).free < 2*2**30:
                raise RuntimeError("Need >=2 GiB free in diagnostic output filesystem")
            started = time.monotonic()
            with torch.no_grad(), h.progress("TARGET FULL FP64: saved root -> weighted matrix -> full SVD -> solves -> four ranks / BF16 proxy"):
                root = h.load_file(root_rec["file"]["path"])["full"].to(device)
                with h.safe_open(str(model_rec["path"]), framework="pt", device="cpu") as f:
                    w = f.get_tensor(TARGET+".weight")
                q = h.load_file(quant["file"]["path"])["weight_q"]
                if w.dtype != torch.bfloat16 or q.dtype != torch.bfloat16 or w.shape != q.shape or list(w.shape) != layer["shape"]:
                    raise RuntimeError("W/Wq dtype/shape mismatch")
                error = (w.float()-q.float()).T.to(device)
                values, rows, checks = solve(root, error)
                validate_candidate(values, layer["shape"])
                for row in rows:
                    h.log("rank_metrics="+str(row))
                h.log("checks="+str(checks))
                candidate = single.atomic_tensors(output/"candidate_factors.safetensors", values)
                single.previous.write_csv(output/"rank_metrics.csv", rows)
                record = {"experiment_identity": identity, "status": "DIAGNOSTIC_COMPLETE", "module": TARGET,
                          "rank_metrics": rows, "checks": checks, "candidate_file": candidate,
                          "metrics_file": single.file_record(output/"rank_metrics.csv"),
                          "factor_bits": {k: single.tensor_record(v) for k,v in values.items()},
                          "root_bits": single.tensor_record(root), "error_bits": single.tensor_record(error),
                          "elapsed_seconds": time.monotonic()-started,
                          "gpu_peak_GiB": torch.cuda.max_memory_allocated(device)/2**30,
                          "deployment_authorized": False,
                          "candidate_status": "NUMERICAL_CHECKS_PASSED" if outcome_code({"checks": checks}) == 0 else "QUARANTINED_NUMERICAL_FAILURE",
                          "note": "Diagnostic candidates only. Failed gates remain failed. No PPL, no automatic deployment or source update."}
                core.atomic_json(output/"report.json", record)
            h.log("TARGET DIAGNOSTIC COMPLETE: "+record["candidate_status"]+"; report="+str(output/"report.json"))
            return outcome_code(record)
        except Exception as exc:
            core.atomic_json(output/"failure.json", {"experiment_identity": identity, "status":"FAILED",
                             "type":type(exc).__name__, "message":str(exc), "source_modified":False})
            raise


if __name__ == "__main__":
    raise SystemExit(main())
