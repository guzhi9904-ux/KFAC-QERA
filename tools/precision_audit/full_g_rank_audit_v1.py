#!/usr/bin/env python3
"""Offline rank audit of frozen Llama MXINT3 Full-G factors, never a repair.

Reads source artifacts only. Writes small JSON/CSV reports to a disjoint directory.
One committed report per module; restart the same command to resume.
No teacher forward/backward, no changed factors, no new damping, no PPL evaluation.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import sys
import threading
import time
import uuid

sys.dont_write_bytecode = True
import torch
from safetensors import safe_open
from safetensors.torch import load_file

VERSION = "full_g_rank_audit_v1"
RANKS = (8, 16, 32, 64)


def log(message):
    print(f"[rank-audit] {time.strftime('%Y-%m-%dT%H:%M:%S%z')} {message}", flush=True)


@contextmanager
def heartbeat(message):
    log(message)
    start, event = time.monotonic(), threading.Event()
    def tick():
        while not event.wait(30):
            log(f"{message} working elapsed={time.monotonic()-start:.0f}s")
    thread = threading.Thread(target=tick, daemon=True)
    thread.start()
    try:
        yield
    finally:
        event.set()
        thread.join()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            value.update(block)
    return value.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(item) for item in value]
    return value


def sync_dir(path):
    if os.name == "posix":
        fd = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name("."+path.name+"."+uuid.uuid4().hex+".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(clean(value), stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    sync_dir(path.parent)


def safe_name(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).replace(".", "__")


def disjoint(output, protected):
    output = Path(output).resolve()
    if len(output.parts) < 3:
        raise RuntimeError("Audit output must be a specific directory")
    for item in protected:
        item = Path(item).resolve()
        if output == item or output in item.parents or item in output.parents:
            raise RuntimeError(f"Audit output overlaps a protected path: {item}")


class Inputs:
    def __init__(self):
        self.verified = set()

    def verify(self, record):
        key = fingerprint(record)
        path = Path(record["path"])
        if key not in self.verified:
            with heartbeat("hash " + str(path)):
                if path.stat().st_size != record["bytes"] or sha(path) != record["sha256"]:
                    raise RuntimeError(f"Input hash/size mismatch: {path}")
            self.verified.add(key)
        return path

    def tensors(self, record):
        return load_file(str(self.verify(record)))


def norm2(value):
    return sum(float(part.double().square().sum()) for part in value.split(256, dim=0))


def rel(value, reference):
    return math.sqrt(norm2(value.double()-reference.double()) / max(norm2(reference), 1e-30))


def apply_a(root, value):
    return root[:, None]*value if root.ndim == 1 else root @ value


def full_root(gram_sum, count, relative_floor=1e-6):
    # Same operation order as frozen full_g_v1/numerics.py; tested against it.
    if count <= 0 or not 0 < relative_floor < 1 or gram_sum.ndim != 2:
        raise ValueError("Invalid full-G root arguments")
    if gram_sum.shape[0] != gram_sum.shape[1] or not torch.isfinite(gram_sum).all():
        raise ValueError("Full-G must be finite and square")
    raw = gram_sum.double()/count
    asymmetry = rel(raw, raw.T)
    if asymmetry > 1e-10:
        raise RuntimeError("Full-G asymmetry exceeds frozen gate")
    raw = (raw+raw.T)*.5
    mean = float(raw.diagonal().mean())
    if mean <= 0:
        raise ValueError("All-zero/negative Full-G trace")
    raw.div_(mean)
    eigenvalues, vectors = torch.linalg.eigh(raw)
    if not torch.isfinite(eigenvalues).all() or not torch.isfinite(vectors).all():
        raise FloatingPointError("Nonfinite G eigendecomposition")
    negative_limit = 1e-10*max(float(eigenvalues.abs().max()), 1.)
    if float(eigenvalues.min()) < -negative_limit:
        raise RuntimeError("Materially non-PSD Full-G")
    effective = eigenvalues.clamp_min(relative_floor)
    root = (vectors*effective.sqrt()[None, :]) @ vectors.T
    root = ((root+root.T)*.5).float()
    if not torch.isfinite(root).all():
        raise FloatingPointError("Nonfinite Full-G root")
    return root, {"raw_mean_diagonal": mean, "relative_floor": relative_floor,
        "floored_eigenvalues": int((eigenvalues < relative_floor).sum()),
        "negative_eigenvalues": int((eigenvalues < 0).sum()),
        "normalized_min_eigenvalue": float(eigenvalues.min()),
        "normalized_max_eigenvalue": float(eigenvalues.max()),
        "asymmetry": asymmetry, "eigh_dtype": "float64", "root_dtype": "float32"}


def flag_rows(rows):
    previous = None
    for row in rows:
        before, after = row["sse_before"], row["sse_after"]
        flags = row.setdefault("flags", [])
        tolerance = before*1e-5+1e-20
        if not all(math.isfinite(x) for x in row.values() if isinstance(x, (int, float))):
            flags.append("NONFINITE")
        if after > before+tolerance:
            flags.append("WORSE_THAN_NO_CORRECTION")
        if previous is not None and after > previous+tolerance:
            flags.append("NONMONOTONIC_RANK")
        previous = after
        if row.get("left_orthogonality", 0.) > 1e-3:
            flags.append("LEFT_ORTHOGONALITY_GT_1E_3_SCREEN")
        if row.get("bf16_factor_rounding_product_relative", 0.) > .01:
            flags.append("BF16_FACTOR_ROUNDING_GT_1PCT_SCREEN")
        if row.get("rank64_saved_sse_difference_over_before", 0.) > 1e-4:
            flags.append("RANK64_REPLAY_DRIFT_SCREEN")
        if "svd_tail_sse" in row and after > row["svd_tail_sse"]+before*1e-3+1e-20:
            flags.append("ABOVE_SVD_TAIL_SCREEN")
    return rows


def measure(error, sa, sg, left, right, ranks=RANKS, precision="fp32", reference=False, saved_after=None):
    dtype = torch.float32 if precision == "fp32" else torch.float64
    e, a, g = error.to(dtype), sa.to(dtype), sg.to(dtype)
    l, b = left.to(dtype), right.to(dtype)
    m = apply_a(a, e) @ g
    before = norm2(m)
    singular = None
    if reference:
        with heartbeat(f"{precision} full SVD reference (NO inverse solve)"):
            u, singular, vh = torch.linalg.svd(m, full_matrices=True)
            del u, vh
    projected_left = apply_a(a, l)
    result = []
    for rank in ranks:
        with heartbeat(f"{precision} rank={rank} direct weighted residual"):
            correction = l[:, :rank] @ b[:rank]
            remainder = e-correction
            after = norm2(apply_a(a, remainder) @ g)
            c_norm = math.sqrt(norm2(correction))
            ul = projected_left[:, :rank]
            identity = torch.eye(rank, dtype=dtype, device=e.device)
            orth = math.sqrt(norm2(ul.T @ ul-identity) / rank)
            row = {"rank": rank, "precision": precision, "sse_before": before, "sse_after": after,
                   "sse_ratio": after/before if before else None,
                   "weight_mse": norm2(remainder)/remainder.numel(), "correction_norm": c_norm,
                   "correction_over_error_norm": c_norm/max(math.sqrt(norm2(e)), 1e-30),
                   "left_max": float(l[:, :rank].abs().max()), "right_max": float(b[:rank].abs().max()),
                   "left_orthogonality": orth, "flags": []}
            if precision == "fp32":
                # A screening proxy, NOT actual BF16 two-GEMM activation execution.
                rounded = l[:, :rank].bfloat16().float() @ b[:rank].bfloat16().float()
                row["bf16_factor_rounding_product_relative"] = rel(rounded, correction)
                del rounded
                if rank == max(ranks) and saved_after is not None:
                    row["rank64_saved_sse_difference_over_before"] = abs(after-saved_after)/max(before, 1e-30)
            if singular is not None:
                row["svd_tail_sse"] = norm2(singular[rank:])
                row["excess_over_svd_tail_over_before"] = (after-row["svd_tail_sse"])/max(before, 1e-30)
            result.append(row)
            del correction, remainder, ul
    return flag_rows(result)


def committed_states(run, manifest):
    states = {}
    protocol = manifest["payload"]["full_g_protocol"]
    for index, shard in enumerate(protocol["shards"]):
        folder = run/"statistics/checkpoints"/f"shard_{index:03d}"
        state = read_json(folder/"CURRENT.json")
        expected_names = {item["name"] for item in shard}
        if (state["manifest_sha256"] != manifest["sha256"] or state["shard"] != index
                or state["windows_completed"] != 256 or state["prediction_tokens"] != 256*2047
                or set(state["files"]) != expected_names or state["representation"] != "full"
                or state["dtype"] != "float64" or not re.fullmatch("gen_[0-9a-f]{32}", state["generation"])):
            raise RuntimeError("Full-G checkpoint coverage/protocol mismatch")
        generation = folder/state["generation"]
        if generation.is_symlink() or read_json(generation/"STATE.json") != state:
            raise RuntimeError("Full-G generation mismatch")
        if read_json(generation/"OWNER.json") != {"manifest_sha256": manifest["sha256"], "shard": index}:
            raise RuntimeError("Full-G generation ownership mismatch")
        for position, item in enumerate(shard):
            name = item["name"]
            record = state["files"][name]
            if Path(record["path"]).resolve() != (generation/f"layer_{position:03d}.safetensors").resolve():
                raise RuntimeError("Full-G checkpoint pointer mismatch")
            if name in states:
                raise RuntimeError("Duplicate Full-G module")
            states[name] = record
    return states


def metadata(run, manifest, group, layer, states):
    name, payload = layer["name"], manifest["payload"]
    baselines = payload["full_g_protocol"]["baseline_inputs"]
    corrections = {}
    for method in ("diag_gf", "full_gf"):
        path = run/"corrections"/method/(safe_name(name)+".safetensors")
        record = read_json(path.with_suffix(".json"))
        if (record["status"] != "PASS" or record["manifest_sha256"] != manifest["sha256"]
                or record["method"] != method or record["layer"] != name or record["rank"] != 64
                or record["raw_g_reference"] != states[name]
                or record["quant_reference"] != baselines["wq"][name]["quant"]
                or Path(record["file"]["path"]).resolve() != path.resolve()):
            raise RuntimeError("Correction/input binding mismatch: "+name)
        corrections[method] = record
    return {"root": group["roots"], "gram": states[name], "corrections": corrections,
            "quant": baselines["wq"][name]["quant"],
            "model": payload["model_files"][layer["weight_file"]], "shape": layer["shape"]}


def audit_module(name, bindings, inputs, device, fp64, reference):
    root_a = inputs.tensors(bindings["root"])
    raw = inputs.tensors(bindings["gram"])
    d_out, d_in = bindings["shape"]
    if (raw["gram"].shape != (d_out, d_out) or raw["gram"].dtype != torch.float64
            or raw["diagonal"].shape != (d_out,) or raw["diagonal"].dtype != torch.float64
            or not torch.isfinite(raw["diagonal"]).all()
            or not torch.isfinite(raw["gram"]).all() or (raw["diagonal"] < 0).any()):
        raise RuntimeError("Invalid raw Full-G tensor")
    diagonal_error = rel(raw["gram"].diagonal(), raw["diagonal"])
    if diagonal_error > 1e-10:
        raise RuntimeError("Raw Full-G/direct diagonal mismatch")
    with heartbeat(name+" rebuild frozen FP64-eigh/FP32 G root"):
        sg, g_info = full_root(raw["gram"].to(device), 256*2047, 1e-6)
    del raw
    quant = inputs.tensors(bindings["quant"])["weight_q"]
    path = inputs.verify(bindings["model"])
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        weight = handle.get_tensor(name+".weight")
    if weight.dtype != torch.bfloat16 or quant.dtype != torch.bfloat16 or weight.shape != quant.shape:
        raise RuntimeError("Expected frozen matching BF16 W and Wq")
    error = (weight.float()-quant.float()).T.to(device)
    del weight, quant
    report = {"module": name, "g_diagnostics": g_info, "gram_diagonal_error": diagonal_error, "rows": []}
    for method, record in bindings["corrections"].items():
        factors = inputs.tensors(record["file"])
        left, right = factors["A"].to(device), factors["B"].to(device)
        sa = root_a[method.split("_")[0]].to(device)
        expected_a = (d_in,) if method == "diag_gf" else (d_in, d_in)
        if (sa.shape != expected_a or left.shape != (d_in, 64) or right.shape != (64, d_out)
                or any(x.dtype != torch.float32 or not torch.isfinite(x).all() for x in (sa, left, right))):
            raise RuntimeError("Saved root/factor shape, dtype or finiteness mismatch")
        saved_g = record["g_diagnostics"]
        if saved_g["relative_floor"] != 1e-6:
            raise RuntimeError("G floor differs from frozen protocol")
        report[method+"_saved_metrics"] = {key: record[key] for key in (
            "weighted_sse_before", "weighted_sse_after", "g_inverse_residual", "g_diagnostics")}
        for precision in (["fp32", "fp64"] if fp64 else ["fp32"]):
            with heartbeat(name+" "+method+" "+precision):
                rows = measure(error, sa, sg, left, right, precision=precision,
                               reference=reference, saved_after=record["weighted_sse_after"])
            for row in rows:
                row.update(module=name, method=method)
                row["g_floored_fraction"] = g_info["floored_eigenvalues"]/d_out
                row["g_root_condition_spectral_proxy"] = math.sqrt(
                    max(g_info["normalized_max_eigenvalue"], 1e-6)/max(g_info["normalized_min_eigenvalue"], 1e-6))
                row["saved_floor_count_matches"] = saved_g["floored_eigenvalues"] == g_info["floored_eigenvalues"]
                if not row["saved_floor_count_matches"]:
                    row["flags"].append("G_ROOT_FLOOR_COUNT_REPLAY_DIFFERENCE")
                log(f"{name} {method} r{row['rank']} {precision} sse_ratio={row['sse_ratio']} flags={row['flags']}")
            report["rows"].extend(rows)
        del factors, left, right, sa
    return report


def summarize(output, identity, tasks):
    rows, modules = [], []
    for name, binding in tasks:
        path = output/"modules"/(safe_name(name)+".json")
        if not path.exists():
            continue
        record = read_json(path)
        if (record["audit_identity"] != identity or record["binding_sha256"] != fingerprint(binding)
                or record["status"] != "MODULE_AUDITED" or record["module"] != name):
            raise RuntimeError("Audit resume binding mismatch")
        modules.append(name)
        rows.extend(record["rows"])
    for filename, selected in (("rank_metrics.csv", rows), ("flagged_rows.csv", [r for r in rows if r["flags"]])):
        keys = sorted({k for r in rows for k in r}) or ["module", "method", "rank", "flags"]
        path = output/filename
        temporary = path.with_name("."+path.name+"."+uuid.uuid4().hex+".tmp")
        with temporary.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            for row in selected:
                writer.writerow({**row, "flags": "|".join(row["flags"])})
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    result = {"status": "AUDIT_COMPLETE" if len(modules) == len(tasks) else "AUDIT_INCOMPLETE",
              "audited_modules": len(modules), "expected_modules": len(tasks),
              "flagged_rows": sum(bool(row["flags"]) for row in rows),
              "note": "Screening flags are NOT experiment gates. No flags does NOT prove global NLL correctness."}
    atomic_json(output/"status.json", result)
    return result


@contextmanager
def audit_lock(output):
    if os.name != "posix":
        raise RuntimeError("Server CLI requires Linux (math tests run on CPU/Windows)")
    import fcntl
    with (output/".audit.lock").open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-modules", type=int)
    parser.add_argument("--max-hours", type=float)
    parser.add_argument("--module", help="Optional one-module investigation; use a SEPARATE output directory")
    parser.add_argument("--fp64-check", action="store_true", help="Re-evaluate SAME stored FP32 roots/factors in FP64; no new solve")
    parser.add_argument("--svd-reference", action="store_true", help="Expensive full-SVD objective reference, preferably only flagged modules")
    args = parser.parse_args(argv)
    if args.max_new_modules is not None and args.max_new_modules <= 0:
        parser.error("--max-new-modules must be positive")
    if args.max_hours is not None and (not math.isfinite(args.max_hours) or args.max_hours <= 0):
        parser.error("--max-hours must be positive finite")
    run, output = args.run_dir.resolve(), args.output_dir.resolve()
    manifest = read_json(run/"manifest.json")
    payload = manifest["payload"]
    if manifest["sha256"] != fingerprint(payload) or read_json(run/"config.json") != payload["config"]:
        raise RuntimeError("Frozen manifest/config mismatch")
    config = payload["config"]
    if config["experiment_variant"] != "mxint3_full_g_v1" or tuple(config["ranks"]) != RANKS or config["g_relative_floor"] != 1e-6:
        raise RuntimeError("Require frozen Llama MXINT3 full_g_v1 r8/16/32/64 protocol")
    protected = [run.parent, Path(__file__).resolve()]
    for dictionary in (config, payload["source_config"]):
        protected += [dictionary[k] for k in ("run_dir", "source_run_dir", "model_path", "qera_source_dir") if k in dictionary]
    for record in payload["code"].values():
        parts = Path(record["path"]).parts
        if "experiments" in parts:
            protected.append(Path(*parts[:parts.index("experiments")]))
    disjoint(output, protected)
    if args.output_dir.is_symlink():
        raise RuntimeError("Output cannot be a symlink")
    inputs = Inputs()
    for record in payload["code"].values():
        inputs.verify(record)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Production audit requires an idle CUDA GPU; CPU is for unit tests")
    free, total = torch.cuda.mem_get_info(device)
    gpu = torch.cuda.get_device_name(device)
    if "4090" not in gpu or free < 18*2**30:
        raise RuntimeError("Use an idle RTX4090 with at least 18 GiB free")
    torch.set_num_threads(14)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(1234)
    if not str(torch.__version__).startswith("2.3.0+cu121"):
        raise RuntimeError("Use original qera-original-a torch 2.3.0+cu121 environment")
    with heartbeat("read checkpoint pointers and correction metadata (no tensor loads yet)"):
        states = committed_states(run, manifest)
        tasks = [(layer["name"], metadata(run, manifest, group, layer, states))
                 for group in payload["groups"] for layer in group["layers"]]
    if len(tasks) != 224 or {name for name, _ in tasks} != set(states):
        raise RuntimeError("Expected exactly 224 frozen Llama modules")
    if args.module:
        tasks = [item for item in tasks if item[0] == args.module]
        if not tasks:
            parser.error("Unknown --module")
    audit_config = {"version": VERSION, "script_sha256": sha(__file__), "source_manifest": manifest["sha256"],
                    "run_dir": str(run), "device": str(device), "gpu": gpu, "torch": str(torch.__version__),
                    "ranks": RANKS, "fp64_check": args.fp64_check, "svd_reference": args.svd_reference,
                    "selected_module": args.module, "tf32": False,
                    "screening_tolerances": {"sse_growth_over_before": 1e-5, "left_orthogonality": 1e-3,
                        "bf16_factor_product": .01, "rank64_drift_over_before": 1e-4, "svd_excess_over_before": 1e-3},
                    "limits": "No true BF16 activation execution, no PPL, no A condition number; FP64 check promotes saved FP32 tensors only."}
    identity = fingerprint(audit_config)
    if output.exists() and not (output/"audit_config.json").exists():
        if any(output.iterdir()):
            raise RuntimeError("Refusing nonempty unowned audit directory")
    output.mkdir(parents=True, exist_ok=True)
    with audit_lock(output):
        init = output/"audit_config.json"
        if init.exists():
            if read_json(init) != clean(audit_config):
                raise RuntimeError("Audit settings changed; use a new output directory")
        else:
            atomic_json(init, audit_config)
        (output/"modules").mkdir(exist_ok=True)
        if (output/"modules").is_symlink():
            raise RuntimeError("Audit modules directory cannot be a symlink")
        summarize(output, identity, tasks)
        stop = {"requested": False}
        def request_stop(number, frame):
            stop["requested"] = True
            log("Stop requested; commit current MODULE before exiting")
        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
        start, new = time.monotonic(), 0
        for index, (name, binding) in enumerate(tasks):
            path = output/"modules"/(safe_name(name)+".json")
            if path.exists():
                log(f"already audited {name}")
                continue
            if (stop["requested"] or (args.max_hours and time.monotonic()-start >= args.max_hours*3600)
                    or (args.max_new_modules is not None and new >= args.max_new_modules)):
                log("PAUSED: restart same command (remove/increase pilot budget) to resume")
                print(json.dumps(summarize(output, identity, tasks)), flush=True)
                return 75
            with heartbeat(f"module={index+1}/{len(tasks)} {name}"), torch.no_grad():
                report = audit_module(name, binding, inputs, device, args.fp64_check, args.svd_reference)
            report.update(status="MODULE_AUDITED", audit_identity=identity,
                          binding_sha256=fingerprint(binding), inputs=binding)
            atomic_json(path, report)
            new += 1
            gc.collect()
            torch.cuda.empty_cache()
            result = summarize(output, identity, tasks)
            log(f"COMMITTED module={name} audited={result['audited_modules']}/{len(tasks)} flagged_rows={result['flagged_rows']}")
        print(json.dumps(summarize(output, identity, tasks)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
