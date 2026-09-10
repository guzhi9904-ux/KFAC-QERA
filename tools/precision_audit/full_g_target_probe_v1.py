#!/usr/bin/env python3
"""Isolated Llama MXINT3 single-module diagnostics and guarded hybrid evaluation.

Requires the unchanged full_g_rank_audit_v1.py beside this file. Source runs and
checkouts are read-only. No statistics recollection, source edits, or new factors
are saved. Numerical re-solves are diagnostics only, never used for evaluation.
"""
from __future__ import annotations

import argparse
import csv
import gc
import importlib.metadata
import math
import os
from pathlib import Path
import signal
import sys
import time
import uuid

sys.dont_write_bytecode = True
import torch
from safetensors import safe_open
import full_g_rank_audit_v1 as core

VERSION = "full_g_target_probe_v1"
HELPER_SHA = "1c39e84f7f2c1ac9dfdcbb4ee7291760cd047c8222cae0cd9404e0dd5276d3e9"
TARGET = "model.layers.0.self_attn.o_proj"
METHODS = ("full_gi", "full_gd", "full_gf")


class Paused(Exception):
    pass


class Budget:
    def __init__(self, hours=None, batches=None):
        self.start = time.monotonic()
        self.hours, self.batches, self.done, self.requested = hours, batches, 0, False

    def signal(self, number, frame):
        self.requested = True
        core.log("Stop requested; finish and commit the current numerical method/evaluation batch")

    def check(self):
        if (self.requested or (self.hours is not None and time.monotonic()-self.start >= self.hours*3600)
                or (self.batches is not None and self.done >= self.batches)):
            raise Paused()


def write_csv(path, rows):
    if not rows:
        return
    fields = sorted(set().union(*(r.keys() for r in rows)))
    temporary = path.with_name("."+path.name+"."+uuid.uuid4().hex+".tmp")
    with temporary.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(core.clean(rows))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    core.sync_dir(path.parent)


def valid_records(records, total):
    if len(records) > total:
        raise RuntimeError("Too many evaluation records")
    for i, row in enumerate(records):
        if (row["window"] != i or row["tokens"] != 2047 or not math.isfinite(row["nll_sum"])
                or row["nll_sum"] < 0):
            raise RuntimeError("Invalid/noncontiguous evaluation records")
    return records


def get_records(path, identity, total=138):
    if not path.exists():
        return []
    value = core.read_json(path)
    if value.get("probe_identity") != identity:
        raise RuntimeError("Checkpoint belongs to a different probe")
    records = valid_records(value["records"], total)
    if value.get("complete") != (len(records) == total):
        raise RuntimeError("Evaluation completion marker mismatch")
    return records


def ppl(records):
    return math.exp(sum(r["nll_sum"] for r in records)/sum(r["tokens"] for r in records))


def route_hybrid(base, replacement, target):
    if target not in base or target not in replacement:
        raise RuntimeError("Replacement target missing")
    if base[target]["quant"] != replacement[target]["quant"]:
        raise RuntimeError("Hybrid must use EXACTLY the same quantized weight")
    result = {name: dict(files) for name, files in base.items()}
    result[target] = dict(replacement[target])
    changed = [name for name in base if base[name] != result[name]]
    if changed != [target]:
        raise RuntimeError("Hybrid must change exactly one module's correction")
    return result


def diagonal_root(raw):
    raw = raw.double()/(256*2047)
    if raw.ndim != 1 or not torch.isfinite(raw).all() or (raw < 0).any() or raw.mean() <= 0:
        raise RuntimeError("Invalid ORIGINAL DG statistics")
    return (raw/raw.mean()).clamp_min(1e-6).sqrt().float()


def metrics(error, a, g, left, right, rank):
    # All measurements FP64; inputs may be saved FP32 or newly re-solved FP64.
    e, sa, sg = error.double(), a.double(), g.double()
    l, b = left[:, :rank].double(), right[:rank].double()
    correction = l @ b
    residual = e-correction
    before = core.norm2(sa @ e @ sg)
    after = core.norm2(sa @ residual @ sg)
    ul = sa @ l
    rounded = l.bfloat16().double() @ b.bfloat16().double()
    rounding_norm = math.sqrt(core.norm2(rounded-correction))
    error_norm = math.sqrt(core.norm2(e))
    return {"rank": rank, "sse_before": before, "sse_after": after,
            "sse_ratio": after/before, "weight_mse": core.norm2(residual)/e.numel(),
            "correction_over_error_norm": math.sqrt(core.norm2(correction))/error_norm,
            "left_orthogonality_fp64": math.sqrt(core.norm2(ul.T @ ul-torch.eye(rank, device=e.device))/rank),
            "bf16_rounding_product_over_error_norm": rounding_norm/error_norm,
            "bf16_rounding_product_relative": rounding_norm/max(math.sqrt(core.norm2(correction)), 1e-30)}


def numerical_method(error, a, g, left, right, ranks=core.RANKS):
    """Same saved roots; no damping/pseudoinverse. Fresh FP64 SVD + true solve.

    Fixed-U A-side comparison isolates inverse precision from SVD basis changes.
    Direct FP32 solve failure is recorded, not silently repaired.
    """
    e, sa, sg = error.double(), a.double(), g.double()
    u, s, vh = torch.linalg.svd(sa @ e @ sg, full_matrices=True)
    k = max(ranks)
    u, vh = u[:, :k].clone(), vh[:k].clone()
    target = s[:k, None]*vh
    l64 = torch.linalg.solve(sa, u)
    b64 = torch.linalg.solve(sg.T, target.T).T
    inverse = {"a_inverse_residual_fp64": core.rel(sa @ l64, u),
               "g_inverse_residual_fp64": core.rel(b64 @ sg, target)}
    try:
        l32 = torch.linalg.solve(a.float(), u.float())
        l64_fixed = torch.linalg.solve(sa, u.float().double())
        inverse.update(a_inverse_residual_fp32_fixed_u=core.rel(sa @ l32.double(), u.float().double()),
                       a_inverse_solution_fp32_vs_fp64=core.rel(l32.double(), l64_fixed))
    except RuntimeError as exc:
        inverse["fp32_solve_error"] = str(exc)
    rows = []
    for rank in ranks:
        saved = metrics(e, sa, sg, left, right, rank)
        fresh = metrics(e, sa, sg, l64, b64, rank)
        tail = core.norm2(s[rank:])
        row = {"rank": rank, "svd_tail_sse_fp64": tail, **inverse}
        row.update({"saved_"+key: val for key, val in saved.items() if key != "rank"})
        row.update({"resolved_fp64_"+key: val for key, val in fresh.items() if key != "rank"})
        row["saved_excess_over_tail_over_before"] = (saved["sse_after"]-tail)/saved["sse_before"]
        row["resolved_excess_over_tail_over_before"] = (fresh["sse_after"]-tail)/fresh["sse_before"]
        row["saved_vs_resolved_product_relative"] = core.rel(
            left[:, :rank].double() @ right[:rank].double(), l64[:, :rank] @ b64[:rank])
        row["relative_singular_gap_at_rank"] = float((s[rank-1]-s[rank])/s[rank-1]) if rank < len(s) else None
        rows.append(row)
    return rows


def diagnostics(manifest, bindings, inputs, device, output, identity, stop):
    destination = output/"diagnostics"
    destination.mkdir(exist_ok=True)
    if destination.is_symlink():
        raise RuntimeError("Symlink report folder")
    for path in destination.glob("*.json"):
        if core.read_json(path).get("probe_identity") != identity:
            raise RuntimeError("Numerical checkpoint identity mismatch")
    if all((destination/(m+".json")).exists() for m in METHODS):
        core.log("All three numerical methods already complete")
    else:
        stop.check()
        a = inputs.tensors(bindings["root"])["full"].to(device)
        raw = inputs.tensors(bindings["gram"])
        if core.rel(raw["gram"].diagonal(), raw["diagonal"]) > 1e-10:
            raise RuntimeError("Full-G direct diagonal mismatch")
        original_state = manifest["payload"]["full_g_protocol"]["diagonal_g_state"]
        inputs.verify(manifest["payload"]["full_g_protocol"]["diagonal_g_state_file"])
        if (original_state["windows_completed"] != 256 or original_state["prediction_tokens"] != 256*2047
                or core.read_json(manifest["payload"]["full_g_protocol"]["diagonal_g_state_file"]["path"]) != original_state):
            raise RuntimeError("Original DG checkpoint changed")
        original_dg = inputs.tensors(original_state["file"])[TARGET]
        dg_drift = core.rel(raw["diagonal"], original_dg)
        if dg_drift > 1e-6:
            raise RuntimeError("Full-G diagonal differs from original DG")
        with core.heartbeat("rebuild frozen Full-G root"):
            gf, g_info = core.full_root(raw["gram"].to(device), 256*2047)
        gd = torch.diag(diagonal_root(original_dg).to(device))
        del raw, original_dg
        weight_path = inputs.verify(bindings["model"])
        with safe_open(str(weight_path), framework="pt", device="cpu") as handle:
            weight = handle.get_tensor(TARGET+".weight")
        quant = inputs.tensors(bindings["quant"])["weight_q"]
        if weight.dtype != torch.bfloat16 or quant.dtype != torch.bfloat16 or weight.shape != quant.shape:
            raise RuntimeError("Frozen W/Wq dtype or shape mismatch")
        error = (weight.float()-quant.float()).T.to(device)
        del weight, quant
        spectrum_path = destination/"a_spectrum.json"
        if not spectrum_path.exists():
            with core.heartbeat("A-root FP64 singular spectrum (no regularization)"):
                values = torch.linalg.svdvals(a.double())
                ratio = values/values.max()
            core.atomic_json(spectrum_path, {"probe_identity": identity, "module": TARGET,
                "a_root_singular_min": float(values.min()), "a_root_singular_max": float(values.max()),
                "a_root_condition": float(values.max()/values.min()), "a_root_asymmetry": core.rel(a, a.T),
                "relative_singular_below_1e_6": int((ratio < 1e-6).sum()),
                "relative_singular_below_1e_8": int((ratio < 1e-8).sum()),
                "g_diagnostics": g_info, "original_vs_full_diagonal_relative": dg_drift,
                "note": "Spectrum of STORED FP32 A root promoted to FP64, not original raw A."})
            del values, ratio
        for method in METHODS:
            path = destination/(method+".json")
            if path.exists():
                continue
            stop.check()
            g = torch.eye(gf.shape[0], device=device) if method == "full_gi" else gd if method == "full_gd" else gf
            factor_file = bindings["baseline_factors"][method]
            factors = inputs.tensors(factor_file)
            left, right = factors["A"].to(device), factors["B"].to(device)
            if (left.shape != (error.shape[0], 64) or right.shape != (64, error.shape[1])
                    or any(t.dtype != torch.float32 or not torch.isfinite(t).all() for t in (a, left, right))):
                raise RuntimeError("Invalid saved factors")
            with torch.no_grad(), core.heartbeat(method+" saved-factor checks / FP64 SVD and inverse re-solve"):
                rows = numerical_method(error, a, g, left, right)
                # Common FA+Identity-G output-reconstruction proxy for all methods.
                for row in rows:
                    residual = error.double()-left[:, :row["rank"]].double() @ right[:row["rank"]].double()
                    row["saved_common_fa_gi_sse"] = core.norm2(a.double() @ residual)
                    row["method"] = method
            core.atomic_json(path, {"probe_identity": identity, "module": TARGET,
                "factor_file": factor_file, "rows": rows,
                "note": "All metrics FP64. Re-solves use stored roots, are not deployed; product drift alone is not a failure gate."})
            del factors, left, right, residual
            gc.collect()
            torch.cuda.empty_cache()
    rows = []
    for method in METHODS:
        path = destination/(method+".json")
        if path.exists():
            rows.extend(core.read_json(path)["rows"])
    write_csv(destination/"comparison.csv", rows)
    core.atomic_json(destination/"status.json", {"probe_identity": identity,
        "status": "DIAGNOSTICS_COMPLETE" if len(rows) == 12 else "INCOMPLETE", "rows": len(rows)})


def correction_hook(left, right):
    def hook(module, args, output):
        return output+(args[0] @ left) @ right
    return hook


def control_check(observed, reference, tolerance):
    if len(observed) != 138 or len(reference) != 138:
        raise RuntimeError("Control needs all 138 windows")
    delta = abs(ppl(observed)-ppl(reference))
    return {"status": "PASS" if delta <= tolerance else "FAIL",
            "ppl_observed": ppl(observed), "ppl_reference": ppl(reference),
            "absolute_ppl_difference": delta, "tolerance": tolerance,
            "max_window_nll_absolute_difference": max(abs(a["nll_sum"]-b["nll_sum"]) for a,b in zip(observed, reference))}


def evaluate_one(legacy, manifest, artifacts, rank, label, windows, inputs, destination, identity, stop):
    path = destination/(label+".json")
    records = get_records(path, identity)
    if len(records) == 138:
        core.log(label+" already complete")
        return records
    # Batch boundaries must match the original run on restart.
    config = manifest["payload"]["config"]
    bs = config["eval_batch_size"]
    if len(records) % bs:
        raise RuntimeError("Resume offset is not an original batch boundary")
    stop.check()
    loading = dict(manifest["payload"]["source_config"])
    loading["max_memory"] = config["eval_max_memory"]
    model, handles, modules = None, [], {}
    module = factors = left = right = None
    try:
        with core.heartbeat("load "+label+" BF16 balanced eager"):
            model = legacy.load_model(loading, "bfloat16", "balanced")
        if not {str(x) for x in model.hf_device_map.values()} <= {"0", "1", "cuda:0", "cuda:1"}:
            raise RuntimeError("Evaluation model must be fully GPU resident")
        modules = dict(model.named_modules())
        with torch.no_grad(), core.heartbeat("install frozen Wq and correction factors "+label):
            for name, files in artifacts.items():
                module = modules[name]
                quant = inputs.tensors(files["quant"])["weight_q"]
                if quant.dtype != torch.bfloat16 or module.weight.dtype != torch.bfloat16 or quant.shape != module.weight.shape:
                    raise RuntimeError("Evaluation Wq mismatch")
                module.weight.copy_(quant.to(module.weight.device))
                factors = inputs.tensors(files["correction"])
                left = factors["A"][:, :rank].to(device=module.weight.device, dtype=module.weight.dtype)
                right = factors["B"][:rank].to(device=module.weight.device, dtype=module.weight.dtype)
                handles.append(module.register_forward_hook(correction_hook(left, right)))
                del quant
        device = legacy._input_device(model)
        with torch.inference_mode():
            for index in range(len(records), 138, bs):
                stop.check()
                end = min(index+bs, 138)
                ids = windows["input_ids"][index:end].to(device)
                mask = windows["attention_mask"][index:end].to(device)
                with core.heartbeat(f"{label} windows={index+1}-{end}/138"):
                    logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
                    values = legacy._chunked_window_nll(logits, ids, mask, config["eval_ce_chunk_tokens"])
                del logits, ids, mask
                for offset, (nll, tokens) in enumerate(values):
                    records.append({"window": index+offset, "tokens": tokens, "nll_sum": nll})
                valid_records(records, 138)
                core.atomic_json(path, {"probe_identity": identity, "configuration": label,
                    "rank": rank, "records": records, "complete": end == 138,
                    "hf_device_map": model.hf_device_map})
                stop.done += 1
                core.log(f"COMMITTED {label} window={end}/138")
    finally:
        for handle in handles:
            handle.remove()
        handles.clear()
        modules.clear()
        model = module = factors = left = right = None
        gc.collect()
        torch.cuda.empty_cache()
    return records


def evaluation(manifest, base, hybrid, inputs, output, identity, stop, only_rank, repo):
    # Import pure loading/NLL helpers only. Do NOT call legacy.evaluate/summarize:
    # those routines write to the original run directory.
    sys.path.insert(0, str(repo/"experiments"))
    from qera_diag_g_isolation import pipeline as legacy
    from qera_original_a_isolation import pipeline as original
    verified = {Path(r["path"]).resolve() for r in manifest["payload"]["code"].values()}
    if any(Path(m.__file__).resolve() not in verified for m in (legacy, original)):
        raise RuntimeError("Imported evaluation helpers are not manifest-bound")
    config = manifest["payload"]["config"]
    windows = inputs.tensors(manifest["payload"]["data"]["wikitext2"])
    if (tuple(windows["input_ids"].shape) != (138, 2048)
            or windows["attention_mask"].shape != windows["input_ids"].shape
            or not (windows["attention_mask"] == 1).all()):
        raise RuntimeError("Expected original 138 x 2048 evaluation tensors with all-valid masks")
    for record in manifest["payload"]["model_files"].values():
        inputs.verify(record)
    for files in hybrid.values():
        for record in files.values():
            inputs.verify(record)
    # Verify the control's target factor too (the hybrid map replaces it).
    inputs.verify(base[TARGET]["correction"])
    destination = output/"evaluation"
    destination.mkdir(exist_ok=True)
    if destination.is_symlink():
        raise RuntimeError("Symlink evaluation folder")
    for rank in core.RANKS:
        if only_rank and rank != only_rank:
            continue
        stop.check()
        name = f"FULL_GF_R{rank}"
        source = Path(config["run_dir"])/"evaluation/configurations"/(name+".json")
        reference_value = core.read_json(source)
        expected = core.fingerprint({"manifest": manifest["sha256"], "name": name, "artifacts": base})
        if reference_value.get("protocol_sha256") != expected or reference_value.get("complete") is not True:
            raise RuntimeError("Original GF evaluation binding/completion mismatch")
        reference = valid_records(reference_value["records"], 138)
        control = evaluate_one(legacy, manifest, base, rank, f"CONTROL_FULL_GF_R{rank}", windows,
                               inputs, destination, identity, stop)
        check = control_check(control, reference, config["control_ppl_tolerance"])
        core.atomic_json(destination/f"control_check_r{rank}.json", {"probe_identity": identity,
            "source_path": str(source), "source_sha256": core.sha(source), **check})
        if check["status"] != "PASS":
            raise RuntimeError(f"CONTROL FAILED at r{rank}; hybrid blocked: {check}")
        replaced = evaluate_one(legacy, manifest, hybrid, rank, f"HYBRID_O0_GD_R{rank}", windows,
                                inputs, destination, identity, stop)
        paired = [{"window": a["window"], "tokens": a["tokens"], "control_nll": a["nll_sum"],
            "hybrid_nll": b["nll_sum"], "hybrid_minus_control_nll": b["nll_sum"]-a["nll_sum"]}
            for a,b in zip(control, replaced)]
        core.atomic_json(destination/f"paired_r{rank}.json", {"probe_identity": identity,
            "rank": rank, "control_ppl": ppl(control), "hybrid_ppl": ppl(replaced),
            "delta_ppl": ppl(replaced)-ppl(control), "delta_mean_nll":
                sum(r["hybrid_minus_control_nll"] for r in paired)/(138*2047),
            "improved_windows": sum(r["hybrid_minus_control_nll"] < 0 for r in paired), "records": paired,
            "note": "Post-hoc diagnostic single-module intervention, not a new method benchmark."})
        summarize_evaluation(destination, identity)
    summarize_evaluation(destination, identity)


def summarize_evaluation(destination, identity):
    summary, pairs = [], []
    for rank in core.RANKS:
        path = destination/f"paired_r{rank}.json"
        if not path.exists():
            continue
        value = core.read_json(path)
        if value["probe_identity"] != identity:
            raise RuntimeError("Paired report identity mismatch")
        summary.append({k: value[k] for k in ("rank", "control_ppl", "hybrid_ppl", "delta_ppl", "delta_mean_nll", "improved_windows")})
        pairs.extend({"rank": rank, **r} for r in value["records"])
    write_csv(destination/"ppl_comparison.csv", summary)
    write_csv(destination/"paired_windows.csv", pairs)
    core.atomic_json(destination/"status.json", {"probe_identity": identity,
        "status": "COMPLETE" if len(summary) == 4 else "INCOMPLETE", "completed_ranks": [r["rank"] for r in summary]})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("diagnose", "evaluate"))
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repo-dir", required=True, type=Path)
    parser.add_argument("--only-rank", type=int, choices=core.RANKS)
    parser.add_argument("--max-hours", type=float, default=10)
    parser.add_argument("--max-new-batches", type=int, help="Evaluation pilot only; remove to continue")
    args = parser.parse_args(argv)
    if (not math.isfinite(args.max_hours) or args.max_hours <= 0
            or (args.max_new_batches is not None and args.max_new_batches < 1)):
        parser.error("Budgets must be positive finite")
    if args.command == "diagnose" and (args.only_rank is not None or args.max_new_batches is not None):
        parser.error("diagnose always checks all four ranks; use only --max-hours")
    if core.sha(core.__file__) != HELPER_SHA:
        raise RuntimeError("Use the original unchanged full_g_rank_audit_v1.py")
    run, output, repo = args.run_dir.resolve(), args.output_dir.resolve(), args.repo_dir.resolve()
    manifest = core.read_json(run/"manifest.json")
    payload = manifest["payload"]
    config = payload["config"]
    if manifest["sha256"] != core.fingerprint(payload) or core.read_json(run/"config.json") != config:
        raise RuntimeError("Frozen manifest/config mismatch")
    if (config["experiment_variant"] != "mxint3_full_g_v1" or config["ranks"] != list(core.RANKS)
            or config["g_relative_floor"] != 1e-6 or config["eval_batch_size"] != 8
            or config["eval_ce_chunk_tokens"] != 256 or Path(config["run_dir"]).resolve() != run):
        raise RuntimeError("Require unchanged original Llama MXINT3 Full-G protocol")
    protected = [run.parent, repo, Path(__file__).resolve(), Path(core.__file__).resolve()]
    for dictionary in (config, payload["source_config"]):
        protected += [dictionary[k] for k in ("run_dir", "source_run_dir", "model_path", "qera_source_dir") if k in dictionary]
    for rec in payload["code"].values():
        parts = Path(rec["path"]).parts
        if "experiments" in parts:
            protected.append(Path(*parts[:parts.index("experiments")]))
    core.disjoint(output, protected)
    if args.output_dir.is_symlink() or (output.exists() and any(p.is_symlink() for p in output.rglob("*"))):
        raise RuntimeError("Output must not contain symlinks")
    inputs = core.Inputs()
    for record in payload["code"].values():
        inputs.verify(record)
    if not torch.cuda.is_available() or not str(torch.__version__).startswith("2.3.0+cu121"):
        raise RuntimeError("Use original CUDA torch 2.3.0+cu121 environment")
    count = 2 if args.command == "evaluate" else 1
    if torch.cuda.device_count() < count:
        raise RuntimeError(f"Need {count} idle visible RTX4090 GPU(s)")
    for index in range(count):
        if "4090" not in torch.cuda.get_device_name(index) or torch.cuda.mem_get_info(index)[0] < 18*2**30:
            raise RuntimeError("Use idle RTX4090s with at least 18 GiB free; do not compete with ongoing jobs")
    torch.set_num_threads(14)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(1234)
    states = core.committed_states(run, manifest)
    tasks = {layer["name"]: core.metadata(run, manifest, group, layer, states)
             for group in payload["groups"] for layer in group["layers"]}
    if len(tasks) != 224 or TARGET not in tasks:
        raise RuntimeError("Expected all 224 modules")
    baseline = payload["full_g_protocol"]["baseline_inputs"]
    base = {name: {"quant": b["quant"], "correction": b["corrections"]["full_gf"]["file"]} for name,b in tasks.items()}
    for name,b in tasks.items():
        record = b["corrections"]["full_gf"]
        if (record.get("gi_reference") != baseline["full_gi"][name]["correction"]
                or record.get("gd_reference") != baseline["full_gd"][name]["correction"]):
            raise RuntimeError("FG factor baseline binding mismatch")
    hybrid = route_hybrid(base, baseline["full_gd"], TARGET)
    bindings = dict(tasks[TARGET])
    bindings["baseline_factors"] = {m: baseline[m][TARGET]["correction"] if m != "full_gf"
                                   else base[TARGET]["correction"] for m in METHODS}
    for m in METHODS[:2]:
        if baseline[m][TARGET]["quant"] != base[TARGET]["quant"]:
            raise RuntimeError("Baseline Wq mismatch")
    probe_config = {"version": VERSION, "script_sha256": core.sha(__file__), "helper_sha256": HELPER_SHA,
        "source_manifest": manifest["sha256"], "run_dir": str(run), "repo_dir": str(repo),
        "module": TARGET, "ranks": list(core.RANKS), "base_artifacts": core.fingerprint(base),
        "hybrid_artifacts": core.fingerprint(hybrid), "numerical_inputs": core.fingerprint(bindings),
        "torch": str(torch.__version__), "transformers": importlib.metadata.version("transformers"),
        "accelerate": importlib.metadata.version("accelerate"), "tf32": False,
        "eval_batch_size": 8, "eval_ce_chunk_tokens": 256,
        "numerics": "FP64 remeasurement AND independent FP64 re-solve of STORED FP32 roots; no regularization",
        "intervention": "FA+GF everywhere; only model.layers.0.self_attn.o_proj uses saved FA+DG at same rank",
        "control_tolerance": config["control_ppl_tolerance"]}
    identity = core.fingerprint(probe_config)
    if output.exists() and not (output/"probe_config.json").exists() and any(output.iterdir()):
        raise RuntimeError("Refusing unowned nonempty output directory")
    output.mkdir(parents=True, exist_ok=True)
    stop = Budget(args.max_hours, args.max_new_batches)
    signal.signal(signal.SIGTERM, stop.signal)
    signal.signal(signal.SIGINT, stop.signal)
    with core.audit_lock(output):
        path = output/"probe_config.json"
        if path.exists() and core.read_json(path) != probe_config:
            raise RuntimeError("Probe settings changed; use a new output directory")
        if not path.exists():
            core.atomic_json(path, probe_config)
        try:
            if args.command == "diagnose":
                diagnostics(manifest, bindings, inputs, "cuda:0", output, identity, stop)
            else:
                evaluation(manifest, base, hybrid, inputs, output, identity, stop, args.only_rank, repo)
        except Paused:
            core.log("PAUSED: checkpoint committed; repeat SAME command, removing pilot batch limit if needed")
            return 75
    core.log("REQUESTED STAGE COMPLETE (not an automatic scientific pass)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
