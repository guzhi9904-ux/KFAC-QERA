#!/usr/bin/env python3
"""Qwen DA/FA + GI: stored-root FP64 solves and frozen token-PPL protocol.

No statistics/root/quantizer changes, DG or old-factor adoption.
Only the new output directory is writable. Each method/module and eval batch
is an atomic transaction. No numerical fallback or precision mixing.
"""
from __future__ import annotations
import argparse
import gc
import importlib
import importlib.metadata
import math
from pathlib import Path
import shutil
import signal
import sys
import time

sys.dont_write_bytecode = True
import torch
import qwen_a_fp64_target_v1 as target

VERSION = "qwen_gi_fp64_v1"
TARGET_SHA = "f35774e507e7906d09fd31ae938b9691223341d4961e31dd0718ede282572ea9"
METHODS = ("diag_gi", "full_gi")
RANKS = (8, 16, 32, 64)
PILOT = ("model.layers.0.mlp.gate_proj", "model.layers.0.mlp.down_proj", target.TARGET)
QUANT = {"name": "mxint", "width": 3, "block_size": 32, "block_axis": -1}


def solve_diag(root, error, ranks=RANKS):
    """Vector representation of DA; no dense diagonal matrix or new floor."""
    if (root.dtype != torch.float32 or error.dtype != torch.float32 or root.ndim != 1
            or error.ndim != 2 or len(root) != len(error) or not torch.isfinite(root).all()
            or not torch.isfinite(error).all() or not (root > 0).all()):
        raise RuntimeError("Require finite FP32 error and strictly positive stored FP32 DA root")
    if not ranks or tuple(sorted(set(ranks))) != tuple(ranks) or not 0 < ranks[0] <= ranks[-1] <= min(error.shape):
        raise ValueError("Invalid ranks")
    s, e = root.double(), error.double()
    weighted = s[:, None]*e
    u, sigma, vh = torch.linalg.svd(weighted, full_matrices=True)
    k = max(ranks)
    u, vh = u[:, :k].clone(), vh[:k].clone()
    left, right = u/s[:, None], sigma[:k, None]*vh
    before, en = target.norm2(weighted), math.sqrt(target.norm2(e))
    del weighted
    rows, previous = [], before
    for r in ranks:
        a, b = left[:, :r], right[:r]
        correction = a @ b
        after = target.norm2(s[:, None]*(e-correction))
        tail = target.norm2(sigma[r:])
        residual = target.relative(s[:, None]*a, u[:, :r])
        gap = (after-tail)/max(before, 1e-30)
        rounded = a.bfloat16().double() @ b.bfloat16().double()
        proxy = target.norm2(s[:, None]*(e-rounded))
        finite = bool(torch.isfinite(rounded).all()) and math.isfinite(proxy)
        good = (all(math.isfinite(x) for x in (after, tail, residual, gap))
                and residual <= target.INVERSE_TOL and abs(gap) <= target.TAIL_TOL
                and after <= previous+max(before, 1e-30)*target.TAIL_TOL)
        rows.append({"rank": r, "weighted_sse_before": before, "weighted_sse_after_fp64": after,
                     "svd_tail_sse_fp64": tail, "relative_tail_difference": gap,
                     "a_inverse_relative_residual_fp64": residual, "g_inverse_relative_residual_fp64": 0.,
                     "numerical_gate_passed": good, "correction_over_error_norm": math.sqrt(target.norm2(correction))/max(en, 1e-30),
                     "bf16_factors_finite": finite, "bf16_rounded_sse_fp64_proxy": proxy,
                     "bf16_proxy_objective_increased_flag": not finite or proxy > before*(1+1e-5)+1e-20})
        previous = after
    info = {"numerical_gate_passed": all(r["numerical_gate_passed"] for r in rows),
            "bf16_factors_finite": all(r["bf16_factors_finite"] for r in rows),
            "bf16_proxy_flags": sum(r["bf16_proxy_objective_increased_flag"] for r in rows),
            "inverse_tolerance": target.INVERSE_TOL, "tail_tolerance": target.TAIL_TOL}
    values = {"A_fp64": left.cpu(), "B_fp64": right.cpu(), "A_bf16": left.bfloat16().cpu(),
              "B_bf16": right.bfloat16().cpu(), "singular_values": sigma.cpu()}
    return values, rows, info


def require_gates(record):
    checks = record["checks"]
    if (not checks["numerical_gate_passed"] or not checks["bf16_factors_finite"]
            or checks["bf16_proxy_flags"] != 0 or len(record["rank_metrics"]) != 4
            or [r["rank"] for r in record["rank_metrics"]] != list(RANKS)):
        raise RuntimeError("Numerical/BF16 proxy gate failed; candidate quarantined, no deployment")
    for row in record["rank_metrics"]:
        if not row["numerical_gate_passed"] or not row["bf16_factors_finite"] or row["bf16_proxy_objective_increased_flag"]:
            raise RuntimeError("Failed per-rank gate")


def configurations():
    return [("BF16", None, None), ("W3_MXINT", "wq", None)] + [
        (f"{m.upper()}_R{r}", m, r) for m in METHODS for r in RANKS]


def read_rows(path, identity, total, h):
    if not path.exists():
        return []
    obj = h.read_json(path)
    if obj.get("identity") != identity:
        raise RuntimeError("Evaluation identity changed")
    rows = obj["records"]
    if len(rows) > total:
        raise RuntimeError("Too many windows")
    for i, row in enumerate(rows):
        if row["window"] != i or row["tokens"] != 2047 or not math.isfinite(row["nll_sum"]) or row["nll_sum"] < 0:
            raise RuntimeError("Invalid/noncontiguous evaluation checkpoint")
    if h.fingerprint(rows) != obj.get("records_sha256"):
        raise RuntimeError("Evaluation records checksum mismatch")
    return rows


def baseline_control(label, rows, entries, manifest, root, h):
    """If a frozen completed BF16/Wq reference exists, require replay agreement."""
    old = Path(manifest["payload"]["config"]["run_dir"])/"evaluation/configurations"/(label+".json")
    if not old.exists():
        return {"status": "NO_FROZEN_REFERENCE", "note": "New baseline, not a claimed regression pass"}
    record = h.read_json(old)
    files = {} if label == "BF16" else {x["layer"]["name"]: {"quant": x["quant"]["file"]} for x in entries}
    expected = h.fingerprint({"manifest": manifest["sha256"], "name": label, "files": files})
    if record.get("protocol") != expected:
        raise RuntimeError("Frozen baseline protocol mismatch")
    reference = record["records"]
    if len(reference) != len(rows):
        return {"status": "INCOMPLETE_FROZEN_REFERENCE", "windows": len(reference)}
    for i, row in enumerate(reference):
        if row["window"] != i or row["tokens"] != 2047 or not math.isfinite(row["nll_sum"]) or row["nll_sum"] < 0:
            raise RuntimeError("Invalid frozen baseline rows")
    maximum = max(abs(a["nll_sum"]-b["nll_sum"]) for a,b in zip(rows, reference))
    ppl_gap = abs(math.exp(sum(r["nll_sum"] for r in rows)/(len(rows)*2047))
                  - math.exp(sum(r["nll_sum"] for r in reference)/(len(rows)*2047)))
    result = {"status": "PASS" if maximum <= 1e-3 and ppl_gap <= 1e-5 else "FAIL",
              "reference_sha256": h.sha256(old), "max_window_nll_difference": maximum,
              "ppl_difference": ppl_gap, "window_tolerance": 1e-3, "ppl_tolerance": 1e-5}
    return result


def source_inventory(run, manifest, h, stop):
    """Hash all actual roots/Wq; raw stats are bound but not consumed/rehashed."""
    entries = []
    for group in manifest["payload"]["groups"]:
        stop.check()
        root = h.owned_record(run, "roots", group["target"], manifest["sha256"])
        for layer in group["layers"]:
            stop.check()
            quant = h.owned_record(run, "quantized", layer["name"], manifest["sha256"])
            if quant.get("quantization") != QUANT or quant.get("fp32_bf16_equal") is not True:
                raise RuntimeError("Missing/mismatched frozen W3")
            entries.append({"layer": layer, "root": root, "quant": quant,
                            "model": manifest["payload"]["model_files"][layer["weight_file"]]})
    if len(entries) != 196 or len({x["layer"]["name"] for x in entries}) != 196:
        raise RuntimeError("Expected 196 unique Qwen targets")
    return entries


def factor_record(output, method, entry, identity, single, h):
    name = entry["layer"]["name"]
    path = output/"factors"/method/(h.safe_name(name)+".safetensors")
    meta = path.with_suffix(".json")
    if not meta.exists():
        return None
    obj = h.read_json(meta)
    if (obj.get("identity") != identity or obj.get("source") != entry or obj.get("method") != method
            or obj.get("status") != "PASS" or Path(obj["file"]["path"]).resolve() != path.resolve()):
        raise RuntimeError("Factor checkpoint identity/status mismatch: "+name)
    require_gates(obj)
    h.verify(obj["file"])
    values = h.load_file(str(path))
    target.validate_candidate(values, entry["layer"]["shape"])
    if obj["factor_bits"] != {k: single.tensor_record(v) for k,v in values.items()}:
        raise RuntimeError("Factor bits changed")
    return obj


def solve_all(entries, output, identity, single, h, stop, pilot=False):
    selected = [x for x in entries if not pilot or x["layer"]["name"] in PILOT]
    all_rows = []
    for i, entry in enumerate(selected):
        for method in METHODS:
            stop.check()
            name = entry["layer"]["name"]
            old = factor_record(output, method, entry, identity, single, h)
            if old:
                all_rows.extend({"module": name, "method": method, **r} for r in old["rank_metrics"])
                continue
            path = output/"factors"/method/(h.safe_name(name)+".safetensors")
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.cuda.empty_cache()
            if torch.cuda.mem_get_info(0)[0] < 18*2**30:
                raise RuntimeError("Need >=18 GiB free on GPU0; no concurrent GPU job")
            started = time.monotonic()
            with torch.no_grad(), h.progress(f"SOLVE {i+1}/{len(selected)} {name} {method} FP64 ranks=8/16/32/64"):
                root = h.load_file(entry["root"]["file"]["path"])["diag" if method == "diag_gi" else "full"].to("cuda:0")
                with h.safe_open(entry["model"]["path"], framework="pt", device="cpu") as f:
                    weight = f.get_tensor(name+".weight")
                q = h.load_file(entry["quant"]["file"]["path"])["weight_q"]
                if weight.dtype != torch.bfloat16 or q.dtype != torch.bfloat16 or list(weight.shape) != entry["layer"]["shape"] or weight.shape != q.shape:
                    raise RuntimeError("Frozen weight dtype/shape mismatch")
                error = (weight.float()-q.float()).T.to("cuda:0")
                values, rows, checks = (solve_diag if method == "diag_gi" else target.solve)(root, error)
                target.validate_candidate(values, entry["layer"]["shape"])
                obj = {"identity": identity, "source": entry, "method": method, "rank_metrics": rows,
                       "checks": checks, "elapsed_seconds": time.monotonic()-started,
                       "factor_bits": {k: single.tensor_record(v) for k,v in values.items()}}
                obj["file"] = single.atomic_tensors(path, values)
                try:
                    require_gates(obj)
                except RuntimeError:
                    single.core.atomic_json(path.with_suffix(".failed.json"), {**obj, "status": "QUARANTINED"})
                    raise
                single.core.atomic_json(path.with_suffix(".json"), {**obj, "status": "PASS"})
                h.log(f"COMMITTED {method} {name}; seconds={obj['elapsed_seconds']:.1f}")
                all_rows.extend({"module": name, "method": method, **r} for r in rows)
                del root, weight, q, error, values
            gc.collect(); torch.cuda.empty_cache()
    single.previous.write_csv(output/("pilot_rank_metrics.csv" if pilot else "rank_metrics.csv"), all_rows)


def import_evaluator(source, payload, h):
    """Import only hash-verified original code; never invoke its write stages."""
    stage_path = source/"experiments/qwen25_base_isolation_v1/stages.py"
    records = {str(Path(x["path"]).resolve()): x for x in payload["code"].values()}
    if str(stage_path) not in records:
        raise RuntimeError("Evaluator checkout is not the frozen Qwen source")
    sys.path.insert(0, str(source/"experiments"))
    stage = importlib.import_module("qwen25_base_isolation_v1.stages")
    for name, module in list(sys.modules.items()):
        if name.split(".")[0] in ("qwen25_base_isolation_v1", "qera_original_a_isolation", "qera_diag_g_isolation"):
            path = str(Path(module.__file__).resolve())
            if path not in records or h.sha256(path) != records[path]["sha256"]:
                raise RuntimeError("Imported evaluator differs from manifest: "+path)
    return stage


def install(model, entries, factors, method, rank, single, h, handles):
    deployed = {}
    for entry in entries:
        name = entry["layer"]["name"]
        module = model.get_submodule(name)
        if tuple(module.weight.shape) != tuple(entry["layer"]["shape"]) or (module.bias is not None) != entry["layer"]["bias"]:
            raise RuntimeError("Deployment module shape/bias policy mismatch")
        bias = None if module.bias is None else single.tensor_record(module.bias)
        if method:
            q = h.load_file(entry["quant"]["file"]["path"])["weight_q"]
            module.weight.data.copy_(q.to(module.weight.device))
            if not torch.equal(module.weight.detach().cpu(), q):
                raise RuntimeError("Deployment Wq mismatch")
        bits = {"weight": single.tensor_record(module.weight), "bias": bias}
        if method in METHODS:
            vals = h.load_file(factors[name]["path"])
            a = vals["A_bf16"][:, :rank].to(module.weight.device)
            b = vals["B_bf16"][:rank].to(module.weight.device)
            handles.append(module.register_forward_hook(lambda _m, args, out, a=a, b=b: out+(args[0]@a)@b))
            bits.update(A=single.tensor_record(a), B=single.tensor_record(b))
        if bias != (None if module.bias is None else single.tensor_record(module.bias)):
            raise RuntimeError("Bias changed")
        deployed[name] = bits
    return deployed


def evaluate(entries, manifest, output, identity, single, h, stop, stage, smoke=False):
    config = dict(manifest["payload"]["config"])
    config["run_dir"] = str(output)  # teacher writes only new runtime diagnostics
    data = h.load_file(manifest["payload"]["data"]["wikitext2"]["path"])
    total = 8 if smoke else len(data["input_ids"])
    root = output/("smoke" if smoke else "evaluation")
    root.mkdir(parents=True, exist_ok=True)
    for sub in ("configurations", "deployment", "controls"):
        (root/sub).mkdir(parents=True, exist_ok=True)
    summary, per_window = [], []
    for label, method, rank in configurations()[:1] if smoke else configurations():
        stop.check()
        factors = {}
        if method in METHODS:
            for entry in entries:
                obj = factor_record(output, method, entry, identity, single, h)
                if obj is None:
                    raise RuntimeError("Complete all FP64 factors before PPL")
                factors[entry["layer"]["name"]] = obj["file"]
        protocol = h.fingerprint({"experiment": identity, "configuration": label, "factor_files": factors,
                                  "smoke": smoke, "windows": total})
        path = root/"configurations"/(label+".json")
        rows = read_rows(path, protocol, total, h)
        if len(rows) < total:
            model = stage.teacher(config, manifest, "bfloat16")
            handles = []
            try:
                deployed = install(model, entries, factors, method, rank, single, h, handles)
                deployment = {"identity": protocol, "tensors": deployed, "hf_device_map": model.hf_device_map}
                dep_path = root/"deployment"/(label+".json")
                if dep_path.exists() and h.read_json(dep_path) != deployment:
                    raise RuntimeError("Actual deployment tensors/device map changed on resume")
                single.core.atomic_json(dep_path, deployment)
                with torch.inference_mode():
                    device = stage._input_device(model)
                    for index in range(len(rows), total, config["eval_batch_size"]):
                        stop.check()
                        end = min(index+config["eval_batch_size"], total)
                        ids, mask = (data[k][index:end].to(device) for k in ("input_ids", "attention_mask"))
                        with h.progress(f"{'SMOKE' if smoke else 'PPL'} {label} windows={index+1}-{end}/{total}"):
                            logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
                            metrics = stage._chunked_window_nll(logits, ids, mask, config["eval_ce_chunk_tokens"])
                        del logits
                        if len(metrics) != end-index or any(t != 2047 or not math.isfinite(n) or n < 0 for n,t in metrics):
                            raise RuntimeError("Invalid PPL batch")
                        rows.extend({"window": index+j, "tokens": t, "nll_sum": n} for j,(n,t) in enumerate(metrics))
                        single.core.atomic_json(path, {"identity": protocol, "records": rows, "records_sha256": h.fingerprint(rows)})
                        h.log(f"COMMITTED {label} window={end}/{total}")
            finally:
                for hook in handles:
                    hook.remove()
                handles.clear()
                del model
                gc.collect(); torch.cuda.empty_cache()
        nll = sum(r["nll_sum"] for r in rows)
        if not smoke and method in (None, "wq"):
            control = baseline_control(label, rows, entries, manifest, root, h)
            single.core.atomic_json(root/"controls"/(label+".json"), control)
            if control["status"] == "FAIL":
                raise RuntimeError("Frozen BF16/Wq baseline regression failed")
        if not smoke and method is None:
            smoke_path = output/"smoke/configurations/BF16.json"
            if smoke_path.exists():
                smoke_id = h.fingerprint({"experiment": identity, "configuration": "BF16", "factor_files": {}, "smoke": True, "windows": 8})
                smoke_rows = read_rows(smoke_path, smoke_id, 8, h)
                if any(abs(x["nll_sum"]-y["nll_sum"]) > 1e-3 for x,y in zip(rows, smoke_rows)):
                    raise RuntimeError("Formal BF16 baseline differs from pilot smoke")
        summary.append({"configuration": label, "method": method or "teacher", "rank": rank or "",
                        "context": 2048, "metric": "token_ppl", "windows": total,
                        "prediction_tokens": total*2047, "nll_sum": nll, "ppl": math.exp(nll/(total*2047))})
        per_window.extend({"configuration": label, **r} for r in rows)
        single.previous.write_csv(root/"ppl_summary_wikitext2.csv", summary)
        single.previous.write_csv(root/"wikitext2_per_window.csv", per_window)
        h.log(f"COMPLETE {label} ppl={summary[-1]['ppl']:.9f}")
    single.core.atomic_json(root/"status.json", {"identity": identity, "status": "SMOKE_COMPLETE" if smoke else "COMPLETE",
                            "configurations": len(summary), "expected_configurations": 1 if smoke else 10,
                            "windows_per_configuration": total, "not_word_ppl": True})


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("pilot", "run", "solve", "evaluate"))
    for key in ("source-code", "run-dir", "output-dir", "harness-source", "official-qera-root", "word-reference-dir"):
        p.add_argument("--"+key, type=Path, required=True)
    p.add_argument("--protocol", choices=("both", "token", "word"), default="both")
    p.add_argument("--max-hours", type=float, default=10.)
    args = p.parse_args(argv)
    if not math.isfinite(args.max_hours) or args.max_hours <= 0:
        p.error("max-hours must be positive and finite")
    directory = Path(__file__).resolve().parent
    single, v2, h = target.load_libraries(directory, directory/"qwen_full_a_audit_v2.py")
    if h.sha256(target.__file__) != TARGET_SHA:
        raise RuntimeError("Frozen target solver changed")
    stop = single.previous.Budget(args.max_hours)
    signal.signal(signal.SIGTERM, stop.signal); signal.signal(signal.SIGINT, stop.signal)
    run, output, source = args.run_dir.resolve(), args.output_dir.resolve(), args.source_code.resolve()
    manifest = h.read_json(run/"manifest.json")
    payload = manifest["payload"]
    config = payload["config"]
    if h.fingerprint(payload) != manifest["sha256"] or config != h.read_json(run/"config.json"):
        raise RuntimeError("Frozen manifest/config mismatch")
    if (config.get("experiment_variant") != "qwen25_base_mxint3_v1" or config.get("ranks") != list(RANKS)
            or config.get("quantization") != QUANT or config.get("eval_batch_size") != 8
            or config.get("eval_ce_chunk_tokens") != 256):
        raise RuntimeError("Unexpected Qwen W3/four-rank/token-PPL protocol")
    protected = [run.parent, source, directory, args.harness_source, args.official_qera_root, args.word_reference_dir]
    for settings in (config, payload["source_config"]):
        protected.extend(settings[k] for k in ("model_path", "run_dir", "source_run_dir", "qera_source_dir") if k in settings)
    for rec in payload["code"].values():
        parts = Path(rec["path"]).parts
        if "experiments" in parts:
            protected.append(Path(*parts[:parts.index("experiments")]))
    single.core.disjoint(output, protected)
    if args.output_dir.is_symlink() or (output.exists() and any(x.is_symlink() for x in output.rglob("*"))):
        raise RuntimeError("Output must not contain symlinks")
    if output.exists() and not (output/"experiment.json").exists() and any(output.iterdir()):
        raise RuntimeError("Refuse unknown output contents; put log outside output")
    environment = v2.environment_report(payload["environment"], {n: importlib.metadata.version(n) for n in v2.PACKAGES}, torch.__version__, torch.version.cuda)
    if environment["status"] != "PASS":
        raise RuntimeError("Require original Qwen environment: "+str(environment))
    if torch.cuda.device_count() != 2 or any("4090" not in torch.cuda.get_device_name(i) for i in (0,1)):
        raise RuntimeError("Require exactly two visible RTX4090 cards")
    torch.set_num_threads(14); torch.manual_seed(1234)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    experiment = {"version": VERSION, "manifest": single.file_record(run/"manifest.json"),
                  "script_sha256": h.sha256(__file__), "target_sha256": TARGET_SHA,
                  "helpers": target.HELPERS, "v2_sha256": target.V2_SHA, "source_code": str(source),
                  "environment": environment, "methods": list(METHODS), "ranks": list(RANKS),
                  "root_policy": "stored FP32 root promoted; no new roots or damping", "solve_dtype": "float64",
                  "deployment_dtype": "bfloat16", "full_svd": True, "inverse_tolerance": 1e-9,
                  "tail_tolerance": 1e-9, "proxy_gate": "no objective increase beyond 1e-5 relative + 1e-20",
                  "new_statistics": False, "reuse_old_corrections": False, "word_ppl": True,
                  "word_script_sha256": h.sha256(directory/"qwen_gi_word_v1.py"),
                  "word_reference": single.file_record(args.word_reference_dir/"protocol.json")}
    identity = h.fingerprint(experiment)
    output.mkdir(parents=True, exist_ok=True)
    with single.core.audit_lock(output):
        exp = output/"experiment.json"
        if exp.exists() and h.read_json(exp) != experiment:
            raise RuntimeError("Experiment identity changed; do not overwrite")
        single.core.atomic_json(exp, experiment)
        try:
            if shutil.disk_usage(output).free < 8*2**30:
                raise RuntimeError("Need >=8 GiB free for new factors and both evaluations")
            h.log("AUDIT source code / model / roots / Wq / evaluation tokens (read-only)")
            for rec in [*payload["code"].values(), *payload["model_files"].values(), payload["data"]["wikitext2"]]:
                stop.check(); h.verify(rec)
            data = h.load_file(payload["data"]["wikitext2"]["path"])
            expected = payload["data_details"]["wikitext2"]["windows"]
            if (data["input_ids"].shape != (expected, 2048) or data["attention_mask"].shape != data["input_ids"].shape
                    or not (data["attention_mask"] == 1).all() or expected != 143):
                raise RuntimeError("Require frozen Qwen 143 x 2048 evaluation windows")
            del data
            entries = source_inventory(run, manifest, h, stop)
            inventory = output/"inputs.json"
            if inventory.exists() and h.read_json(inventory) != entries:
                raise RuntimeError("Source inputs changed")
            single.core.atomic_json(inventory, entries)
            stage = import_evaluator(source, payload, h)
            import qwen_gi_word_v1 as word
            word_ctx = word.prepare(args, manifest, output, identity, single, h) if args.protocol != "token" else None
            if args.command in ("pilot", "run", "solve"):
                solve_all(entries, output, identity, single, h, stop, args.command == "pilot")
            if args.command in ("pilot", "run", "evaluate"):
                if args.protocol != "word":
                    evaluate(entries, manifest, output, identity, single, h, stop, stage, args.command == "pilot")
                if word_ctx:
                    word.evaluate(word_ctx, entries, manifest, output, single, h, stop, sys.modules[__name__], args.command == "pilot")
            h.log("PILOT COMPLETE: 3 modules x DA/FA, all ranks; token smoke 8 windows / full BF16 word baseline if selected. Use run."
                  if args.command == "pilot" else f"REQUESTED STAGE COMPLETE: {args.command}")
            return 0
        except single.previous.Paused:
            h.log("PAUSED: budget/signal reached; restart SAME command. Committed work retained.")
            return 75
        except Exception as exc:
            single.core.atomic_json(output/"failure.json", {"identity": identity, "type": type(exc).__name__, "message": str(exc)})
            raise


if __name__ == "__main__":
    raise SystemExit(main())
