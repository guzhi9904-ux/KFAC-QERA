#!/usr/bin/env python3
"""DA+GI/GD/GF, stored-root FP64 rank64 solves and frozen dual PPL.

Only this process binds the frozen evaluator's artifact reader. No source file,
shared environment, statistics, Wq, or previous checkpoint is modified.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import math
from pathlib import Path
import shutil
import signal
import sys
import time
from types import SimpleNamespace

sys.dont_write_bytecode = True
import torch
from safetensors import safe_open
import full_a_all_ranks_dual_ppl_v1 as dual

parent, single, core, prev = dual.parent, dual.single, dual.core, dual.prev
VERSION = "diag_a_fp64_dual_v1"
DUAL_SHA = "e7d331c417fbb1d114f93d62583580e7aa949e5179a557f2d6d14bc28572ebf7"
METHODS = ("diag_gi", "diag_gd", "diag_gf")
RANKS = (8, 16, 32, 64)
CONFIGS = [("teacher", None), ("wq", None)] + [(m, r) for m in METHODS for r in RANKS]


def solve_diagonal(error, a, g, ranks=RANKS):
    """FP64 full SVD; diagonal A inverse, dense G solve, no new floor.

    Reject zero/negative A instead of changing the old inverse's epsilon policy.
    For positive stored A the official diagonal solve is exactly this problem.
    """
    if a.dtype != torch.float32 or a.ndim != 1 or a.shape[0] != error.shape[0]:
        raise RuntimeError("Need stored FP32 diagonal A root")
    if not torch.isfinite(a).all() or (a <= 0).any():
        raise RuntimeError("Nonpositive DA root: stop for explicit inverse-policy audit; no automatic floor")
    if g.dtype != torch.float32 or g.shape != (error.shape[1], error.shape[1]):
        raise RuntimeError("Need frozen-protocol FP32 G root")
    if not torch.isfinite(error).all() or not torch.isfinite(g).all():
        raise RuntimeError("Nonfinite solve input")
    k = max(ranks)
    if not 0 < min(ranks) <= k <= min(error.shape):
        raise ValueError("Invalid ranks")
    e, sa, sg = error.double(), a.double(), g.double()
    y = (sa[:, None] * e) @ sg
    u, s, vh = torch.linalg.svd(y, full_matrices=True)
    u, vh = u[:, :k].clone(), vh[:k].clone()
    left = u / sa[:, None]
    target = s[:k, None] * vh
    right = torch.linalg.solve(sg.T, target.T).T
    inverse = {"a_residual": core.rel(sa[:, None] * left, u),
               "g_residual": core.rel(right @ sg, target)}
    if not all(math.isfinite(v) and v <= 1e-9 for v in inverse.values()):
        raise RuntimeError("FP64 inverse residual failed")
    if not torch.isfinite(left).all() or not torch.isfinite(right).all():
        raise RuntimeError("Nonfinite factors")
    before = core.norm2(y)
    rows, last = [], before
    for rank in ranks:
        correction = left[:, :rank] @ right[:rank]
        after = core.norm2((sa[:, None] * (e - correction)) @ sg)
        tail = core.norm2(s[rank:])
        excess = (after - tail) / max(before, 1e-30)
        if not math.isfinite(excess) or abs(excess) > 1e-9 or after > last + max(before, 1e-30)*1e-9:
            raise RuntimeError("FP64 objective/tail/monotonicity gate failed")
        lr, br = left[:, :rank].bfloat16().double(), right[:rank].bfloat16().double()
        if not torch.isfinite(lr).all() or not torch.isfinite(br).all():
            raise RuntimeError("BF16 factor overflow")
        rounded = core.norm2((sa[:, None] * (e - lr @ br)) @ sg)
        if not math.isfinite(rounded):
            raise RuntimeError("Nonfinite rounded objective")
        rows.append({"rank": rank, "sse_before": before, "sse_after_fp64": after,
                     "svd_tail_sse": tail, "relative_tail_difference": excess,
                     "bf16_rounded_sse_proxy": rounded,
                     "bf16_objective_increased_flag": rounded > before*(1+1e-5)+1e-20})
        last = after
    return left, right, rows, inverse


def record_path(ctx, method, name):
    return ctx.output/"factors"/method/(core.safe_name(name)+".json")


def read_checked(ctx, method, name):
    key = (method, name)
    if key in ctx.checked:
        return ctx.checked[key]
    path = record_path(ctx, method, name)
    if not path.exists():
        return None
    r = core.read_json(path)
    if (r.get("experiment_identity") != ctx.identity or r.get("status") != "PASS"
            or r.get("module") != name or r.get("method") != method
            or r.get("binding_sha256") != core.fingerprint(ctx.tasks[name])
            or [x["rank"] for x in r["metrics"]] != list(RANKS)):
        raise RuntimeError("Factor identity/binding mismatch")
    for field, suffix in (("file", ".bf16.safetensors"), ("fp64_file", ".fp64.safetensors")):
        if Path(r[field]["path"]).resolve() != path.with_suffix(suffix).resolve():
            raise RuntimeError("Factor file escaped owned path")
    v, f = ctx.inputs.tensors(r["file"]), ctx.inputs.tensors(r["fp64_file"])
    d_out, d_in = ctx.tasks[name]["shape"]
    if set(v) != {"A", "B"} or set(f) != {"A", "B"}:
        raise RuntimeError("Wrong factor schema")
    for key, shape in (("A", (d_in, 64)), ("B", (64, d_out))):
        if (f[key].dtype != torch.float64 or f[key].shape != shape or not torch.isfinite(f[key]).all()
                or v[key].dtype != torch.bfloat16 or v[key].shape != shape
                or not torch.isfinite(v[key]).all() or not torch.equal(f[key].bfloat16(), v[key])
                or single.tensor_record(v[key]) != r["bits"][key]):
            raise RuntimeError("Factor dtype/shape/direct BF16 cast mismatch")
    ctx.checked[(method, name)] = r
    return r


def prepare_one(ctx, name, method):
    if read_checked(ctx, method, name) is not None:
        return
    b = ctx.tasks[name]
    path = record_path(ctx, method, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with torch.no_grad(), core.heartbeat(f"DA FP64 SOLVE {method} {name} rank64; checks={RANKS}"):
        a = ctx.inputs.tensors(b["root"])["diag"].to(ctx.device, copy=True)
        with safe_open(str(ctx.inputs.verify(b["model"])), framework="pt", device="cpu") as f:
            w = f.get_tensor(name+".weight").clone()
        q = ctx.inputs.tensors(b["quant"])["weight_q"]
        if w.dtype != torch.bfloat16 or q.dtype != torch.bfloat16 or tuple(w.shape) != tuple(b["shape"]) or w.shape != q.shape:
            raise RuntimeError("Frozen W/Wq mismatch")
        e = (w.float()-q.float()).T.to(ctx.device)
        del w, q
        # Same G construction and precision as completed FA FP64 experiments.
        g, g_info = parent.make_g_root(method.replace("diag_", "full_", 1), b, ctx.dg[name], ctx.inputs, ctx.device)
        if single.tensor_record(g) != ctx.g_references[method][name]["bits"]:
            raise RuntimeError("Rebuilt G root bits differ from completed FA FP64 certificate; no silent migration")
        left, right, metrics, inverse = solve_diagonal(e, a, g)
        values = {"A": left.cpu(), "B": right.cpu()}
        bf16 = {k: v.bfloat16().contiguous() for k, v in values.items()}
        record = {"experiment_identity": ctx.identity, "status": "PASS", "method": method, "module": name,
                  "binding_sha256": core.fingerprint(b), "root_bits": {"a": single.tensor_record(a), "g": single.tensor_record(g)},
                  "metrics": metrics, "inverse": inverse, "g_diagnostics": g_info,
                  "fa_g_certificate": ctx.g_references[method][name],
                  "solve_rank": 64, "bits": {k: single.tensor_record(v) for k, v in bf16.items()},
                  "fp64_file": single.atomic_tensors(path.with_suffix(".fp64.safetensors"), values),
                  "file": single.atomic_tensors(path.with_suffix(".bf16.safetensors"), bf16),
                  "elapsed_seconds": time.monotonic()-started,
                  "note": "Same stored DA root; unchanged G-root policy. BF16 objective is only a rounding proxy."}
        core.atomic_json(path, record)  # sole commit marker; orphan tensors may be recomputed
    del a, g, e, left, right, values, bf16
    dual.clear_gpu()
    read_checked(ctx, method, name)
    core.log(f"COMMITTED DA {method} {name} elapsed={record['elapsed_seconds']:.1f}s")


@contextmanager
def evaluation_binding(old_token=False):
    """Explicit, reversible process-local dependency injection; files untouched.

    OLD token controls use old FP32 factors through the same BF16 deployment.
    The old FA+GD word control keeps its original frozen routing unchanged.
    """
    saved = {k: getattr(dual, k) for k in ("METHODS", "read_checked", "artifact_routes", "install")}
    def routes(ctx, method, rank, old=False):
        if method.startswith("old_diag_"):
            if not old_token:
                raise RuntimeError("OLD token route outside control binding")
            return saved["artifact_routes"](ctx, method[4:], rank, old=True)
        return saved["artifact_routes"](ctx, method, rank, old)
    def install(model, route, rank, inputs, old=False):
        return saved["install"](model, route, rank, inputs, old=old or old_token)
    dual.METHODS, dual.read_checked = METHODS, read_checked
    dual.artifact_routes, dual.install = routes, install
    try:
        yield
    finally:
        for k, value in saved.items():
            setattr(dual, k, value)


def token_controls(ctx, stop):
    with evaluation_binding(old_token=True):
        for method in METHODS:
            state = dual.token_evaluate(ctx, ctx.legacy, "old_"+method, 8, stop)
            reference = ctx.references[method]
            check = prev.control_check(state["records"], reference["records"], 1e-5)
            check["max_window_difference"] = max(abs(a["nll_sum"]-b["nll_sum"]) for a, b in zip(state["records"], reference["records"]))
            if check["max_window_difference"] > 1e-3:
                check["status"] = "FAIL"
            dual.freeze_json(ctx.output/"token_ppl"/(method+"_old_control.json"), {"experiment_identity": ctx.identity, **check})
            if check["status"] != "PASS":
                raise RuntimeError("OLD DA token replay failed; new evaluation blocked")


def summarize(ctx):
    checks = [dict(method=m, module=n, **row) for m in METHODS for n in ctx.tasks
              if (r := read_checked(ctx, m, n)) is not None for row in r["metrics"]]
    prev.write_csv(ctx.output/"rank_checks.csv", checks)
    core.atomic_json(ctx.output/"solve_status.json", {"experiment_identity": ctx.identity,
                     "completed": len(checks)//4, "expected": len(ctx.tasks)*3,
                     "status": "COMPLETE" if len(checks) == len(ctx.tasks)*12 else "INCOMPLETE"})
    for protocol in ("token_ppl", "word_ppl"):
        folder = ctx.output/protocol
        if not folder.exists() or (protocol == "word_ppl" and ctx.word is None):
            continue
        summary, details, pairs, comparisons, records = [], [], [], [], {}
        for m, rank in CONFIGS:
            name = dual.label(m, rank)
            if protocol == "token_ppl":
                state = single.read_state(folder, name, ctx.identity, ctx.inputs)
                if not state["complete"]:
                    continue
                rows = state["records"]
                values = {"ppl": prev.ppl(rows), "nll_sum": sum(r["nll_sum"] for r in rows),
                          "windows": 138, "prediction_tokens": 282486, "context": 2048}
            else:
                state = dual.word_existing(ctx, name)
                if state is None:
                    continue
                rows = core.read_json(state["documents_file"]["path"])
                values = {**state["summary"], "context": 4096}
            summary.append({"configuration": name, "method": m, "rank": rank, "metric": protocol, **values})
            records[name] = (rows, values)
            details.extend({"configuration": name, **row} for row in rows)
        for rank in RANKS:
            for first, second in ((METHODS[0], METHODS[1]), (METHODS[0], METHODS[2]), (METHODS[1], METHODS[2])):
                a, b = dual.label(first, rank), dual.label(second, rank)
                if a not in records or b not in records:
                    continue
                aa, av = records[a]; bb, bv = records[b]
                unit = "window" if protocol == "token_ppl" else "document"
                if len(aa) != len(bb):
                    raise RuntimeError("Pair coverage mismatch")
                for x, y in zip(aa, bb):
                    keys = (unit, "tokens") if unit == "window" else (unit, "words", "document_sha256")
                    if any(x[k] != y[k] for k in keys):
                        raise RuntimeError("Pair inputs differ")
                    pairs.append({"comparison": b+"_minus_"+a, unit: x[unit], "delta_nll": y["nll_sum"]-x["nll_sum"]})
                comparisons.append({"comparison": b+"_minus_"+a, "rank": rank,
                                    "delta_nll": bv["nll_sum"]-av["nll_sum"], "delta_ppl": bv["ppl"]-av["ppl"],
                                    "improved_units": sum(y["nll_sum"] < x["nll_sum"] for x, y in zip(aa, bb)), "units": len(aa)})
        for name, rows in (("ppl_summary", summary), ("per_unit", details), ("comparisons", comparisons), ("paired_units", pairs)):
            prev.write_csv(folder/(name+".csv"), rows)
        core.atomic_json(folder/"status.json", {"experiment_identity": ctx.identity, "completed": len(summary), "expected": 14,
                         "status": "COMPLETE" if len(summary) == 14 else "INCOMPLETE",
                         "note": "Separate protocols; no IID-token significance claims."})


def setup(args):
    if core.sha(dual.__file__) != DUAL_SHA or core.sha(parent.__file__) != dual.PARENT_SHA:
        raise RuntimeError("Frozen evaluator/helper changed")
    ctx = parent.setup(SimpleNamespace(**{**vars(args), "command": "run"}))
    core.disjoint(ctx.output, [args.word_reference_dir, args.harness_source, args.official_qera_root, args.source_fp64_dir])
    ctx.checked = {}
    original_experiment = ctx.experiment
    source = args.source_fp64_dir.resolve()
    if core.read_json(source/"experiment.json") != original_experiment:
        raise RuntimeError("Completed FA FP64 experiment does not match frozen source/environment")
    ctx.g_references = {m: {} for m in METHODS}
    # Capture before extending the old FA bindings with DA factor routes.
    for m in METHODS:
        fm = m.replace("diag_", "full_", 1)
        for name, b in ctx.tasks.items():
            r = parent.solved_record(source, fm, name, b, ctx.identity, ctx.inputs)
            if r is None:
                raise RuntimeError("Completed FA FP64 factors/certificates missing")
            ctx.g_references[m][name] = {"bits": r["root_bits"]["g"],
                                         "certificate": single.file_record(parent.factor_paths(source, fm, name)[1])}
    baseline = ctx.manifest["payload"]["full_g_protocol"]["baseline_inputs"]
    ref_files = {}
    for m in METHODS:
        artifacts = {}
        for name, b in ctx.tasks.items():
            gf = b["corrections"]["diag_gf"]
            if gf["gi_reference"] != baseline["diag_gi"][name]["correction"] or gf["gd_reference"] != baseline["diag_gd"][name]["correction"]:
                raise RuntimeError("DA GF baseline references differ")
            for key in ("identity_product_drift", "diagonal_product_drift"):
                drift = gf[key]
                tol = ctx.manifest["payload"]["config"]["identity_product_tolerance"]
                if set(drift) != {str(r) for r in RANKS} or any(not math.isfinite(v) or not 0 <= v <= tol for v in drift.values()):
                    raise RuntimeError("Original DA regression gate failed")
            f = gf["file"] if m == "diag_gf" else baseline[m][name]["correction"]
            if m != "diag_gf" and baseline[m][name]["quant"] != b["quant"]:
                raise RuntimeError("DA Wq differs")
            b["baseline_factors"][m] = f
            ctx.inputs.verify(f)
            artifacts[name] = {"quant": b["quant"], "correction": f}
        label = dual.label(m, 8)
        path = args.run_dir.resolve()/"evaluation/configurations"/(label+".json")
        r = core.read_json(path)
        expected = core.fingerprint({"manifest": ctx.manifest["sha256"], "name": label, "artifacts": artifacts})
        if r.get("protocol_sha256") != expected or not r.get("complete") or len(r["records"]) != 138:
            raise RuntimeError("Missing/mismatched original DA r8 reference")
        prev.valid_records(r["records"], 138)
        ctx.references[m] = r
        ref_files[m] = single.file_record(path)
    ctx.legacy = parent.load_legacy(ctx, args)
    ctx.word = dual.prepare_word(ctx, args) if args.protocol in ("both", "word") else None
    ctx.experiment = {"version": VERSION, "script_sha256": core.sha(__file__), "dual_sha256": DUAL_SHA,
                      "frozen_audit": original_experiment, "bindings_sha256": core.fingerprint(ctx.tasks),
                      "references": ref_files, "methods": list(METHODS), "ranks": list(RANKS), "solve_rank": 64,
                      "fa_source_experiment": single.file_record(source/"experiment.json"), "g_references": ctx.g_references,
                      "protocol_selection": args.protocol, "word_protocol": None if ctx.word is None else ctx.word.protocol,
                      "token_protocol": {"windows": 138, "context": 2048, "batch": 8, "ce_chunk": 256},
                      "a_policy": "stored FP32 diag root -> FP64; nonpositive root aborts; no new floor",
                      "g_policy": "same as FA FP64: frozen normalization/floor -> FP32 root -> FP64",
                      "solve": "FP64 full SVD, diagonal A inverse, dense G solve, no fallback",
                      "deployment": "direct rank64 BF16 factors, prefix 8/16/32/64, frozen two-GEMM hooks",
                      "new_statistics": False, "new_root_precision": False}
    ctx.identity = core.fingerprint(ctx.experiment)
    return ctx


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("doctor", "pilot", "solve", "run", "evaluate", "summarize"))
    for key in ("run-dir", "repo-dir", "output-dir", "source-fp64-dir", "official-qera-root", "harness-source", "word-reference-dir"):
        p.add_argument("--"+key, required=True, type=Path)
    p.add_argument("--protocol", choices=("both", "token", "word"), default="both")
    p.add_argument("--max-hours", type=float, default=10)
    p.add_argument("--max-new-units", type=int)
    args = p.parse_args(argv)
    if not math.isfinite(args.max_hours) or args.max_hours <= 0 or (args.max_new_units is not None and args.max_new_units < 1):
        p.error("Positive finite budgets required")
    stop = prev.Budget(args.max_hours, args.max_new_units)
    signal.signal(signal.SIGINT, stop.signal); signal.signal(signal.SIGTERM, stop.signal)
    ctx = setup(args)
    if args.command == "doctor":
        core.log("DA DOCTOR PASS: frozen provenance/environment; no numerical solve or PPL performed")
        return 0
    if ctx.output.exists() and not (ctx.output/"experiment.json").exists() and any(ctx.output.iterdir()):
        raise RuntimeError("Refusing unowned nonempty output directory")
    ctx.output.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(ctx.output).free < 16*2**30:
        raise RuntimeError("Need >=16 GiB free for factors and evaluation checkpoints")
    with core.audit_lock(ctx.output):
        dual.freeze_json(ctx.output/"experiment.json", ctx.experiment)
        try:
            if args.command == "summarize":
                summarize(ctx); return 0
            if args.command in ("pilot", "solve", "run"):
                names = parent.PILOT if args.command == "pilot" else list(ctx.tasks)
                for i, name in enumerate(names, 1):
                    for m in METHODS:
                        if read_checked(ctx, m, name) is not None:
                            continue
                        stop.check()
                        core.log(f"DA module={i}/{len(names)} method={m} {name}")
                        prepare_one(ctx, name, m); stop.done += 1
                summarize(ctx)
            if args.command == "solve":
                return 0
            with evaluation_binding():
                if args.command == "pilot":
                    if ctx.word is not None:
                        dual.word_gate(ctx, "BF16", dual.word_evaluate(ctx, args, "teacher", None, stop))
                    core.log("DA PILOT COMPLETE: 2 large modules x 3 G; all-rank numerical checks. Use run.")
                    return 0
                if any(read_checked(ctx, m, n) is None for m in METHODS for n in ctx.tasks):
                    raise RuntimeError("Incomplete factors: run solve/run before evaluate")
                if args.protocol in ("token", "both"):
                    token_controls(ctx, stop)
                    for m, rank in CONFIGS:
                        dual.token_evaluate(ctx, ctx.legacy, m, rank, stop); summarize(ctx)
                if ctx.word is not None:
                    dual.word_gate(ctx, "BF16", dual.word_evaluate(ctx, args, "teacher", None, stop))
                    dual.word_gate(ctx, "OLD_FULL_GD_R8", dual.word_evaluate(ctx, args, "full_gd", 8, stop, old=True))
                    for m, rank in CONFIGS:
                        dual.word_evaluate(ctx, args, m, rank, stop); summarize(ctx)
            summarize(ctx)
            core.log("DA REQUESTED PROTOCOLS COMPLETE; inspect controls, not an automatic scientific pass")
        except prev.Paused:
            summarize(ctx)
            core.log("DA PAUSED: SAME run command resumes; solve=current module/method, token<=8 windows, word=current configuration")
            return 75
        except Exception as exc:
            core.atomic_json(ctx.output/"last_failure.json", {"experiment_identity": ctx.identity,
                             "type": type(exc).__name__, "message": str(exc), "source_files_modified": False})
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
