#!/usr/bin/env python3
"""Rank64 GI/DG/GF single-module interventions in a common BF16 background.

Reuses frozen evaluated factors. No collection, SVD, merging, or group deletion.
One checkpoint contains a complete paired window for all three methods.
"""
from __future__ import annotations

import argparse
import contextlib
import math
from pathlib import Path
import shutil
import signal
import sys
import tarfile

sys.dont_write_bytecode = True
import numpy as np
import torch
import torch.nn.functional as F
import grouped_rank_ablation_v1 as frozen

dual, single, core, prev = frozen.dual, frozen.single, frozen.core, frozen.prev
VERSION = "local_output_kl_v1"
METHODS = ("full_gi", "full_gd", "full_gf")
LAYERS = (0, 10, 20, 31)
PROJECTIONS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
               "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
MODULES = tuple(f"model.layers.{layer}.{proj}" for layer in LAYERS for proj in PROJECTIONS)
PILOT_MODULES = ("model.layers.0.self_attn.o_proj", "model.layers.31.mlp.down_proj")
WINDOWS, LENGTH = frozen.WINDOWS, frozen.LENGTH
HELPERS = ("grouped_rank_ablation_v1.py", *frozen.HELPERS)


def nll_tokens(logits, ids, chunk=128):
    """Explicit FP32 CE, FP64 storage/reduction; batch1, not legacy batch8 NLL."""
    if logits.ndim != 3 or logits.shape[:2] != ids.shape or ids.shape[0] != 1:
        raise ValueError("Expected one full sequence")
    parts = []
    for start in range(0, ids.shape[1]-1, chunk):
        end = min(start+chunk, ids.shape[1]-1)
        values = F.cross_entropy(logits[0, start:end].float(),
                                 ids[0, start+1:end+1].to(logits.device), reduction="none")
        parts.append(values.double().cpu())
    result = torch.cat(parts)
    if not torch.isfinite(result).all() or (result < 0).any():
        raise RuntimeError("Invalid NLL")
    return result


def local_tokens(reference, actual, chunk=128):
    """Actual BF16 outputs, squared difference in FP64, excluding final position."""
    if reference.shape != actual.shape or reference.ndim != 3 or reference.shape[0] != 1:
        raise ValueError("Local output shape mismatch")
    errors, energies = [], []
    for start in range(0, reference.shape[1]-1, chunk):
        end = min(start+chunk, reference.shape[1]-1)
        ref = reference[0, start:end].to(actual.device, dtype=torch.float64)
        out = actual[0, start:end].double()
        errors.append((out-ref).square().sum(-1).cpu())
        energies.append(ref.square().sum(-1).cpu())
    error, energy = torch.cat(errors), torch.cat(energies)
    if not torch.isfinite(error).all() or not torch.isfinite(energy).all():
        raise RuntimeError("Nonfinite actual output error")
    return error, energy


@contextlib.contextmanager
def capture(module):
    values = {}
    def hook(_module, args, output):
        if values:
            raise RuntimeError("Target module executed more than once per forward")
        values.update(x=args[0].detach(), y=output.detach())
    handle = module.register_forward_hook(hook)
    try:
        yield values
        if set(values) != {"x", "y"}:
            raise RuntimeError("Target module was not executed")
    finally:
        handle.remove()


@contextlib.contextmanager
def replacement(module, quant, left, right):
    """Restore weight and hooks even on failure; correction uses two BF16 GEMMs."""
    if (module.weight.dtype != torch.bfloat16 or quant.dtype != torch.bfloat16
            or module.weight.shape != quant.shape
            or left.shape != (module.in_features, 64) or right.shape != (64, module.out_features)):
        raise ValueError("Invalid rank64 intervention shape/dtype")
    left, right = frozen.masked_factors(left, right, 0)
    if not torch.isfinite(quant).all():
        raise ValueError("Nonfinite Wq")
    old = module.weight.detach().clone()
    handle = None
    try:
        with torch.no_grad():
            module.weight.copy_(quant.to(module.weight.device))
        if not torch.equal(module.weight, quant.to(module.weight.device)):
            raise RuntimeError("Wq copy changed bits")
        handle = module.register_forward_hook(prev.correction_hook(
            left.to(module.weight.device), right.to(module.weight.device)))
        yield
    finally:
        if handle is not None:
            handle.remove()
        with torch.no_grad():
            module.weight.copy_(old)
        if not torch.equal(module.weight, old):
            raise RuntimeError("BF16 weight restoration failed")


def compare_window(teacher, student, name, factors, quant, ids, mask, self_check=False):
    """Common teacher input/output and logits; all arms are independently restored."""
    tmod, smod = teacher.get_submodule(name), student.get_submodule(name)
    td, sd = next(teacher.parameters()).device, next(student.parameters()).device
    if not torch.equal(tmod.weight.to(sd), smod.weight):
        raise RuntimeError("Target not in common BF16 background")
    with torch.inference_mode():
        with capture(tmod) as ref:
            tl = teacher(input_ids=ids.to(td), attention_mask=mask.to(td), use_cache=False).logits
        result = {"teacher_nll": nll_tokens(tl, ids)}
        self_max = 0.
        if self_check:
            with capture(smod) as control:
                sl = student(input_ids=ids.to(sd), attention_mask=mask.to(sd), use_cache=False).logits
            self_values, _ = frozen.kl_tokens(tl, sl)
            self_max = float(self_values.max())
            if (self_max > 1e-9 or not torch.equal(ref["x"].to(sd), control["x"])
                    or not torch.equal(ref["y"].to(sd), control["y"])
                    or not torch.equal(result["teacher_nll"], nll_tokens(sl, ids))):
                raise RuntimeError("BF16 common-background self control failed")
            del sl, control, self_values
        for method in METHODS:
            left, right = factors[method]
            with replacement(smod, quant, left, right):
                # Registered AFTER the correction hook: measures corrected output.
                with capture(smod) as actual:
                    sl = student(input_ids=ids.to(sd), attention_mask=mask.to(sd), use_cache=False).logits
                if not torch.equal(ref["x"].to(sd), actual["x"]):
                    raise RuntimeError("Methods did not receive identical module inputs")
                err, energy = local_tokens(ref["y"], actual["y"])
                kl, _ = frozen.kl_tokens(tl, sl)
                result[method+"_sse"] = err
                result[method+"_kl"] = kl[0]
                result[method+"_nll"] = nll_tokens(sl, ids)
                if "reference_energy" in result and not torch.equal(result["reference_energy"], energy):
                    raise RuntimeError("Local reference changed between methods")
                result["reference_energy"] = energy
                del sl, actual, kl
        del tl, ref
    return result, self_max


def binding(ctx, name):
    return {"experiment_identity": ctx.identity, "module": name,
            "quant": ctx.tasks[name]["quant"],
            "factors": {m: ctx.rank64_checked[m][name]["file"] for m in METHODS}}


def state_path(ctx, name, pilot=False):
    return ctx.output/("pilot_windows" if pilot else "windows")/(core.safe_name(name)+".json")


def tensor_rows(values, index, count=None):
    count = LENGTH-1 if count is None else count
    keys = {"teacher_nll", "reference_energy"} | {m+"_"+k for m in METHODS for k in ("sse", "kl", "nll")}
    if set(values) != keys:
        raise RuntimeError("Paired window payload keys mismatch")
    for value in values.values():
        if (value.dtype != torch.float64 or value.shape != (count,)
                or not torch.isfinite(value).all() or (value < 0).any()):
            raise RuntimeError("Invalid paired token payload")
    return {"window": index, "tokens": count, **{k+"_sum": float(v.sum()) for k, v in values.items()}}


def read_state(ctx, name, pilot=False, verify=True):
    path = state_path(ctx, name, pilot)
    limit = 2 if pilot else WINDOWS
    if not path.exists():
        return {"binding": binding(ctx, name), "records": [], "complete": False, "expected_windows": limit}
    state = core.read_json(path)
    rr = state["records"]
    if (state.get("binding") != binding(ctx, name) or state.get("expected_windows") != limit
            or len(rr) > limit or state["complete"] != (len(rr) == limit)):
        raise RuntimeError("Paired window identity/completion mismatch")
    for i, row in enumerate(rr):
        expected = path.parent/core.safe_name(name)/f"window_{i:04d}.safetensors"
        if (row["window"] != i or row["tokens"] != LENGTH-1
                or not 0 <= row.get("self_kl_max", math.inf) <= 1e-9
                or Path(row["file"]["path"]).resolve() != expected.resolve()):
            raise RuntimeError("Paired checkpoint order/path mismatch")
        if verify:
            actual = tensor_rows(ctx.inputs.tensors(row["file"]), i)
            if {k: row[k] for k in actual} != actual:
                raise RuntimeError("Paired checkpoint sums mismatch")
        elif not all(math.isfinite(v) for k, v in row.items() if k.endswith("_sum")):
            raise RuntimeError("Nonfinite window summary")
    return state


def deployment(model):
    return {"parameter_bits": {n: single.tensor_record(t) for n, t in model.named_parameters()},
            "buffer_bits": {n: single.tensor_record(t) for n, t in model.named_buffers()}}


def certify_background(ctx, teacher, student):
    expected = ctx.rank64_deployments["teacher"]["deployment"]
    for model in (teacher, student):
        current = deployment(model)
        if any(current[k] != expected[k] for k in current):
            raise RuntimeError("Resident model differs from frozen BF16 deployment")
    dual.freeze_json(ctx.output/"background.json", {"experiment_identity": ctx.identity,
        "deployment": {k: expected[k] for k in ("parameter_bits", "buffer_bits")}})


def evaluate_module(ctx, teacher, student, name, stop, pilot=False):
    state = read_state(ctx, name, pilot)
    if state["complete"]:
        return
    factors = {}
    for method in METHODS:
        checked = ctx.inputs.tensors(ctx.rank64_checked[method][name]["file"])
        raw = ctx.inputs.tensors(ctx.factors[method][name]["file"])
        if any(not torch.equal(checked[k], raw[k+"_fp64"].bfloat16()) for k in ("A", "B")):
            raise RuntimeError("Factor is not the evaluated direct FP64-to-BF16 rank64 endpoint")
        factors[method] = (checked["A"], checked["B"])
    del raw, checked
    quant = ctx.inputs.tensors(ctx.tasks[name]["quant"])["weight_q"]
    data = frozen.windows(ctx)
    path = state_path(ctx, name, pilot)
    directory = path.parent/core.safe_name(name)
    directory.mkdir(parents=True, exist_ok=True)
    for i in range(len(state["records"]), state["expected_windows"]):
        stop.check()
        with core.heartbeat(f"LOCAL {name} window={i+1}/{state['expected_windows']} GI/DG/GF"):
            values, self_max = compare_window(teacher, student, name, factors, quant,
                data["input_ids"][i:i+1], data["attention_mask"][i:i+1], self_check=True)
        # Shared teacher per-window certificate also detects cross-module contamination.
        dual.freeze_json(ctx.output/"teacher_windows"/f"window_{i:04d}.json",
            {"experiment_identity": ctx.identity, "window": i,
             "teacher_nll_bits": single.tensor_record(values["teacher_nll"])})
        row = tensor_rows(values, i)
        row["self_kl_max"] = self_max
        row["file"] = single.atomic_tensors(directory/f"window_{i:04d}.safetensors", values)
        state["records"].append(row)
        state["complete"] = i+1 == state["expected_windows"]
        core.atomic_json(path, state)
        stop.done += 1


def paired_stats(rows, baseline, key, draws=2000, block=8):
    """GF minus baseline; circular window blocks, exploratory intervals."""
    delta = np.array([r["full_gf_"+key+"_sum"]-r[baseline+"_"+key+"_sum"] for r in rows])
    count = np.array([r["tokens"] for r in rows], dtype=np.float64)
    n = len(rows)
    rng = np.random.default_rng(20260912)
    starts = rng.integers(0, n, size=(draws, math.ceil(n/block)))
    ix = ((starts[:, :, None]+np.arange(block)) % n).reshape(draws, -1)[:, :n]
    low, high = np.quantile(delta[ix].sum(1)/count[ix].sum(1), [.025, .975])
    return {"delta": float(delta.sum()/count.sum()), "ci95_low": float(low), "ci95_high": float(high)}


def summarize(ctx, verify=True):
    summary, paired, detail = [], [], []
    completed = 0
    for name in ctx.modules:
        state = read_state(ctx, name, verify=verify)
        if not state["complete"]:
            continue
        completed += 1
        rows = state["records"]
        count = sum(r["tokens"] for r in rows)
        energy = sum(r["reference_energy_sum"] for r in rows)
        teacher_nll = sum(r["teacher_nll_sum"] for r in rows)/count
        dim = ctx.tasks[name]["shape"][0]
        means = {}
        for method in METHODS:
            means[method] = {k: sum(r[method+"_"+k+"_sum"] for r in rows)/count for k in ("sse", "kl", "nll")}
            a = means[method]
            summary.append({"module": name, "method": method, "rank": 64, "tokens": count,
                "output_dimension": dim, "output_sse_per_position": a["sse"],
                "output_mse_per_element": a["sse"]/dim,
                "output_nmse": a["sse"]*count/energy if energy > 0 else None,
                "teacher_kl_per_token": a["kl"], "nll_per_token_batch1": a["nll"],
                "teacher_nll_per_token_batch1": teacher_nll, "delta_nll_vs_teacher": a["nll"]-teacher_nll})
        for baseline in ("full_gi", "full_gd"):
            stats = {k: paired_stats(rows, baseline, k) for k in ("sse", "kl", "nll")}
            r = means[baseline]["sse"]
            relative = stats["sse"]["delta"]/r if r > 0 else None
            paired.append({"module": name, "contrast": "full_gf_minus_"+baseline,
                "relative_output_sse_change": relative,
                "larger_local_error_lower_kl": stats["sse"]["delta"] > 0 and stats["kl"]["delta"] < 0,
                **{k+"_"+field: value for k, s in stats.items() for field, value in s.items()}})
        detail.extend({"module": name, **{k: v for k, v in row.items() if k != "file"}} for row in rows)
    for file, rows in (("summary.csv", summary), ("paired.csv", paired), ("per_window.csv", detail)):
        prev.write_csv(ctx.output/file, rows)
    core.atomic_json(ctx.output/"status.json", {"experiment_identity": ctx.identity,
        "complete": completed == len(ctx.modules), "completed_modules": completed,
        "expected_modules": len(ctx.modules), "completed_arms": len(summary),
        "expected_arms": len(ctx.modules)*3,
        "note": "Single-module BF16 background; batch1 CE FP32, sums FP64; exploratory paired block intervals, no multiplicity correction."})


def pack_summary(ctx):
    summarize(ctx)
    destination = ctx.output/(VERSION+"_summary.tar.gz")
    names = ("experiment.json", "status.json", "summary.csv", "paired.csv", "per_window.csv", "pilot.json", "background.json")
    with tarfile.open(destination, "w:gz") as archive:
        for name in names:
            path = ctx.output/name
            if path.exists():
                archive.add(path, arcname=name, recursive=False)
    core.log("SUMMARY PACKAGE: "+str(destination))


def setup(args):
    ctx = frozen.setup(args)  # Frozen source audit only; no grouped-ablation execution.
    if any(name not in ctx.tasks for name in MODULES):
        raise RuntimeError("Prespecified Llama modules missing")
    ctx.modules = MODULES
    ctx.experiment = {"version": VERSION, "script_sha256": core.sha(__file__),
        "helpers": {n: core.sha(Path(__file__).with_name(n)) for n in HELPERS},
        "frozen_source_audit": ctx.experiment, "modules": list(MODULES), "methods": list(METHODS),
        "rank": 64, "background": "BF16 except one target Wq plus two-GEMM correction",
        "protocol": {"windows": WINDOWS, "length": LENGTH, "batch": 1,
            "local_positions": "0..2046; same as logits predicting ids[1:]",
            "local_error": "actual corrected BF16 outputs; FP64 difference, squares, sums",
            "kl": "full vocabulary teacher||student; FP64 centered logsumexp; chunk128",
            "nll": "explicit FP32 cross_entropy chunk128, FP64 sum; new batch1 protocol",
            "self_control": "each paired window before intervention; KL<=1e-9, exact inputs/outputs/NLL",
            "checkpoint": "all three methods for one module/window; source factors unchanged"},
        "bootstrap": {"seed": 20260912, "draws": 2000, "circular_block_windows": 8,
                      "scope": "exploratory per-module intervals, no multiple-comparison adjustment"},
        "pilot": {"modules": list(PILOT_MODULES), "windows": 2, "separate_checkpoints": True}}
    ctx.identity = core.fingerprint(ctx.experiment)
    return ctx


def run(ctx, stop, pilot=False):
    names = PILOT_MODULES if pilot else ctx.modules
    if all(read_state(ctx, name, pilot)["complete"] for name in names):
        return
    teacher = student = None
    try:
        stop.check()
        with core.heartbeat("LOCAL load BF16 teacher GPU0 and student GPU1"):
            teacher = frozen.load_kl_model(ctx, 0)
            student = frozen.load_kl_model(ctx, 1)
            certify_background(ctx, teacher, student)
        for name in names:
            evaluate_module(ctx, teacher, student, name, stop, pilot)
            if not pilot:
                summarize(ctx, verify=False)
        certify_background(ctx, teacher, student)
    finally:
        teacher = student = None
        dual.clear_gpu()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("doctor", "pilot", "run", "summarize", "pack"))
    for key in ("run-dir", "repo-dir", "output-dir", "source-fp64-dir", "source-rank64-dir",
                "official-qera-root", "harness-source", "word-reference-dir"):
        parser.add_argument("--"+key, required=True, type=Path)
    parser.add_argument("--max-hours", type=float, default=10)
    parser.add_argument("--max-new-units", type=int)
    args = parser.parse_args(argv)
    if not math.isfinite(args.max_hours) or args.max_hours <= 0 or (args.max_new_units is not None and args.max_new_units < 1):
        parser.error("Budgets must be positive")
    stop = prev.Budget(args.max_hours, args.max_new_units)
    signal.signal(signal.SIGINT, stop.signal)
    signal.signal(signal.SIGTERM, stop.signal)
    ctx = setup(args)
    if args.command == "doctor":
        core.log("DOCTOR PASS: sources verified; 28 modules x 3 rank64 methods; no GPU forward executed")
        return 0
    if ctx.output.exists() and not (ctx.output/"experiment.json").exists() and any(ctx.output.iterdir()):
        raise RuntimeError("Refusing unowned nonempty output")
    ctx.output.mkdir(parents=True, exist_ok=True)
    with core.audit_lock(ctx.output):
        dual.freeze_json(ctx.output/"experiment.json", ctx.experiment)
        try:
            if args.command in ("summarize", "pack"):
                summarize(ctx) if args.command == "summarize" else pack_summary(ctx)
                return 0
            if shutil.disk_usage(ctx.output).free < 4*2**30:
                raise RuntimeError("Need at least 4 GiB free for paired token checkpoints")
            if args.command == "run":
                gate = core.read_json(ctx.output/"pilot.json")
                if gate != {"experiment_identity": ctx.identity, "status": "PASS", "modules": list(PILOT_MODULES), "windows": 2}:
                    raise RuntimeError("Run matching pilot first")
            run(ctx, stop, pilot=args.command == "pilot")
            if args.command == "pilot":
                dual.freeze_json(ctx.output/"pilot.json", {"experiment_identity": ctx.identity,
                    "status": "PASS", "modules": list(PILOT_MODULES), "windows": 2})
                core.log("PILOT COMPLETE: two modules x three arms x two windows; restoration and self controls passed")
            else:
                pack_summary(ctx)
                core.log("LOCAL EXPERIMENT COMPLETE; inspect all modules, including contrary outcomes")
        except prev.Paused:
            summarize(ctx, verify=False)
            core.log("PAUSED (75): rerun identical command/output; current paired window is atomic")
            return 75
        except Exception as exc:
            core.atomic_json(ctx.output/"last_failure.json", {"experiment_identity": ctx.identity,
                "type": type(exc).__name__, "error": str(exc)})
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
