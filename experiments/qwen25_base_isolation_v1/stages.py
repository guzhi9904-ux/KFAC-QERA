from __future__ import annotations

import csv
import gc
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from qera_original_a_isolation.common import import_official_qera
from qera_original_a_isolation.pipeline import load_model, _input_device, _real_sqrtm_root, _chunked_window_nll
from qera_diag_g_isolation.math_ops import ce_hidden_gradient, diagonal_increment, diagonal_scale, correction_drift
from qera_diag_g_isolation.full_svd_v1.solver import solve_weighted
from qera_diag_g_isolation.mxint3_v1.pipeline import official_identity
from qera_diag_g_isolation.full_g_v1.checkpoint import Paused
from qera_diag_g_isolation.full_g_v1.collect import memory_status
from qera_diag_g_isolation.storage import (
    atomic_json, checked_tensors, fingerprint, heartbeat, layers, load_manifest, log, read_json, verify, verify_model, weight_tensor,
)
from .state import Store, artifact
from .protocol import QUANT


def a_shards(manifest):
    groups = manifest["payload"]["groups"]
    size = manifest["payload"]["config"]["a_groups_per_shard"]
    return [groups[i:i+size] for i in range(0, len(groups), size)]


def a_store(config, manifest, i):
    schema = {g["target"]: {"full": ((g["in_features"],)*2, torch.float64),
                            "diag": ((g["in_features"],), torch.float32)} for g in a_shards(manifest)[i]}
    return Store(Path(config["run_dir"]) / "statistics/a" / f"shard_{i:03d}", manifest["sha256"], schema, 2048)


def g_store(config, manifest):
    schema = {x["name"]: {"diag": ((x["shape"][0],), torch.float64)} for x in layers(manifest)}
    return Store(Path(config["run_dir"]) / "statistics/dg", manifest["sha256"], schema, 2047)


def accumulate_a(x, full, diag):
    """Verbatim old arithmetic and accumulator dtypes, including FP32 diagonal."""
    x = x.reshape(-1, x.shape[-1]).to(torch.float32)
    if full.dtype != torch.float64 or diag.dtype != torch.float32:
        raise RuntimeError("A accumulator dtype changed")
    full.add_((x.T @ x).to(device="cpu", dtype=torch.float64))
    diag.add_(x.square().sum(0).to(device="cpu", dtype=torch.float64))


def teacher(config, manifest, dtype):
    loading = dict(manifest["payload"]["source_config"])
    loading["max_memory"] = config["collect_max_memory"] if dtype == "float32" else config["eval_max_memory"]
    with heartbeat("qwen-model", f"loading {dtype}"):
        model = load_model(loading, dtype, "balanced")
    if not {str(v) for v in model.hf_device_map.values()} <= {"0", "1", "cuda:0", "cuda:1"}:
        raise RuntimeError("Model weights must fit on both GPUs; no silent CPU/disk offload")
    for layer in layers(manifest):
        module = model.get_submodule(layer["name"])
        if tuple(module.weight.shape) != tuple(layer["shape"]) or (module.bias is not None) != layer["bias"]:
            raise RuntimeError("Runtime Qwen module/bias mismatch")
    model.requires_grad_(False)
    actual = {"dtype": dtype, "hf_device_map": model.hf_device_map,
              "torch": torch.__version__, "attention": "eager", "use_cache": False,
              "targets": len(layers(manifest)), "bias_policy": "unchanged"}
    atomic_json(Path(config["run_dir"]) / "diagnostics" / f"latest_runtime_{dtype}.json", actual)
    log("qwen-model", f"runtime={actual}")
    return model


def collect(config, stop, kind, max_new_windows=None):
    manifest = load_manifest(config)
    verify_model(manifest)
    data = checked_tensors(manifest["payload"]["data"]["calibration"])
    stores = [a_store(config, manifest, i) for i in range(len(a_shards(manifest)))] if kind == "a" else [g_store(config, manifest)]
    pending = []
    for i, store in enumerate(stores):
        store.cleanup()
        state = store.current()
        if state is None or state["windows"] < 256:
            pending.append(i)
    if not pending:
        log("collect-qwen", f"{kind} already complete")
        return
    stop.check()
    model = teacher(config, manifest, "float32")
    handles = []
    try:
        modules, device = dict(model.named_modules()), _input_device(model)
        if kind == "dg":
            if model.get_output_embeddings().bias is not None:
                raise RuntimeError("Expected bias-free lm_head for analytic CE seed")
            handles.append(model.get_input_embeddings().register_forward_hook(lambda _m, _a, out: out.requires_grad_(True)))
        new = 0
        for i in pending:
            stop.check()
            store = stores[i]
            with heartbeat("collect-qwen", f"{kind} shard={i+1} loading checkpoint"):
                sums, start, nll = store.load()
            seen, valid = set(), None
            def hook_for(name):
                def hook(_module, args, output):
                    if kind == "a":
                        if name in seen:
                            raise RuntimeError("Duplicate A hook")
                        accumulate_a(args[0], sums[name]["full"], sums[name]["diag"])
                        seen.add(name)
                    else:
                        def backward(gradient):
                            if name in seen:
                                raise RuntimeError("Duplicate DG hook")
                            sums[name]["diag"].add_(diagonal_increment(gradient, valid, 128))
                            seen.add(name)
                        output.register_hook(backward)
                return hook
            shard_handles = [modules[name].register_forward_hook(hook_for(name)) for name in sums]
            batch = config["a_batch_size"] if kind == "a" else 1
            interval = config["a_checkpoint_windows"] if kind == "a" else config["checkpoint_every_windows"]
            started, last_saved = time.monotonic(), start
            try:
                for index in range(start, 256, batch):
                    end = min(index+batch, 256)
                    ids = data["input_ids"][index:end].to(device)
                    mask = data["attention_mask"][index:end].to(device)
                    if kind == "dg":
                        mask = mask.bool()
                    seen.clear()
                    label = f"{kind} shard={i+1}/{len(stores)} windows={index+1}-{end}/256"
                    with heartbeat("collect-qwen", label):
                        if kind == "a":
                            with torch.inference_mode():
                                model(input_ids=ids, attention_mask=mask, use_cache=False)
                        else:
                            valid = torch.zeros_like(mask)
                            valid[:, :-1] = mask[:, :-1] & mask[:, 1:]
                            with torch.autograd.graph.save_on_cpu(pin_memory=False, device_type="cuda"):
                                output = model.model(input_ids=ids, attention_mask=mask, use_cache=False)
                                hidden = output.last_hidden_state.to(model.get_output_embeddings().weight.device)
                                seed, loss, tokens = ce_hidden_gradient(hidden, model.get_output_embeddings().weight,
                                                                       ids, mask, config["ce_gradient_chunk_tokens"])
                                hidden.backward(seed)
                            if tokens != 2047 or any(p.grad is not None for p in model.parameters()):
                                raise RuntimeError("DG token count/frozen parameter gradient mismatch")
                            nll += loss
                            del output, hidden, seed
                    if seen != set(sums):
                        raise RuntimeError("Incomplete hook coverage: partial batch must NOT be committed")
                    new += end-index
                    limit = max_new_windows is not None and new >= max_new_windows
                    if end % interval == 0 or end == 256 or limit or stop.requested():
                        with heartbeat("checkpoint-qwen", label):
                            store.save(sums, end, nll)
                        last_saved = end
                    elapsed = time.monotonic()-started
                    log("collect-qwen", f"{label} elapsed={elapsed:.0f}s shard_eta={elapsed/(end-start)*(256-end):.0f}s memory={memory_status()}")
                    if stop.requested() and last_saved != end:
                        store.save(sums, end, nll)
                    stop.check()
                    if limit:
                        raise Paused("Qwen pilot completed and checkpoint committed")
                    del ids, mask
            finally:
                for h in shard_handles:
                    h.remove()
                del sums
                gc.collect()
    finally:
        for h in handles:
            h.remove()
        del model
        gc.collect()
        torch.cuda.empty_cache()


def roots(config, stop):
    manifest = load_manifest(config)
    official = import_official_qera(manifest["payload"]["source_config"])
    for i, groups in enumerate(a_shards(manifest)):
        store = a_store(config, manifest, i)
        s = store.current()
        if s is None or s["windows"] != 256:
            raise RuntimeError("Complete A256 before roots")
        for group in groups:
            stop.check()
            name = group["target"]
            old = artifact(config, manifest, "roots", name)
            if old:
                if old.get("raw_reference") != s["files"][name]:
                    raise RuntimeError("A root input binding changed")
                continue
            raw = store.read(s, name)
            matrix = raw["full"].numpy()
            matrix = (matrix + matrix.T)*.5
            with heartbeat("roots-qwen", f"{name} scipy sqrtm dimension={len(matrix)}"):
                result = official["sqrtm_scipy"](matrix)
                root, imaginary, normalized_imaginary = _real_sqrtm_root(result, 524288)
                covariance = matrix / 524288
                residual = float(np.linalg.norm(root @ root-covariance) / max(np.linalg.norm(covariance), np.finfo(float).eps))
            if not math.isfinite(residual):
                raise RuntimeError("Nonfinite A root residual")
            if imaginary > config["sqrtm_max_imaginary"]:
                log("roots-qwen", f"complex diagnostic raw={imaginary}; discard imaginary as in pinned QERA")
            full = torch.from_numpy(root).float()
            diag = (raw["diag"] / 524288).clamp_min(0).sqrt().float()
            if not torch.isfinite(full).all() or not torch.isfinite(diag).all():
                raise RuntimeError("Nonfinite A roots")
            artifact(config, manifest, "roots", name, {"full": full, "diag": diag}, raw_reference=s["files"][name],
                     residual=residual, raw_max_imaginary=imaginary, normalized_max_imaginary=normalized_imaginary)
            log("roots-qwen", f"saved {name} residual={residual:.3g}")
            del raw, matrix, result, root, covariance, full, diag


def quantize(config, stop):
    manifest = load_manifest(config)
    verify_model(manifest)
    quantizer = import_official_qera(manifest["payload"]["source_config"])["mxint_quantizer"]
    for layer in layers(manifest):
        stop.check()
        if artifact(config, manifest, "quantized", layer["name"]):
            continue
        weight = weight_tensor(manifest, layer).to(config["solve_device"])
        if weight.dtype != torch.bfloat16:
            raise RuntimeError("Expected BF16 checkpoint")
        q32 = quantizer(weight.float(), width=3, block_size=32, block_axis=-1)
        q16 = quantizer(weight, width=3, block_size=32, block_axis=-1)
        if not torch.isfinite(q32).all() or not torch.equal(q32, q16.float()):
            raise RuntimeError("FP32/BF16 MXINT3 quantization mismatch")
        artifact(config, manifest, "quantized", layer["name"], {"weight_q": q16},
                 quantization=QUANT, fp32_bf16_equal=True)
        log("quantize-qwen", layer["name"])
        del weight, q32, q16


def valid_drift(drift, config):
    return set(drift) == {str(r) for r in config["ranks"]} and all(
        math.isfinite(v) and 0 <= v <= config["identity_product_tolerance"] for v in drift.values())


def solve(config, stop, kind):
    manifest = load_manifest(config)
    verify_model(manifest)
    official = import_official_qera(manifest["payload"]["source_config"])
    from qera.approximate import _compute_scale_inv_dot_U
    g = None
    if kind == "gd":
        store = g_store(config, manifest)
        g = store.current()
        if g is None or g["windows"] != 256:
            raise RuntimeError("Complete DG256 first")
    device, rank = config["solve_device"], max(config["ranks"])
    with torch.no_grad():
        for group in manifest["payload"]["groups"]:
            stop.check()
            root_record = artifact(config, manifest, "roots", group["target"])
            if root_record is None:
                raise RuntimeError("Missing A roots")
            scales = checked_tensors(root_record["file"])
            for layer in group["layers"]:
                name = layer["name"]
                stop.check()
                quant = artifact(config, manifest, "quantized", name)
                if quant is None or quant.get("fp32_bf16_equal") is not True or quant.get("quantization") != QUANT:
                    raise RuntimeError("Missing audited W3")
                weight = weight_tensor(manifest, layer).float().to(device)
                q = checked_tensors(quant["file"])["weight_q"].float().to(device)
                error = (weight.cpu()-q.cpu()).T.to(device)
                del q
                scale_g = torch.ones(layer["shape"][0], device=device)
                if g:
                    value = store.read(g, name)["diag"]
                    scale_g, gd_diagnostics = diagonal_scale(value, 256*2047, config["g_relative_floor"])
                    scale_g = scale_g.to(device)
                for a_kind in ("diag", "full"):
                    method = a_kind + "_" + kind
                    inputs = {"quant": quant["file"], "roots": root_record["file"]}
                    gi = None
                    if g:
                        inputs["g"] = g["files"][name]
                        gi = artifact(config, manifest, f"corrections/{a_kind}_gi", name)
                        if gi is None or not valid_drift(gi.get("identity_drift", {}), config):
                            raise RuntimeError("Missing passing GI baseline")
                        inputs["gi"] = gi["file"]
                    old = artifact(config, manifest, f"corrections/{method}", name)
                    if old:
                        if old.get("inputs") != inputs or not valid_drift(old.get("identity_drift", {}), config):
                            raise RuntimeError("Resumed correction input/gate mismatch")
                        continue
                    scale_a = scales[a_kind].to(device)
                    with heartbeat("solve-qwen", f"{name} {method} full SVD rank={rank}"):
                        if kind == "gi":
                            ref_a, ref_b, mse = official_identity(official, name, weight, scale_a, rank)
                            a, b, _ = solve_weighted(error, scale_a, scale_g, rank, _compute_scale_inv_dot_U)
                            drift = correction_drift(a, b, ref_a, ref_b, config["ranks"])
                            del a, b
                            if not valid_drift(drift, config):
                                raise RuntimeError(f"Identity-G regression failed: {name} {a_kind} {drift}")
                            a, b, metrics = ref_a, ref_b, {"official_mse": mse}
                        else:
                            a, b, metrics = solve_weighted(error, scale_a, scale_g, rank, _compute_scale_inv_dot_U)
                            drift = gi["identity_drift"]
                            metrics["g_diagnostics"] = gd_diagnostics
                    artifact(config, manifest, f"corrections/{method}", name, {"A": a, "B": b},
                             method=method, rank=rank, quantization=QUANT, inputs=inputs,
                             identity_drift=drift, **metrics)
                    log("solve-qwen", f"saved {name} {method} {metrics}")
                    del a, b, scale_a
                    if kind == "gi":
                        del ref_a, ref_b
                    stop.check()
                del weight, error, scale_g
                torch.cuda.empty_cache()


def choices(config):
    return [("BF16", None, None), ("W3_MXINT", "wq", None)] + [
        (f"{method.upper()}_R{rank}", method, rank)
        for method in ("diag_gi", "full_gi", "diag_gd", "full_gd") for rank in config["ranks"]]


def evaluation_inputs(config, manifest, method, check=False):
    result = {}
    for layer in layers(manifest) if method else []:
        name = layer["name"]
        q = artifact(config, manifest, "quantized", name, check=check)
        if q is None or q.get("fp32_bf16_equal") is not True or q.get("quantization") != QUANT:
            raise RuntimeError("Missing audited Wq")
        result[name] = {"quant": q["file"]}
        if method != "wq":
            c = artifact(config, manifest, f"corrections/{method}", name, check=check)
            if (c is None or not valid_drift(c.get("identity_drift", {}), config) or c["inputs"]["quant"] != q["file"]
                    or c.get("method") != method or c.get("rank") != max(config["ranks"]) or c.get("quantization") != QUANT):
                raise RuntimeError("Missing/invalid correction")
            if method.endswith("_gd"):
                gi = artifact(config, manifest, f"corrections/{method[:-2]}gi", name, check=check)
                if gi is None or c["inputs"].get("gi") != gi["file"]:
                    raise RuntimeError("DG correction references a different GI baseline")
            result[name]["correction"] = c["file"]
    return result


def evaluation_rows(path, digest, total):
    if not path.exists():
        return []
    state = read_json(path)
    rows = state["records"]
    if state.get("protocol") != digest or len(rows) > total:
        raise RuntimeError("Evaluation protocol/count changed")
    for i, row in enumerate(rows):
        if row["window"] != i or row["tokens"] != 2047 or not math.isfinite(row["nll_sum"]) or row["nll_sum"] < 0:
            raise RuntimeError("Invalid evaluation rows")
    return rows


def evaluate(config, stop, phase):
    manifest = load_manifest(config)
    verify_model(manifest)
    windows = checked_tensors(manifest["payload"]["data"]["wikitext2"])
    total = len(windows["input_ids"])
    cache = {}
    for name, method, rank in choices(config):
        if (phase == "gi" and method in ("diag_gd", "full_gd")) or (phase == "gd" and method not in ("diag_gd", "full_gd")):
            continue
        stop.check()
        files = evaluation_inputs(config, manifest, method)
        digest = fingerprint({"manifest": manifest["sha256"], "name": name, "files": files})
        path = Path(config["run_dir"]) / "evaluation/configurations" / f"{name}.json"
        rows = evaluation_rows(path, digest, total)
        if len(rows) == total:
            continue
        model = teacher(config, manifest, "bfloat16")
        handles = []
        try:
            for module_name, records in files.items():
                module = model.get_submodule(module_name)
                bias_before = None if module.bias is None else module.bias.detach().clone()
                identity = fingerprint(records["quant"])
                if module_name not in cache:
                    cache[module_name] = (identity, checked_tensors(records["quant"])["weight_q"])
                if cache[module_name][0] != identity:
                    raise RuntimeError("Wq identity changed during evaluation")
                module.weight.data.copy_(cache[module_name][1].to(module.weight.device))
                if bias_before is not None and not torch.equal(module.bias, bias_before):
                    raise RuntimeError("Bias changed during weight quantization")
                if "correction" in records:
                    factors = checked_tensors(records["correction"])
                    a = factors["A"][:, :rank].to(device=module.weight.device, dtype=module.weight.dtype)
                    b = factors["B"][:rank].to(device=module.weight.device, dtype=module.weight.dtype)
                    handles.append(module.register_forward_hook(lambda _m, args, out, a=a, b=b: out+(args[0]@a)@b))
            with torch.inference_mode():
                device = _input_device(model)
                for index in range(len(rows), total, config["eval_batch_size"]):
                    end = min(index+config["eval_batch_size"], total)
                    ids = windows["input_ids"][index:end].to(device)
                    mask = windows["attention_mask"][index:end].to(device)
                    with heartbeat("evaluate-qwen", f"{name} windows={index+1}-{end}/{total}"):
                        logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
                        metrics = _chunked_window_nll(logits, ids, mask, config["eval_ce_chunk_tokens"])
                    del logits
                    if len(metrics) != end-index:
                        raise RuntimeError("Evaluation batch coverage mismatch")
                    for offset, (nll, tokens) in enumerate(metrics):
                        if tokens != 2047 or not math.isfinite(nll) or nll < 0:
                            raise RuntimeError("Invalid evaluation metric")
                        rows.append({"window": index+offset, "tokens": tokens, "nll_sum": nll})
                    atomic_json(path, {"protocol": digest, "records": rows})
                    log("evaluate-qwen", f"{name} window={end}/{total}")
                    stop.check()
        finally:
            for handle in handles:
                handle.remove()
            handles.clear()
            module = a = b = factors = None
            del model
            gc.collect()
            torch.cuda.empty_cache()
        summarize(config)
    return summarize(config)


def summarize(config):
    manifest = load_manifest(config)
    total = manifest["payload"]["data_details"]["wikitext2"]["windows"]
    summary, per_window = [], []
    root = Path(config["run_dir"]) / "evaluation"
    for name, method, rank in choices(config):
        path = root / "configurations" / f"{name}.json"
        if not path.exists():
            continue
        files = evaluation_inputs(config, manifest, method)
        digest = fingerprint({"manifest": manifest["sha256"], "name": name, "files": files})
        rows = evaluation_rows(path, digest, total)
        if len(rows) != total:
            continue
        nll = sum(x["nll_sum"] for x in rows)
        summary.append({"configuration": name, "method": method or "teacher", "rank": rank or "", "windows": total,
                        "prediction_tokens": total*2047, "nll_sum": nll, "ppl": math.exp(nll/(total*2047))})
        per_window.extend({"configuration": name, **x} for x in rows)
    root.mkdir(parents=True, exist_ok=True)
    for name, rows in (("ppl_summary_wikitext2.csv", summary), ("wikitext2_per_window.csv", per_window)):
        if rows:
            path = root / name
            temporary = path.with_suffix(".csv.tmp")
            with temporary.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temporary, path)
    result = {"status": "PASS" if len(summary) == 18 else "INCOMPLETE", "configurations": len(summary),
              "expected_configurations": 18, "windows_per_configuration": total}
    atomic_json(root / "status.json", result)
    return result
