from __future__ import annotations

import csv
import gc
import math
import os
import time
from pathlib import Path

import torch

from qera_original_a_isolation.common import import_official_qera, safe_name
from qera_original_a_isolation.pipeline import (
    _chunked_window_nll, _input_device, _reset_evaluation_memory_peaks, load_model,
)
from .math_ops import ce_hidden_gradient, correction_drift, diagonal_increment, diagonal_scale, solve_weighted
from .storage import (
    atomic_json, atomic_tensors, checked_tensors, file_record, fingerprint, gpu_check,
    heartbeat, layers, load_manifest, log, read_json, verify, verify_model, weight_tensor,
)


def artifact_path(config, directory, name):
    return Path(config["run_dir"]) / directory / f"{safe_name(name)}.safetensors"


def completed_artifact(path, manifest, check=True):
    metadata = path.with_suffix(".json")
    if not metadata.exists():
        return None
    saved = read_json(metadata)
    if saved.get("status") != "PASS" or saved.get("manifest_sha256") != manifest["sha256"]:
        raise RuntimeError(f"Artifact protocol mismatch: {metadata}")
    if check:
        verify(saved["file"])
    return saved


def save_artifact(path, manifest, tensors, **details):
    atomic_tensors(path, tensors)
    record = {"status": "PASS", "manifest_sha256": manifest["sha256"],
              "file": file_record(path), **details}
    atomic_json(path.with_suffix(".json"), record)
    return record


def quantize(config):
    manifest = load_manifest(config)
    verify_model(manifest)
    qera = import_official_qera(manifest["payload"]["source_config"])
    quantizer = qera["mxint_quantizer"]
    records = {}
    with torch.no_grad():
        for index, layer in enumerate(layers(manifest), 1):
            path = artifact_path(config, "quantized", layer["name"])
            record = completed_artifact(path, manifest)
            if record is None:
                weight = weight_tensor(manifest, layer).to(config["solve_device"])
                if weight.dtype != torch.bfloat16:
                    raise ValueError("Expected original BF16 checkpoint weights")
                q32 = quantizer(weight.float(), width=4, block_size=32, block_axis=-1)
                q16 = quantizer(weight, width=4, block_size=32, block_axis=-1)
                if not torch.equal(q32, q16.float()):
                    raise RuntimeError(f"FP32 solve / BF16 eval quantization mismatch: {layer['name']}")
                record = save_artifact(path, manifest, {"weight_q": q16},
                                       layer=layer["name"], fp32_bf16_quantization_equal=True)
                del weight, q32, q16
            records[layer["name"]] = record["file"]
            log("quantize", f"{index}/224 saved/verified {layer['name']}")
    atomic_json(Path(config["run_dir"]) / "quantized" / "complete.json",
                {"manifest_sha256": manifest["sha256"], "files": records, "status": "PASS"})


def _load_g_checkpoint(config, manifest):
    state_path = Path(config["run_dir"]) / "statistics" / "collect_state.json"
    if not state_path.exists():
        return {x["name"]: torch.zeros(x["shape"][0], dtype=torch.float64) for x in layers(manifest)}, 0, 0.0
    state = read_json(state_path)
    if state["manifest_sha256"] != manifest["sha256"]:
        raise RuntimeError("G checkpoint belongs to a different protocol")
    count = int(state["windows_completed"])
    if not 0 <= count <= 256 or state["prediction_tokens"] != count * 2047:
        raise RuntimeError("Invalid G checkpoint counters")
    if not math.isfinite(float(state["teacher_nll_sum"])) or float(state["teacher_nll_sum"]) < 0:
        raise RuntimeError("Invalid checkpoint teacher NLL")
    tensors = checked_tensors(state["file"])
    expected = {x["name"]: (x["shape"][0],) for x in layers(manifest)}
    if set(tensors) != set(expected):
        raise RuntimeError("G checkpoint module coverage mismatch")
    for name, shape in expected.items():
        value = tensors[name]
        if tuple(value.shape) != shape or value.dtype != torch.float64 or not torch.isfinite(value).all() or (value < 0).any():
            raise RuntimeError(f"Invalid G checkpoint tensor: {name}")
    return tensors, count, float(state["teacher_nll_sum"])


def _save_g_checkpoint(config, manifest, sums, count, nll):
    root = Path(config["run_dir"]) / "statistics"
    path = root / "checkpoints" / f"window_{count:04d}.safetensors"
    atomic_tensors(path, sums)
    state = {"status": "PASS" if count == 256 else "IN_PROGRESS",
             "manifest_sha256": manifest["sha256"], "windows_completed": count,
             "prediction_tokens": count * 2047, "teacher_nll_sum": nll,
             "representation": "diagonal", "accumulator_dtype": "float64", "file": file_record(path)}
    # A crash before publishing state leaves only an unreferenced complete snapshot.
    atomic_json(path.with_suffix(".json"), state)
    atomic_json(root / "collect_state.json", state)
    log("checkpoint", f"retained G window={count}/256 file={path}")


def collect(config):
    manifest = load_manifest(config)
    sums, start, nll = _load_g_checkpoint(config, manifest)
    if start == 256:
        log("collect", "G already complete; raw statistics retained")
        return
    gpu_check()
    verify_model(manifest)
    data = checked_tensors(manifest["payload"]["data"]["calibration"])
    loading = dict(manifest["payload"]["source_config"])
    loading["max_memory"] = config["collect_max_memory"]
    _reset_evaluation_memory_peaks([0, 1])
    with heartbeat("collect", "loading FP32 teacher"):
        model = load_model(loading, "float32", "balanced")
    placements = {str(x) for x in model.hf_device_map.values()}
    if not placements <= {"0", "1", "cuda:0", "cuda:1"}:
        raise RuntimeError(f"FP32 teacher must fit across both GPUs, not CPU/disk weights: {model.hf_device_map}")
    model.requires_grad_(False)
    if model.get_output_embeddings().bias is not None:
        raise RuntimeError("Analytic CE seed currently supports a bias-free Llama lm_head")
    modules = dict(model.named_modules())
    increment = {}
    active_mask = None
    def make_hook(name):
        def forward(_module, _inputs, output):
            def backward(gradient):
                if name in increment:
                    raise RuntimeError(f"Duplicate G hook: {name}")
                increment[name] = diagonal_increment(gradient, active_mask)
            output.register_hook(backward)
        return forward
    handles = [model.get_input_embeddings().register_forward_hook(lambda _m, _a, out: out.requires_grad_(True))]
    for layer in layers(manifest):
        module = modules[layer["name"]]
        if tuple(module.weight.shape) != tuple(layer["shape"]) or module.weight.dtype != torch.float32:
            raise RuntimeError("Projection shape/dtype mismatch")
        handles.append(module.register_forward_hook(make_hook(layer["name"])))
    atomic_json(Path(config["run_dir"]) / "statistics" / "runtime.json",
                {"hf_device_map": {k: str(v) for k, v in model.hf_device_map.items()},
                 "torch": torch.__version__, "batch_size": 1, "dtype": "float32",
                 "save_on_cpu": True, "start_window": start})
    started = time.monotonic()
    device = _input_device(model)
    try:
        for index in range(start, 256):
            increment.clear()
            ids = data["input_ids"][index:index + 1].to(device)
            mask = data["attention_mask"][index:index + 1].to(device).bool()
            active_mask = torch.zeros_like(mask)
            active_mask[:, :-1] = mask[:, :-1] & mask[:, 1:]
            log("collect", f"starting window={index + 1}/256 targets=224")
            with heartbeat("collect", f"window={index + 1}/256 forward/backward"):
                with torch.autograd.graph.save_on_cpu(pin_memory=False, device_type="cuda"):
                    output = model.model(input_ids=ids, attention_mask=mask, use_cache=False)
                    hidden = output.last_hidden_state.to(model.get_output_embeddings().weight.device)
                    seed, window_nll, tokens = ce_hidden_gradient(
                        hidden, model.get_output_embeddings().weight, ids, mask,
                        config["ce_gradient_chunk_tokens"],
                    )
                    hidden.backward(seed)
            if set(increment) != set(sums) or tokens != 2047:
                raise RuntimeError("Gradient coverage/token count mismatch; window NOT committed")
            if any(p.grad is not None for p in model.parameters()):
                raise RuntimeError("Frozen teacher accumulated parameter gradients")
            for name, value in increment.items():
                sums[name].add_(value)
            nll += window_nll
            completed = index + 1
            if completed % config["checkpoint_every_windows"] == 0 or completed in (64, 128, 256):
                _save_g_checkpoint(config, manifest, sums, completed, nll)
            elapsed = time.monotonic() - started
            eta = elapsed / (completed - start) * (256 - completed)
            log("collect", f"window={completed}/256 elapsed={elapsed:.0f}s eta={eta:.0f}s "
                f"gpu_peak_GiB={[round(torch.cuda.max_memory_allocated(i) / 2**30, 2) for i in range(2)]}")
            del output, hidden, seed, ids, mask
    finally:
        for handle in handles:
            handle.remove()
        del model
        gc.collect()
        torch.cuda.empty_cache()


def solve(config):
    manifest = load_manifest(config)
    sums, count, _ = _load_g_checkpoint(config, manifest)
    if count != 256:
        raise RuntimeError("Collect all 256 G windows before solving")
    verify_model(manifest)
    import_official_qera(manifest["payload"]["source_config"])
    from qera.approximate import _compute_scale_inv_dot_U
    device = config["solve_device"]
    rank = max(config["ranks"])
    completed = 0
    with torch.no_grad():
        for group in manifest["payload"]["groups"]:
            roots = checked_tensors(group["roots"])
            for layer in group["layers"]:
                completed += 1
                name = layer["name"]
                log("solve", f"module={completed}/224 {name}")
                quant_path = artifact_path(config, "quantized", name)
                quant_state = completed_artifact(quant_path, manifest)
                if quant_state is None:
                    raise RuntimeError("Run quantize first")
                error_t = (weight_tensor(manifest, layer).float() - checked_tensors(quant_state["file"])["weight_q"].float()).T.to(device)
                scale_g, diagnostics = diagonal_scale(sums[name], count * 2047, config["g_relative_floor"])
                g_path = artifact_path(config, "statistics/effective_g", name)
                if completed_artifact(g_path, manifest) is None:
                    save_artifact(g_path, manifest, {"g_sum": sums[name], "sqrt_g_effective": scale_g},
                                  count=count * 2047, **diagnostics)
                scale_g = scale_g.to(device)
                for method in ("diag", "full"):
                    path = artifact_path(config, f"corrections/{method}_gd", name)
                    if completed_artifact(path, manifest) is not None:
                        log("solve", f"{method}_gd already complete")
                        continue
                    scale_a = roots[method].to(device)
                    reference = checked_tensors(layer["gi"][method])
                    with heartbeat("solve", f"{name} {method} identity-G regression SVD"):
                        left_i, right_i, _ = solve_weighted(error_t, scale_a, torch.ones_like(scale_g), rank, _compute_scale_inv_dot_U)
                        drift = correction_drift(left_i, right_i, reference["A"].to(device), reference["B"].to(device), config["ranks"])
                    if max(drift.values()) > config["identity_product_tolerance"]:
                        raise RuntimeError(f"Identity-G regression failed: {name} {method} {drift}; do not interpret as G effect")
                    del left_i, right_i, reference
                    with heartbeat("solve", f"{name} {method} diagonal-G SVD"):
                        left, right, metrics = solve_weighted(error_t, scale_a, scale_g, rank, _compute_scale_inv_dot_U)
                    record = save_artifact(path, manifest, {"A": left, "B": right}, layer=name, rank=rank,
                                           method=method + "_gd", identity_product_drift=drift,
                                           g_diagnostics=diagnostics, **metrics)
                    log("solve", f"saved {method}_gd {name} sse={metrics['weighted_sse_before']:.6g}->{metrics['weighted_sse_after']:.6g}")
                    del left, right, scale_a, record
                del error_t, scale_g
                torch.cuda.empty_cache()
            del roots
    atomic_json(Path(config["run_dir"]) / "solve_complete.json", {"status": "PASS", "manifest_sha256": manifest["sha256"], "modules": completed})


def configurations(config):
    result = [("BF16", None, None), ("W4_MXINT", "wq", None)]
    for method in ("diag_gi", "full_gi", "diag_gd", "full_gd"):
        result.extend((f"{method.upper()}_R{rank}", method, rank) for rank in config["ranks"])
    return result


def evaluation_records(path, protocol, total):
    if not path.exists():
        return []
    state = read_json(path)
    if state.get("protocol_sha256") != protocol:
        raise RuntimeError("Evaluation protocol/artifact mismatch")
    records = state["records"]
    if len(records) > total:
        raise RuntimeError("Too many evaluation records")
    for index, row in enumerate(records):
        if row["window"] != index or row["tokens"] != 2047 or not math.isfinite(row["nll_sum"]) or row["nll_sum"] < 0:
            raise RuntimeError("Invalid/non-contiguous evaluation checkpoint")
    return records


def check_control(config, manifest, name, records):
    reference = manifest["payload"].get("control_reference_ppl", {}).get(name)
    if reference is None or len(records) != 138:
        return
    observed = math.exp(sum(r["nll_sum"] for r in records) / sum(r["tokens"] for r in records))
    difference = abs(observed - reference)
    record = {"configuration": name, "reference_ppl": reference, "observed_ppl": observed,
              "absolute_difference": difference, "tolerance": config["control_ppl_tolerance"],
              "status": "PASS" if difference <= config["control_ppl_tolerance"] else "FAIL",
              "note": "Engineering regression against our original token-PPL run, NOT QERA paper's word PPL"}
    atomic_json(Path(config["run_dir"]) / "evaluation/control_checks" / f"{name}.json", record)
    if record["status"] != "PASS":
        raise RuntimeError(f"Control PPL regression failed: {record}. Review before interpreting GD.")


def _evaluation_inputs(config, manifest, method, check=True):
    artifacts = {}
    for layer in layers(manifest) if method is not None else []:
        name = layer["name"]
        quant = completed_artifact(artifact_path(config, "quantized", name), manifest, check=check)
        if quant is None:
            raise RuntimeError("Missing frozen quantized weight")
        value = {"quant": quant["file"]}
        if method not in (None, "wq"):
            if method.endswith("_gi"):
                correction = layer["gi"][method.split("_")[0]]
                if check:
                    verify(correction)
            else:
                state = completed_artifact(artifact_path(config, f"corrections/{method}", name), manifest, check=check)
                if state is None:
                    raise RuntimeError("Missing GD correction")
                correction = state["file"]
            value["correction"] = correction
        artifacts[name] = value
    return artifacts


def evaluate(config, only=None):
    manifest = load_manifest(config)
    gpu_check()
    choices = configurations(config)
    if only and only not in [x[0] for x in choices]:
        raise ValueError(f"Unknown configuration {only}")
    verify_model(manifest)
    windows = checked_tensors(manifest["payload"]["data"]["wikitext2"])
    total = len(windows["input_ids"])
    # Keep one verified BF16 Wq copy in host RAM (~14 GiB), not repeated shared-
    # filesystem reads for 17 quantized evaluations. No extra GPU resident copy.
    quant_cache = {}
    for name, method, rank in choices:
        if only and only != name:
            continue
        artifacts = _evaluation_inputs(config, manifest, method, check=False)
        protocol = fingerprint({"manifest": manifest["sha256"], "name": name, "artifacts": artifacts})
        path = Path(config["run_dir"]) / "evaluation" / "configurations" / f"{name}.json"
        records = evaluation_records(path, protocol, total)
        if len(records) == total:
            check_control(config, manifest, name, records)
            log("evaluate", f"{name} already complete")
            continue
        loading = dict(manifest["payload"]["source_config"])
        loading["max_memory"] = config["eval_max_memory"]
        _reset_evaluation_memory_peaks([0, 1])
        with heartbeat("evaluate", f"loading {name}"):
            model = load_model(loading, "bfloat16", "balanced")
        if not {str(x) for x in model.hf_device_map.values()} <= {"0", "1", "cuda:0", "cuda:1"}:
            raise RuntimeError("Evaluation weights must be GPU resident")
        handles = []
        modules = {}
        module = tensors = left = right = hook = None
        try:
            modules = dict(model.named_modules())
            for module_name, files in artifacts.items():
                module = modules[module_name]
                identity = fingerprint(files["quant"])
                if module_name not in quant_cache:
                    quant_cache[module_name] = (identity, checked_tensors(files["quant"])["weight_q"])
                cached_identity, weight_q = quant_cache[module_name]
                if identity != cached_identity:
                    raise RuntimeError("Quantized weight identity changed during evaluation")
                module.weight.data.copy_(weight_q.to(module.weight.device))
                if "correction" in files:
                    tensors = checked_tensors(files["correction"])
                    left = tensors["A"][:, :rank].to(device=module.weight.device, dtype=module.weight.dtype)
                    right = tensors["B"][:rank].to(device=module.weight.device, dtype=module.weight.dtype)
                    def hook(_module, args, output, a=left, b=right):
                        return output + (args[0] @ a) @ b
                    handles.append(module.register_forward_hook(hook))
            started, start = time.monotonic(), len(records)
            device = _input_device(model)
            with torch.inference_mode():
                for index in range(start, total, config["eval_batch_size"]):
                    end = min(index + config["eval_batch_size"], total)
                    ids = windows["input_ids"][index:end].to(device)
                    mask = windows["attention_mask"][index:end].to(device)
                    with heartbeat("evaluate", f"{name} windows={index + 1}-{end}/{total}"):
                        logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
                        metrics = _chunked_window_nll(logits, ids, mask, config["eval_ce_chunk_tokens"])
                    del logits, ids, mask
                    for offset, (nll, tokens) in enumerate(metrics):
                        if tokens != 2047 or not math.isfinite(nll) or nll < 0:
                            raise RuntimeError("Invalid evaluation metric")
                        records.append({"window": index + offset, "tokens": tokens, "nll_sum": nll})
                    atomic_json(path, {"protocol_sha256": protocol, "manifest_sha256": manifest["sha256"],
                                       "configuration": name, "records": records, "complete": end == total})
                    elapsed = time.monotonic() - started
                    log("evaluate", f"{name} window={end}/{total} elapsed={elapsed:.0f}s eta={elapsed / (end-start) * (total-end):.0f}s")
        finally:
            for handle in handles:
                handle.remove()
            handles.clear()
            modules.clear()
            module = tensors = left = right = hook = None
            del model
            gc.collect()
            torch.cuda.empty_cache()
        check_control(config, manifest, name, records)
        summarize(config)
    return summarize(config)


def summarize(config):
    manifest = load_manifest(config)
    root = Path(config["run_dir"]) / "evaluation"
    root.mkdir(parents=True, exist_ok=True)
    summary, per_window = [], []
    for name, method, rank in configurations(config):
        path = root / "configurations" / f"{name}.json"
        if not path.exists():
            continue
        # A report summarizes recorded evaluations, not a new weight-file audit.
        # Input identities are still bound into the evaluation protocol digest.
        artifacts = _evaluation_inputs(config, manifest, method, check=False)
        protocol = fingerprint({"manifest": manifest["sha256"], "name": name, "artifacts": artifacts})
        records = evaluation_records(path, protocol, 138)
        # Partial results are kept for resume, NEVER published as a final PPL.
        if len(records) != 138:
            continue
        check_control(config, manifest, name, records)
        nll = sum(row["nll_sum"] for row in records)
        tokens = sum(row["tokens"] for row in records)
        summary.append({"configuration": name, "method": method or "teacher", "rank": rank or "",
                        "windows": len(records), "prediction_tokens": tokens, "nll_sum": nll,
                        "ppl": math.exp(nll / tokens)})
        per_window.extend({"configuration": name, **row} for row in records)
    for filename, rows in (("ppl_summary_wikitext2.csv", summary), ("wikitext2_per_window.csv", per_window)):
        if rows:
            path = root / filename
            temporary = path.with_suffix(".csv.tmp")
            with temporary.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
    result = {"status": "PASS" if len(summary) == 18 else "INCOMPLETE", "configurations": len(summary),
              "expected_configurations": 18, "summary_path": str(root / "ppl_summary_wikitext2.csv")}
    atomic_json(root / "status.json", result)
    return result
