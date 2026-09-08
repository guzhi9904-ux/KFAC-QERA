"""Module-sharded collection using the frozen diagonal-G teacher computation."""
from __future__ import annotations

import gc
import math
import os
import shutil
import time
from pathlib import Path

import torch

from qera_original_a_isolation.pipeline import load_model, _input_device, _reset_evaluation_memory_peaks
from qera_diag_g_isolation.math_ops import ce_hidden_gradient
from qera_diag_g_isolation.storage import (
    atomic_json, checked_tensors, heartbeat, load_manifest, log, read_json, verify_model,
)
from .checkpoint import Checkpoints, Paused
from .numerics import accumulate_gram, check_diagonal, relative_error


def checkpoints(config, manifest, index):
    shard = manifest["payload"]["full_g_protocol"]["shards"][index]
    return Checkpoints(Path(config["run_dir"]) / "statistics/checkpoints", manifest["sha256"], index,
                       {x["name"]: x["shape"][0] for x in shard})


def memory_status():
    result = {}
    if os.name == "posix":
        import resource
        result["cpu_process_peak_GiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    for path, key in (("/sys/fs/cgroup/memory.current", "cgroup_current_GiB"),
                      ("/sys/fs/cgroup/memory.peak", "cgroup_peak_GiB"),
                      ("/sys/fs/cgroup/memory.max", "cgroup_limit_GiB")):
        try:
            value = Path(path).read_text().strip()
            if value.isdigit():
                result[key] = int(value) / 2**30
        except OSError:
            pass
    if torch.cuda.is_available():
        result["gpu_peak_GiB"] = [round(torch.cuda.max_memory_allocated(i) / 2**30, 3) for i in range(2)]
    return result


def compare_parent(config, manifest, index, store, state):
    if state is None or state["windows_completed"] != 256:
        raise RuntimeError("Require all 256 complete windows in each shard")
    protocol = manifest["payload"]["full_g_protocol"]
    parent = checked_tensors(protocol["diagonal_g_state"]["file"])
    differences, gram_errors = {}, {}
    for name in store.dimensions:
        tensors = store.read_layer(state, name)
        gram_errors[name] = check_diagonal(tensors["gram"], tensors["diagonal"])
        differences[name] = relative_error(tensors["diagonal"], parent[name])
        if differences[name] > protocol["parent_diagonal_tolerance"]:
            raise RuntimeError(f"Recollected diagonal G differs from frozen G256: {name} {differences[name]}")
        del tensors
    reference_nll = protocol["diagonal_g_state"]["teacher_nll_sum"]
    nll_drift = abs(state["teacher_nll_sum"] - reference_nll) / max(abs(reference_nll), 1e-30)
    if nll_drift > protocol["teacher_nll_relative_tolerance"]:
        raise RuntimeError(f"Teacher NLL differs from frozen G256: {nll_drift}")
    report = {"status": "PASS", "manifest_sha256": manifest["sha256"], "shard": index,
              "checkpoint": state, "vs_parent_diagonal_relative": differences,
              "gram_vs_direct_diagonal_relative": gram_errors, "teacher_nll_relative_drift": nll_drift}
    atomic_json(Path(config["run_dir"]) / "statistics" / f"audit_shard_{index:03d}.json", report)
    log("audit-full-g", f"shard={index + 1} PASS diagonal max drift={max(differences.values()):.3g}")
    return report


def completed_audit(config, manifest, index):
    store = checkpoints(config, manifest, index)
    state = store.load(tensors=False)
    path = Path(config["run_dir"]) / "statistics" / f"audit_shard_{index:03d}.json"
    if state is None or state["windows_completed"] != 256:
        raise RuntimeError("Collect all Full-G shards before solve/evaluate")
    if path.exists():
        report = read_json(path)
        if (report.get("status") != "PASS" or report.get("manifest_sha256") != manifest["sha256"]
                or report.get("checkpoint") != state or report.get("shard") != index):
            raise RuntimeError("Full-G audit/checkpoint mismatch")
        protocol = manifest["payload"]["full_g_protocol"]
        for key, tolerance in (("vs_parent_diagonal_relative", protocol["parent_diagonal_tolerance"]),
                               ("gram_vs_direct_diagonal_relative", 1e-10)):
            differences = report.get(key, {})
            if set(differences) != set(store.dimensions) or any(
                    not math.isfinite(v) or not 0 <= v <= tolerance for v in differences.values()):
                raise RuntimeError("Full-G audit has no passing diagonal check")
        drift = report.get("teacher_nll_relative_drift", float("nan"))
        if not math.isfinite(drift) or not 0 <= drift <= protocol["teacher_nll_relative_tolerance"]:
            raise RuntimeError("Full-G audit has no passing teacher NLL check")
    else:
        report = compare_parent(config, manifest, index, store, state)
    return store, state, report


def collect(config, stop, max_new_windows=None):
    manifest = load_manifest(config)
    protocol = manifest["payload"]["full_g_protocol"]
    stores = [checkpoints(config, manifest, i) for i in range(len(protocol["shards"]))]
    pending = []
    for index, store in enumerate(stores):
        store.cleanup()
        state = store.load(tensors=False)
        if state is not None and state["windows_completed"] == 256:
            completed_audit(config, manifest, index)
            log("collect-full-g", f"shard={index + 1}/{len(stores)} already complete")
        else:
            pending.append(index)
    if not pending:
        return
    stop.check()
    verify_model(manifest)
    data = checked_tensors(manifest["payload"]["data"]["calibration"])
    if data["input_ids"].shape != (256, 2048) or not (data["attention_mask"] == 1).all():
        raise RuntimeError("Expected frozen 256 unpadded length-2048 calibration windows")
    loading = dict(manifest["payload"]["source_config"])
    loading["max_memory"] = config["collect_max_memory"]
    with heartbeat("collect-full-g", "loading unchanged FP32 teacher"):
        model = load_model(loading, "float32", "balanced")
    handles = []
    try:
        placements = {k: str(v) for k, v in model.hf_device_map.items()}
        if not set(placements.values()) <= {"0", "1", "cuda:0", "cuda:1"}:
            raise RuntimeError("Teacher weights must be GPU resident on the same two GPUs")
        if placements != protocol["teacher_device_map"]:
            raise RuntimeError("Teacher device map changed from original diagonal-G run")
        model.requires_grad_(False)
        if model.get_output_embeddings().bias is not None:
            raise RuntimeError("Expected bias-free Llama lm_head")
        modules = dict(model.named_modules())
        handles.append(model.get_input_embeddings().register_forward_hook(lambda _m, _a, out: out.requires_grad_(True)))
        device = _input_device(model)
        total_new = 0
        for index in pending:
            stop.check()
            store = stores[index]
            # Space for a new active-shard snapshot, while the old one remains valid.
            shard_bytes = sum((d*d + d)*8 for d in store.dimensions.values())
            if shutil.disk_usage(config["run_dir"]).free < shard_bytes + 5 * 2**30:
                raise RuntimeError("Need free disk for a whole new shard checkpoint plus 5 GiB reserve")
            with heartbeat("collect-full-g", f"loading shard={index + 1} checkpoint"):
                sums, start, nll = store.load()
            last_saved = start
            seen, active_mask = set(), None
            def make_hook(name):
                def forward(_module, _inputs, output):
                    def backward(gradient):
                        if name in seen:
                            raise RuntimeError(f"Duplicate gradient hook: {name}")
                        accumulate_gram(gradient, active_mask, sums[name]["gram"], sums[name]["diagonal"],
                                        protocol["gram_row_tile"])
                        seen.add(name)
                    output.register_hook(backward)
                return forward
            shard_handles = []
            try:
                for layer in protocol["shards"][index]:
                    module = modules[layer["name"]]
                    if tuple(module.weight.shape) != tuple(layer["shape"]) or module.weight.dtype != torch.float32:
                        raise RuntimeError("Teacher target shape/dtype mismatch")
                    shard_handles.append(module.register_forward_hook(make_hook(layer["name"])))
                _reset_evaluation_memory_peaks([0, 1])
                started = time.monotonic()
                for window in range(start, 256):
                    # No check here: an already begun shard/window is committed first.
                    seen.clear()
                    ids = data["input_ids"][window:window + 1].to(device)
                    mask = data["attention_mask"][window:window + 1].to(device).bool()
                    active_mask = torch.zeros_like(mask)
                    active_mask[:, :-1] = mask[:, :-1] & mask[:, 1:]
                    label = f"shard={index + 1}/{len(stores)} window={window + 1}/256"
                    log("collect-full-g", f"starting {label} targets={len(sums)} gram=FP64")
                    with heartbeat("collect-full-g", label + " forward/backward/Gram"):
                        with torch.autograd.graph.save_on_cpu(pin_memory=False, device_type="cuda"):
                            output = model.model(input_ids=ids, attention_mask=mask, use_cache=False)
                            hidden = output.last_hidden_state.to(model.get_output_embeddings().weight.device)
                            seed, window_nll, tokens = ce_hidden_gradient(
                                hidden, model.get_output_embeddings().weight, ids, mask, config["ce_gradient_chunk_tokens"])
                            hidden.backward(seed)
                    if seen != set(sums) or tokens != 2047 or any(p.grad is not None for p in model.parameters()):
                        raise RuntimeError("Incomplete/invalid window: RAM discarded, checkpoint NOT advanced")
                    for tensors in sums.values():
                        check_diagonal(tensors["gram"], tensors["diagonal"])
                    nll += window_nll
                    total_new += 1
                    completed = window + 1
                    # Release graph/activation references before large disk serialization.
                    del output, hidden, seed, ids, mask
                    limit = max_new_windows is not None and total_new >= max_new_windows
                    if completed % config["checkpoint_every_windows"] == 0 or completed == 256 or stop.requested() or limit:
                        with heartbeat("checkpoint-full-g", label + " writing transaction"):
                            store.save(sums, completed, nll)
                            last_saved = completed
                    elapsed = time.monotonic() - started
                    remaining = 256 - completed + 256 * (len(stores) - index - 1)
                    runtime = {"shard": index, "windows_completed": completed, "elapsed_seconds": elapsed,
                               "remaining_window_passes": remaining,
                               "rough_collect_eta_seconds": elapsed / (completed - start) * remaining,
                               "hf_device_map": placements, **memory_status()}
                    atomic_json(Path(config["run_dir"]) / "statistics/runtime.json", runtime)
                    log("collect-full-g", f"{label} elapsed={elapsed:.0f}s rough_collect_eta="
                        f"{runtime['rough_collect_eta_seconds']:.0f}s memory={memory_status()}")
                    if stop.requested() and last_saved != completed:
                        # A signal may arrive during runtime logging, after the
                        # regular checkpoint decision. Preserve this whole window too.
                        with heartbeat("checkpoint-full-g", label + " saving on stop"):
                            store.save(sums, completed, nll)
                    stop.check()
                    if limit:
                        raise Paused("Pilot/window budget complete and checkpoint committed; use run to continue")
            finally:
                for handle in shard_handles:
                    handle.remove()
                del sums
                gc.collect()
                torch.cuda.empty_cache()
            completed_audit(config, manifest, index)
    finally:
        for handle in handles:
            handle.remove()
        del model
        gc.collect()
        torch.cuda.empty_cache()
