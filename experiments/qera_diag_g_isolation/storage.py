from __future__ import annotations

import contextlib
import csv
import hashlib
import json
import math
import os
import threading
import time
import uuid
from pathlib import Path

import torch
import yaml
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from qera_original_a_isolation.common import import_official_qera, safe_name, sha256_file


def log(stage, message):
    print(f"[{stage}] {time.strftime('%Y-%m-%dT%H:%M:%S%z')} {message}", flush=True)


@contextlib.contextmanager
def heartbeat(stage, message):
    start = time.monotonic()
    stop = threading.Event()
    def tick():
        while not stop.wait(30):
            log(stage, f"{message} working elapsed={time.monotonic() - start:.0f}s")
    worker = threading.Thread(target=tick, daemon=True)
    worker.start()
    try:
        yield
    finally:
        stop.set()
        worker.join()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    sync_parent(path)


def sync_parent(path):
    if os.name == "posix":
        descriptor = os.open(str(Path(path).parent), os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def atomic_tensors(path, tensors):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    save_file({k: v.detach().cpu().contiguous() for k, v in tensors.items()}, str(temporary))
    with temporary.open("r+b") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    sync_parent(path)


def disjoint_roots(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("New output and source must be disjoint directories, not ancestors of each other")
    if destination == Path(destination.anchor) or len(destination.parts) < 3:
        raise ValueError("Refusing a broad output directory")


def config_from_file(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for key in ("source_run_dir", "run_dir"):
        config[key] = str(Path(os.path.expandvars(config[key])).expanduser().resolve())
    disjoint_roots(config["source_run_dir"], config["run_dir"])
    required = {"num_calibration_windows": 256, "sequence_length": 2048,
                "ranks": [8, 16, 32, 64], "teacher_dtype": "float32",
                "collect_batch_size": 1, "retain_raw_g": True, "save_activations_on_cpu": True}
    for key, value in required.items():
        if config.get(key) != value:
            raise ValueError(f"Round-one protocol fixes {key}={value}")
    for key in ("checkpoint_every_windows", "ce_gradient_chunk_tokens", "eval_batch_size",
                "eval_ce_chunk_tokens", "cpu_threads"):
        if int(config[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    if not 0 < float(config["g_relative_floor"]) < 1:
        raise ValueError("Invalid G floor")
    if not 0 < float(config["identity_product_tolerance"]) < 1 or not 0 < float(config["control_ppl_tolerance"]) < 1:
        raise ValueError("Invalid identity tolerance")
    return config


def file_record(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    with heartbeat("audit", f"hash {path.name}"):
        digest = sha256_file(path)
    return {"path": str(path), "sha256": digest, "bytes": path.stat().st_size}


def verify(record):
    path = Path(record["path"])
    if not path.is_file() or path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
        raise RuntimeError(f"Immutable input/checkpoint changed: {path}")
    return path


def checked_tensors(record):
    return load_file(str(verify(record)))


def load_manifest(config):
    saved = read_json(Path(config["run_dir"]) / "manifest.json")
    if saved["sha256"] != fingerprint(saved["payload"]) or saved["payload"]["config"] != config:
        raise RuntimeError("Manifest/config mismatch: keep protocol fixed or use a new output directory")
    # Avoid rehashing 100+ GiB every subcommand; stage-specific inputs are checked
    # again when consumed. Source code is always checked, including on resume.
    for record in saved["payload"]["code"].values():
        verify(record)
    return saved


@contextlib.contextmanager
def run_lock(config):
    """OS lock is released on process death; no stale PID deletion is needed."""
    root = Path(config["run_dir"])
    root.mkdir(parents=True, exist_ok=True)
    if os.name != "posix":
        raise RuntimeError("Server commands require Linux; mathematical tests also run on CPU/Windows")
    import fcntl
    with (root / ".run.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another process owns this experiment directory") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def prepare(config):
    """Read-only audit of source artifacts. Only the NEW output is written."""
    root, source = Path(config["run_dir"]), Path(config["source_run_dir"])
    source_config = read_json(source / "config_resolved.json")
    for key, expected in {"sequence_length": 2048, "num_calibration_samples": 256,
                          "profiling_dtype": "float32", "solve_dtype": "float32",
                          "eval_dtype": "bfloat16", "ranks": [8, 16, 32, 64]}.items():
        if source_config.get(key) != expected:
            raise ValueError(f"Original experiment mismatch: {key}")
    qera = import_official_qera(source_config)
    if source_config.get("quantization") != {"name": "mxint", "width": 4, "block_size": 32, "block_axis": -1}:
        raise ValueError("Original quantization must be MXINT4 with block_size=32")
    plan = read_json(source / "plan.json")
    model_dir = Path(source_config["model_path"])
    disjoint_roots(model_dir, root)
    index = read_json(model_dir / "model.safetensors.index.json")["weight_map"]
    payload = {"config": config, "source_config": source_config, "qera_commit": qera["commit"],
               "source_plan": file_record(source / "plan.json"), "groups": [], "code": {},
               "model_files": {}, "data": {}, "protocol_version": 1,
               "g_definition": "mean over valid prediction positions of squared d(sequence CE SUM)/d(projection output)",
               "a_token_count": 524288, "g_prediction_positions": 524032}
    code_paths = list(Path(__file__).parent.glob("*.py"))
    old = Path(__file__).parent.parent / "qera_original_a_isolation"
    code_paths += [old / "pipeline.py", old / "common.py"]
    official = Path(source_config["qera_source_dir"])
    code_paths += [official / p for p in ("src/qera/approximate.py", "src/qera/quantize/quantizers/mxint.py")]
    for path in code_paths:
        payload["code"][str(path.resolve())] = file_record(path)
    # Pin actual model weights, configuration and tokenizer files, never HF tokens.
    for path in sorted(model_dir.glob("*.json")) + sorted(model_dir.glob("*.safetensors")):
        log("audit", f"model {path.name}")
        payload["model_files"][path.name] = file_record(path)
    model_config = read_json(model_dir / "config.json")
    required_model = {"model_type": "llama", "num_hidden_layers": 32, "hidden_size": 4096,
                      "intermediate_size": 14336, "num_attention_heads": 32, "num_key_value_heads": 8}
    if any(model_config.get(key) != value for key, value in required_model.items()):
        raise ValueError("First round is fixed to Llama-3.1-8B (32 layers)")
    # The earlier harness baseline recorded the same model's file identity. Its
    # score need not agree with the paper to be useful as a historical checksum.
    historical = source / "evaluation_harness_word_ppl/protocol.json"
    if historical.exists():
        previous_hashes = read_json(historical)["model"]["sha256"]
        for filename, record in payload["model_files"].items():
            if filename in previous_hashes and previous_hashes[filename] != record["sha256"]:
                raise RuntimeError(f"Model file changed since original BF16 audit: {filename}")
        payload["historical_model_protocol"] = file_record(historical)
    summary_path = source / "evaluation/ppl_summary_wikitext2.csv"
    payload["source_ppl_summary"] = file_record(summary_path)
    controls = {}
    with summary_path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            name = row["configuration"]
            if name.startswith("QERA_DIAG_R"):
                name = name.replace("QERA_DIAG_R", "DIAG_GI_R")
            elif name.startswith("QERA_FULL_R"):
                name = name.replace("QERA_FULL_R", "FULL_GI_R")
            if name in controls or int(row["windows"]) != 138 or int(row["prediction_tokens"]) != 282486:
                raise RuntimeError("Original control evaluation is incomplete/duplicated")
            ppl = float(row["ppl"])
            if not math.isfinite(ppl) or ppl <= 0:
                raise RuntimeError("Invalid original control PPL")
            controls[name] = ppl
    expected_controls = {"BF16", "W4_MXINT"} | {f"{a}_GI_R{r}" for a in ("DIAG", "FULL") for r in (8, 16, 32, 64)}
    if set(controls) != expected_controls:
        raise RuntimeError("Require all ten completed original token-PPL configurations")
    payload["control_reference_ppl"] = controls
    for role, count in (("calibration", 256), ("wikitext2", 138)):
        record = file_record(source / "data" / f"{role}.safetensors")
        metadata = read_json(source / "data" / f"{role}.json")
        if metadata.get("status") != "PASS" or metadata.get("sha256") != record["sha256"]:
            raise RuntimeError(f"Frozen {role} metadata does not match")
        tensors = checked_tensors(record)
        if tensors["input_ids"].shape != (count, 2048) or not (tensors["attention_mask"] == 1).all():
            raise ValueError("Require original complete, unpadded frozen windows")
        payload["data"][role] = record
    for shard_index, shard in enumerate(plan["shards"]):
        for stage in ("collect", "roots", "solve"):
            if read_json(source / "state" / f"{stage}_shard_{shard_index:03d}.json").get("status") != "PASS":
                raise RuntimeError(f"Source {stage} shard {shard_index} incomplete")
        for group in shard:
            log("audit", f"A group {len(payload['groups']) + 1}/128 {group['target']}")
            raw = file_record(source / "statistics" / "raw" / f"{safe_name(group['target'])}.safetensors")
            roots = file_record(source / "statistics" / "roots" / f"{safe_name(group['target'])}.safetensors")
            metadata = read_json(Path(roots["path"]).with_suffix(".json"))
            if metadata.get("raw_sha256") != raw["sha256"] or metadata.get("root_sha256") != roots["sha256"]:
                raise RuntimeError("Source A/root hash mismatch")
            with safe_open(raw["path"], framework="pt", device="cpu") as handle:
                if handle.get_tensor("sample_count").item() != 524288 or handle.get_tensor("windows_completed").item() != 256:
                    raise RuntimeError("Source A did not use the expected 256 full windows")
            item = {"target": group["target"], "raw_a": raw, "roots": roots, "layers": []}
            for name in (group["target"], *group["shares"]):
                key = name + ".weight"
                filename = index[key]
                if filename not in payload["model_files"]:
                    raise RuntimeError("Weight index references a file outside the pinned model")
                with safe_open(str(model_dir / filename), framework="pt", device="cpu") as handle:
                    shape = handle.get_slice(key).get_shape()
                if shape[1] != group["in_features"]:
                    raise RuntimeError("Source A/weight dimension mismatch")
                layer = {"name": name, "shape": shape, "weight_file": filename, "gi": {}}
                for method in ("diag", "full"):
                    record = file_record(source / "corrections" / method / f"{safe_name(name)}.safetensors")
                    state = read_json(Path(record["path"]).with_suffix(".json"))
                    if (state.get("status") != "PASS" or state.get("sha256") != record["sha256"]
                            or state.get("rank") != 64 or state.get("layer") != name or state.get("method") != method):
                        raise RuntimeError(f"Incomplete/changed source correction: {name} {method}")
                    with safe_open(record["path"], framework="pt", device="cpu") as handle:
                        if handle.get_slice("A").get_shape() != [shape[1], 64] or handle.get_slice("B").get_shape() != [64, shape[0]]:
                            raise RuntimeError(f"Invalid original correction dimensions: {name}")
                    layer["gi"][method] = record
                item["layers"].append(layer)
            payload["groups"].append(item)
    names = [x["name"] for group in payload["groups"] for x in group["layers"]]
    if len(names) != 224 or len(set(names)) != 224 or len(payload["groups"]) != 128:
        raise RuntimeError("Expected 128 A groups and 224 distinct G targets")
    saved = {"sha256": fingerprint(payload), "payload": payload}
    path = root / "manifest.json"
    if path.exists() and read_json(path) != saved:
        raise RuntimeError("Existing manifest differs; do not overwrite an existing experiment")
    atomic_json(path, saved)
    log("audit", f"PASS frozen source manifest={saved['sha256']} targets=224")
    return saved


def layers(manifest):
    return [layer for group in manifest["payload"]["groups"] for layer in group["layers"]]


def verify_model(manifest):
    for name, record in manifest["payload"]["model_files"].items():
        log("audit", f"verify model {name}")
        with heartbeat("audit", name):
            verify(record)


def weight_tensor(manifest, layer):
    path = manifest["payload"]["model_files"][layer["weight_file"]]["path"]
    with safe_open(path, framework="pt", device="cpu") as handle:
        return handle.get_tensor(layer["name"] + ".weight")


def gpu_check():
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES=0,1; this run requires exactly two visible GPUs")
    devices = [{"index": i, "name": torch.cuda.get_device_name(i),
                "total_GiB": torch.cuda.get_device_properties(i).total_memory / 2**30}
               for i in range(2)]
    if any(d["total_GiB"] < 20 for d in devices):
        raise RuntimeError("Requires two GPUs with at least 20 GiB each")
    return devices
