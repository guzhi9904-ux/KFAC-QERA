#!/usr/bin/env python3
"""Evaluate existing A-isolation artifacts using QERA's pinned lm-eval task.

No token-window CE, detokenizer, rolling likelihood, or PPL aggregator is
implemented here. Those operations are performed by the unmodified harness.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import sys
import time

HARNESS_COMMIT = "3823cfec41c016378acbcc8616dd1ac92c15edd4"
HARNESS_HASHES = {
    "lm_eval/tasks/wikitext/wikitext.yaml": "6339b85890637da1b064a484f13d9ecbecb542211e3ecdcf6e761569b03143a7",
    "lm_eval/tasks/wikitext/preprocess_wikitext.py": "a44020374b172159bf5be80a64d2c4194c9d5c77284f696ac666361338c8da7d",
    "lm_eval/models/huggingface.py": "c98387a0c0386c1f1f4536f8b2e64f5d8c3059ab9859d83991efc9a2c4d4eca8",
    "lm_eval/evaluator.py": "21838d5716795fff091e9482a070b3c58760e6a3b1babe46bccea46a39899b69",
    "lm_eval/api/metrics.py": "aa5a2939b8e8e448b87361a43c43c869b6b4d3ae8e4eca0c1417ed68891aa054",
    "lm_eval/utils.py": "31524c4be78e1b0445b79d7a7ef952dd3207e10180a910420e1e4670c78816b9",
}
PROTECTED_PACKAGES = (
    "torch", "transformers", "accelerate", "datasets", "numpy", "scipy",
    "pandas", "safetensors", "peft", "huggingface-hub", "tokenizers", "pyarrow",
)
REFERENCE = {"BF16": 7.55, "W4_MXINT": 8.78, "QERA_DIAG_R32": 8.45, "QERA_FULL_R32": 8.33}


def log(message):
    print(f"[harness] {time.strftime('%Y-%m-%dT%H:%M:%S%z')} {message}", flush=True)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(temporary, path)


def core_versions():
    return {name: importlib.metadata.version(name) for name in PROTECTED_PACKAGES}


def verify_harness(source, check_import=False):
    source = Path(source).resolve()
    for relative, expected in HARNESS_HASHES.items():
        path = source / relative
        if not path.is_file():
            raise RuntimeError(f"Missing pinned harness file: {path}; run setup_harness.sh first")
        actual = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Pinned harness hash mismatch: {relative}")
    if check_import:
        sys.path.insert(0, str(source))
        import lm_eval
        if Path(lm_eval.__file__).resolve() != source / "lm_eval" / "__init__.py":
            raise RuntimeError(f"Wrong lm_eval imported: {lm_eval.__file__}")
    return source


def model_identity(path):
    path = Path(path)
    weights = sorted(set(path.glob("model*.safetensors")) | set(path.glob("pytorch_model*.bin")))
    if not weights:
        raise RuntimeError(f"No original model weight files found: {path}")
    metadata = sorted(set(path.glob("*.json")) | set(path.glob("*.model")) | set(path.glob("*.tiktoken")))
    hashes = {}
    for item in weights + metadata:
        log(f"hashing model/tokenizer file={item.name} size_MiB={item.stat().st_size / 2**20:.1f}")
        hashes[item.name] = file_hash(item)
    return {"path": str(path.resolve()), "sha256": hashes}


def dataset_identity(task):
    digest = hashlib.sha256()
    words = 0
    docs = task.eval_docs
    for doc in docs:
        payload = doc["page"].encode("utf-8")
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
        # Audit only. The actual metric is computed by the pinned task.
        words += len(re.split(r"\s+", doc["page"]))
    return {
        "dataset": "EleutherAI/wikitext_document_level", "subset": "wikitext-2-raw-v1",
        "split": "test", "documents": len(docs), "raw_words_audit": words,
        "raw_pages_sha256": digest.hexdigest(), "datasets_fingerprint": getattr(docs, "_fingerprint", None),
    }


def extract_word_ppl(result):
    metrics = result["results"]["wikitext"]
    values = [value for key, value in metrics.items() if key == "word_perplexity" or key.startswith("word_perplexity,")]
    if len(values) != 1 or not math.isfinite(float(values[0])) or float(values[0]) <= 0:
        raise RuntimeError(f"Missing or invalid harness word_perplexity: {metrics}")
    count = result["n-samples"]["wikitext"]
    if int(count["original"]) <= 0 or count["original"] != count["effective"]:
        raise RuntimeError(f"Incomplete WikiText evaluation: {count}")
    return float(values[0])


def baseline_gate(ppl, tolerance):
    return {
        "reference_word_ppl": REFERENCE["BF16"], "observed_word_ppl": ppl,
        "absolute_difference": abs(ppl - REFERENCE["BF16"]),
        "tolerance": tolerance, "close_to_reference": abs(ppl - REFERENCE["BF16"]) <= tolerance,
        "note": "Engineering comparison threshold, not a tolerance specified by the QERA paper",
    }


def existing_result(root, name, protocol_hash):
    directory = root / name
    state = directory / "complete.json"
    if not state.exists():
        return None
    record = json.loads(state.read_text(encoding="utf-8"))
    if record.get("protocol_sha256") != protocol_hash or record.get("status") != "PASS":
        raise RuntimeError(f"Completed result has a different protocol: {state}")
    result_path = directory / "results.json"
    if record.get("results_sha256") != file_hash(result_path):
        raise RuntimeError(f"Result checksum mismatch: {result_path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if extract_word_ppl(result) != record["word_ppl"]:
        raise RuntimeError(f"Completed metric does not match saved harness output: {state}")
    return record


def correction_identity(config, method, rank=32):
    from safetensors import safe_open
    from qera_original_a_isolation.common import safe_name

    run_root = Path(config["run_dir"])
    plan = json.loads((run_root / "plan.json").read_text())
    hashes = {}
    for shard in plan["shards"]:
        for group in shard:
            for name in (group["target"], *group["shares"]):
                path = run_root / "corrections" / method / f"{safe_name(name)}.safetensors"
                record = json.loads(path.with_suffix(".json").read_text())
                digest = file_hash(path)
                if (record.get("status") != "PASS" or record.get("sha256") != digest
                        or record.get("layer") != name or record.get("method") != method
                        or record.get("rank", 0) < rank):
                    raise RuntimeError(f"Invalid saved correction: {path}")
                with safe_open(path, framework="pt", device="cpu") as tensors:
                    a = tensors.get_slice("A").get_shape()
                    b = tensors.get_slice("B").get_shape()
                if len(a) != 2 or len(b) != 2 or a[1] != b[0] or a[1] < rank or a[0] != group["in_features"]:
                    raise RuntimeError(f"Invalid correction shapes for rank {rank}: {path}, A={a}, B={b}")
                hashes[name] = digest
    if len(hashes) != plan["target_module_count"]:
        raise RuntimeError("Saved correction module count does not match original plan")
    log(f"verified saved corrections method={method} modules={len(hashes)} rank={rank}; no roots recomputed")
    return hashes


def evaluate_one(config, task, name, root, protocol_hash, args):
    corrections = None
    if name in ("QERA_DIAG_R32", "QERA_FULL_R32"):
        corrections = correction_identity(config, "diag" if name == "QERA_DIAG_R32" else "full")
    previous = existing_result(root, name, protocol_hash)
    if previous:
        if previous.get("correction_sha256") != corrections:
            raise RuntimeError(f"Saved correction files changed since evaluation: {name}")
        log(f"configuration={name} already complete word_ppl={previous['word_ppl']:.6f}")
        return previous
    import torch
    import transformers
    from accelerate import dispatch_model
    from lm_eval.models.huggingface import HFLM
    from lm_eval.evaluator import simple_evaluate
    from qera.utils import create_device_map
    from qera_original_a_isolation.common import import_official_qera
    from qera_original_a_isolation.pipeline import _attach_corrections, _quantize_model

    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    log(f"configuration={name} loading BF16 model; context=2048 attention=eager harness_batch_size={args.batch_size}")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        config["model_path"], torch_dtype=torch.bfloat16, local_files_only=True,
        _attn_implementation="eager", max_position_embeddings=2048,
    )
    model.eval()
    model.config.use_cache = False
    # Use the same dispatch helper as official ptq_bf16_baseline.py.
    device_map = create_device_map(model, "auto-balanced")
    model = dispatch_model(model, device_map=device_map)
    log(f"configuration={name} device_map={device_map}")
    handles = []
    lm = None
    try:
        if name != "BF16":
            qera = import_official_qera(config)
            names = _quantize_model(model, qera["mxint_quantizer"])
            plan = json.loads((Path(config["run_dir"]) / "plan.json").read_text())
            if len(names) != plan["target_module_count"]:
                raise RuntimeError(f"Quantized module count mismatch: {len(names)}")
            log(f"configuration={name} official_MXINT4_modules={len(names)}")
            if name in ("QERA_DIAG_R32", "QERA_FULL_R32"):
                method = "diag" if name == "QERA_DIAG_R32" else "full"
                handles = _attach_corrections(model, config, method, 32)
                log(f"configuration={name} reused rank32 prefixes from saved rank64 corrections")
        # HFLM, task preprocessing, rolling likelihood and aggregation are
        # unmodified pinned harness code. No old token-window evaluator here.
        lm = HFLM(model, batch_size=args.batch_size, max_length=2048)
        log(f"configuration={name} actual_harness_max_length={lm.max_length} actual_batch_size={lm.batch_size}")
        with torch.no_grad():
            result = simple_evaluate(
                model=lm, tasks=[task], num_fewshot=0, limit=None,
                bootstrap_iters=0, log_samples=True, apply_chat_template=False,
                random_seed=0, numpy_random_seed=1234, torch_random_seed=1234,
                fewshot_random_seed=1234,
            )
        ppl = extract_word_ppl(result)
        result_path = directory / "results.json"
        write_json(result_path, result)
        record = {
            "status": "PASS", "configuration": name, "metric": "word_perplexity",
            "word_ppl": ppl, "paper_word_ppl": REFERENCE[name],
            "difference_from_paper": ppl - REFERENCE[name],
            "documents": result["n-samples"]["wikitext"]["effective"],
            "elapsed_seconds": time.monotonic() - started,
            "protocol_sha256": protocol_hash, "results_sha256": file_hash(result_path),
            "device_map": device_map,
            "correction_sha256": corrections,
        }
        write_json(directory / "complete.json", record)
        log(f"configuration={name} PASS word_ppl={ppl:.6f} paper={REFERENCE[name]:.2f} elapsed={record['elapsed_seconds']:.0f}s")
        return record
    finally:
        for handle in handles:
            handle.remove()
        handles.clear()
        del lm, model
        gc.collect()
        torch.cuda.empty_cache()


def write_summary(root, records, protocol_hash):
    by_name = {record["configuration"]: record for record in records}
    for name in REFERENCE:
        if name not in by_name:
            previous = existing_result(root, name, protocol_hash)
            if previous:
                by_name[name] = previous
    keys = ["configuration", "metric", "word_ppl", "paper_word_ppl", "difference_from_paper", "documents", "elapsed_seconds"]
    temporary = root / "word_ppl_summary.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(by_name[name] for name in REFERENCE if name in by_name)
    os.replace(temporary, root / "word_ppl_summary.csv")


def execute_stages(config, task, root, protocol_hash, args):
    records = []
    if args.stage == "compare":
        baseline = existing_result(root, "BF16", protocol_hash)
        if baseline is None:
            raise RuntimeError("Run --stage bf16 first, or use --stage all")
    else:
        baseline = evaluate_one(config, task, "BF16", root, protocol_hash, args)
    records.append(baseline)
    write_summary(root, records, protocol_hash)
    gate = baseline_gate(baseline["word_ppl"], args.bf16_tolerance)
    write_json(root / "bf16_reference_check.json", gate)
    log(f"BF16 reference check: {json.dumps(gate, ensure_ascii=False)}")
    if args.stage != "bf16" and gate["close_to_reference"]:
        for name in ("W4_MXINT", "QERA_DIAG_R32", "QERA_FULL_R32"):
            records.append(evaluate_one(config, task, name, root, protocol_hash, args))
            write_summary(root, records, protocol_hash)
    outcome = {
        "status": "PASS" if gate["close_to_reference"] else "NEEDS_REVIEW",
        "stage": args.stage, "baseline_gate": gate, "configurations": len(records),
        "summary_path": str(root / "word_ppl_summary.csv"),
    }
    write_json(root / "status.json", outcome)
    print(json.dumps(outcome, indent=2, ensure_ascii=False))
    return 0 if gate["close_to_reference"] else 2


def run(args):
    import torch
    from qera_original_a_isolation.common import load_config, validate_config, import_official_qera
    from qera_original_a_isolation.pipeline import create_plan

    if args.batch_size <= 0 or not math.isfinite(args.bf16_tolerance) or args.bf16_tolerance < 0:
        raise ValueError("batch_size must be positive and bf16_tolerance finite and nonnegative")
    if not torch.cuda.is_available():
        raise RuntimeError("This server evaluation requires CUDA")
    for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if args.allow_download:
            os.environ.pop(key, None)
        else:
            os.environ[key] = "1"
    config = load_config(args.config)
    validate_config(config)
    if not (Path(config["run_dir"]) / "plan.json").is_file():
        raise RuntimeError("Original plan.json is missing; point CONFIG to the completed A-isolation run")
    plan = create_plan(config)
    source = args.harness_source or Path(config["qera_source_dir"]).parent / "QERA-harness-3823cfe"
    verify_harness(source, check_import=True)
    qera = import_official_qera(config)
    from lm_eval.tasks import TaskManager, get_task_dict

    log("loading pinned harness WikiText document-level test set (not the 138 frozen token windows)")
    task = get_task_dict(["wikitext"], TaskManager(verbosity="INFO"))["wikitext"]
    data = dataset_identity(task)
    log(f"dataset documents={data['documents']} raw_words={data['raw_words_audit']} pages_sha256={data['raw_pages_sha256']}")
    identity = model_identity(config["model_path"])
    protocol = {
        "harness_commit": HARNESS_COMMIT, "harness_hashes": HARNESS_HASHES,
        "qera_commit": qera["commit"], "original_plan_fingerprint": plan["fingerprint"],
        "model": identity, "dataset": data, "packages": core_versions(),
        "harness_distribution": importlib.metadata.version("lm_eval"),
        "runner_sha256": file_hash(__file__), "context_length": 2048,
        "artifact_loader_sha256": file_hash(Path(__file__).with_name("pipeline.py")),
        "common_sha256": file_hash(Path(__file__).with_name("common.py")),
        "attention": "eager", "dtype": "bfloat16", "batch_size": args.batch_size,
        "num_fewshot": 0, "chat_template": False, "bootstrap_iters": 0,
        "seeds": [0, 1234, 1234, 1234],
        "correction_origin": "saved A-isolation rank64 factors, exact rank32 prefixes",
    }
    root = args.output_dir or Path(config["run_dir"]) / "evaluation_harness_word_ppl"
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    protocol_hash = fingerprint(protocol)
    protocol_path = root / "protocol.json"
    if protocol_path.exists() and fingerprint(json.loads(protocol_path.read_text())) != protocol_hash:
        raise RuntimeError(f"Existing harness output uses a different protocol; choose a new --output-dir: {root}")
    write_json(protocol_path, protocol)
    return execute_stages(config, task, root, protocol_hash, args)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    protect = commands.add_parser("protect-env")
    protect.add_argument("--constraints", required=True, type=Path)
    protect.add_argument("--snapshot", required=True, type=Path)
    verify = commands.add_parser("verify")
    verify.add_argument("--source", required=True, type=Path)
    verify.add_argument("--snapshot", type=Path)
    verify.add_argument("--check-import", action="store_true")
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--config", required=True, type=Path)
    evaluation.add_argument("--stage", choices=["bf16", "compare", "all"], default="bf16")
    evaluation.add_argument("--batch-size", type=int, default=1)
    evaluation.add_argument("--bf16-tolerance", type=float, default=0.05)
    evaluation.add_argument("--harness-source", type=Path)
    evaluation.add_argument("--output-dir", type=Path)
    evaluation.add_argument("--allow-download", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "protect-env":
        versions = core_versions()
        args.constraints.write_text("".join(f"{name}=={version}\n" for name, version in versions.items()), encoding="utf-8")
        write_json(args.snapshot, versions)
        print(json.dumps(versions, indent=2))
        return 0
    if args.command == "verify":
        verify_harness(args.source, args.check_import)
        if args.snapshot and json.loads(args.snapshot.read_text()) != core_versions():
            raise RuntimeError("Core package versions changed during installation")
        log(f"PASS pinned harness={HARNESS_COMMIT}; core environment verified")
        return 0
    return run(args)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    raise SystemExit(main())
