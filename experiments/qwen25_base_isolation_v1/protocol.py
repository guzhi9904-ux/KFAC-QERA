from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import torch
from safetensors import safe_open

from qera_diag_g_isolation.storage import file_record, fingerprint, read_json, verify, layers, log
from qera_original_a_isolation.common import import_official_qera
from qera_original_a_isolation.pipeline import _groups_from_config

VERSION = "qwen25_base_mxint3_v1"
QUANT = {"name": "mxint", "width": 3, "block_size": 32, "block_axis": -1}


def read_manifest(path):
    value = read_json(path)
    if value["sha256"] != fingerprint(value["payload"]):
        raise RuntimeError(f"Invalid source manifest: {path}")
    return value


def assert_separate(code_root, output, protected_code_roots, protected_outputs):
    code_root, output = Path(code_root).resolve(), Path(output).resolve()
    for path in protected_code_roots:
        path = Path(path).resolve()
        if code_root == path or code_root in path.parents or path in code_root.parents:
            raise RuntimeError("Deploy Qwen in a separate code checkout, outside the protected Llama checkout")
    for path in [*protected_outputs, code_root]:
        path = Path(path).resolve()
        if output == path or output in path.parents or path in output.parents:
            raise RuntimeError("Qwen output must be disjoint from source outputs/models/code")


def verify_protected(records):
    # Read-only; never acquire a writer lock in the Llama run or call its stages.
    for record in records:
        verify(record)


def verify_private_helpers(code_root, records):
    """Copied legacy helpers must be byte-identical to the running release."""
    for record in records:
        parts = Path(record["path"]).parts
        if "experiments" not in parts:
            continue
        relative = Path(*parts[parts.index("experiments"):])
        private = Path(code_root) / relative
        if file_record(private)["sha256"] != record["sha256"]:
            raise RuntimeError(f"Private legacy helper differs from frozen Llama release: {relative}")


def audit_reference(settings, enforce_isolation=True):
    original = Path(settings["llama_g_run"])
    dg = read_manifest(original / "manifest.json")
    mx = read_manifest(original / "mxint3_v1/manifest.json")
    fg = read_manifest(original / "mxint3_full_g_v1/manifest.json")
    if (mx["payload"]["quantization_transition"]["parent_manifest_sha256"] != dg["sha256"]
            or fg["payload"]["full_g_protocol"]["baseline_manifest_sha256"] != mx["sha256"]):
        raise RuntimeError("Llama reference chain is inconsistent")
    a_dir = Path(dg["payload"]["config"]["source_run_dir"])
    a = read_json(a_dir / "config_resolved.json")
    if a != dg["payload"]["source_config"]:
        raise RuntimeError("Llama A config changed since G256 was prepared")
    required_a = {"profiling_dtype": "float32", "solve_dtype": "float32", "eval_dtype": "bfloat16",
                  "num_calibration_samples": 256, "sequence_length": 2048, "ranks": [8, 16, 32, 64],
                  "calibration_acquisition": "streaming_prefix"}
    if any(a.get(k) != v for k, v in required_a.items()):
        raise RuntimeError("Unexpected Llama A protocol; review instead of guessing defaults")
    c = dg["payload"]["config"]
    required_g = {"teacher_dtype": "float32", "collect_batch_size": 1, "ce_gradient_chunk_tokens": 128,
                  "eval_batch_size": 8, "eval_ce_chunk_tokens": 256, "g_relative_floor": 1e-6,
                  "identity_product_tolerance": .001, "save_activations_on_cpu": True,
                  "num_calibration_windows": 256, "sequence_length": 2048, "ranks": [8, 16, 32, 64],
                  "cpu_threads": 14, "checkpoint_every_windows": 8}
    if any(c.get(k) != v for k, v in required_g.items()):
        raise RuntimeError("Unexpected Llama DG protocol")
    if mx["payload"]["source_config"]["quantization"] != QUANT:
        raise RuntimeError("Reference must be the completed MXINT3 experiment")
    if mx["payload"]["quantization_transition"].get("svd_full_matrices") is not True:
        raise RuntimeError("Reference must be the audited full-SVD MXINT3 release")
    protected = list(fg["payload"]["code"].values())
    protected += [file_record(original / p) for p in (
        "manifest.json", "mxint3_v1/manifest.json", "mxint3_full_g_v1/manifest.json")]
    protected += [file_record(a_dir / "config_resolved.json")]
    roots = set()
    for record in fg["payload"]["code"].values():
        path = Path(record["path"])
        if "experiments" in path.parts:
            roots.add(str(Path(*path.parts[:path.parts.index("experiments")])))
    if enforce_isolation:
        assert_separate(Path(__file__).resolve().parents[2], settings["run_dir"], roots,
                        [original, a_dir, settings["model_path"], a["model_path"], a["qera_source_dir"]])
    verify_protected(protected)
    # Validate the saved A implementation's actual accumulator dtype, not just YAML.
    for group in dg["payload"]["groups"]:
        with safe_open(group["raw_a"]["path"], framework="pt", device="cpu") as handle:
            if (handle.get_slice("rxx_sum").get_dtype() != "F64"
                    or handle.get_slice("diag_sum").get_dtype() != "F32"
                    or int(handle.get_tensor("sample_count").item()) != 524288):
                raise RuntimeError("Llama A statistic dtype/count differs from audited implementation")
    import_official_qera(a)
    return {"a_config": a, "dg_config": c, "protected": protected,
            "llama_a_run": str(a_dir), "llama_g_manifest": dg, "llama_mx3_manifest": mx,
            "a_calibration": file_record(a_dir / "data/calibration.json"),
            "a_wikitext2": file_record(a_dir / "data/wikitext2.json")}


def model_inventory(path):
    root = Path(path).resolve()
    config = read_json(root / "config.json")
    expected = {"model_type": "qwen2", "num_hidden_layers": 28, "hidden_size": 3584,
                "intermediate_size": 18944, "num_attention_heads": 28, "num_key_value_heads": 4,
                "vocab_size": 152064, "torch_dtype": "bfloat16", "tie_word_embeddings": False}
    if any(config.get(k) != v for k, v in expected.items()):
        raise RuntimeError("Require original BF16 Qwen2.5-7B architecture, not a pre-quantized checkpoint")
    if config.get("quantization_config") or "instruct" in root.name.lower():
        raise RuntimeError("Require Qwen2.5-7B Base, not Instruct/quantized weights")
    # Explicit provenance evidence; dimensions alone cannot distinguish Base/Instruct.
    readme = root / "README.md"
    if not readme.exists() or "contains the base 7b qwen2.5 model" not in readme.read_text(encoding="utf-8").lower():
        raise RuntimeError("Missing upstream Base-model README provenance; inspect download before proceeding")
    index = read_json(root / "model.safetensors.index.json")["weight_map"]
    filenames = set(index.values()) | {"config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json", "README.md"}
    filenames.update(p.name for p in root.glob("*.json"))
    filenames.update(p.name for p in root.glob("*.txt"))
    records = {}
    for name in sorted(filenames):
        target = root / name
        if Path(name).name != name or not target.is_file() or target.stat().st_size == 0:
            raise RuntimeError(f"Incomplete/unsafe model file: {name}")
        records[name] = file_record(target)
    shapes, dtypes = {}, {}
    for filename in sorted(set(index.values())):
        with safe_open(str(root / filename), framework="pt", device="cpu") as handle:
            assigned = {k for k, v in index.items() if v == filename}
            if not assigned <= set(handle.keys()):
                raise RuntimeError("Weight index references missing tensors")
            for name in assigned:
                tensor = handle.get_slice(name)
                shapes[name], dtypes[name] = tensor.get_shape(), tensor.get_dtype()
                if dtypes[name] != "BF16":
                    raise RuntimeError("Expected BF16 checkpoint tensor: " + name)
    from types import SimpleNamespace
    for name in ("model.embed_tokens.weight", "lm_head.weight"):
        if shapes.get(name) != [152064, 3584]:
            raise RuntimeError("Missing/invalid unquantized embedding or lm_head: " + name)
    norms = ["model.norm.weight"] + [f"model.layers.{i}.{kind}.weight"
             for i in range(28) for kind in ("input_layernorm", "post_attention_layernorm")]
    if any(shapes.get(name) != [3584] for name in norms):
        raise RuntimeError("Missing/invalid unquantized norm weights")
    groups = []
    for group in _groups_from_config(SimpleNamespace(**config)):
        item = {"target": group.target, "in_features": group.in_features, "layers": []}
        for name in group.all_layers:
            suffix = name.rsplit(".", 1)[-1]
            out = 512 if suffix in ("k_proj", "v_proj") else 18944 if suffix in ("gate_proj", "up_proj") else 3584
            if shapes.get(name + ".weight") != [out, group.in_features]:
                raise RuntimeError("Qwen projection shape mismatch: " + name)
            bias = suffix in ("q_proj", "k_proj", "v_proj")
            if (name + ".bias" in index) != bias or (bias and shapes[name + ".bias"] != [out]):
                raise RuntimeError("Qwen QKV bias coverage mismatch")
            item["layers"].append({"name": name, "shape": [out, group.in_features],
                                   "weight_file": index[name + ".weight"], "bias": bias})
        groups.append(item)
    if len(groups) != 112 or sum(len(g["layers"]) for g in groups) != 196 or "lm_head.bias" in index:
        raise RuntimeError("Unexpected Qwen module coverage/lm_head bias")
    return config, records, groups


def resolved_config(settings, reference):
    config = deepcopy(reference["dg_config"])
    config.update(run_dir=str(Path(settings["run_dir"]).resolve()), model_path=str(Path(settings["model_path"]).resolve()),
                  experiment_variant=VERSION, quantization=QUANT, a_groups_per_shard=8,
                  a_batch_size=reference["a_config"]["calibration_batch_size"],
                  a_checkpoint_windows=reference["a_config"]["checkpoint_every_windows"],
                  num_workers=reference["a_config"]["num_workers"],
                  sqrtm_max_imaginary=reference["a_config"]["sqrtm_max_imaginary"])
    return config
