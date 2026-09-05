from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .utils import get_module


_LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def torch_dtype(name: str) -> torch.dtype:
    mapping = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    try:
        return mapping[name]
    except KeyError as error:
        raise ValueError(f"Unsupported dtype: {name}") from error


def _model_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    model = config["model"]
    kwargs: dict[str, Any] = {
        "local_files_only": bool(model.get("local_files_only", True)),
        "trust_remote_code": bool(model.get("trust_remote_code", False)),
    }
    if model.get("revision"):
        kwargs["revision"] = model["revision"]
    return kwargs


def inspect_model_config(config: Mapping[str, Any]) -> dict[str, Any]:
    source = str(config["model"]["name_or_path"])
    observed = AutoConfig.from_pretrained(source, **_model_kwargs(config))
    return {
        "architecture": (getattr(observed, "architectures", None) or [None])[0],
        "model_type": getattr(observed, "model_type", None),
        "num_hidden_layers": getattr(observed, "num_hidden_layers", None),
        "hidden_size": getattr(observed, "hidden_size", None),
        "intermediate_size": getattr(observed, "intermediate_size", None),
        "vocab_size": getattr(observed, "vocab_size", None),
        "torch_dtype": str(getattr(observed, "torch_dtype", None)),
    }


def load_tokenizer(config: Mapping[str, Any]) -> Any:
    source = str(config["model"]["name_or_path"])
    tokenizer = AutoTokenizer.from_pretrained(source, use_fast=True, **_model_kwargs(config))
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(config: Mapping[str, Any], *, require_input_grads: bool = False) -> tuple[nn.Module, Any, torch.device]:
    if not torch.cuda.is_available() and str(config["model"]["device"]).startswith("cuda"):
        raise RuntimeError("CUDA was requested but is not available")
    source = str(config["model"]["name_or_path"])
    device = torch.device(str(config["model"]["device"]))
    hf_config = AutoConfig.from_pretrained(source, **_model_kwargs(config))
    hf_config.use_cache = False
    kwargs = _model_kwargs(config)
    kwargs.update(
        {
            "config": hf_config,
            "torch_dtype": torch_dtype(str(config["model"]["dtype"])),
            "low_cpu_mem_usage": True,
            "device_map": {"": str(device)},
        }
    )
    attention = config["model"].get("attention_implementation")
    if attention:
        kwargs["attn_implementation"] = attention
    model = AutoModelForCausalLM.from_pretrained(source, **kwargs)
    model.eval().requires_grad_(False)
    model.config.use_cache = False
    if require_input_grads:
        model.enable_input_require_grads()
    tokenizer = load_tokenizer(config)
    names = discover_target_modules(model, config)
    for name in names:
        if not isinstance(get_module(model, name), nn.Linear):
            raise TypeError(f"Target is not torch.nn.Linear: {name}")
    return model, tokenizer, device


def layer_index(name: str) -> int:
    match = _LAYER_PATTERN.search(name)
    if not match:
        raise ValueError(f"Cannot determine decoder layer index: {name}")
    return int(match.group(1))


def projection_name(name: str) -> str:
    return name.rsplit(".", 1)[-1]


def discover_target_modules(model: nn.Module, config: Mapping[str, Any]) -> list[str]:
    suffixes = tuple(str(value) for value in config["model"]["target_suffixes"])
    layer_filter = config["model"].get("layers", "all")
    allowed = None if layer_filter == "all" else {int(value) for value in layer_filter}
    order = {suffix: index for index, suffix in enumerate(suffixes)}
    matches: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        suffix = next((candidate for candidate in suffixes if name.endswith(candidate)), None)
        if suffix is None:
            continue
        layer = layer_index(name)
        if allowed is None or layer in allowed:
            matches.append(name)
    matches.sort(key=lambda name: (layer_index(name), order[next(s for s in suffixes if name.endswith(s))]))
    if not matches:
        raise RuntimeError("No target Linear modules matched model.target_suffixes/model.layers")
    selected_layers = sorted({layer_index(name) for name in matches})
    for layer in selected_layers:
        present = {suffix for suffix in suffixes if any(layer_index(name) == layer and name.endswith(suffix) for name in matches)}
        missing = set(suffixes) - present
        if missing:
            raise RuntimeError(f"Layer {layer} is missing configured projections: {sorted(missing)}")
    return matches


def module_manifest(model: nn.Module, names: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in names:
        module = get_module(model, name)
        if not isinstance(module, nn.Linear):
            raise TypeError(name)
        dense_bytes = 8 * (module.in_features**2 + module.out_features**2 + module.in_features + module.out_features)
        rows.append(
            {
                "module": name,
                "layer": layer_index(name),
                "projection": projection_name(name),
                "in_features": module.in_features,
                "out_features": module.out_features,
                "weight_parameters": module.weight.numel(),
                "weight_dtype": str(module.weight.dtype).removeprefix("torch."),
                "estimated_dense_accumulator_bytes": dense_bytes,
                # The raw artifact contains the FP64 A/G sums and diagonals,
                # plus one FP32 quantization-error matrix.
                "estimated_raw_statistics_bytes": dense_bytes + 4 * module.weight.numel(),
            }
        )
    return rows


def base_model(model: nn.Module, config: Mapping[str, Any]) -> nn.Module:
    attribute = str(config["model"].get("base_model_attribute", "model"))
    current: nn.Module = model
    for part in attribute.split("."):
        current = getattr(current, part)
    return current


def output_embedding(model: nn.Module) -> nn.Module:
    module = model.get_output_embeddings()
    if module is None or not hasattr(module, "weight"):
        raise RuntimeError("Model does not expose a weighted output embedding")
    return module
