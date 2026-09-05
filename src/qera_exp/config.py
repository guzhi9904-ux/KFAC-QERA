from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigError(ValueError):
    pass


def _merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if key == "extends":
            continue
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    return value


def _read(path: Path, seen: set[Path]) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved in seen:
        raise ConfigError(f"Circular config inheritance: {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    seen.add(resolved)
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError(f"Config root must be a mapping: {resolved}")
    parent = raw.get("extends")
    if parent:
        parent_path = Path(str(parent))
        if not parent_path.is_absolute():
            parent_path = resolved.parent / parent_path
        result = _merge(_read(parent_path, seen), raw)
    else:
        result = dict(raw)
    seen.remove(resolved)
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    config = _expand(_read(source, set()))
    config["_config_path"] = str(source)
    validate_config(config)
    return config


def _required(config: Mapping[str, Any], dotted: str) -> Any:
    value: Any = config
    for key in dotted.split("."):
        if not isinstance(value, Mapping) or key not in value:
            raise ConfigError(f"Missing config key: {dotted}")
        value = value[key]
    return value


def validate_config(config: Mapping[str, Any]) -> None:
    if int(_required(config, "schema_version")) != 1:
        raise ConfigError("Only schema_version=1 is supported")
    for key in (
        "experiment.output_dir",
        "model.name_or_path",
        "model.device",
        "model.dtype",
        "model.target_suffixes",
        "quantization.block_size",
        "statistics.methods",
        "statistics.ranks",
        "statistics.maximum_rank",
    ):
        value = _required(config, key)
        if isinstance(value, str) and ("${" in value or "$" in value):
            raise ConfigError(f"Unexpanded environment variable in {key}: {value}")
    if str(config["model"]["dtype"]) != "bfloat16":
        raise ConfigError("model.dtype must be bfloat16 for this experiment")
    if int(config["quantization"]["width"]) != 4:
        raise ConfigError("This repository currently implements MXINT4 only")
    ranks = [int(value) for value in config["statistics"]["ranks"]]
    maximum_rank = int(config["statistics"]["maximum_rank"])
    if not ranks or min(ranks) <= 0 or max(ranks) > maximum_rank:
        raise ConfigError("statistics.ranks must be positive and <= maximum_rank")
    valid_methods = {"AD_GI", "AD_GD", "AD_GF", "AF_GI", "AF_GD", "AF_GF"}
    methods = set(config["statistics"]["methods"])
    if not methods or not methods <= valid_methods:
        raise ConfigError(f"Unknown method(s): {sorted(methods - valid_methods)}")
    for role in ("calibration", "wikitext2", "c4"):
        spec = _required(config, f"data.{role}")
        for key in ("split", "text_column", "sequence_length", "windows"):
            if key not in spec:
                raise ConfigError(f"Missing config key: data.{role}.{key}")


def output_root(config: Mapping[str, Any]) -> Path:
    return Path(str(config["experiment"]["output_dir"])).resolve()
