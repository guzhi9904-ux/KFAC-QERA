from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml


OFFICIAL_QERA_COMMIT = "bd7fc86a2e44d41f95b9b0421f27f5624dd37064"
OFFICIAL_FILE_SHA256 = {
    "src/qera/approximate.py": "6bd5e119271b04050991d0bde20c92ecaf32ee75e2c274381486dc05f9fb0bc2",
    "src/qera/datasets/slim_pajama.py": "85f8ba293df4862d0d000efc2f3c459a91695a2cdf741f554f0c9e6be834e749",
    "src/qera/datasets/wikitext2.py": "1991a661b1f2a2f336a22622f6693135caf96d92ad944788bd701e0c94fc0555",
    "src/qera/quantize/quantizers/mxint.py": "74643c6306fd7109a57d76c559ee8dfe82d702e3285af6cddc8296722d68c9c4",
    "src/qera/statistic_profiler/scale.py": "c5e4fe599531b89b8d905918f34154b74ff5c17968c362544ad10c0d926734c7",
}
TARGET_SUFFIXES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["_config_path"] = str(path)
    config["run_dir"] = os.path.expanduser(os.path.expandvars(config["run_dir"]))
    config["model_path"] = os.path.expanduser(os.path.expandvars(config["model_path"]))
    config["qera_source_dir"] = os.path.expanduser(os.path.expandvars(config["qera_source_dir"]))
    return config


def run_dir(config: dict[str, Any]) -> Path:
    return Path(config["run_dir"])


def save_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".tmp", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path: str | Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_source_file(path: str | Path) -> str:
    """Hash source bytes after CRLF normalization, matching the Git blob/codeload form."""
    content = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest()


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).replace(".", "__")


def git_value(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={repo}", "-C", str(repo), *args], text=True
    ).strip()


def import_official_qera(config: dict[str, Any]) -> dict[str, Any]:
    source = Path(config["qera_source_dir"]).resolve()
    if not (source / "src" / "qera").is_dir():
        raise RuntimeError(f"Official QERA source not found: {source}")
    expected = config.get("official_qera_commit", OFFICIAL_QERA_COMMIT)
    if (source / ".git").exists():
        actual = git_value(source, "rev-parse", "HEAD")
        if git_value(source, "status", "--porcelain"):
            raise RuntimeError(f"Official QERA checkout must be clean: {source}")
    elif (source / ".qera_commit").is_file():
        actual = (source / ".qera_commit").read_text(encoding="utf-8").strip()
    else:
        raise RuntimeError(f"QERA source has neither .git metadata nor .qera_commit marker: {source}")
    if actual != expected:
        raise RuntimeError(f"Official QERA commit mismatch: expected={expected}, actual={actual}")
    for relative, digest in OFFICIAL_FILE_SHA256.items():
        path = source / relative
        found = sha256_source_file(path) if path.is_file() else "missing"
        if found != digest:
            raise RuntimeError(f"Official QERA source hash mismatch: {relative}, expected={digest}, actual={found}")
    source_path = str(source / "src")
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    approximate = importlib.import_module("qera.approximate")
    datasets = importlib.import_module("qera.datasets")
    mxint = importlib.import_module("qera.quantize.quantizers.mxint")
    scale = importlib.import_module("qera.statistic_profiler.scale")
    return {
        "commit": actual,
        "compute_ab": approximate._compute_scales_and_error_for_fc,
        "get_data_module": datasets.get_data_module,
        "preprocess_data_module": datasets.preprocess_data_module,
        "mxint_quantizer": mxint.mxint_quantizer,
        "sqrtm_scipy": scale.sqrtm_scipy,
    }


def torch_dtype(name: str) -> torch.dtype:
    try:
        value = getattr(torch, name)
    except AttributeError as error:
        raise ValueError(f"Unknown torch dtype: {name}") from error
    if not isinstance(value, torch.dtype):
        raise ValueError(f"Not a torch dtype: {name}")
    return value


def get_module(model: torch.nn.Module, name: str) -> torch.nn.Module:
    module = model
    for part in name.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


@dataclass(frozen=True)
class InputGroup:
    target: str
    shares: tuple[str, ...]
    in_features: int

    @property
    def all_layers(self) -> tuple[str, ...]:
        return (self.target, *self.shares)


def discover_input_groups(model: torch.nn.Module) -> list[InputGroup]:
    modules = dict(model.named_modules())
    target_names = {
        name
        for name, module in modules.items()
        if isinstance(module, torch.nn.Linear) and name.endswith(TARGET_SUFFIXES)
    }
    groups: list[InputGroup] = []
    seen: set[str] = set()
    for name in sorted(target_names, key=_natural_key):
        if name in seen:
            continue
        if name.endswith("self_attn.k_proj"):
            prefix = name[: -len("k_proj")]
            candidates = (prefix + "q_proj", prefix + "v_proj")
            shares = tuple(candidate for candidate in candidates if candidate in target_names)
        elif name.endswith("mlp.gate_proj"):
            prefix = name[: -len("gate_proj")]
            shares = tuple(candidate for candidate in (prefix + "up_proj",) if candidate in target_names)
        elif name.endswith(("self_attn.q_proj", "self_attn.v_proj", "mlp.up_proj")):
            continue
        else:
            shares = ()
        layer = modules[name]
        groups.append(InputGroup(name, shares, int(layer.in_features)))
        seen.update((name, *shares))
    missing = target_names - seen
    if missing:
        raise RuntimeError(f"Target linear layers were not grouped: {sorted(missing)}")
    return groups


def _natural_key(value: str) -> list[Any]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value)]


def qera_layer_config(rank: int) -> dict[str, Any]:
    return {
        "rank": rank,
        "name": "qera",
        "is_ptq": True,
        "x_quantizer": {"name": "bypass"},
        "w_quantizer": {"name": "mxint", "width": 4, "block_size": 32, "block_axis": -1},
        "b_quantizer": {"name": "bypass"},
    }


def validate_config(config: dict[str, Any]) -> None:
    ranks = config["ranks"]
    if ranks != sorted(set(ranks)) or any(int(rank) <= 0 for rank in ranks):
        raise ValueError("ranks must contain unique positive integers in ascending order")
    if ranks != [8, 16, 32, 64]:
        raise ValueError("This isolation experiment is fixed to ranks [8, 16, 32, 64]")
    if int(config["sequence_length"]) != 2048:
        raise ValueError("Source-aligned sequence_length must be 2048")
    if int(config["num_calibration_samples"]) != 256:
        raise ValueError("Source-aligned num_calibration_samples must be 256")
    for key in ("calibration_batch_size", "eval_batch_size"):
        if int(config[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    if int(config["checkpoint_every_windows"]) % int(config["calibration_batch_size"]) != 0:
        raise ValueError("checkpoint_every_windows must be divisible by calibration_batch_size")
    if config.get("calibration_acquisition", "official_full") not in ("official_full", "streaming_prefix"):
        raise ValueError("calibration_acquisition must be official_full or streaming_prefix")
    quant = config["quantization"]
    if quant != {"name": "mxint", "width": 4, "block_size": 32, "block_axis": -1}:
        raise ValueError("Source-aligned quantization must be MXINT4, block_size=32, block_axis=-1")
