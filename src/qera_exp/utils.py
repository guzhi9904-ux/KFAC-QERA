from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from safetensors.torch import load_file, save_file


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    raw = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def atomic_text(path: str | Path, content: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_json(path: str | Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def save_csv(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = [dict(row) for row in rows]
    if not materialized:
        atomic_text(path, "")
        return
    fields: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fields:
                fields.append(key)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(materialized)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_csv(path: str | Path) -> list[dict[str, str]]:
    source = Path(path)
    if not source.exists() or source.stat().st_size == 0:
        return []
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def upsert_csv(path: str | Path, rows: Iterable[Mapping[str, Any]], keys: tuple[str, ...]) -> None:
    incoming = [dict(row) for row in rows]
    replaced = {tuple(str(row[key]) for key in keys) for row in incoming}
    kept = [row for row in read_csv(path) if tuple(str(row.get(key, "")) for key in keys) not in replaced]
    save_csv(path, kept + incoming)


def atomic_safetensors(path: str | Path, tensors: Mapping[str, torch.Tensor]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    save_file({key: value.detach().cpu().contiguous() for key, value in tensors.items()}, temporary)
    os.replace(temporary, destination)


def load_safetensors(path: str | Path) -> dict[str, torch.Tensor]:
    return load_file(Path(path), device="cpu")


def safe_name(value: str) -> str:
    return value.replace(".", "__")


def get_module(model: torch.nn.Module, name: str) -> torch.nn.Module:
    current: Any = model
    for part in name.split("."):
        current = current[int(part)] if part.isdigit() and isinstance(current, (torch.nn.ModuleList, torch.nn.Sequential)) else getattr(current, part)
    return current


def ensure_layout(root: Path) -> None:
    for relative in (
        "data",
        "quantization",
        "statistics/raw",
        "statistics/metrics",
        "corrections",
        "evaluation",
        "analysis/figures",
        "state",
        "logs",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)


def log(root: Path, message: str, name: str = "run") -> None:
    line = f"{utc_now()} {message}"
    print(f"[{name}] {message}", flush=True)
    destination = root / "logs" / f"{name}.log"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def deterministic_runtime(seed: int, allow_tf32: bool = False) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
    torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")
