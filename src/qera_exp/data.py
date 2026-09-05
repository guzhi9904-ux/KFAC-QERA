from __future__ import annotations

import hashlib
import random
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import torch

from .config import output_root
from .modeling import load_tokenizer
from .utils import (
    atomic_safetensors,
    canonical_sha256,
    ensure_layout,
    heartbeat,
    load_safetensors,
    log,
    save_json,
    sha256_file,
    tensor_sha256,
    utc_now,
)


def _load_source(spec: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
    from datasets import load_dataset, load_from_disk

    local_path = spec.get("local_path")
    if local_path:
        location = Path(str(local_path)).resolve()
        loaded = load_from_disk(str(location))
        split = str(spec["split"])
        dataset = loaded[split] if hasattr(loaded, "keys") and split in loaded else loaded
        return dataset, {"access": "load_from_disk", "path": str(location), "split": split}
    dataset = load_dataset(
        str(spec["dataset"]),
        spec.get("subset"),
        split=str(spec["split"]),
        streaming=bool(spec.get("streaming", False)),
    )
    return dataset, {
        "access": "huggingface",
        "dataset": spec["dataset"],
        "subset": spec.get("subset"),
        "split": spec["split"],
        "streaming": bool(spec.get("streaming", False)),
        "fingerprint": getattr(dataset, "_fingerprint", None),
    }


def _iter_text(dataset: Any, column: str) -> Iterable[tuple[int, str]]:
    for index, row in enumerate(dataset):
        text = str(row.get(column, ""))
        if text:
            yield index, text


def _token_stream(
    dataset: Any,
    tokenizer: Any,
    spec: Mapping[str, Any],
    *,
    required: int | None,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    separator = tokenizer(
        str(spec.get("separator", "\n\n")),
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )["input_ids"]
    tokens: list[int] = []
    rows: list[dict[str, Any]] = []
    for row_index, text in _iter_text(dataset, str(spec["text_column"])):
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            truncation=False,
            verbose=False,
        )["input_ids"]
        if not encoded:
            continue
        tokens.extend(encoded)
        tokens.extend(separator)
        rows.append(
            {
                "source_row_index": row_index,
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "tokens": len(encoded),
            }
        )
        if progress is not None and (len(rows) == 1 or len(rows) % 1000 == 0):
            progress(len(rows), len(tokens))
        if required is not None and len(tokens) >= required:
            break
    if required is not None and len(tokens) < required:
        raise RuntimeError(f"Dataset yielded {len(tokens)}/{required} required tokens")
    return torch.tensor(tokens, dtype=torch.long), rows


def _windowize(stream: torch.Tensor, spec: Mapping[str, Any]) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    length = int(spec["sequence_length"])
    blocks = int(stream.numel() // length)
    requested = spec["windows"]
    count = blocks if requested == "all" else int(requested)
    if count > blocks:
        raise RuntimeError(f"Requested {count} windows but only {blocks} complete windows are available")
    order = list(range(blocks))
    if str(spec.get("sampling", "prefix")) == "random_blocks":
        random.Random(int(spec.get("seed", 0))).shuffle(order)
    selected = order[:count]
    ids = torch.stack([stream[index * length : (index + 1) * length] for index in selected]).contiguous()
    rows = []
    for window_id, block in enumerate(selected):
        value = ids[window_id]
        rows.append(
            {
                "window_id": window_id,
                "source_block_index": block,
                "start": block * length,
                "end": (block + 1) * length,
                "token_ids_sha256": tensor_sha256(value),
                "effective_prediction_tokens": length - 1,
            }
        )
    return ids, rows


def prepare_role(config: Mapping[str, Any], role: str) -> dict[str, Any]:
    from datasets import IterableDataset

    root = output_root(config)
    ensure_layout(root)
    if role not in config["data"]:
        raise KeyError(f"Unknown data role: {role}")
    spec = config["data"][role]
    started = time.time()
    log(root, f"role={role} loading dataset source", "data")
    with heartbeat(root, f"role={role} loading dataset source", "data"):
        dataset, provenance = _load_source(spec)
    log(root, f"role={role} dataset ready; loading tokenizer", "data")
    if isinstance(dataset, IterableDataset) and spec.get("shuffle_buffer_size"):
        dataset = dataset.shuffle(seed=int(spec.get("seed", 0)), buffer_size=int(spec["shuffle_buffer_size"]))
    length = int(spec["sequence_length"])
    requested = spec["windows"]
    sampling = str(spec.get("sampling", "prefix"))
    # Random non-overlapping blocks must be sampled from the complete finite
    # split, not only from a prefix with exactly the requested token count.
    required = None if requested == "all" or sampling == "random_blocks" else int(requested) * length
    with heartbeat(root, f"role={role} loading tokenizer", "data"):
        tokenizer = load_tokenizer(config)
    log(root, f"role={role} tokenization started required_tokens={required if required is not None else 'complete split'}", "data")

    def report(source_row_count: int, token_count: int) -> None:
        log(
            root,
            f"role={role} tokenizing rows={source_row_count} tokens={token_count} elapsed={time.time() - started:.1f}s",
            "data",
        )

    with heartbeat(root, f"role={role} tokenizing", "data"):
        stream, source_rows = _token_stream(dataset, tokenizer, spec, required=required, progress=report)
    log(root, f"role={role} tokenization complete rows={len(source_rows)} tokens={stream.numel()}", "data")
    ids, window_rows = _windowize(stream, spec)
    mask = torch.ones_like(ids)
    artifact = root / "data" / f"{role}.safetensors"
    atomic_safetensors(artifact, {"input_ids": ids, "attention_mask": mask})
    manifest = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "role": role,
        "dataset": spec.get("dataset"),
        "subset": spec.get("subset"),
        "split": spec["split"],
        "sampling": spec.get("sampling"),
        "seed": spec.get("seed"),
        "sequence_length": length,
        "windows": int(ids.shape[0]),
        "effective_prediction_tokens": int(ids.shape[0]) * (length - 1),
        "input_ids_sha256": tensor_sha256(ids),
        "attention_mask_sha256": tensor_sha256(mask),
        "artifact": str(artifact),
        "artifact_sha256": sha256_file(artifact),
        "tokenizer_name_or_path": str(tokenizer.name_or_path),
        "tokenizer_class": type(tokenizer).__name__,
        "vocab_size": int(tokenizer.vocab_size),
        "provenance": provenance,
        "source_rows_sha256": canonical_sha256(source_rows),
        "source_row_count": len(source_rows),
        "windows_manifest": window_rows,
    }
    save_json(root / "data" / f"{role}_manifest.json", manifest)
    log(root, f"role={role} saved windows={ids.shape[0]} elapsed={time.time() - started:.1f}s", "data")
    return manifest


def prepare_all(config: Mapping[str, Any]) -> dict[str, Any]:
    manifests = {role: prepare_role(config, role) for role in ("calibration", "wikitext2", "c4")}
    result = {"status": "PASS", "created_at_utc": utc_now(), "roles": manifests}
    save_json(output_root(config) / "state" / "data_complete.json", result)
    return result


def load_windows(config: Mapping[str, Any], role: str) -> list[dict[str, Any]]:
    root = output_root(config)
    artifact = root / "data" / f"{role}.safetensors"
    manifest_path = root / "data" / f"{role}_manifest.json"
    if not artifact.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"Prepared data is absent for role={role}; run prepare-data first")
    tensors = load_safetensors(artifact)
    ids, mask = tensors["input_ids"], tensors["attention_mask"]
    result = []
    for index in range(ids.shape[0]):
        row_ids = ids[index : index + 1].contiguous()
        row_mask = mask[index : index + 1].contiguous()
        result.append(
            {
                "window_id": index,
                "input_ids": row_ids,
                "attention_mask": row_mask,
                "token_ids_sha256": tensor_sha256(row_ids),
                "attention_mask_sha256": tensor_sha256(row_mask),
                "window_hash": canonical_sha256(
                    {"token_ids_sha256": tensor_sha256(row_ids), "attention_mask_sha256": tensor_sha256(row_mask)}
                ),
            }
        )
    return result
