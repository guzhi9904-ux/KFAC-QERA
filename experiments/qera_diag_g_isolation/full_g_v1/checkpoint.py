"""Whole-window transactions with bounded disk usage and cooperative stopping."""
from __future__ import annotations

from contextlib import contextmanager
import math
from pathlib import Path
import re
import signal
import time
import uuid

import torch

from qera_diag_g_isolation.storage import (
    atomic_json, atomic_tensors, checked_tensors, file_record, log, read_json, sync_parent,
)
from .numerics import check_diagonal


class Paused(RuntimeError):
    """Intentional stop after durable progress; executable exits with code 75."""


class Stop:
    def __init__(self, max_seconds=None):
        self.deadline = None if max_seconds is None else time.monotonic() + max_seconds
        self.signal_number = None

    def requested(self):
        return self.signal_number is not None or (self.deadline is not None and time.monotonic() >= self.deadline)

    def check(self):
        if self.requested():
            raise Paused("Time budget/signal reached; restart the SAME command to resume")

    @contextmanager
    def installed(self):
        previous = {}
        def receive(number, _frame):
            self.signal_number = number
            log("stop", "Stop requested; finishing the current safe unit and committing progress")
        for number in (signal.SIGINT, signal.SIGTERM):
            previous[number] = signal.signal(number, receive)
        try:
            yield self
        finally:
            for number, handler in previous.items():
                signal.signal(number, handler)


class Checkpoints:
    """Only CURRENT.json publishes a generation. Unpublished work is discarded.

    Each shard has ONE committed generation plus a temporary new generation.
    A crash during save retains the old pointer. A crash during backward never
    advances the pointer. The commit includes *all* target tensors and counters.
    """
    def __init__(self, root, manifest_sha256, shard, dimensions):
        self.root = Path(root) / f"shard_{shard:03d}"
        self.manifest = manifest_sha256
        self.shard = shard
        self.dimensions = dict(dimensions)
        self.root.mkdir(parents=True, exist_ok=True)
        if self.root.is_symlink():
            raise RuntimeError("Checkpoint directory cannot be a symlink")

    def _state(self):
        path = self.root / "CURRENT.json"
        if not path.exists():
            return None
        state = read_json(path)
        n = state.get("windows_completed")
        if (state.get("manifest_sha256") != self.manifest or state.get("shard") != self.shard
                or type(n) is not int or not 1 <= n <= 256 or state.get("prediction_tokens") != n * 2047
                or not math.isfinite(state.get("teacher_nll_sum", float("nan")))
                or state["teacher_nll_sum"] < 0 or set(state.get("files", {})) != set(self.dimensions)
                or state.get("representation") != "full" or state.get("dtype") != "float64"):
            raise RuntimeError("Checkpoint protocol/counters/coverage mismatch")
        generation = state.get("generation", "")
        if not re.fullmatch(r"gen_[0-9a-f]{32}", generation):
            raise RuntimeError("Invalid checkpoint generation")
        directory = self.root / generation
        if directory.is_symlink() or directory.resolve().parent != self.root.resolve():
            raise RuntimeError("Checkpoint generation escapes owned directory")
        owner = read_json(directory / "OWNER.json")
        if owner != {"manifest_sha256": self.manifest, "shard": self.shard}:
            raise RuntimeError("Checkpoint owner mismatch")
        for index, name in enumerate(self.dimensions):
            if Path(state["files"][name]["path"]).resolve() != (directory / f"layer_{index:03d}.safetensors").resolve():
                raise RuntimeError("Checkpoint file points outside its generation")
        if read_json(directory / "STATE.json") != state:
            raise RuntimeError("Committed checkpoint metadata differs")
        return state

    def load(self, tensors=True):
        state = self._state()
        if not tensors:
            return state
        if state is None:
            return {n: {"gram": torch.zeros(d, d, dtype=torch.float64),
                        "diagonal": torch.zeros(d, dtype=torch.float64)} for n, d in self.dimensions.items()}, 0, 0.
        values = {}
        for name in self.dimensions:
            # Mutable accumulation must not retain mmap-backed snapshot storage.
            # Clone one module at a time (not an extra copy of the entire shard).
            loaded = self.read_layer(state, name)
            values[name] = {key: value.clone() for key, value in loaded.items()}
            del loaded
        return values, state["windows_completed"], state["teacher_nll_sum"]

    def read_layer(self, state, name):
        tensors = checked_tensors(state["files"][name])
        d = self.dimensions[name]
        if set(tensors) != {"gram", "diagonal"}:
            raise RuntimeError("Checkpoint tensor coverage mismatch")
        for key, shape in (("gram", (d, d)), ("diagonal", (d,))):
            value = tensors[key]
            if tuple(value.shape) != shape or value.dtype != torch.float64 or not torch.isfinite(value).all():
                raise RuntimeError("Invalid checkpoint tensor")
        if (tensors["diagonal"] < 0).any():
            raise RuntimeError("Negative checkpoint diagonal")
        check_diagonal(tensors["gram"], tensors["diagonal"])
        return tensors

    def save(self, values, windows, nll):
        if set(values) != set(self.dimensions) or type(windows) is not int or not 1 <= windows <= 256:
            raise ValueError("Invalid whole-window checkpoint")
        if not math.isfinite(nll) or nll < 0:
            raise ValueError("Invalid teacher NLL")
        old = self._state()
        if old is not None and windows <= old["windows_completed"]:
            raise RuntimeError("Checkpoint counters must advance")
        generation = "gen_" + uuid.uuid4().hex
        directory = self.root / generation
        directory.mkdir()
        atomic_json(directory / "OWNER.json", {"manifest_sha256": self.manifest, "shard": self.shard})
        files = {}
        for index, name in enumerate(self.dimensions):
            value = values[name]
            d = self.dimensions[name]
            if (set(value) != {"gram", "diagonal"} or value["gram"].shape != (d, d)
                    or value["diagonal"].shape != (d,) or any(
                        t.dtype != torch.float64 or not torch.isfinite(t).all() for t in value.values())):
                raise ValueError("Invalid checkpoint values")
            check_diagonal(value["gram"], value["diagonal"])
            path = directory / f"layer_{index:03d}.safetensors"
            atomic_tensors(path, value)
            files[name] = file_record(path)
        state = {"manifest_sha256": self.manifest, "shard": self.shard, "generation": generation,
                 "windows_completed": windows, "prediction_tokens": windows * 2047,
                 "teacher_nll_sum": nll, "files": files, "representation": "full", "dtype": "float64"}
        atomic_json(directory / "STATE.json", state)
        atomic_json(self.root / "CURRENT.json", state)
        log("checkpoint-full-g", f"COMMITTED shard={self.shard + 1} window={windows}/256")
        self.cleanup()
        return state

    def cleanup(self):
        """Prune ONLY this writer's obsolete/partial snapshots, after validation.

        Never recursively delete. Unexpected contents/symlinks cause a refusal.
        Raw statistics for a completed shard's current generation remain forever.
        """
        current = self._state()
        keep = None if current is None else current["generation"]
        for directory in self.root.glob("gen_*"):
            if directory.name == keep:
                continue
            if (not re.fullmatch(r"gen_[0-9a-f]{32}", directory.name) or directory.is_symlink()
                    or directory.resolve().parent != self.root.resolve()):
                raise RuntimeError("Unsafe obsolete checkpoint path")
            owner = directory / "OWNER.json"
            # An interruption before OWNER was committed leaves an empty/unowned
            # directory. Do not adopt or delete it automatically.
            if not owner.exists():
                continue
            if read_json(owner) != {"manifest_sha256": self.manifest, "shard": self.shard}:
                raise RuntimeError("Refusing to remove foreign checkpoint")
            children = list(directory.iterdir())
            allowed = r"(?:OWNER\.json|STATE\.json|layer_[0-9]{3}\.safetensors|\.(?:OWNER\.json|STATE\.json|layer_[0-9]{3}\.safetensors)\.[0-9a-f]{32}\.tmp)"
            if any(p.is_symlink() or not p.is_file() or not re.fullmatch(allowed, p.name) for p in children):
                raise RuntimeError("Unexpected files in obsolete checkpoint; refusing cleanup")
            for path in children:
                path.unlink()
            directory.rmdir()
            sync_parent(directory)
            log("checkpoint-full-g", f"pruned superseded/uncommitted generation {directory.name}")
