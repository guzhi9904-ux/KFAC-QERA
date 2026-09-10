"""Qwen-owned transactional checkpoints. No Llama paths are writable here."""
from pathlib import Path
import math
import re
import uuid

import torch

from qera_diag_g_isolation.storage import atomic_json, atomic_tensors, checked_tensors, file_record, read_json, log


class Store:
    def __init__(self, root, digest, schema, tokens_per_window):
        self.root, self.digest, self.schema, self.tokens = Path(root), digest, schema, tokens_per_window
        self.root.mkdir(parents=True, exist_ok=True)
        if self.root.is_symlink():
            raise RuntimeError("Checkpoint directory must not be a symlink")

    def current(self):
        path = self.root / "CURRENT.json"
        if not path.exists():
            return None
        s = read_json(path)
        n = s.get("windows")
        if (s.get("manifest") != self.digest or type(n) is not int or not 1 <= n <= 256
                or s.get("tokens") != n*self.tokens or set(s.get("files", {})) != set(self.schema)
                or not math.isfinite(s.get("nll", float("nan"))) or s["nll"] < 0):
            raise RuntimeError("Checkpoint protocol/count mismatch")
        if not re.fullmatch("gen_[0-9a-f]{32}", s.get("generation", "")):
            raise RuntimeError("Invalid generation name")
        folder = self.root / s["generation"]
        if folder.is_symlink() or folder.resolve().parent != self.root.resolve():
            raise RuntimeError("Invalid generation path")
        if read_json(folder / "OWNER.json") != {"manifest": self.digest} or read_json(folder / "STATE.json") != s:
            raise RuntimeError("Checkpoint ownership/state mismatch")
        for i, name in enumerate(self.schema):
            if Path(s["files"][name]["path"]).resolve() != (folder / f"part_{i:03d}.safetensors").resolve():
                raise RuntimeError("Checkpoint pointer escapes generation")
        return s

    def validate(self, name, value):
        if set(value) != set(self.schema[name]):
            raise RuntimeError("Checkpoint key mismatch")
        for key, (shape, dtype) in self.schema[name].items():
            x = value[key]
            if tuple(x.shape) != tuple(shape) or x.dtype != dtype or not torch.isfinite(x).all():
                raise RuntimeError("Checkpoint shape/dtype/nonfinite mismatch")
            if key != "full" and (x < 0).any():
                raise RuntimeError("Negative diagonal statistic")

    def read(self, state, name):
        value = checked_tensors(state["files"][name])
        self.validate(name, value)
        return value

    def load(self):
        s = self.current()
        if s is None:
            return {name: {key: torch.zeros(shape, dtype=dtype) for key, (shape, dtype) in fields.items()}
                    for name, fields in self.schema.items()}, 0, 0.
        values = {}
        for name in self.schema:
            tensors = self.read(s, name)
            values[name] = {k: v.clone() for k, v in tensors.items()}
            del tensors
        return values, s["windows"], s["nll"]

    def save(self, values, windows, nll=0.):
        old = self.current()
        if (set(values) != set(self.schema) or type(windows) is not int or not 1 <= windows <= 256
                or (old and windows <= old["windows"]) or not math.isfinite(nll) or nll < 0):
            raise RuntimeError("Invalid/non-advancing whole-window commit")
        folder = self.root / ("gen_" + uuid.uuid4().hex)
        folder.mkdir()
        atomic_json(folder / "OWNER.json", {"manifest": self.digest})
        files = {}
        for i, name in enumerate(self.schema):
            self.validate(name, values[name])
            path = folder / f"part_{i:03d}.safetensors"
            atomic_tensors(path, values[name])
            files[name] = file_record(path)
        s = {"manifest": self.digest, "generation": folder.name, "windows": windows,
             "tokens": windows*self.tokens, "nll": nll, "files": files}
        atomic_json(folder / "STATE.json", s)
        atomic_json(self.root / "CURRENT.json", s)
        log("checkpoint-qwen", f"COMMITTED {self.root.name} window={windows}/256")
        self.cleanup()
        return s

    def cleanup(self):
        current = self.current()
        for folder in self.root.glob("gen_*"):
            if current and folder.name == current["generation"]:
                continue
            if folder.is_symlink() or folder.resolve().parent != self.root.resolve() or not re.fullmatch("gen_[0-9a-f]{32}", folder.name):
                raise RuntimeError("Unsafe checkpoint cleanup path")
            owner = folder / "OWNER.json"
            if not owner.exists():
                continue
            if read_json(owner) != {"manifest": self.digest}:
                raise RuntimeError("Foreign checkpoint owner")
            contents = list(folder.iterdir())
            pattern = r"(OWNER\.json|STATE\.json|part_\d{3}\.safetensors|\.(OWNER\.json|STATE\.json|part_\d{3}\.safetensors)\.[0-9a-f]{32}\.tmp)"
            if any(p.is_symlink() or not p.is_file() or not re.fullmatch(pattern, p.name) for p in contents):
                raise RuntimeError("Unexpected file; refusing checkpoint cleanup")
            for p in contents:
                p.unlink()
            folder.rmdir()


def artifact(config, manifest, directory, name, tensors=None, check=True, **details):
    from qera_diag_g_isolation.pipeline import artifact_path, completed_artifact, save_artifact
    path = artifact_path(config, directory, name)
    record = completed_artifact(path, manifest, check=check)
    if record is not None:
        if Path(record["file"]["path"]).resolve() != path.resolve():
            raise RuntimeError("Qwen artifact must belong to this output directory")
        return record
    if tensors is None:
        return None
    return save_artifact(path, manifest, tensors, **details)
