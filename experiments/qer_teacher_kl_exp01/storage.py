"""Atomic single-file commits; tensors and their progress are committed together."""
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile

from safetensors import safe_open
from safetensors.torch import load_file, save_file


def sha_file(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1048576),b""):h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_bytes(path, data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=".pending-",dir=path.parent)
    try:
        with os.fdopen(fd,"wb") as f:f.write(data);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def save_json(path, value):
    atomic_bytes(path,(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n").encode())


def save_tensors(path, tensors, metadata):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=".pending-",dir=path.parent);os.close(fd)
    try:
        save_file({k:v.detach().contiguous().cpu() for k,v in tensors.items()},tmp,
                  metadata={"record":json.dumps(metadata,allow_nan=False)})
        with open(tmp,"r+b") as f:os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def read_tensors(path, device="cpu"):
    with safe_open(str(path),framework="pt",device="cpu") as f:
        metadata=json.loads(f.metadata()["record"])
    return load_file(str(path),device=device),metadata


def save_csv(path, rows):
    import io
    if not rows:return
    columns=list(dict.fromkeys(k for row in rows for k in row))
    output=io.StringIO();writer=csv.DictWriter(output,fieldnames=columns);writer.writeheader()
    for row in rows:
        writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})
    atomic_bytes(path,output.getvalue().encode())
