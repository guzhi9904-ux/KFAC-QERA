"""Acquire the exact public calibration prefix on a connected computer."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import time

DATASET = "DKYoon/SlimPajama-6B"
REVISION = "b5f90f419b7489cdba26fdbc8c022fcb5562f968"
ROWS = 5120
TEXT_SHA = "84067fe5da1470cca552135d1ee3e2089231b249053336b842721877c1ba03ab"
SHARD = "data/train-00000-of-00048-ab2b35705f029d94.parquet"
SHARD_SHA = "6c1e83b5f990a245a9e3fe95e5a1d5c3af8ce4d1f823791f59390c903656bb3f"
SHARD_SIZE = 292286056


def sha_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(folder):
    import requests
    import pyarrow.parquet as pq
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    shard = folder / Path(SHARD).name
    url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{SHARD}"
    if not shard.exists():
        temporary = shard.with_suffix(".download")
        if temporary.exists():
            raise RuntimeError("Partial download exists; preserve/inspect it before retrying")
        start = last = time.monotonic()
        size = 0
        with requests.get(url, stream=True, timeout=(15, 40)) as response:
            response.raise_for_status()
            with temporary.open("xb") as f:
                for block in response.iter_content(2**20):
                    f.write(block)
                    size += len(block)
                    if time.monotonic() - last > 15:
                        print(f"download {size/2**20:.1f}/{SHARD_SIZE/2**20:.1f} MiB elapsed={time.monotonic()-start:.0f}s", flush=True)
                        last = time.monotonic()
        if size != SHARD_SIZE or sha_file(temporary) != SHARD_SHA:
            raise RuntimeError("Downloaded Parquet identity mismatch")
        temporary.rename(shard)
    if shard.stat().st_size != SHARD_SIZE or sha_file(shard) != SHARD_SHA:
        raise RuntimeError("Cached Parquet identity mismatch")
    rows = []
    for batch in pq.ParquetFile(shard).iter_batches(batch_size=ROWS):
        rows.extend(batch.to_pylist()[:ROWS-len(rows)])
        if len(rows) == ROWS:
            break
    digest = hashlib.sha256()
    for row in rows:
        encoded = row["text"].encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    if len(rows) != ROWS or digest.hexdigest() != TEXT_SHA:
        raise RuntimeError(f"Original prefix mismatch: rows={len(rows)} hash={digest.hexdigest()}")
    output = folder / "calibration.jsonl.gz"
    with output.open("xb") as f:
        with gzip.GzipFile(fileobj=f, mode="wb", mtime=0, filename="") as zipped:
            for row in rows:
                zipped.write((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))
    metadata = {"dataset_id": DATASET, "resolved_revision": REVISION, "raw_prefix_rows": ROWS,
                "raw_text_sha256": TEXT_SHA, "file_sha256": sha_file(output), "source_url": url,
                "source_parquet_sha256": SHARD_SHA}
    with (folder / "calibration.source.json").open("x", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(json.dumps({"status": "PASS", "rows": len(rows), "raw_text_sha256": digest.hexdigest(),
                      "gzip_MiB": output.stat().st_size/2**20, "file": str(output)}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    fetch(parser.parse_args().destination)
