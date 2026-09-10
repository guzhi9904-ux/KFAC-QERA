"""Build an isolated ZIP; never check out, modify or publish the legacy release."""
import argparse
import hashlib
import io
from pathlib import Path
import subprocess
import zipfile

BASE = "8b49f28f9b08c67caab0c3d7e4eccd3992d82059"
PREFIX = "KFAC-QERA-qwen25-base-v1/"


def build(destination):
    folder = Path(__file__).resolve().parent
    repo = folder.parents[1]
    raw = subprocess.check_output([
        "git", "-c", f"safe.directory={repo.as_posix()}", "-C", str(repo), "archive", "--format=zip", BASE,
        "experiments/qera_original_a_isolation", "experiments/qera_diag_g_isolation",
    ])
    # Exclusive creation: a published/deployed release must not be overwritten.
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as target:
        with zipfile.ZipFile(io.BytesIO(raw)) as source:
            for name in source.namelist():
                if not name.endswith("/"):
                    target.writestr(PREFIX + name, source.read(name))
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix in {".py", ".yaml", ".sh", ".md"}:
                # Server source is canonical LF, like git archive.
                target.writestr(PREFIX + path.relative_to(repo).as_posix(), path.read_bytes().replace(b"\r\n", b"\n"))
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("ZIP verification failed")
        assert all(name.startswith(PREFIX) and ".." not in Path(name).parts for name in archive.namelist())
    print(f"{destination}\nSHA256 {hashlib.sha256(Path(destination).read_bytes()).hexdigest()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    build(parser.parse_args().destination)
