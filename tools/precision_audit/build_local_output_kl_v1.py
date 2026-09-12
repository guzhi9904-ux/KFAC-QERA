#!/usr/bin/env python3
"""Build an LF-normalized deterministic offline bundle, without third-party dependencies."""
import argparse
import hashlib
from pathlib import Path
import zipfile

FILES = ("local_output_kl_v1.py", "run_local_output_kl_v1.sh", "test_local_output_kl_v1.py",
         "README_local_output_kl_v1.md", "build_local_output_kl_v1.py",
         "grouped_rank_ablation_v1.py", "full_a_all_ranks_dual_ppl_v1.py",
         "full_a_all_precision_r8_v1.py", "full_g_precision_r8_v1.py",
         "full_g_target_probe_v1.py", "full_g_rank_audit_v1.py")
SUMS = "SHA256SUMS.local_output_kl"


def payload(root):
    data = {n: (root/n).read_bytes().replace(b'\r\n', b'\n') for n in FILES}
    data[SUMS] = ''.join(hashlib.sha256(data[n]).hexdigest()+'  '+n+'\n' for n in FILES).encode()
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--update-checksums', action='store_true')
    parser.add_argument('--server-profile', type=Path,
                        help='Optional private configuration, included only in the offline ZIP')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    data = payload(root)
    if args.update_checksums:
        (root/SUMS).write_bytes(data[SUMS])
    if (root/SUMS).read_bytes().replace(b'\r\n', b'\n') != data[SUMS]:
        raise RuntimeError('Release files changed: review/test before updating checksums')
    if args.server_profile:
        value = args.server_profile.read_bytes().replace(b'\r\n', b'\n')
        data['server_profile.sh'] = value
        data['SHA256SUMS.local_output_kl_profile'] = (hashlib.sha256(value).hexdigest()+'  server_profile.sh\n').encode()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir/'local_output_kl_v1.zip'
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in data.items():
            info = zipfile.ZipInfo(name, date_time=(2026,9,12,0,0,0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, value)
    with zipfile.ZipFile(destination) as archive:
        if (archive.testzip() is not None or set(archive.namelist()) != set(data)
                or any(archive.read(n) != value for n, value in data.items())):
            raise RuntimeError('Bundle verification failed')
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    destination.with_suffix('.zip.sha256').write_bytes((digest+'  '+destination.name+'\n').encode())
    print(str(destination.resolve()))
    print(digest+'  '+str(destination.stat().st_size)+' bytes; '+str(len(data))+' files')


if __name__ == '__main__': main()
