#!/usr/bin/env python3
"""Build the deterministic offline release; private paths stay in a profile."""
import argparse
import hashlib
from pathlib import Path
import zipfile

FILES = ('llama_rank_increment_v1.py', 'run_llama_rank_increment_v1.sh',
         'test_llama_rank_increment_v1.py', 'README_llama_rank_increment_v1.md',
         'build_llama_rank_increment_v1.py', 'diag_a_fp64_dual_v1.py',
         'full_a_all_ranks_dual_ppl_v1.py', 'full_a_all_precision_r8_v1.py',
         'full_g_precision_r8_v1.py', 'full_g_target_probe_v1.py',
         'full_g_rank_audit_v1.py', 'frozen_harness_word_ppl.py')
SUMS = 'SHA256SUMS.llama_rank_increment'


def payload(root):
    data = {n:(root/n).read_bytes().replace(b'\r\n',b'\n') for n in FILES}
    data[SUMS] = ''.join(hashlib.sha256(data[n]).hexdigest()+'  '+n+'\n' for n in FILES).encode()
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--server-profile',type=Path)
    p.add_argument('--update-checksums',action='store_true')
    args = p.parse_args()
    root = Path(__file__).resolve().parent
    data = payload(root)
    if args.update_checksums:
        (root/SUMS).write_bytes(data[SUMS])
    if (root/SUMS).read_bytes().replace(b'\r\n',b'\n') != data[SUMS]:
        raise RuntimeError('Release changed; review and test before updating checksums')
    if args.server_profile:
        value = args.server_profile.read_bytes().replace(b'\r\n',b'\n')
        data['server_profile.sh'] = value
        data[SUMS+'_profile'] = (hashlib.sha256(value).hexdigest()+'  server_profile.sh\n').encode()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    dest = args.output_dir/'llama_rank_increment_v1.zip'
    with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,value in data.items():
            info = zipfile.ZipInfo(name,date_time=(2026,9,13,0,0,0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info,value)
    with zipfile.ZipFile(dest) as archive:
        if archive.testzip() or set(archive.namelist())!=set(data) or any(archive.read(n)!=v for n,v in data.items()):
            raise RuntimeError('ZIP verification failed')
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    dest.with_suffix('.zip.sha256').write_bytes((digest+'  '+dest.name+'\n').encode())
    print(str(dest.resolve()))
    print(f'{digest}  {dest.stat().st_size} bytes; {len(data)} files')


if __name__ == '__main__':
    main()
