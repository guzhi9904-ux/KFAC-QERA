#!/usr/bin/env python3
"""Build a deterministic, LF-normalized offline tool bundle using only stdlib."""
import argparse
import hashlib
from pathlib import Path
import zipfile

FILES=("grouped_rank_ablation_v1.py","run_grouped_rank_ablation_v1.sh",
       "test_grouped_rank_ablation_v1.py","README_grouped_rank_ablation_v1.md",
       "build_grouped_rank_ablation_v1.py","full_a_all_ranks_dual_ppl_v1.py",
       "full_a_all_precision_r8_v1.py","full_g_precision_r8_v1.py",
       "full_g_target_probe_v1.py","full_g_rank_audit_v1.py")
SUMS="SHA256SUMS.grouped_rank_ablation"


def payload(root):
    data={n:(root/n).read_bytes().replace(b'\r\n',b'\n') for n in FILES}
    data[SUMS]=''.join(hashlib.sha256(data[n]).hexdigest()+'  '+n+'\n' for n in FILES).encode()
    return data


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--update-checksums',action='store_true',help='Developer release step; changes only the new checksum file')
    args=parser.parse_args();root=Path(__file__).resolve().parent;data=payload(root)
    if args.update_checksums:
        (root/SUMS).write_bytes(data[SUMS])
    if (root/SUMS).read_bytes().replace(b'\r\n',b'\n')!=data[SUMS]:
        raise RuntimeError('Source changed: review/test before --update-checksums')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    destination=args.output_dir/'grouped_rank_ablation_v1.zip'
    with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,value in data.items():
            info=zipfile.ZipInfo(name,date_time=(2026,9,11,0,0,0))
            info.create_system=3;info.external_attr=0o100644<<16
            info.compress_type=zipfile.ZIP_DEFLATED
            archive.writestr(info,value)
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None or set(archive.namelist())!=set(data):
            raise RuntimeError('Bundle integrity failure')
        if any(archive.read(n)!=v for n,v in data.items()):raise RuntimeError('Bundle byte mismatch')
    digest=hashlib.sha256(destination.read_bytes()).hexdigest()
    destination.with_suffix('.zip.sha256').write_bytes((digest+'  '+destination.name+'\n').encode())
    print(str(destination.resolve()))
    print(digest+'  '+str(destination.stat().st_size)+' bytes; '+str(len(data))+' files')


if __name__=='__main__':main()
