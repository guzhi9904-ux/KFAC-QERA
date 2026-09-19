#!/usr/bin/env python3
"""Private, checksummed parent asset transport. Never uploads anything to GitHub."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import tarfile
import time


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''): result.update(block)
    return result.hexdigest()


def safe_name(name):
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name or not name or ':' in name:
        raise ValueError('Unsafe bundle name: ' + name)
    if str(p) != name: raise ValueError('Noncanonical bundle name')
    return p


def build(exp01, exp03, vendor, output):
    assert os.getuid() == 1001 and os.environ.get('USER') == 'cck', 'Build only as cck'
    sources = {'exp01': exp01.resolve(), 'exp03': exp03.resolve(), 'vendor': vendor.resolve()}
    output = output.resolve()
    for root in sources.values():
        assert root.is_dir() and not output.is_relative_to(root)
    output.mkdir(parents=True, exist_ok=False)
    records = {}; files = []; total = 0
    for group, root in sources.items():
        for path in sorted(root.rglob('*')):
            rel = path.relative_to(root)
            if '.git' in rel.parts or '__pycache__' in rel.parts: continue
            if path.is_symlink(): raise ValueError('Source symlink requires explicit handling: '+str(path))
            if not path.is_file(): continue
            size = path.stat().st_size
            if group == 'vendor' and size > 32*2**20: raise ValueError('Unexpected large vendor asset')
            name = group+'/'+rel.as_posix(); safe_name(name)
            records[name] = {'bytes': size, 'sha256': sha(path)}
            files.append((name, path)); total += size
    identities = {key: json.loads((sources[key]/'identity.json').read_text())['identity'] for key in ('exp01', 'exp03')}
    manifest = {'schema': 1, 'created_at': time.time(), 'source_paths': {k: str(v) for k,v in sources.items()},
                'parent_identities': identities, 'files': records, 'file_count': len(files), 'bytes': total}
    data = (json.dumps(manifest, sort_keys=True, indent=2)+'\n').encode()
    manifest_path = output/'manifest.json'; manifest_path.write_bytes(data)
    temporary = output/'parents.tar.partial'
    with tarfile.open(temporary, 'w') as archive:
        for name, path in files:
            before = path.stat()
            assert before.st_size == records[name]['bytes']
            # TarFile copies in bounded chunks; no complete tensor is deserialized.
            info = archive.gettarinfo(str(path), arcname=name)
            info.uid = info.gid = 0; info.uname = info.gname = ''; info.mode = 0o600
            with path.open('rb') as stream: archive.addfile(info, stream)
            after = path.stat()
            assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
        info = tarfile.TarInfo('manifest.json'); info.size = len(data); info.mode = 0o600
        archive.addfile(info, io.BytesIO(data))
    archive_path = output/'parents.tar'; temporary.rename(archive_path)
    receipt = {'output': str(output), 'archive_bytes': archive_path.stat().st_size,
               'manifest_sha256': hashlib.sha256(data).hexdigest(), 'files': len(files),
               'payload_GiB': total/2**30, 'compute_started': False}
    (output/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt), flush=True)


def extract(archive_path, manifest_path, manifest_sha256, destination):
    data = manifest_path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == manifest_sha256, 'Manifest SHA256 mismatch'
    manifest = json.loads(data)
    assert manifest['schema'] == 1 and manifest['file_count'] == len(manifest['files'])
    assert sum(v['bytes'] for v in manifest['files'].values()) == manifest['bytes']
    for name in manifest['files']: safe_name(name)
    destination = destination.resolve(); destination.mkdir(parents=True, exist_ok=False)
    seen = set(); saw_manifest = False
    with tarfile.open(archive_path, 'r|') as archive:
        for member in archive:
            safe_name(member.name)
            if not member.isfile() or member.name in seen: raise ValueError('Unexpected type or duplicate')
            seen.add(member.name); stream = archive.extractfile(member)
            if member.name == 'manifest.json':
                assert member.size == len(data) and stream.read() == data
                saw_manifest = True; continue
            expected = manifest['files'].get(member.name)
            assert expected is not None and member.size == expected['bytes'], member.name
            target = destination/Path(member.name)
            assert target.resolve().is_relative_to(destination)
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            with target.open('xb') as output:
                for chunk in iter(lambda: stream.read(2**20), b''):
                    output.write(chunk); digest.update(chunk)
            assert digest.hexdigest() == expected['sha256'], ('File SHA256 mismatch', member.name)
    assert saw_manifest and seen == set(manifest['files']) | {'manifest.json'}
    (destination/'migration_manifest.json').write_bytes(data)
    (destination/'migration_verified.json').write_text(json.dumps({'passed': True,
        'manifest_sha256': manifest_sha256, 'files': len(manifest['files']),
        'parent_identities': manifest['parent_identities']}, indent=2)+'\n')
    print(json.dumps({'passed': True, 'destination': str(destination), 'files': len(manifest['files'])}))


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('build')
    for name in ('exp01', 'exp03', 'vendor', 'output'): p.add_argument('--'+name, type=Path, required=True)
    p = sub.add_parser('extract')
    for name in ('archive', 'manifest', 'destination'): p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--manifest-sha256', required=True)
    args = parser.parse_args()
    if args.command == 'build': build(args.exp01, args.exp03, args.vendor, args.output)
    else: extract(args.archive, args.manifest, args.manifest_sha256, args.destination)


if __name__ == '__main__': main()
