"""Self-checking atomic records and immutable freezes; no in-place result edits."""
from common import digest, read, save_json, require, sha_file


def load_record(path, identity):
    row = read(path)
    checksum = row.pop('record_sha256')
    require(digest(row) == checksum, 'Atomic record checksum differs: '+str(path))
    require(row['identity'] == identity, 'Atomic record identity differs: '+str(path))
    return row


def commit(path, identity, **fields):
    row = dict(identity=identity, **fields)
    if path.exists():
        require(load_record(path, identity) == row, 'Refusing to replace frozen record: '+str(path))
    else:
        save_json(path, dict(row, record_sha256=digest(row)))
    return row


def checked_files(root, files):
    for path, expected in files.items():
        require(sha_file(root/path) == expected, 'Frozen file changed: '+str(root/path))


def file_table(root, paths):
    return {p.relative_to(root).as_posix(): sha_file(p) for p in sorted(paths)}
