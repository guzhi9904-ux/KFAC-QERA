"""Bounded worker-log excerpts for controller failures."""
from pathlib import Path


def worker_failure(phase, jobs, paths):
    details = []
    for job, path in zip(jobs, paths):
        if job.poll() in (None, 0):
            continue
        path = Path(path)
        try:
            with path.open('rb') as stream:
                stream.seek(0, 2)
                stream.seek(max(0, stream.tell()-8192))
                tail = stream.read().decode('utf-8', errors='replace')
            excerpt = '\n'.join(tail.splitlines()[-18:])
        except OSError as error:
            excerpt = f'Cannot read worker log: {error}'
        details.append(f'Worker exit={job.returncode}; log={path}\n{excerpt}')
    return RuntimeError(f'{phase}: worker failed\n'+'\n'.join(details))
