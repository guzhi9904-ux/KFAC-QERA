#!/usr/bin/env python3
"""Warm CUDA linalg in this process, then resume the unchanged frozen L10 code.

PyTorch 2.3's lazy CUDA linalg dispatch must not first execute concurrently.
The pilot and formal shell stages are separate processes: the pilot's warmup
does not initialize the formal process. No tensors, thresholds, source manifest,
or old records are rewritten by this launcher.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import threading
import time
import uuid
import torch

EXPERIMENT=Path(__file__).resolve().parents[2]/'experiments/qer_kronecker_l10_v2'


def exercise(device,size=128):
    """Cover eigvalsh/eigh/gesvd/solve using known SPD values, without RNG."""
    if str(device).startswith('cuda'):torch.cuda.set_device(device)
    diagonal=1+torch.arange(1,size+1,device=device,dtype=torch.float64)/size
    matrix=torch.diag(diagonal);rhs=torch.ones((size,3),device=device,dtype=torch.float64)
    vals=torch.linalg.eigvalsh(matrix)
    eigen,vectors=torch.linalg.eigh(matrix)
    options={'driver':'gesvd'} if matrix.is_cuda else {}
    u,s,vh=torch.linalg.svd(matrix,full_matrices=False,**options)
    solution=torch.linalg.solve(matrix,rhs)
    checks=dict(eigenvalues=float((vals-diagonal).abs().max()),
        eigen_reconstruction=float(((vectors*eigen)@vectors.T-matrix).norm()/matrix.norm()),
        svd_reconstruction=float(((u*s)@vh-matrix).norm()/matrix.norm()),
        solve=float((matrix@solution-rhs).norm()/rhs.norm()))
    if not all(value<=1e-12 for value in checks.values()):raise RuntimeError(('Linalg warmup failed',checks))
    if matrix.is_cuda:torch.cuda.synchronize(device)
    return dict(device=str(device),size=size,passed=True,**checks)


def warm_and_probe(devices,probe_size=512,rounds=3):
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError('Initialization must happen on the main thread before workers')
    # Complete all lazy registrations serially, before creating any worker thread.
    serial=[exercise(device) for device in devices]
    barrier=threading.Barrier(len(devices))
    def probe(device):
        results=[]
        for _ in range(rounds):
            barrier.wait(timeout=60)
            results.append(exercise(device,probe_size))
        return results
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futures=[pool.submit(probe,device) for device in devices]
        parallel=[future.result() for future in futures]
    for device in devices:
        if str(device).startswith('cuda'):
            with torch.cuda.device(device):torch.cuda.empty_cache()
    return dict(passed=True,serial=serial,parallel=parallel,
        description='Serial initialization, then concurrent cold-worker validation, in the controller process')


def validate_manifest(config,root):
    sys.path.insert(0,str(EXPERIMENT))
    from common import source_identity,read,require
    manifest=source_identity(config)
    require(read(root/'manifest.json')==manifest,'Frozen experiment code/config mismatch; recovery may not bypass identity')
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('config',type=Path);p.add_argument('output',type=Path)
    p.add_argument('stage',choices=('pilot','run'))
    args=p.parse_args();config=json.loads(args.config.read_text());root=args.output.resolve()
    if sys.flags.optimize:raise RuntimeError('Python -O is forbidden')
    manifest=validate_manifest(config,root)
    from common import save_json,sha_file
    from runtime_io import lock
    if not torch.cuda.is_available() or torch.cuda.device_count()!=2:raise RuntimeError('Require two visible GPUs')
    if not all('4090' in torch.cuda.get_device_name(i) for i in range(2)):raise RuntimeError('Require dual4090')
    torch.set_num_threads(config['cpu_threads']);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    with lock(root):
        path=root/'runtime_recovery'/('linalg_'+uuid.uuid4().hex+'.json')
        record=dict(identity=manifest['identity'],manifest_sha256=sha_file(root/'manifest.json'),
            launcher=str(Path(__file__).resolve()),launcher_sha256=sha_file(__file__),pid=os.getpid(),
            stage=args.stage,started_utc=time.strftime('%FT%TZ',time.gmtime()),torch=torch.__version__,cuda=torch.version.cuda,
            action='Initialize CUDA linalg serially before unchanged frozen controller in same process',
            algorithm_source_unchanged=True,checkpoint_identity_unchanged=True)
        started=time.monotonic();print('SERIAL_LINALG_INITIALIZATION',flush=True)
        try:
            record['validation']=warm_and_probe(['cuda:0','cuda:1'])
            record['passed']=True
        except Exception as exc:
            record.update(passed=False,error=repr(exc));raise
        finally:
            record['prelude_seconds']=time.monotonic()-started;save_json(path,record)
        print('LINALG_COLD_PROCESS_AND_DUAL_WORKER_CHECK_PASS',str(path),flush=True)
    # Deliberately no subprocess: it would lose the initialization just performed.
    import controller
    sys.argv=[str(EXPERIMENT/'controller.py'),str(args.config),str(root),args.stage]
    controller.main()


if __name__=='__main__':main()
