"""One-worker lock, cumulative active wall budget and atomic-boundary stops."""
import contextlib
import os
import shutil
import signal
import socket
import time
import torch
from common import read, save_json, require


class BudgetReached(RuntimeError): pass


class Resources:
    def __init__(self, root, identity, hours, disk_gib):
        self.root = root; self.identity = identity; self.hours = hours; self.disk_gib = disk_gib
        self.path = root/'resource_usage.json'; self.started = time.monotonic(); self.stop = None
        self.previous = read(self.path) if self.path.exists() else dict(identity=identity, attempts=[], timings=[])
        require(self.previous['identity'] == identity, 'Resource identity changed')
        self.base_seconds = sum(a['seconds'] for a in self.previous['attempts'])
        self.previous['attempts'].append(dict(pid=os.getpid(), hostname=socket.gethostname(),
            started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), seconds=0., status='RUNNING'))
        self.old_handlers = {}
        for sig in (signal.SIGINT, signal.SIGTERM):
            self.old_handlers[sig] = signal.signal(sig, lambda signum, frame: setattr(self, 'stop', signum))
        self.flush()

    def flush(self, status=None):
        last = self.previous['attempts'][-1]
        last['seconds'] = time.monotonic()-self.started
        if status: last['status'] = status
        used = sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file())
        gpu_peaks=dict(self.previous.get('max_GPU_allocated_GiB',{}))
        if torch.cuda.is_initialized():
            for i in range(torch.cuda.device_count()):
                gpu_peaks[str(i)]=max(gpu_peaks.get(str(i),0),torch.cuda.max_memory_allocated(i)/2**30)
        self.previous.update(active_seconds=self.base_seconds+last['seconds'], budget_hours=self.hours,
            output_GiB=used/2**30, output_limit_GiB=self.disk_gib,
            max_GPU_allocated_GiB=gpu_peaks,
            budget_note='Cumulative active time over attempts; check between atomic operations. A kernel may exceed deadline. Idle time is excluded.')
        try:
            import resource
            self.previous['host_peak_RSS_GiB'] = max(self.previous.get('host_peak_RSS_GiB',0),resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20)
        except ImportError: pass
        save_json(self.path, self.previous)

    def boundary(self):
        self.flush()
        if self.stop is not None: raise BudgetReached('Signal received; checkpointed at atomic boundary')
        if self.previous['active_seconds'] >= self.hours*3600: raise BudgetReached('Cumulative wall budget reached')
        if self.previous['output_GiB'] >= self.disk_gib: raise BudgetReached('Output disk budget reached')
        if shutil.disk_usage(self.root).free < 2*2**30: raise BudgetReached('Less than 2 GiB free disk headroom')

    @contextlib.contextmanager
    def io(self):
        start=time.monotonic()
        try: yield
        finally:
            self.previous['atomic_output_IO_seconds']=self.previous.get('atomic_output_IO_seconds',0.)+time.monotonic()-start

    def estimate(self, modules):
        """Measured stage extrapolation; only filled once relevant observations exist."""
        by_module={}
        stages={'gradient_and_all_projections':256,'dense_SVD_candidate':9,
                'host_staged_Rx':48,'actual_KL':64,'reference_and_self_KL':48}
        for name in modules:
            rows=[r for r in self.previous['timings'] if r.get('module')==name and r['completed']]
            stage_estimates={}
            for stage,cap in stages.items():
                samples=[r['seconds'] for r in rows if r['stage']==stage]
                if samples:
                    stage_estimates[stage]=dict(measured=len(samples),mean_seconds=sum(samples)/len(samples),
                        conservative_remaining_units=max(0,cap-len(samples)),
                        remaining_seconds=max(0,cap-len(samples))*sum(samples)/len(samples))
            by_module[name]=stage_estimates
        self.previous['stage_extrapolation']=by_module
        self.previous['estimate_note']='Extrapolate measured stages only; no estimate for unobserved modules. KL upper bound four unique candidates; extra reload/label/I/O/replay overhead remains. Completed retries can inflate measurement counts.'
        self.flush()

    @contextlib.contextmanager
    def timed(self, stage, **details):
        self.boundary(); start = time.monotonic(); passed = False
        print(time.strftime('%F %T'), 'START', stage, details, flush=True)
        try:
            yield
            passed = True
        finally:
            if torch.cuda.is_initialized():
                for i in range(torch.cuda.device_count()): torch.cuda.synchronize(i)
            seconds = time.monotonic()-start
            self.previous['timings'].append(dict(stage=stage, seconds=seconds, completed=passed, **details))
            self.flush()
            print(time.strftime('%F %T'), 'COST', stage, round(seconds, 3), details, flush=True)

    def close(self, status):
        self.flush(status)
        for sig, handler in self.old_handlers.items(): signal.signal(sig, handler)


@contextlib.contextmanager
def single_worker(root):
    # POSIX advisory lock is released by the OS even after SIGKILL; the file is harmless.
    import fcntl
    root.mkdir(parents=True, exist_ok=True)
    with (root/'.run.lock').open('a') as f:
        try: fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('Another process holds this run directory') from None
        try: yield
        finally: fcntl.flock(f.fileno(), fcntl.LOCK_UN)
