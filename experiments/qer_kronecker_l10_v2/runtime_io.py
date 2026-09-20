"""Immutable tensor commits, verified bounded CPU LRU, resource budget and process lock."""
import collections
import contextlib
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import socket
import subprocess
import time
import torch
from common import PLAN,read,save_json,save_tensors,read_tensors,sha_file,mo,require


class BudgetReached(RuntimeError):pass


class TensorStore:
    def __init__(self,identity,cache_gib=4):
        self.identity=identity;self.cap=cache_gib*2**30;self.used=0;self.cache=collections.OrderedDict();self.verified={}
    def clear(self):self.cache.clear();self.used=0
    def put(self,path,tensors,**metadata):
        path=Path(path)
        if path.exists():
            old,meta=self.get(path)
            require(all(mo.digest_tensor(v)==meta['tensor_hashes'].get(k) for k,v in tensors.items()),'Attempt to overwrite tensor commit')
            return meta
        meta=dict(identity=self.identity,tensor_hashes={k:mo.digest_tensor(v) for k,v in tensors.items()},**metadata)
        save_tensors(path,tensors,meta)
        receipt=dict(file_sha256=sha_file(path),**meta)
        save_json(path.with_suffix('.json'),receipt)
        self.verified[str(path)]=(path.stat().st_size,path.stat().st_mtime_ns,receipt['file_sha256'])
        return receipt
    def get(self,path):
        path=Path(path);key=str(path);stat=path.stat();signature=(stat.st_size,stat.st_mtime_ns)
        previous=self.verified.get(key)
        if previous is not None:require(signature==previous[:2],'Frozen tensor file changed: '+key)
        if key in self.cache:
            self.cache.move_to_end(key);return self.cache[key][0:2]
        t,m=read_tensors(path);require(m['identity']==self.identity,'Tensor identity mismatch: '+key)
        if previous is None:
            require(all(mo.digest_tensor(v)==m['tensor_hashes'].get(k) for k,v in t.items()),'Tensor content hash mismatch')
            filehash=sha_file(path);receipt=path.with_suffix('.json')
            if receipt.exists():require(read(receipt)==dict(file_sha256=filehash,**m),'Tensor receipt changed')
            else:save_json(receipt,dict(file_sha256=filehash,**m))
            self.verified[key]=(*signature,filehash)
        size=sum(v.numel()*v.element_size() for v in t.values())
        if size<=self.cap:
            while self.cache and self.used+size>self.cap:
                _,(_,_,n)=self.cache.popitem(last=False);self.used-=n
            self.cache[key]=(t,m,size);self.used+=size
        return t,m


def memory_limits():
    """Smallest visible memory cap, paired with usage from that same scope.

    Resolve cgroup membership through mountinfo (including bind-mount roots),
    then inspect ancestors only as far as each mount exposes. Hidden ancestors
    cannot be inspected. Physical RAM also caps an otherwise unlimited cgroup.
    """
    pairs=[];seen=set()

    def add_group(directory,version,inherited=False):
        limit=directory/('memory.max' if version==2 else 'memory.limit_in_bytes')
        if str(limit) in seen:return
        seen.add(str(limit))
        # A v1 parent with hierarchical accounting disabled does not cap children.
        hierarchy=directory/'memory.use_hierarchy'
        if version==1 and inherited and hierarchy.exists() and hierarchy.read_text().strip()=='0':return
        if not limit.exists():return
        raw=limit.read_text().strip()
        if raw=='max':return
        total=int(raw)
        # v1 represents "unlimited" by a page-aligned LONG_MAX on these 64-bit hosts.
        if version==1 and total>=1<<60:return
        require(total>=0,'Negative cgroup memory limit')
        usage=directory/('memory.current' if version==2 else 'memory.usage_in_bytes')
        current=int(usage.read_text())  # A known cap without readable usage must fail closed.
        require(current>=0,'Negative cgroup memory usage')
        pairs.append((total,current))

    membership=Path('/proc/self/cgroup');mountinfo=Path('/proc/self/mountinfo')
    if membership.exists() and mountinfo.exists():
        groups={}
        for line in membership.read_text().splitlines():
            _,controllers,name=line.split(':',2)
            version=2 if not controllers else 1 if 'memory' in controllers.split(',') else None
            if version is not None:groups[version]=PurePosixPath(name)
        # mountinfo encodes whitespace and backslashes using octal escapes.
        unescape=lambda value:re.sub(r'\\([0-7]{3})',lambda match:chr(int(match[1],8)),value)
        mounts=[]
        for line in mountinfo.read_text().splitlines():
            before,separator,after=line.partition(' - ')
            if not separator:continue
            fields=before.split();filesystem=after.split()
            if len(fields)<6 or len(filesystem)<3:continue
            version=2 if filesystem[0]=='cgroup2' else 1 if filesystem[0]=='cgroup' else None
            if version not in groups:continue
            if version==1 and 'memory' not in set(','.join((fields[5],filesystem[1],filesystem[2])).split(',')):continue
            group=groups[version];mount_root=PurePosixPath(unescape(fields[3]));mount=PurePosixPath(unescape(fields[4]))
            if any(not p.is_absolute() or '..' in p.parts for p in (group,mount_root,mount)):continue
            mounts.append((version,group,mount_root,mount))
        rooted={version for version,group,mount_root,_ in mounts if group.is_relative_to(mount_root)}
        for version,group,mount_root,mount in mounts:
            if group.is_relative_to(mount_root):relative=group.relative_to(mount_root)
            elif group==PurePosixPath('/') and version not in rooted:
                # A cgroup namespace may report / for a non-root subtree mount.
                relative=PurePosixPath('.')
            else:continue  # This mount does not expose the process's cgroup.
            leaf=mount/relative;directory=leaf
            while True:
                add_group(Path(str(directory)),version,inherited=directory!=leaf)
                if directory==mount:break
                directory=directory.parent
    else:
        # Preserve the old mount-root fallback when proc membership metadata is absent.
        add_group(Path('/sys/fs/cgroup'),2)
        add_group(Path('/sys/fs/cgroup/memory'),1)
    if Path('/proc/meminfo').exists():
        d={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()}
        pairs.append((d['MemTotal'],d['MemTotal']-d['MemAvailable']))
    # Equal caps can have different usage (e.g. parent plus siblings); use the fuller scope.
    return min(pairs,key=lambda pair:(pair[0],-pair[1])) if pairs else (None,None)


def hardware(root):
    def query(args):return subprocess.check_output(args,text=True).strip()
    limit,current=memory_limits()
    return dict(host=socket.gethostname(),GPU=query(['nvidia-smi','--query-gpu=index,uuid,name,driver_version,memory.total,memory.free,memory.used','--format=csv']),
        processes=query(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv']),
        cgroup=Path('/proc/self/cgroup').read_text(),memory_limit_bytes=limit,memory_current_bytes=current,
        cpu_affinity=len(os.sched_getaffinity(0)),cpu_max=Path('/sys/fs/cgroup/cpu.max').read_text() if Path('/sys/fs/cgroup/cpu.max').exists() else None,
        free_disk_GiB=shutil.disk_usage(root).free/2**30,torch=torch.__version__,cuda=torch.version.cuda)


class Resources:
    def __init__(self,root,identity,hours):
        self.root=root;self.identity=identity;self.hours=hours;self.started=time.monotonic();self.stop=None;self.last_check=0.
        self.path=root/'resource_usage.json';self.data=read(self.path) if self.path.exists() else dict(identity=identity,attempts=[],timings=[])
        require(self.data['identity']==identity,'Resource identity changed')
        self.base=sum(a['seconds'] for a in self.data['attempts']);self.data['attempts'].append(dict(started_utc=time.strftime('%FT%TZ',time.gmtime()),pid=os.getpid(),seconds=0.,status='RUNNING'))
        self.handlers={}
        for sig in (signal.SIGINT,signal.SIGTERM):self.handlers[sig]=signal.signal(sig,lambda number,frame:setattr(self,'stop',number))
        self.flush()
    def flush(self,status=None):
        last=self.data['attempts'][-1];last['seconds']=time.monotonic()-self.started
        if status:last['status']=status
        self.data.update(active_seconds=self.base+last['seconds'],budget_hours=self.hours)
        if torch.cuda.is_initialized():
            peaks=self.data.setdefault('GPU_peaks',{})
            for i in range(torch.cuda.device_count()):
                old=peaks.setdefault(str(i),dict(allocated_GiB=0.,reserved_GiB=0.))
                old['allocated_GiB']=max(old['allocated_GiB'],torch.cuda.max_memory_allocated(i)/2**30)
                old['reserved_GiB']=max(old['reserved_GiB'],torch.cuda.max_memory_reserved(i)/2**30)
        limit,current=memory_limits()
        if current is not None:self.data.update(memory_limit_GiB=limit/2**30,memory_current_GiB=current/2**30,memory_peak_observed_GiB=max(current/2**30,self.data.get('memory_peak_observed_GiB',0)))
        save_json(self.path,self.data)
    def boundary(self,force=False):
        if self.stop:raise BudgetReached('Signal requested; stopping at committed boundary')
        if self.base+time.monotonic()-self.started>=3600*self.hours:raise BudgetReached('Cumulative active time budget reached')
        if not force and time.monotonic()-self.last_check<20:return
        self.last_check=time.monotonic();self.flush()
        limit,current=memory_limits()
        if limit is not None and current>limit*PLAN['host_memory_fraction']:raise BudgetReached('Observed cgroup memory exceeds 85% limit; includes file cache')
        free=shutil.disk_usage(self.root).free/2**30
        if free<PLAN['disk_headroom_GiB']:raise BudgetReached('Disk recovery headroom exhausted')
        used=sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file())/2**30
        self.data.update(output_GiB=used,free_disk_GiB=free)
        if used>PLAN['disk_limit_GiB']:raise BudgetReached('Declared output disk limit exceeded')
    @contextlib.contextmanager
    def timed(self,stage,**details):
        self.boundary();start=time.monotonic();passed=False
        print(time.strftime('%F %T'),'START',stage,details,flush=True)
        try:yield;passed=True
        finally:
            if torch.cuda.is_initialized():
                for i in range(torch.cuda.device_count()):torch.cuda.synchronize(i)
            elapsed=time.monotonic()-start
            self.data['timings'].append(dict(stage=stage,seconds=elapsed,completed=passed,**details));self.flush()
            print(time.strftime('%F %T'),'COST',stage,round(elapsed,3),details,flush=True)
    def close(self,status):
        self.flush(status)
        for sig,handler in self.handlers.items():signal.signal(sig,handler)


@contextlib.contextmanager
def lock(root):
    import fcntl
    root.mkdir(parents=True,exist_ok=True)
    with (root/'.run.lock').open('a') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another worker owns this run') from None
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)
