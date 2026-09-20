"""Thread-safe stage timings and cumulative resource bounds."""
import contextlib
import shutil
import threading
import time
import torch
from bridge import read,save_json,require,memory_limits


class Resources:
    def __init__(self,root,identity,config):
        self.root=root;self.identity=identity;self.config=config;self.mutex=threading.RLock()
        self.started=time.monotonic();self.last_check=0.;self.path=root/'resources.json'
        self.data=read(self.path) if self.path.exists() else dict(identity=identity,active_seconds=0.,timings=[])
        require(self.data['identity']==identity,'Resource identity changed');self.base=self.data['active_seconds']
    def flush(self):
        with self.mutex:
            self.data['active_seconds']=self.base+time.monotonic()-self.started
            if torch.cuda.is_initialized():
                self.data['GPU_peak_allocated_GiB']={str(i):max(self.data.get('GPU_peak_allocated_GiB',{}).get(str(i),0.),torch.cuda.max_memory_allocated(i)/2**30) for i in range(torch.cuda.device_count())}
            save_json(self.path,self.data)
    def boundary(self):
        with self.mutex:
            require(self.base+time.monotonic()-self.started<3600*self.config['budget_hours'],'Cumulative time limit reached; resume only with unchanged scientific configuration')
            if time.monotonic()-self.last_check<30:return
            self.last_check=time.monotonic();limit,current=memory_limits()
            require(limit is None or current<limit*.85,'Host/cgroup memory exceeds85%')
            require(shutil.disk_usage(self.root).free>32*2**30,'Insufficient disk recovery headroom')
            used=sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file())/2**30
            require(used<self.config['disk_limit_GiB'],'Run disk limit reached')
            self.data.update(output_GiB=used,memory_current_GiB=None if current is None else current/2**30);self.flush()
    @contextlib.contextmanager
    def timed(self,stage,**details):
        self.boundary();start=time.monotonic();passed=False
        print(time.strftime('%F %T'),'START',stage,details,flush=True)
        try:yield;passed=True
        finally:
            if torch.cuda.is_initialized():
                devices=[details['device']] if 'device' in details else range(torch.cuda.device_count())
                for device in devices:torch.cuda.synchronize(device)
            seconds=time.monotonic()-start
            with self.mutex:
                self.data['timings'].append(dict(stage=stage,seconds=seconds,completed=passed,**details));self.flush()
            print(time.strftime('%F %T'),'COST',stage,round(seconds,3),details,flush=True)


class DeviceResources:
    def __init__(self,parent,device,module,stop=None):self.parent=parent;self.device=device;self.module=module;self.stop=stop
    def boundary(self):
        require(self.stop is None or not self.stop.is_set(),'Another offline worker failed; stopping at a checkpoint boundary')
        self.parent.boundary()
    def timed(self,stage,**details):
        self.boundary();return self.parent.timed(stage,device=self.device,module=self.module,**details)
