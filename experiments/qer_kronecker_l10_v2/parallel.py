"""Two offline CUDA workers; one teacher process; disjoint candidate writes."""
import contextlib
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import torch


class WorkerResources:
    def __init__(self,parent,mutex,device):
        self.parent=parent;self.mutex=mutex;self.device=device
    def boundary(self):
        with self.mutex:self.parent.boundary()
    @contextlib.contextmanager
    def timed(self,stage,**details):
        self.boundary();start=time.monotonic();passed=False
        print(time.strftime('%F %T'),'START',stage,dict(device=self.device,**details),flush=True)
        try:yield;passed=True
        finally:
            if self.device.startswith('cuda'):torch.cuda.synchronize(self.device)
            elapsed=time.monotonic()-start
            with self.mutex:
                self.parent.data['timings'].append(dict(stage=stage,seconds=elapsed,completed=passed,device=self.device,**details))
                self.parent.flush()
            print(time.strftime('%F %T'),'COST',stage,round(elapsed,3),dict(device=self.device,**details),flush=True)


def offline_map(e,items,function):
    """Main-thread caller owns all shared writes before/after this join barrier."""
    if e.config.get('offline_workers',1)==1:
        return [function(e,item) for item in items]
    from experiment import Experiment
    mutex=threading.RLock();queue=list(enumerate(items));result=[None]*len(items);cancel=threading.Event()
    devices=e.config.get('offline_devices',['cuda:0','cuda:1'])
    def consume(device):
        if device.startswith('cuda'):torch.cuda.set_device(device)
        resources=WorkerResources(e.resources,mutex,device)
        worker=Experiment(e.root,e.config,e.identity,resources,device=device)
        try:
            while not cancel.is_set():
                with mutex:
                    if not queue:break
                    index,item=queue.pop(0)
                result[index]=function(worker,item)
        except BaseException:
            cancel.set();raise
        finally:worker.store.clear()
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures=[executor.submit(consume,d) for d in devices]
        for future in futures:future.result()
    return result
