import shutil
import time
from ko_common import require
from resources import Resources as Base
from bridge import memory_limits


class Resources(Base):
    """Keep caps, but scan only this new run every10min, not old338GiB every30s."""
    def __init__(self,*args):
        super().__init__(*args);self.last_disk=0.;self.stop=False
    def boundary(self):
        with self.mutex:
            require(not self.stop,'Stop requested; committed data can be resumed')
            require(self.base+time.monotonic()-self.started<3600*self.config['budget_hours'],'Cumulative time limit reached')
            now=time.monotonic()
            if now-self.last_check<30:return
            self.last_check=now;limit,current=memory_limits()
            require(limit is None or current<.85*limit,'Host/cgroup memory exceeds85%')
            require(shutil.disk_usage(self.root).free>32*2**30,'Insufficient disk recovery headroom')
            if now-self.last_disk>=600:
                used=sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file())/2**30
                require(used<self.config['disk_limit_GiB'],'New-run disk cap exceeded')
                self.data['output_GiB']=used;self.last_disk=now
            self.data['memory_current_GiB']=None if current is None else current/2**30
            self.flush()
