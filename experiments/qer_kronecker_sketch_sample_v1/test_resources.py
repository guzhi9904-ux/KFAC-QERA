"""CPU-only resource tests; every filesystem probe uses a read-only memory fixture."""
import contextlib
from pathlib import PurePosixPath
import unittest
from unittest.mock import patch

import runtime_io as runtime


GIB=2**30


@contextlib.contextmanager
def filesystem(files):
    probes=[]

    class ReadOnlyPath(PurePosixPath):
        def exists(self):
            probes.append(str(self))
            return str(self) in files

        def read_text(self):
            probes.append(str(self))
            if str(self) not in files:raise FileNotFoundError(str(self))
            return files[str(self)]

    with patch.object(runtime,'Path',ReadOnlyPath):
        yield probes


def meminfo(total=256,used=80):
    return {'/proc/meminfo':f'MemTotal: {total*2**20} kB\nMemAvailable: {(total-used)*2**20} kB\n'}


def group(path,version,limit,used):
    names=('memory.max','memory.current') if version==2 else ('memory.limit_in_bytes','memory.usage_in_bytes')
    value=str(limit*GIB) if isinstance(limit,int) else limit
    return {path+'/'+names[0]:value,path+'/'+names[1]:str(used*GIB)}


class MemoryLimitsTests(unittest.TestCase):
    def check(self,files,expected):
        with filesystem(files) as probes:
            self.assertEqual(runtime.memory_limits(),tuple(v*GIB for v in expected))
        return probes

    def v2(self,membership='/team/job',root='/',mount='/sys/fs/cgroup'):
        return {'/proc/self/cgroup':f'0::{membership}\n',
                '/proc/self/mountinfo':f'30 20 0:25 {root} {mount} rw - cgroup2 cgroup rw\n',**meminfo()}

    def v1(self,membership='/team/job',root='/',mount='/sys/fs/cgroup/memory'):
        return {'/proc/self/cgroup':f'3:cpu,cpuacct:/other\n5:memory:{membership}\n',
                '/proc/self/mountinfo':f'29 20 0:24 / /sys/fs/cgroup/cpu rw - cgroup cgroup rw,cpu,cpuacct\n30 20 0:25 {root} {mount} rw - cgroup cgroup rw,memory\n',**meminfo()}

    def test_v2_nested_limit_not_mount_root(self):
        files={**self.v2(),**group('/sys/fs/cgroup/team/job',2,64,60),
               **group('/sys/fs/cgroup/team',2,128,70),**group('/sys/fs/cgroup',2,'max',80)}
        self.check(files,(64,60))

    def test_v2_parent_cap_uses_parent_usage_including_siblings(self):
        files={**self.v2(),**group('/sys/fs/cgroup/team/job',2,128,8),
               **group('/sys/fs/cgroup/team',2,64,60),**group('/sys/fs/cgroup',2,'max',80)}
        self.check(files,(64,60))
        with filesystem(files):
            resources=object.__new__(runtime.Resources)
            resources.stop=None;resources.base=0.;resources.hours=1.;resources.started=0.
            resources.last_check=0.;resources.flush=lambda:None
            with patch.object(runtime.time,'monotonic',return_value=1.):
                with self.assertRaisesRegex(runtime.BudgetReached,'85%'):
                    resources.boundary(force=True)

    def test_v2_unlimited_leaf_still_obeys_grandparent(self):
        files={**self.v2(),**group('/sys/fs/cgroup/team/job',2,'max',4),
               **group('/sys/fs/cgroup/team',2,'max',5),**group('/sys/fs/cgroup',2,32,29)}
        self.check(files,(32,29))

    def test_v2_nonroot_mount_stops_at_mount_boundary(self):
        files={**self.v2('/tenant/job','/tenant','/run/memory'),
               **group('/run/memory/job',2,128,8),**group('/run/memory',2,64,60),
               **group('/run',2,1,1)}
        probes=self.check(files,(64,60))
        self.assertNotIn('/run/memory.max',probes)

    def test_namespace_root_at_nonroot_mount(self):
        files={**self.v2('/','/tenant','/run/memory'),**group('/run/memory',2,64,60)}
        self.check(files,(64,60))

    def test_root_membership_ignores_unrelated_bind_when_root_mount_exists(self):
        files={**self.v2('/'),**group('/sys/fs/cgroup',2,64,60),**group('/unrelated',2,1,1)}
        files['/proc/self/mountinfo']+='31 20 0:25 /other /unrelated rw - cgroup2 cgroup rw\n'
        probes=self.check(files,(64,60))
        self.assertFalse(any(p.startswith('/unrelated/') for p in probes))

    def test_escaped_mount_root_and_mountpoint(self):
        files={**self.v2('/tenant space/job',r'/tenant\040space',r'/run/mem\040space'),
               **group('/run/mem space/job',2,'max',8),**group('/run/mem space',2,64,60)}
        self.check(files,(64,60))

    def test_v1_nested_memory_controller_and_parent_minimum(self):
        files={**self.v1(),**group('/sys/fs/cgroup/memory/team/job',1,128,8),
               **group('/sys/fs/cgroup/memory/team',1,64,60),
               '/sys/fs/cgroup/memory/team/memory.use_hierarchy':'1',
               **group('/sys/fs/cgroup/cpu/other',1,1,1)}
        probes=self.check(files,(64,60))
        self.assertFalse(any(p.startswith('/sys/fs/cgroup/cpu/') for p in probes))

    def test_v1_nonroot_mount(self):
        files={**self.v1('/tenant/job','/tenant','/run/mem'),
               **group('/run/mem/job',1,'9223372036854771712',8),**group('/run/mem',1,64,60)}
        probes=self.check(files,(64,60))
        self.assertNotIn('/run/memory.limit_in_bytes',probes)

    def test_v1_nonhierarchical_parent_is_not_inherited(self):
        files={**self.v1(),**group('/sys/fs/cgroup/memory/team/job',1,64,8),
               **group('/sys/fs/cgroup/memory/team',1,32,30),
               '/sys/fs/cgroup/memory/team/memory.use_hierarchy':'0'}
        self.check(files,(64,8))

    def test_v1_unlimited_falls_back_to_host(self):
        files={**self.v1(),**group('/sys/fs/cgroup/memory/team/job',1,'9223372036854771712',8)}
        self.check(files,(256,80))

    def test_physical_ram_caps_larger_cgroup_limit(self):
        files={**self.v2(),**meminfo(64,60),**group('/sys/fs/cgroup/team/job',2,128,8)}
        self.check(files,(64,60))

    def test_equal_caps_choose_larger_usage(self):
        files={**self.v2(),**group('/sys/fs/cgroup/team/job',2,64,8),**group('/sys/fs/cgroup/team',2,64,60)}
        self.check(files,(64,60))

    def test_unrelated_subtree_mount_is_not_used(self):
        files={**self.v2('/tenant/job','/other','/run/mem'),**group('/run/mem',2,1,1)}
        probes=self.check(files,(256,80))
        self.assertFalse(any(p.startswith('/run/mem/') for p in probes))

    def test_second_mount_can_expose_restrictive_ancestor(self):
        files={**self.v2('/tenant/job','/tenant','/run/mem'),
               **group('/run/mem/job',2,128,8),**group('/run/mem',2,64,20),
               **group('/all',2,32,30)}
        files['/proc/self/mountinfo']+='31 20 0:25 / /all rw - cgroup2 cgroup rw\n'
        self.check(files,(32,30))

    def test_missing_usage_for_known_limit_fails_closed(self):
        files={**self.v2(),'/sys/fs/cgroup/team/job/memory.max':str(64*GIB)}
        with filesystem(files):
            with self.assertRaises(FileNotFoundError):runtime.memory_limits()

    def test_legacy_root_paths_when_proc_metadata_absent(self):
        self.check({**meminfo(),**group('/sys/fs/cgroup',2,64,60)},(64,60))
        self.check({**meminfo(),**group('/sys/fs/cgroup/memory',1,64,60)},(64,60))

    def test_zero_limit_is_preserved(self):
        self.check({**self.v2(),**group('/sys/fs/cgroup/team/job',2,0,0)},(0,0))

    def test_no_memory_information(self):
        with filesystem({}):self.assertEqual(runtime.memory_limits(),(None,None))


if __name__=='__main__':unittest.main()
