import copy
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
import torch
import resume


class RecoveryTests(unittest.TestCase):
    def test_serial_precedes_workers_and_rng_unchanged(self):
        state=torch.get_rng_state().clone();calls=[]
        original=resume.exercise
        def track(device,size=128):
            calls.append((threading.current_thread() is threading.main_thread(),size))
            return original(device,size)
        with patch.object(resume,'exercise',track):result=resume.warm_and_probe(['cpu','cpu'],probe_size=32,rounds=2)
        self.assertTrue(result['passed']);self.assertEqual(calls[:2],[(True,128),(True,128)])
        self.assertTrue(all(not main for main,size in calls[2:]));self.assertTrue(torch.equal(state,torch.get_rng_state()))
    def test_numerical_failure_stops_before_workers(self):
        with patch.object(resume,'exercise',side_effect=RuntimeError('not accepted')),patch.object(resume,'ThreadPoolExecutor') as pool:
            with self.assertRaisesRegex(RuntimeError,'not accepted'):resume.warm_and_probe(['cpu','cpu'])
            pool.assert_not_called()
    def test_config_and_source_identity_not_bypassed(self):
        import sys
        sys.path.insert(0,str(resume.EXPERIMENT))
        from common import source_identity,save_json
        config={'test':'synthetic'}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest=source_identity(config);save_json(root/'manifest.json',manifest)
            self.assertEqual(resume.validate_manifest(config,root),manifest)
            with self.assertRaisesRegex(RuntimeError,'mismatch'):resume.validate_manifest({'test':'different'},root)
            modified=copy.deepcopy(manifest);modified['source']['parallel.py']='changed';save_json(root/'manifest.json',modified)
            with self.assertRaisesRegex(RuntimeError,'mismatch'):resume.validate_manifest(config,root)

if __name__=='__main__':
    torch.set_num_threads(2);unittest.main(verbosity=2)
