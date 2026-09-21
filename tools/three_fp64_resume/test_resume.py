import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch
import resume_eval as entry
from resume_eval import runtime_config
import resources


class Tests(unittest.TestCase):
    def test_only_runtime_allowance_changes_and_elapsed_is_retained(self):
        original=dict(budget_hours=10.,methods=['Marginal','Token-joint','Sequence-one-step'],disk_limit_GiB=512.)
        changed=runtime_config(original,16.,36001.)
        self.assertEqual(original['budget_hours'],10.)
        self.assertEqual(changed,dict(original,budget_hours=16.))
        with self.assertRaisesRegex(RuntimeError,'already exhausted'):runtime_config(original,16.,57600.)
        with self.assertRaisesRegex(RuntimeError,'cannot reduce'):runtime_config(original,9.,100.)
        with self.assertRaises(RuntimeError):runtime_config(original,float('inf'),100.)
    def test_launcher_preserves_frozen_config_and_cumulative_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config=dict(budget_hours=10.,disk_limit_GiB=512.)
            cp=root/'config.json';entry.save_json(cp,config)
            frozen=entry.identity(config);entry.save_json(root/'manifest.json',frozen)
            entry.save_json(root/'candidate_freeze.json',{})
            entry.save_json(root/'resources.json',dict(identity=frozen['identity'],active_seconds=36001.,timings=[]))
            before=cp.read_bytes();manifest_before=(root/'manifest.json').read_bytes()
            def simulated_runner():
                self.assertEqual(entry.sys.argv[-1],'evaluate')
                resource=entry.runner.Resources(root,frozen['identity'],config)
                self.assertEqual(resource.base,36001.)
                self.assertEqual(resource.config['budget_hours'],16.)
                resource.boundary()
            with patch.object(entry.runner,'main',side_effect=simulated_runner),patch.object(entry.runner,'Resources',resources.Resources),patch.object(resources,'memory_limits',return_value=(None,None)),patch.object(entry.sys,'argv',['resume_eval.py',str(cp),str(root),'--total-hours','16']):
                entry.main()
            self.assertEqual(cp.read_bytes(),before);self.assertEqual((root/'manifest.json').read_bytes(),manifest_before)
            self.assertGreaterEqual(entry.read(root/'resources.json')['active_seconds'],36001.)
            self.assertEqual(len(list((root/'runtime_extensions').glob('*.json'))),1)


if __name__=='__main__':unittest.main(verbosity=2)
