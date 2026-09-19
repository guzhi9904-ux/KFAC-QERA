import json
from pathlib import Path
import struct
import tempfile
import unittest
from preflight_4090 import model_info, scan, tensor_header


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='qer-metadata-test-')
        self.root = Path(self.temp.name).resolve()
        self.assertEqual(self.root.parent, Path(tempfile.gettempdir()).resolve())
        self.addCleanup(self.temp.cleanup)

    def tensor(self, name='model.safetensors', offsets=(0, 8)):
        header = json.dumps({'weight': {'shape': [2, 2], 'dtype': 'BF16', 'data_offsets': offsets}}).encode()
        (self.root/name).write_bytes(struct.pack('<Q', len(header)) + header + b'\x00'*8)

    def test_header_and_shape_do_not_claim_weight_identity(self):
        self.tensor()
        (self.root/'config.json').write_text('{"model_type":"llama"}')
        r = model_info(self.root/'config.json')
        self.assertEqual(r['config']['model_type'], 'llama')
        self.assertEqual(r['tensor_count'], 1)
        self.assertFalse(r['weights_content_verified'])
        self.assertFalse(r['download_manifest_present'])
        self.assertEqual(r['fp32_parameter_GiB'], 16/2**30)

    def test_truncated_and_oversized_tensor_offsets_rejected(self):
        self.tensor(offsets=(0, 80))
        with self.assertRaises(ValueError): tensor_header(self.root/'model.safetensors')
        (self.root/'model.safetensors').write_bytes(b'x')
        with self.assertRaises(ValueError): tensor_header(self.root/'model.safetensors')

    def test_checkpoint_index_traversal_rejected(self):
        (self.root/'config.json').write_text('{}')
        (self.root/'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'weight': '../model.safetensors'}}))
        with self.assertRaises(ValueError):
            model_info(self.root/'config.json')

    def test_index_header_mismatch_rejected(self):
        self.tensor()
        (self.root/'config.json').write_text('{}')
        (self.root/'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'another_weight': 'model.safetensors'}}))
        with self.assertRaises(ValueError):
            model_info(self.root/'config.json')

    def test_bounded_scan_reports_truncation_and_absence(self):
        for i in range(5):
            d = self.root/str(i); d.mkdir(); (d/'config.json').write_text('{}')
        _, result = scan(self.root, 'config.json', max_entries=2)
        self.assertTrue(result['truncated'])
        found, result = scan(self.root/'absent', 'identity.json')
        self.assertEqual(found, [])
        self.assertTrue(result['errors'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
