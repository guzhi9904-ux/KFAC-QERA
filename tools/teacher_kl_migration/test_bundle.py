import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from parent_bundle import extract


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='qer-bundle-test-')
        self.root = Path(self.temp.name).resolve()
        self.assertEqual(self.root.parent, Path(tempfile.gettempdir()).resolve())
        self.addCleanup(self.temp.cleanup)

    def fixture(self, name='exp01/test.txt', body=b'hello', actual=None, link=False):
        record = {'schema': 1, 'file_count': 1, 'bytes': len(body), 'parent_identities': {},
                  'files': {name: {'bytes': len(body), 'sha256': hashlib.sha256(body).hexdigest()}}}
        manifest = json.dumps(record).encode(); path = self.root/'manifest.json'; path.write_bytes(manifest)
        archive = self.root/'test.tar'
        with tarfile.open(archive, 'w') as t:
            item = tarfile.TarInfo(name); item.size = len(body)
            if link: item.type = tarfile.SYMTYPE; item.linkname = '/tmp/outside'; item.size = 0
            t.addfile(item, io.BytesIO(body if actual is None else actual))
            item = tarfile.TarInfo('manifest.json'); item.size = len(manifest); t.addfile(item, io.BytesIO(manifest))
        return archive, path, hashlib.sha256(manifest).hexdigest()

    def test_roundtrip_and_no_overwrite(self):
        args = self.fixture(); destination = self.root/'new'
        extract(*args, destination)
        self.assertEqual((destination/'exp01/test.txt').read_bytes(), b'hello')
        self.assertTrue(json.loads((destination/'migration_verified.json').read_text())['passed'])
        with self.assertRaises(FileExistsError): extract(*args, destination)

    def test_changed_payload_rejected(self):
        args = self.fixture(actual=b'xxxxx')
        with self.assertRaises(AssertionError): extract(*args, self.root/'bad')
        self.assertFalse((self.root/'bad/migration_verified.json').exists())

    def test_path_escape_and_symlink_rejected(self):
        args = self.fixture(name='../outside')
        with self.assertRaises(ValueError): extract(*args, self.root/'escape')
        args = self.fixture(link=True)
        with self.assertRaises(ValueError): extract(*args, self.root/'link')


if __name__ == '__main__': unittest.main(verbosity=2)
