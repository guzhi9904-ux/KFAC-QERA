"""Verify archived payload bytes, including frozen helper dependencies."""
import hashlib
from pathlib import Path
import unittest


class ReleaseIntegrityTests(unittest.TestCase):
    def test_frozen_payload_manifest(self):
        root = Path(__file__).resolve().parent
        entries = (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
        self.assertGreater(len(entries), 10)
        names = set()
        for line in entries:
            digest, name = line.split("  ", 1)
            self.assertEqual(Path(name).name, name)
            self.assertNotIn(name, names)
            names.add(name)
            self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), digest, name)


if __name__ == "__main__":
    unittest.main()
