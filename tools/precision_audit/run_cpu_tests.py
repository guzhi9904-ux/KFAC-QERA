"""Run frozen standalone tests from their consolidated repository location."""
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
root = Path(__file__).resolve().parent
sys.path.insert(0, str(root.parents[1] / "experiments"))
sys.path.insert(0, str(root))
suite = unittest.defaultTestLoader.discover(str(root), pattern="test_*.py")
raise SystemExit(not unittest.TextTestRunner(verbosity=1).run(suite).wasSuccessful())
