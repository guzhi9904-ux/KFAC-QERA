import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from failure_reporting import worker_failure


def job(code):
    return SimpleNamespace(returncode=code, poll=lambda: code)


class FailureTests(unittest.TestCase):
    def test_failure_includes_actual_cause_and_path(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'pilot.log'
            path.write_text('x'*10000+'\nAssertionError: Parent S replay\n', encoding='utf-8')
            error = str(worker_failure('pilot', [job(1), job(0), job(None)], [path]*3))
            self.assertIn(str(path), error)
            self.assertIn('AssertionError: Parent S replay', error)
            self.assertEqual(error.count('Worker exit='), 1)
            self.assertLess(len(error), 9000)

    def test_missing_log_retains_exit_status(self):
        with tempfile.TemporaryDirectory() as folder:
            error = str(worker_failure('pilot', [job(-9)], [Path(folder)/'missing']))
            self.assertIn('exit=-9', error)
            self.assertIn('Cannot read worker log', error)


if __name__ == '__main__':
    unittest.main(verbosity=2)
