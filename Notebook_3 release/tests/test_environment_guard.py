import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('environment_guard',
    Path(__file__).resolve().parents[1] / 'tools/check_environment.py')
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)

class EnvironmentGuardTests(unittest.TestCase):
    def test_loaded_version_mismatch_requires_restart(self):
        with patch.object(guard.metadata, 'version', return_value='0.0.0'):
            with self.assertRaisesRegex(RuntimeError, 'Restart session'):
                guard.check_numpy()

    def test_numpy_testing_import_failure_requires_restart(self):
        with patch.dict('sys.modules', {'numpy.testing': None}):
            with self.assertRaisesRegex(RuntimeError, 'keep this VM'):
                guard.check_numpy()

if __name__ == '__main__':
    unittest.main()
