import contextlib
import io
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from shared import training

def observer():
    source = (ROOT / 'train_dae_progress_cell.txt').read_text(encoding='utf-8')
    namespace = {}
    exec(source.split('\nTRAIN_STAGE = ')[0], namespace)
    return namespace['training_progress']

class TrainingProgressTests(unittest.TestCase):
    def test_original_dae_updates_and_rng_preserved(self):
        model = torch.nn.Linear(1, 1)
        other = torch.nn.Linear(1, 1)
        other.load_state_dict(model.state_dict())
        batches = [(torch.ones(2, 1), torch.zeros(2, 1)),
                   (torch.ones(1, 1), torch.zeros(1, 1))]
        criterion = torch.nn.MSELoss()
        original = training._reconstruction_epoch
        expected = original(model, batches, criterion, 'cpu', torch.optim.SGD(model.parameters(), .1), training._scaler('cpu', False))
        state = torch.get_rng_state().clone()
        with contextlib.redirect_stdout(io.StringIO()):
            with observer()(42, 'dae', interval=.01):
                actual = training._reconstruction_epoch(other, batches, criterion, 'cpu', torch.optim.SGD(other.parameters(), .1), training._scaler('cpu', False))
        self.assertEqual(expected, actual)
        for a, b in zip(model.parameters(), other.parameters()):
            self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertIs(training._reconstruction_epoch, original)

    def test_heartbeat_reports_completed_batches(self):
        token = object()
        def slow(model, batches, *args, **kwargs):
            for batch in batches:
                time.sleep(.025)
            time.sleep(.025)
            return token
        output = io.StringIO()
        with patch.object(training, '_reconstruction_epoch', slow):
            with contextlib.redirect_stdout(output):
                with observer()(42, 'dae', interval=.005):
                    actual = training._reconstruction_epoch(None, [(torch.ones(2, 1), None)], None, 'cpu')
            self.assertIs(training._reconstruction_epoch, slow)
        self.assertIs(actual, token)
        self.assertIn('batch 1/1', output.getvalue())
        self.assertIn('images/s', output.getvalue())

    def test_exception_restores_all_hooks(self):
        names = ('train_dae', 'train_condition', 'save_torch', '_completed', '_reconstruction_epoch')
        originals = {name: getattr(training, name) for name in names}
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, 'original failure'):
                with observer()(42, 'dae'):
                    raise ValueError('original failure')
        for name in names:
            self.assertIs(getattr(training, name), originals[name])

if __name__ == '__main__':
    unittest.main()
