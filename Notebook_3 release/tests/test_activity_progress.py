import contextlib
import io
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from shared import xai, reporting
from shared.preprocessing import ReleasePreprocessor

def observer():
    source = (ROOT / 'activity_progress_cell.txt').read_text(encoding='utf-8')
    namespace = {}
    exec(source.split('\nPIPELINE_STAGE = ')[0], namespace)
    return namespace['activity_progress']

class ActivityProgressTests(unittest.TestCase):
    def test_explainer_arguments_results_and_hooks_preserved(self):
        token = object()
        inputs = object()
        calls = []
        def explain(value):
            calls.append(value)
            time.sleep(.03)
            return token
        def evaluate(exp, frames, criterion, condition_id, device='cuda', *, gradcam_fn=explain, shap_maps_fn=explain):
            self.assertEqual(device, 'cpu')
            return shap_maps_fn(inputs)
        frames = {'xai': pd.DataFrame({'image': ['a.png']})}
        output = io.StringIO()
        original_prep = ReleasePreprocessor.prepare
        with patch.object(xai, 'evaluate_condition', evaluate):
            with contextlib.redirect_stdout(output):
                with observer()(42, 'shap', interval=.005):
                    result = xai.evaluate_condition(None, frames, 'SHAP', 'baseline', device='cpu')
            self.assertIs(xai.evaluate_condition, evaluate)
        self.assertIs(result, token)
        self.assertEqual(calls, [inputs])
        self.assertIs(ReleasePreprocessor.prepare, original_prep)
        self.assertIn('computing SHAP attributions', output.getvalue())

    def test_report_failure_restores_all_hooks(self):
        originals = (xai.verify_record, xai.publish_file, reporting.report_all,
                     reporting.patient_bootstrap, ReleasePreprocessor.prepare)
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, 'original failure'):
                with observer()(42, 'reports'):
                    raise ValueError('original failure')
        self.assertEqual(originals, (xai.verify_record, xai.publish_file, reporting.report_all,
                                    reporting.patient_bootstrap, ReleasePreprocessor.prepare))

if __name__ == '__main__':
    unittest.main()
