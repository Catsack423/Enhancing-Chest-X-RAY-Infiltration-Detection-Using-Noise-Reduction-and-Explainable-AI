import importlib.util
from pathlib import Path
import unittest
import tempfile
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

spec = importlib.util.spec_from_file_location('gpu_batch', Path(__file__).resolve().parents[1]/'tools/dae_gpu_batch.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class GpuBatchTests(unittest.TestCase):
    def test_excludes_over_budget_and_uses_fast_large_batch(self):
        rows = [{'batch_size':16,'peak_bytes':20,'images_per_second':100},
                {'batch_size':32,'peak_bytes':60,'images_per_second':98},
                {'batch_size':64,'peak_bytes':110,'images_per_second':120}]
        self.assertEqual(module.choose_batch(rows,90)['batch_size'],32)
    def test_slower_large_batch_is_not_chosen(self):
        rows = [{'batch_size':16,'peak_bytes':20,'images_per_second':100},
                {'batch_size':32,'peak_bytes':60,'images_per_second':80}]
        self.assertEqual(module.choose_batch(rows,90)['batch_size'],16)
    def test_no_safe_batch_fails(self):
        with self.assertRaisesRegex(RuntimeError,'No DAE batch fits'):
            module.choose_batch([{'batch_size':16,'peak_bytes':100,'images_per_second':1}],90)
    def test_new_batch_uses_separate_outputs_and_preserves_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            config = {'run_mode':'full','dae':{'batch_size':16}}
            old = SimpleNamespace(base_output_root=base/'release_full', data_root=base/'data',
                                  cache_root=base/'cache', config=config)
            opened = []
            def new_exp(root,data_root,output_root,**kwargs):
                exp = SimpleNamespace(config=deepcopy(config),cache_root=base/'cache',output_root=output_root/'seed_42',
                                      artifacts=output_root/'seed_42/shared/artifacts',lock=lambda:None)
                opened.append(exp)
                return exp
            with patch('shared.experiment.Experiment.open',side_effect=new_exp), \
                 patch('shared.data.prepare_manifests',return_value={}), \
                 patch('shared.runtime_cache.RuntimeCache') as cache:
                cache.return_value.prepare.return_value = object()
                actual = module.open_batch_runs(ROOT,{42:(old,{})},{'batch_size':64})
            self.assertEqual(old.config['dae']['batch_size'],16)
            self.assertEqual(actual[42][0].config['dae']['batch_size'],64)
            self.assertEqual(actual[42][0].output_root,base/'release_full_dae_b064_gpu/seed_42')
            self.assertNotEqual(actual[42][0].output_root.parent,old.base_output_root)

if __name__ == '__main__':
    unittest.main()
