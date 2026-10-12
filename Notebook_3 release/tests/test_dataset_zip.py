import importlib.util
from pathlib import Path
import tempfile
import unittest
import json

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
def load(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class DatasetZipTests(unittest.TestCase):
    def test_roundtrip_resume_and_corruption(self):
        packer, stager = load('pack_dataset'), load('stage_dataset')
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root, data, output, vm = [base / p for p in ('root', 'data', 'zip', 'vm')]
            (root / 'manifests').mkdir(parents=True)
            relative = 'images_001/images/a.png'
            (data / relative).parent.mkdir(parents=True)
            (data / relative).write_bytes(b'original PNG bytes')
            for name in ('Data_Entry_2017.csv', 'BBox_List_2017.csv', 'train_val_list.txt', 'test_list.txt'):
                (data / name).write_bytes(name.encode())
            (root / 'manifests/train.csv').write_text('image,relative_path\na.png,' + relative + '\n')
            catalog = packer.pack(root, data, output)
            self.assertEqual(catalog['images'], 1)
            self.assertEqual(stager.stage_dataset(root, output, vm), vm.resolve())
            self.assertEqual((vm / relative).read_bytes(), (data / relative).read_bytes())
            stager.stage_dataset(root, output, vm)
            (vm / relative).write_bytes(b'corrupt')
            stager.stage_dataset(root, output, vm)
            self.assertEqual((vm / relative).read_bytes(), (data / relative).read_bytes())
            self.assertFalse(list((vm / '.zip_state').glob('*.zip')))
            # Missing local image plus damaged transfer must fail before publishing it.
            (vm / relative).unlink()
            archive_path = output / catalog['archives'][0]['name']
            original_archive = archive_path.read_bytes()
            archive_path.write_bytes(b'corrupt ZIP')
            with self.assertRaisesRegex(ValueError, 'Incomplete/corrupt ZIP'):
                stager.stage_dataset(root, output, vm)
            self.assertFalse((vm / relative).exists())
            archive_path.write_bytes(original_archive)
            stager.stage_dataset(root, output, vm)
            index = output / 'dataset_zip_index.json'
            catalog['archives'][0]['files']['../escape'] = {'bytes': 0, 'sha256': '0'*64}
            index.write_text(json.dumps(catalog))
            with self.assertRaisesRegex(ValueError, 'Unsafe ZIP member'):
                stager.stage_dataset(root, output, vm)

if __name__ == '__main__':
    unittest.main()
