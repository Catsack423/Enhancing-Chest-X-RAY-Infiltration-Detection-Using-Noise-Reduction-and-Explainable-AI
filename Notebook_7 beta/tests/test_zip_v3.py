from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib
import json
import shutil
import unittest
from unittest.mock import patch
import zipfile

from v7.colab import prepare_from_v3_zip
from v7.release.experiment import file_hash

ROOT = Path(__file__).resolve().parents[1]


def make_catalog(root, archives, files):
    archives.mkdir(parents=True)
    records = []
    groups = {}
    for name, payload in files.items():
        groups.setdefault(name.split('/')[0] if name.endswith('.png') else 'metadata', {})[name] = payload
    for group, contents in groups.items():
        target = archives / (group + '.zip')
        with zipfile.ZipFile(target, 'w', zipfile.ZIP_STORED) as archive:
            for name, payload in contents.items():
                archive.writestr(name, payload)
        records.append({'name': target.name, 'bytes': target.stat().st_size, 'sha256': file_hash(target),
            'unpacked_bytes': sum(map(len, contents.values())),
            'files': {name: {'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}
                      for name, payload in contents.items()}})
    catalog = {'schema': 1, 'images': sum(name.endswith('.png') for name in files),
        'manifest_sha256': {p.name: file_hash(p) for p in (root / 'v3_zip_reference/manifests').glob('*.csv')},
        'archives': records}
    (archives / 'dataset_zip_index.json').write_text(json.dumps(catalog))
    return catalog


class V3ZipTests(unittest.TestCase):
    def test_legacy_loader_bytes_and_reference_hashes(self):
        self.assertEqual(file_hash(ROOT / 'v3_zip_reference/stage_dataset.py'),
                         '5870c0c32c79d62c2170ee9431cc435f586097540ac2b195ba84532ed60a9265')
        for name in ('train', 'validation', 'test', 'xai'):
            self.assertEqual(file_hash(ROOT / f'partition_reference/{name}.csv'),
                             file_hash(ROOT / f'v3_zip_reference/manifests/{name}.csv'))

    def test_zip_stage_resume_repair_corruption_and_escape(self):
        with TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, archives, vm, work = [base / name for name in ('root', 'archives', 'vm', 'work')]
            (root / 'configs').mkdir(parents=True)
            (root / 'configs/protocol.json').write_text(json.dumps({'expected_audit_images': 1, 'audit_scope': 'v3_zip_originals'}))
            (root / 'partition_reference').mkdir()
            (root / 'v3_zip_reference/manifests').mkdir(parents=True)
            shutil.copyfile(ROOT / 'v3_zip_reference/stage_dataset.py', root / 'v3_zip_reference/stage_dataset.py')
            for name in ('train', 'validation', 'test', 'xai', 'background'):
                payload = 'image\na.png\n'
                (root / f'v3_zip_reference/manifests/{name}.csv').write_text(payload)
                (root / f'partition_reference/{name}.csv').write_text(payload)
            relative = 'images_001/images/a.png'
            files = {relative: b'original image bytes', 'Data_Entry_2017.csv': b'original metadata',
                     'train_val_list.txt': b'a.png', 'test_list.txt': b''}
            catalog = make_catalog(root, archives, files)
            with patch('v7.colab.prepare_on_drive') as engine:
                prepare_from_v3_zip(root, archives, vm, work)
                self.assertEqual(engine.call_args.kwargs['image_ids'], ['a.png'])
                self.assertEqual((vm / relative).read_bytes(), files[relative])
                prepare_from_v3_zip(root, archives, vm, work)
                self.assertFalse(list((vm / '.zip_state').glob('*.zip')))
                (vm / relative).write_bytes(b'corrupt local')
                prepare_from_v3_zip(root, archives, vm, work)
                self.assertEqual((vm / relative).read_bytes(), files[relative])
                (vm / relative).unlink()
                part = archives / 'images_001.zip'
                saved = part.read_bytes()
                part.write_bytes(b'corrupt ZIP')
                with self.assertRaisesRegex(ValueError, 'Incomplete/corrupt ZIP'):
                    prepare_from_v3_zip(root, archives, vm, work)
                self.assertFalse((vm / relative).exists())
                part.write_bytes(saved)
                catalog['archives'][0]['files']['../escape'] = {'bytes': 0, 'sha256': '0'*64}
                (archives / 'dataset_zip_index.json').write_text(json.dumps(catalog))
                with self.assertRaisesRegex(ValueError, 'Unsafe ZIP member'):
                    prepare_from_v3_zip(root, archives, vm, work)
