"""Pack frozen selected original PNGs; no resizing, resplitting or recompression."""
import argparse
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import time
import zipfile

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()

def pack(root, data, output):
    root, data, output = map(Path, (root, data, output))
    output.mkdir(parents=True, exist_ok=True)
    selected = {}
    manifest_hashes = {}
    for manifest in sorted((root / 'manifests').glob('*.csv')):
        manifest_hashes[manifest.name] = sha(manifest)
        with manifest.open(encoding='utf-8', newline='') as f:
            for row in csv.DictReader(f):
                relative = row['relative_path']
                path = PurePosixPath(relative)
                if path.is_absolute() or '..' in path.parts or '\\' in relative:
                    raise ValueError('Unsafe manifest path')
                if row['image'] in selected and selected[row['image']] != relative:
                    raise ValueError('Conflicting manifest paths')
                selected[row['image']] = relative
    groups = {}
    for relative in sorted(selected.values()):
        groups.setdefault(relative.split('/')[0], []).append(relative)
    groups['metadata'] = ['Data_Entry_2017.csv', 'BBox_List_2017.csv', 'train_val_list.txt', 'test_list.txt']
    total = sum((data / p).stat().st_size for paths in groups.values() for p in paths)
    print(f'ZIP: {len(selected):,} original images, {total/1024**3:.2f} GiB; PNG stored without recompression', flush=True)
    catalog = {'schema': 1, 'images': len(selected), 'manifest_sha256': manifest_hashes, 'archives': []}
    started = time.monotonic()
    for group, paths in sorted(groups.items()):
        target = output / (group + '.zip')
        record_path = target.with_suffix('.record.json')
        if target.exists() and record_path.exists():
            record = json.loads(record_path.read_text())
            if (record.get('manifest_sha256') == manifest_hashes and record.get('sha256') == sha(target)
                    and list(record.get('files', {})) == paths
                    and all(sha(data / p) == record['files'][p]['sha256'] for p in paths)):
                catalog['archives'].append(record)
                print(f'ZIP: reused verified {target.name}', flush=True)
                continue
        required = sum((data / p).stat().st_size for p in paths)
        if shutil.disk_usage(output).free < required + 512 * 1024**2:
            raise OSError(f'Insufficient disk for {target.name}; choose another output drive')
        temporary = target.with_suffix('.zip.tmp')
        files = {}
        last_log = 0
        with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_STORED, allowZip64=True) as archive:
            for i, relative in enumerate(paths, 1):
                source = data / relative
                h = hashlib.sha256()
                with source.open('rb') as src, archive.open(relative, 'w', force_zip64=True) as dst:
                    for block in iter(lambda: src.read(8 * 1024**2), b''):
                        h.update(block)
                        dst.write(block)
                files[relative] = {'bytes': source.stat().st_size, 'sha256': h.hexdigest()}
                now = time.monotonic()
                if now - last_log >= 10 or i == len(paths):
                    print(f'ZIP {group}: {i:,}/{len(paths):,} | elapsed {now-started:.0f}s', flush=True)
                    last_log = now
        temporary.replace(target)
        record = {'name': target.name, 'bytes': target.stat().st_size, 'sha256': sha(target),
                  'unpacked_bytes': sum(x['bytes'] for x in files.values()),
                  'files': files, 'manifest_sha256': manifest_hashes}
        record_path.write_text(json.dumps(record), encoding='utf-8')
        catalog['archives'].append(record)
    temporary = output / 'dataset_zip_index.json.tmp'
    temporary.write_text(json.dumps(catalog), encoding='utf-8')
    temporary.replace(output / 'dataset_zip_index.json')
    print(f'DONE: upload the {len(groups)} ZIPs and dataset_zip_index.json from {output}', flush=True)
    return catalog

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    pack(Path(__file__).resolve().parents[1], args.data_root, args.output)
