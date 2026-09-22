"""Export a small, matched NIH pilot for manual Teachable Machine experiments.

No model training. Existing denoising implementations are loaded without DAE.
"""
import csv
import importlib.util
import json
from pathlib import Path
import random
import zipfile

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / 'teachable_machine_pilot'
SEED = 42
SIZE = 256


def load_method(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / 'denoise' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    if OUT.exists():
        raise FileExistsError(f'Refusing to overwrite {OUT}')
    methods = {name: load_method(name) for name in ('baseline', 'median', 'clahe_dwt')}
    variants = [('baseline', 'baseline', 1)] + [
        (f'{name}_L{level}', name, level)
        for name in ('median', 'clahe_dwt') for level in (1, 2, 3)
    ]
    selected = []
    used_patients = set()
    rng = random.Random(SEED)
    # One image per patient across classes AND splits; keep official split membership.
    for split, count in [('train', 200), ('test', 50)]:
        for cls, source in [('Infiltration', 'infiltration'), ('Normal', 'normal')]:
            with (ROOT / 'data' / 'manifests' / f'{split}_{source}.csv').open(newline='') as f:
                rows = list(csv.DictReader(f))
            rng.shuffle(rows)
            chosen = []
            for row in rows:
                patient = row['Patient ID']
                if patient in used_patients:
                    continue
                assert row['split'] == split
                assert Path(row['path']).is_file(), row['path']
                used_patients.add(patient)
                chosen.append({**row, 'class': cls})
                if len(chosen) == count:
                    break
            assert len(chosen) == count
            selected.extend(chosen)
    # Verify labels and membership against original metadata and official lists.
    with (ROOT / 'data/versions/3/Data_Entry_2017.csv').open(newline='') as f:
        labels = {r['Image Index']: r['Finding Labels'] for r in csv.DictReader(f)}
    official = {
        s: set((ROOT / 'data/versions/3' / filename).read_text().splitlines())
        for s, filename in [('train', 'train_val_list.txt'), ('test', 'test_list.txt')]
    }
    for row in selected:
        assert row['Image Index'] in official[row['split']]
        assert labels[row['Image Index']] == ('No Finding' if row['class'] == 'Normal' else 'Infiltration')
    OUT.mkdir(parents=True)
    with (OUT / 'selection.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    preview_rows = []
    preview_counts = {'Normal': 0, 'Infiltration': 0}
    for index, row in enumerate(selected):
        original = cv2.imread(row['path'], cv2.IMREAD_GRAYSCALE)
        assert original is not None
        tiles = []
        for folder, name, level in variants:
            processed = methods[name].denoise(original, level=level)
            assert processed.shape == original.shape and processed.dtype == np.uint8
            if name == 'baseline':
                assert np.array_equal(processed, original)
            resized = cv2.resize(processed, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
            dest = OUT / folder / row['split'] / row['class'] / row['Image Index']
            dest.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(resized).convert('RGB').save(dest)
            # Verify the saved PNG is readable, lossless, and has the expected dimensions.
            with Image.open(dest) as check:
                assert check.size == (SIZE, SIZE) and check.mode == 'RGB'
                assert np.array_equal(np.asarray(check)[:, :, 0], resized)
            tiles.append(Image.fromarray(resized).convert('RGB'))
        if row['split'] == 'train' and preview_counts[row['class']] < 2:
            preview_rows.append((row, tiles))
            preview_counts[row['class']] += 1
        if (index + 1) % 50 == 0:
            print(f'Prepared {index + 1}/{len(selected)} source images', flush=True)
    canvas = Image.new('RGB', (SIZE * len(variants), 60 + len(preview_rows) * (SIZE + 35)), 'white')
    draw = ImageDraw.Draw(canvas)
    for column, (folder, _, _) in enumerate(variants):
        draw.text((column * SIZE + 8, 15), folder, fill='black')
    for i, (row, tiles) in enumerate(preview_rows):
        y = 60 + i * (SIZE + 35)
        for j, tile in enumerate(tiles):
            canvas.paste(tile, (j * SIZE, y + 30))
            draw.text((j * SIZE + 8, y + 5), f"{row['class']} | {row['Image Index']}", fill='black')
    canvas.save(OUT / 'before_after.png')
    config = {
        'seed': SEED, 'train_per_class': 200, 'test_per_class': 50,
        'image_size': [SIZE, SIZE], 'patients': len(used_patients),
        'processing': 'grayscale native resolution -> denoise -> resize INTER_AREA -> RGB PNG',
        'median_kernel': methods['median'].LEVELS,
        'clahe_dwt_joint_presets': methods['clahe_dwt'].LEVELS,
        'wavelet': 'db1', 'clahe_grid': [8, 8], 'order': 'DWT then CLAHE',
        'versions': {'opencv': cv2.__version__, 'numpy': np.__version__, 'pywavelets': methods['clahe_dwt'].pywt.__version__},
    }
    (OUT / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    for folder, _, _ in variants:
        files = sorted((OUT / folder).rglob('*.png'))
        assert len(files) == 500
        with zipfile.ZipFile(OUT / f'{folder}.zip', 'w', zipfile.ZIP_STORED) as archive:
            for file in files:
                archive.write(file, file.relative_to(OUT / folder))
        with zipfile.ZipFile(OUT / f'{folder}.zip') as archive:
            assert archive.testzip() is None
    print(f'COMPLETE: {OUT} | 500 sources, 3500 PNGs, 7 ZIPs; no training performed.', flush=True)


if __name__ == '__main__':
    main()
