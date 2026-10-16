"""Reuse a completed DAE from an earlier run when its training is provably the same.

The earlier run's provenance differs only because CNN code/config changed. Before copying,
this checks that everything the DAE depends on matches: config outside `cnn`, manifests,
source pixels, package versions, every shared source file except CNN/report-only ones, and,
for shared/training.py, every DAE function against the exact file that trained it
(release_full_training_reference.py, verified by the SHA-256 the old run recorded).
The copy is re-signed with the new provenance and keeps an `imported_from` record.
Lives in tools/ so it is outside the source fingerprint.
"""
import ast
import json
import shutil
from copy import deepcopy
from pathlib import Path

REFERENCE = Path(__file__).with_name('release_full_training_reference.py')
# shared/denoising.py exactly as release_v4 ran it; the L4 version only adds denoising conditions.
DENOISING_REFERENCE = Path(__file__).with_name('release_v4_denoising_reference.py')
# Source files that may differ: none of them is used to train the DAE.
NON_DAE_SOURCES = {'shared/training.py', 'shared/evaluation.py', 'shared/reporting.py', 'build_notebooks.py'}
# Everything train_dae runs; _completed is excluded because only its CNN branch changed.
DAE_FUNCTIONS = ('train_dae', 'DenoiseDataset', 'synthetic_noise', 'loader', 'capture_rng_state',
                 'restore_rng_state', '_autocast', '_scaler', '_finite_loss', '_write_csv',
                 '_resume_checkpoint', '_repair_epoch_writes', '_restore_training', '_reconstruction_epoch',
                 '_dae_class_reconstruction', '_training_backend', '_ensure_cache', 'checkpoint_state')
# denoising.py functions that build DAE training inputs (runtime_cache imports working_image).
DENOISING_DAE_FUNCTIONS = ('read_gray', 'working_image')
# Source files whose DAE-relevant functions are compared one by one against the code that trained the old DAE.
CHECKED_SOURCES = {'shared/training.py': (REFERENCE, DAE_FUNCTIONS),
                   'shared/denoising.py': (DENOISING_REFERENCE, DENOISING_DAE_FUNCTIONS)}


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _definitions(source):
    tree = ast.parse(source)
    return {node.name: ast.dump(node) for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}


def _dae_scope(config):
    """Config fields that can influence the DAE; CNN settings, conditions and output paths cannot.

    `conditions` only choose CNN inputs; train_dae never reads them.
    """
    scope = deepcopy(config)
    for key in ('cnn', 'conditions', 'output_subdirectory', 'source_code_hashes'):
        scope.pop(key, None)
    scope['data'].pop('colab_output_root', None)
    return scope


def _differences(a, b, prefix=''):
    if isinstance(a, dict) and isinstance(b, dict):
        return [d for k in sorted(set(a) | set(b)) for d in _differences(a.get(k), b.get(k), f'{prefix}{k}.')]
    return [] if a == b else [prefix.rstrip('.')]


def verify(exp, old_seed_root):
    """Return (old completion, old best checkpoint) or raise naming the first mismatch."""
    import torch
    from shared.experiment import digest, file_hash
    old = Path(old_seed_root) / 'shared/artifacts/dae_checkpoint'
    completed = _read(old / 'complete.json')
    for name, key in (('best.pt', 'checkpoint_hash'), ('history.csv', 'history_hash')):
        if file_hash(old / name) != completed[key]:
            raise ValueError(f'Old DAE artifact changed after completion: {old / name}')
    saved = completed['provenance']
    if digest(saved['config']) != saved['config_hash']:
        raise ValueError('Old DAE provenance is internally inconsistent')
    checkpoint = torch.load(old / 'best.pt', map_location='cpu', weights_only=True)
    if checkpoint['provenance']['config_hash'] != saved['config_hash'] or int(checkpoint['epoch']) < 1:
        raise ValueError('Old DAE checkpoint does not belong to its completion record')
    current = exp.provenance()
    if completed.get('epochs_run') != exp.config['dae']['epochs']:
        raise ValueError('Old DAE did not finish the configured epoch budget')
    differences = _differences(_dae_scope(saved['config']), _dae_scope(current['config']))
    if differences:
        raise ValueError(f'DAE-relevant config differs: {differences}')
    for key in ('manifest_hashes', 'input_source_hash', 'versions'):
        if saved.get(key) != current.get(key):
            raise ValueError(f'Old DAE {key} differs from this run; train a new DAE')
    old_sources, new_sources = saved['config']['source_code_hashes'], current['config']['source_code_hashes']
    changed = {k for k in set(old_sources) | set(new_sources) if old_sources.get(k) != new_sources.get(k)}
    unchecked = changed - NON_DAE_SOURCES - set(CHECKED_SOURCES)
    if unchecked:
        raise ValueError(f'Source files used by DAE training changed: {sorted(unchecked)}')
    for source, (reference, functions) in CHECKED_SOURCES.items():
        if source not in changed:
            continue
        if file_hash(reference) != old_sources[source]:
            raise ValueError(f'Reference for {source} is not the code that trained the old DAE')
        before = _definitions(reference.read_text(encoding='utf-8'))
        after = _definitions((Path(exp.root) / source).read_text(encoding='utf-8'))
        altered = [name for name in functions if before.get(name) is None or before.get(name) != after.get(name)]
        if altered:
            raise ValueError(f'DAE training code changed in {source}: {altered}')
    return completed, checkpoint


def import_dae(exp, old_seed_root):
    from shared.experiment import file_hash, save_torch, write_json
    from shared.denoising import load_dae
    import torch
    old_seed_root = Path(old_seed_root)
    old = old_seed_root / 'shared/artifacts/dae_checkpoint'
    folder = exp.artifacts / 'dae_checkpoint'
    source_marker = file_hash(old / 'complete.json')
    if (folder / 'complete.json').is_file():
        existing = _read(folder / 'complete.json').get('imported_from', {})
        if existing.get('complete_json_hash') == source_marker:
            load_dae(exp, 'cpu')  # Re-verify the earlier import end to end.
            print(f'DAE seed {exp.config["seed"]}: already imported from {old_seed_root}', flush=True)
            return folder / 'best.pt'
        raise ValueError(f'{folder} already holds a different completed DAE; not overwriting it')
    if folder.is_dir() and any(folder.glob('*.pt')):
        raise ValueError(f'{folder} holds a partially trained DAE; finish it or move it aside before importing')
    completed, checkpoint = verify(exp, old_seed_root)
    record = {'source': str(old_seed_root), 'config_hash': completed['provenance']['config_hash'],
              'checkpoint_hash': completed['checkpoint_hash'], 'complete_json_hash': source_marker,
              'reference_training_sha256': file_hash(REFERENCE),
              'reference_denoising_sha256': file_hash(DENOISING_REFERENCE),
              'reason': 'DAE code/config/data/versions identical; only CNN protocol or conditions changed'}
    provenance = exp.provenance()
    for name in ('best.pt', 'last.pt'):
        state = torch.load(old / name, map_location='cpu', weights_only=True)
        state['provenance'], state['imported_from'] = provenance, record
        save_torch(folder / name, state)
    shutil.copyfile(old / 'history.csv', folder / 'history.csv')
    write_json(folder / 'complete.json', {'provenance': provenance, 'checkpoint_hash': file_hash(folder / 'best.pt'),
                                          'history_hash': file_hash(folder / 'history.csv'),
                                          'best_epoch': int(checkpoint['epoch']), 'epochs_run': completed['epochs_run'],
                                          'imported_from': record})
    load_dae(exp, 'cpu')
    print(f'DAE seed {exp.config["seed"]}: imported best epoch {checkpoint["epoch"]} from {old_seed_root}', flush=True)
    return folder / 'best.pt'


def import_if_available(exp, old_seed_root):
    """Import when the old seed finished its DAE; otherwise leave training to train_dae."""
    if not (Path(old_seed_root) / 'shared/artifacts/dae_checkpoint/complete.json').is_file():
        print(f'DAE seed {exp.config["seed"]}: no completed DAE in {old_seed_root}; will train a new one', flush=True)
        return None
    return import_dae(exp, old_seed_root)
