"""Coordinate Colab sessions that each own one seed of one shared release run.

Lives in tools/ so source fingerprints (root *.py, shared/*.py) and config_hash stay unchanged.
Seeds share nothing below seed_<n>/, so one seed per session needs no cross-session writes
except claims and the final seed summary.
"""
import json
import socket
import threading
import time
import uuid
from pathlib import Path

STALE_SECONDS = 15 * 60
MIN_DAE_VRAM_GIB = 20  # DAE batch 64 peaked at 17.8 GiB on L4


def _read(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _conditions(root):
    return [c['id'] for c in json.loads((Path(root) / 'configs/methods_release_controlled.json').read_text(encoding='utf-8'))]


def preflight(base, seed, all_seeds):
    """Fail before training when this VM cannot join the existing run."""
    import torch
    from shared.experiment import versions
    base = Path(base)
    if seed not in all_seeds:
        raise ValueError(f'SESSION_SEED must be one of {all_seeds}')
    if not torch.cuda.is_available():
        raise RuntimeError('Select a GPU runtime first')
    current = versions()
    for other in all_seeds:
        saved = _read(base / f'seed_{other}/shared/artifacts/experiment.json')
        if saved is None:
            continue
        diff = {k: (saved['versions'].get(k), current.get(k)) for k in set(saved['versions']) | set(current)
                if saved['versions'].get(k) != current.get(k)}
        if diff:
            raise RuntimeError(f'Package versions differ from seed_{other} (saved, this VM): {diff}. '
                               'Seed summary needs one environment; use a runtime with the saved versions.')
    gpu = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    dae_done = (base / f'seed_{seed}/shared/artifacts/dae_checkpoint/complete.json').is_file()
    if not dae_done and vram < MIN_DAE_VRAM_GIB:
        raise RuntimeError(f'{gpu} has {vram:.1f} GiB; DAE batch 64 needs >= {MIN_DAE_VRAM_GIB} GiB. Choose L4 or A100.')
    others = {p.stem: (_read(p) or {}).get('gpu') for p in (base / 'parallel_claims').glob('seed_*.json')
              if p.stem != f'seed_{seed}'}
    if any(name and name != gpu for name in others.values()):
        print(f'WARNING: this VM is {gpu}; other seeds ran on {others}. Non-deterministic cuDNN makes '
              'cross-GPU seeds differ more; prefer the same GPU type.', flush=True)
    print(f'Preflight ok | seed {seed} | {gpu} {vram:.1f} GiB | versions match | {base}', flush=True)


def recent_writes(seed_root, seconds=STALE_SECONDS):
    """Age of the newest DAE/CNN checkpoint write, to catch a writer that holds no claim."""
    seed_root = Path(seed_root)
    newest = 0.0
    folders = [seed_root / 'shared/artifacts/dae_checkpoint', *(seed_root / 'CNN/results').glob('*')]
    for folder in folders:
        if folder.is_dir():
            for path in folder.iterdir():
                if path.is_file():
                    newest = max(newest, path.stat().st_mtime)
    age = time.time() - newest
    return age if age < seconds else None


class SeedClaim:
    """Best-effort Drive lease: one live session per seed, refreshed every minute.

    Released claims stay on Drive (released=True) so later sessions can compare GPU types.
    """

    def __init__(self, base, seed, interval=60):
        self.base, self.seed, self.interval = Path(base), int(seed), interval
        self.path = self.base / 'parallel_claims' / f'seed_{self.seed}.json'
        self.token = uuid.uuid4().hex
        self.started = time.time()
        self._stop = threading.Event()
        self._thread = None
        self.error = None

    def _record(self, released=False):
        import torch
        return {'token': self.token, 'seed': self.seed, 'host': socket.gethostname(),
                'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
                'started': self.started, 'heartbeat': time.time(), 'released': released}

    def _write(self, released=False):
        from shared.experiment import write_json
        write_json(self.path, self._record(released))

    def acquire(self, force=False):
        existing = _read(self.path)
        held = existing and not existing.get('released') and existing.get('token') != self.token
        if held and not force:
            age = time.time() - existing['heartbeat']
            if age < STALE_SECONDS:
                raise RuntimeError(f"seed {self.seed} is claimed by {existing['host']} ({existing['gpu']}), "
                                   f"heartbeat {age/60:.1f} min ago. Pick another seed, or wait {STALE_SECONDS//60} min "
                                   'after that session stops; FORCE_CLAIM=True only if it is surely dead.')
        age = recent_writes(self.base / f'seed_{self.seed}')
        if age is not None and not force and not (existing and existing.get('released')):
            raise RuntimeError(f'seed_{self.seed} checkpoints were written {age/60:.1f} min ago by a session without a '
                               'live claim. Stop that cell, then rerun with FORCE_CLAIM=True.')
        self._write()
        time.sleep(5)  # Let a racing session's write land, then confirm who won.
        if (_read(self.path) or {}).get('token') != self.token:
            raise RuntimeError(f'Another session claimed seed {self.seed} at the same time; pick another seed')
        self._thread = threading.Thread(target=self._beat, daemon=True)
        self._thread.start()
        print(f'Claimed seed {self.seed} | {self.path}', flush=True)
        return self

    def _beat(self):
        while not self._stop.wait(self.interval):
            try:
                if (_read(self.path) or {}).get('token') != self.token:
                    self.error = 'claim taken over by another session'
                    return
                self._write()
            except OSError as error:  # Transient Drive errors; next beat retries.
                print(f'[claim seed={self.seed}] heartbeat write failed: {error}', flush=True)

    def check(self):
        """Call before each stage; refuse to keep writing after losing the claim."""
        if self.error or self._thread is None or not self._thread.is_alive():
            raise RuntimeError(f'Lost claim on seed {self.seed}: {self.error or "heartbeat stopped"}. '
                               'Rerun the claim cell before continuing.')
        if (_read(self.path) or {}).get('token') != self.token:
            raise RuntimeError(f'seed {self.seed} is now claimed by another session; stop here')

    def release(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        if (_read(self.path) or {}).get('token') == self.token:
            self._write(released=True)
        print(f'Released seed {self.seed}', flush=True)


def seed_status(root, base, seeds):
    base = Path(base)
    conditions = _conditions(root)
    rows = {}
    for seed in seeds:
        folder = base / f'seed_{seed}'
        done = {criterion: sum((folder / criterion / 'results' / c / 'complete.json').is_file() for c in conditions)
                for criterion in ('CNN', 'Grad-CAM', 'SHAP')}
        claim = _read(base / 'parallel_claims' / f'seed_{seed}.json')
        live = claim and not claim.get('released') and time.time() - claim['heartbeat'] < STALE_SECONDS
        rows[seed] = {'dae': (folder / 'shared/artifacts/dae_checkpoint/complete.json').is_file(), **done,
                      'report': (folder / 'comparison/complete.json').is_file(),
                      'claim': f"{claim['host']} {claim['gpu']}" if live else '-'}
    return rows, len(conditions)


def print_status(root, base, seeds):
    rows, total = seed_status(root, base, seeds)
    print(f'{"seed":>4} | DAE | CNN   | Grad-CAM | SHAP  | report | claim', flush=True)
    for seed, r in rows.items():
        print(f"{seed:>4} | {'yes' if r['dae'] else 'no ':3} | {r['CNN']:>2}/{total} | {r['Grad-CAM']:>5}/{total} | "
              f"{r['SHAP']:>2}/{total} | {'yes' if r['report'] else 'no':6} | {r['claim']}", flush=True)
    return rows


def next_seed(root, base, seeds):
    """First seed with no report and no live claim, for a session that finished its own seed."""
    rows, _ = seed_status(root, base, seeds)
    return next((seed for seed, r in rows.items() if not r['report'] and r['claim'] == '-'), None)


def missing_reports(base, seeds):
    return [s for s in seeds if not (Path(base) / f'seed_{s}/comparison/complete.json').is_file()]
