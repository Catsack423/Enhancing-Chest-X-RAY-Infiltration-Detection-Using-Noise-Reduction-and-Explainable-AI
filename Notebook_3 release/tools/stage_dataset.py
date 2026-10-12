"""Copy verified ZIPs from Drive, extract originals to VM, delete local ZIPs."""
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

def stage_dataset(root, archive_root, local_root):
    root, archive_root, local_root = map(Path, (root, archive_root, local_root))
    local_root = local_root.resolve()
    if '/content/drive/' in local_root.as_posix():
        raise ValueError('Extract dataset to VM disk, outside mounted Drive')
    index = json.loads((archive_root / 'dataset_zip_index.json').read_text(encoding='utf-8'))
    if index.get('schema') != 1:
        raise ValueError('Unsupported ZIP catalog')
    expected = {p.name: sha(p) for p in (root / 'manifests').glob('*.csv')}
    if index['manifest_sha256'] != expected:
        raise ValueError('ZIP dataset does not match frozen release manifests')
    local_root.mkdir(parents=True, exist_ok=True)
    state = local_root / '.zip_state'
    state.mkdir(exist_ok=True)
    archives = index['archives']
    needed = sum(a['unpacked_bytes'] for a in archives) + max(a['bytes'] for a in archives) + 6 * 1024**3
    existing = sum(p.stat().st_size for p in local_root.rglob('*') if p.is_file())
    if shutil.disk_usage(local_root).free + existing < needed:
        raise OSError(f'Need about {needed/1024**3:.1f} GiB for originals, largest ZIP and working cache')
    print(f'ZIP staging: {index["images"]:,} images -> {local_root}', flush=True)
    for number, record in enumerate(archives, 1):
        name = record['name']
        if Path(name).name != name or '\\' in name:
            raise ValueError('Unsafe archive name')
        for relative in record['files']:
            path = PurePosixPath(relative)
            if path.is_absolute() or '..' in path.parts or '\\' in relative or not path.parts:
                raise ValueError('Unsafe ZIP member path')
            if not local_root.joinpath(*path.parts).resolve().is_relative_to(local_root):
                raise ValueError('ZIP member escapes local dataset')
        marker = state / (name + '.done')
        if marker.exists() and marker.read_text() == record['sha256']:
            if all((local_root / p).is_file() and sha(local_root / p) == info['sha256'] for p, info in record['files'].items()):
                print(f'ZIP {number}/{len(archives)}: verified existing {name}', flush=True)
                continue
        marker.unlink(missing_ok=True)
        local_zip = state / name
        if not local_zip.exists() or sha(local_zip) != record['sha256']:
            tmp = local_zip.with_suffix('.partial')
            started, last, copied = time.monotonic(), 0, 0
            h = hashlib.sha256()
            print(f'ZIP {number}/{len(archives)}: copying {name} ({record["bytes"]/1024**3:.2f} GiB)', flush=True)
            with (archive_root / name).open('rb') as source, tmp.open('wb') as target:
                for block in iter(lambda: source.read(8 * 1024**2), b''):
                    target.write(block)
                    h.update(block)
                    copied += len(block)
                    now = time.monotonic()
                    if now - last >= 10:
                        rate = copied / max(now - started, .001) / 1024**2
                        print(f'  copy {copied/record["bytes"]:.1%} | {rate:.1f} MiB/s', flush=True)
                        last = now
            if copied != record['bytes'] or h.hexdigest() != record['sha256']:
                raise ValueError(f'Incomplete/corrupt ZIP: {name}')
            tmp.replace(local_zip)
        print(f'  extracting {name} on VM (no per-image Drive reads)', flush=True)
        with zipfile.ZipFile(local_zip) as archive:
            members = archive.infolist()
            if len(members) != len(record['files']) or {m.filename for m in members} != set(record['files']):
                raise ValueError('ZIP members differ from catalog')
            last = 0
            for i, member in enumerate(members, 1):
                info = record['files'][member.filename]
                if member.is_dir() or member.file_size != info['bytes']:
                    raise ValueError('Unexpected ZIP geometry')
                target = local_root / member.filename
                target.parent.mkdir(parents=True, exist_ok=True)
                tmp = target.with_suffix(target.suffix + '.tmp')
                h = hashlib.sha256()
                with archive.open(member) as source, tmp.open('wb') as dst:
                    for block in iter(lambda: source.read(8 * 1024**2), b''):
                        dst.write(block)
                        h.update(block)
                if h.hexdigest() != info['sha256']:
                    raise ValueError(f'Corrupt original image: {member.filename}')
                tmp.replace(target)
                now = time.monotonic()
                if now - last >= 10 or i == len(members):
                    print(f'  extract {i:,}/{len(members):,}', flush=True)
                    last = now
        marker.write_text(record['sha256'])
        local_zip.unlink()  # exact known local ZIP; keep extracted originals and verified cache
    print('ZIP staging DONE; DATA_ROOT now reads VM disk', flush=True)
    return local_root
