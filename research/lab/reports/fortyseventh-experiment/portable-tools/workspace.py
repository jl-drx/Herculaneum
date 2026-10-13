"""Fetch, verify and restore selected research bundles without loading models."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile

RESEARCH = Path(__file__).resolve().parents[1]
REPO = RESEARCH.parent
INDEX = RESEARCH / 'workspace/MANIFEST.json'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def safe_path(root, name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or '..' in relative.parts or str(relative) != name or not relative.parts:
        raise ValueError(f'Unsafe artifact path: {name}')
    path = root.joinpath(*relative.parts)
    if any(p.is_symlink() for p in [path, *path.parents] if p != root.parent):
        raise ValueError(f'Symlink in artifact destination: {name}')
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Artifact escapes destination: {name}')
    return path


def checked(path, record):
    return path.is_file() and path.stat().st_size == record['bytes'] and sha(path) == record['sha256']


def restore_bundle(archive, destination, records):
    """Only declared regular files; verify before atomically placing each file."""
    expected = {r['path']: r for r in records}; seen = set(); restored = reused = 0
    with tarfile.open(archive, 'r|gz') as tar:
        for member in tar:
            if not member.isfile() or member.name not in expected or member.name in seen:
                raise ValueError(f'Unexpected or nonregular archive member: {member.name}')
            r = expected[member.name]; target = safe_path(destination, member.name)
            if member.size != r['bytes']:
                raise ValueError(f'Archive length mismatch: {member.name}')
            seen.add(member.name)
            if target.exists():
                if not checked(target, r):
                    raise ValueError(f'Existing file differs; refusing overwrite: {target}')
                reused += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + '.plinius-partial')
            h = hashlib.sha256(); count = 0
            created = False
            try:
                with temporary.open('xb') as output:
                    created = True
                    with tar.extractfile(member) as source:
                        while chunk := source.read(1024 * 1024):
                            output.write(chunk); h.update(chunk); count += len(chunk)
                if count != r['bytes'] or h.hexdigest() != r['sha256']:
                    raise ValueError(f'Artifact checksum mismatch: {member.name}')
                os.replace(temporary, target)
                restored += 1
            finally:
                if created:
                    temporary.unlink(missing_ok=True)
    if seen != set(expected):
        raise ValueError('Archive is missing declared files')
    return {'restored': restored, 'reused': reused}


def run_git(*args):
    return subprocess.run(['git', '-C', str(REPO), *args], check=True, text=True, capture_output=True).stdout


def resolve_archive(bundle, offline):
    relative = 'research/workspace/bundles/' + bundle['name']
    workfile = REPO / relative
    if workfile.is_file() and workfile.stat().st_size == bundle['bytes']:
        if not checked(workfile, bundle):
            raise ValueError('Working archive checksum mismatch: ' + bundle['name'])
        return workfile, False
    environment = run_git('lfs', 'env')
    media = next((s.removeprefix('LocalMediaDir=') for s in environment.splitlines() if s.startswith('LocalMediaDir=')), None)
    if media is None:
        raise RuntimeError('Git LFS did not report its local object directory')
    oid = bundle['sha256']; cached = Path(media) / oid[:2] / oid[2:4] / oid
    if not cached.is_file():
        if offline:
            raise FileNotFoundError('Bundle is not downloaded: ' + bundle['name'])
        subprocess.run(['git', '-C', str(REPO), 'lfs', 'fetch', 'origin', 'HEAD',
                        '--include=' + relative, '--exclude='], check=True)
    if not checked(cached, bundle):
        raise ValueError('Git LFS archive checksum mismatch: ' + bundle['name'])
    return cached, True


def restore_input_views(root):
    work = root / 'reports/fortysixth-experiment/work'
    name = 'concat_w047-w048_R5B2_z9500-11000_66'
    for phase, suffix in [('plain', ''), ('padded', '_padded')]:
        view = work / f'{phase}-level0.zarr'
        source = work / (name + suffix + '.zarr') / '0'
        if not (view / '.zgroup').is_file() or not (source / '.zarray').is_file():
            raise ValueError('Prepared input view is incomplete')
        link = view / '0'
        if link.is_symlink() and link.resolve() == source.resolve():
            continue
        if link.exists() or link.is_symlink():
            raise ValueError('Refusing to replace existing input view: ' + str(link))
        link.symlink_to(Path('..') / source.parent.name / '0', target_is_directory=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['list', 'restore', 'verify'])
    parser.add_argument('--groups', nargs='+', default=['core'])
    parser.add_argument('--destination', type=Path)
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--drop-cache', action='store_true', help='Remove each verified local LFS download after restoring it; remote objects remain available.')
    args = parser.parse_args(); raw = INDEX.read_bytes(); manifest = json.loads(raw)
    assert manifest['schema_version'] == 1
    available = set(manifest['groups'])
    if args.command == 'list':
        for group in sorted(available):
            bundles = [b for b in manifest['bundles'] if b['group'] == group]
            print(f'{group:20} {sum(b["bytes"] for b in bundles)/1024**3:6.2f} GiB download  '
                  f'{sum(b["expanded_bytes"] for b in bundles)/1024**3:6.2f} GiB restored  '
                  f'{sum(b["files"] for b in bundles):6} files')
        return
    selected = available if args.groups == ['all'] else set(args.groups)
    if selected - available:
        parser.error('Unknown groups: ' + ', '.join(sorted(selected - available)))
    if args.destination is None:
        parser.error('--destination is required; no existing lab is modified implicitly')
    root = args.destination.resolve(); records = [r for r in manifest['files'] if r['group'] in selected]
    if args.command == 'verify':
        for record in records:
            if not checked(safe_path(root, record['path']), record):
                raise ValueError('Missing or altered file: ' + record['path'])
        print(json.dumps({'status': 'PASS', 'verified_files': len(records), 'groups': sorted(selected)}))
        return
    marker = root / '.plinius-workspace-restore.json'
    if root.exists() and any(root.iterdir()) and not marker.is_file():
        raise ValueError('Destination is nonempty and is not a managed restore; choose a fresh directory')
    identity = hashlib.sha256(raw).hexdigest()
    state = json.loads(marker.read_text()) if marker.exists() else {'manifest_sha256': identity, 'completed_bundles': []}
    if state['manifest_sha256'] != identity:
        raise ValueError('Destination belongs to a different snapshot; choose a fresh directory')
    root.mkdir(parents=True, exist_ok=True)
    bundles = [b for b in manifest['bundles'] if b['group'] in selected]
    missing_bytes = sum(r['bytes'] for r in records if not safe_path(root, r['path']).exists())
    download_budget = max((b['bytes'] for b in bundles), default=0) if args.drop_cache else sum(b['bytes'] for b in bundles)
    required = missing_bytes + download_budget + 1024**3
    if shutil.disk_usage(root).free < required:
        raise RuntimeError(f'Insufficient free space: conservative requirement {required/1024**3:.2f} GiB; use --drop-cache or fewer groups')
    marker.write_text(json.dumps(state, indent=2) + '\n')
    for bundle in bundles:
        archive, is_cache = resolve_archive(bundle, args.offline)
        result = restore_bundle(archive, root, [r for r in records if r['bundle'] == bundle['name']])
        if bundle['name'] not in state['completed_bundles']:
            state['completed_bundles'].append(bundle['name'])
        temporary = marker.with_suffix('.partial'); temporary.write_text(json.dumps(state, indent=2) + '\n'); temporary.replace(marker)
        if args.drop_cache and is_cache:
            archive.unlink()  # Only the hash-verified Git LFS cache object just used.
        print(json.dumps({'bundle': bundle['name'], **result}), flush=True)
    if 'pherc343-inputs' in selected:
        restore_input_views(root)
    print(json.dumps({'status': 'PASS', 'groups': sorted(selected), 'files': len(records), 'destination': str(root)}))


if __name__ == '__main__':
    main()
