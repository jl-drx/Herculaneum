"""Package a bounded cache delta or apply it to a new, verified restore copy."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile


BASE_ARCHIVE_SHA256 = '4d16cece37b134da339775306d00dbd9c8bf1df2ea6d8e416561719714f72917'
BASE_MANIFEST_SHA256 = '665c69cb1e6fb890ce06685679910f0293799d52d63dbcb935b852b13ba92899'
MUTABLE = {'work/' + prefix + name for prefix in ['', 'original/']
           for name in ['ACQUISITION_FAILURE.json', 'ACQUISITION_STATUS.json']}


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def identity(path):
    return dict(bytes=path.stat().st_size, sha256=sha(path))


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def relative(name):
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or str(p) != name:
        raise ValueError('Unsafe relative path: ' + name)
    return p


def inventory(root):
    result = {}
    for path in sorted(root.rglob('*')):
        if '__pycache__' in path.parts:
            continue
        if path.is_symlink():
            raise ValueError('Symlink in checkpoint: ' + str(path))
        if path.is_file():
            result[str(path.relative_to(root))] = identity(path)
    return result


def verify(root, expected_manifest=None):
    if expected_manifest and sha(root / 'MANIFEST.json') != expected_manifest:
        raise ValueError('Unexpected base manifest')
    files = json.loads((root / 'MANIFEST.json').read_text())['files']
    for name, expected in files.items():
        path = root / relative(name)
        if path.is_symlink() or identity(path) != expected:
            raise ValueError('Checkpoint mismatch: ' + name)
    actual = set(inventory(root))
    if actual != set(files) | {'MANIFEST.json', 'SHA256SUMS'}:
        raise ValueError('Unexpected checkpoint files')
    sums = ''.join(f"{r['sha256']}  {name}\n" for name, r in files.items())
    if (root / 'SHA256SUMS').read_text() != sums:
        raise ValueError('Checksum manifest mismatch')
    return files


def manifest(root, scope):
    files = inventory(root)
    write(root / 'MANIFEST.json', dict(schema=1, scope=scope, files=files))
    (root / 'SHA256SUMS').write_text(''.join(f"{r['sha256']}  {n}\n" for n, r in files.items()))
    return files


def package(a):
    if a.archive.exists() or a.receipt.exists():
        raise FileExistsError('Use new archive and receipt paths')
    base = verify(a.base, BASE_MANIFEST_SHA256)
    with tempfile.TemporaryDirectory(prefix='plinius-cache-delta-', dir='/tmp') as temp:
        root = Path(temp)
        entries = {}
        for name, after in inventory(a.work).items():
            if name.endswith('.partial'):
                raise ValueError('Unfinished source write')
            dest = 'work/' + name
            before = base.get(dest)
            if before == after:
                continue
            if before is not None and dest not in MUTABLE:
                raise ValueError('Frozen source changed: ' + dest)
            entries[dest] = dict(before=before, after=after)
            target = root / 'overlay' / dest
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(a.work / name, target)
        for name in ['STATUS.json', 'ACCESS_RECHECK.json', 'ACQUISITION_FAILURE.json',
                     'RETRY_PLAN.json', 'SOURCE_PROBE.json', 'ORIGINAL_CHECKPOINT_REMOTE_RESTORE.json',
                     'PARENT_CHECKPOINT_REMOTE_RESTORE.json', 'SOURCE_QUALIFICATION_RESULTS.json']:
            dest = root / 'evidence' / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(a.packet / name, dest)
        (root / 'code').mkdir()
        shutil.copyfile(__file__, root / 'code/checkpoint_delta.py')
        write(root / 'INDEX.json', dict(schema=1, base_archive_sha256=BASE_ARCHIVE_SHA256,
            base_manifest_sha256=BASE_MANIFEST_SHA256, entries=entries,
            scope='Acquisition and partial added-source qualification; original qualifications unchanged.'))
        (root / 'README.md').write_text(
            '# Private intake and partial qualification update\n\n'
            'Apply to the verified combined 21:47 restore (manifest pinned in INDEX).\n'
            'Base is the first 490.9 MiB archive, 329.0 MiB original completion,\n'
            'and 276.71 MiB cumulative retry. Do not also apply the older addenda.\n'
            'Includes 1009 new sources, the failed retry and first complete case\n'
            'mechanical teacher/support qualification. Full cohort remains unadmitted.\n'
            'Verify with verify_bundle.py, then run code/checkpoint_delta.py apply\n'
            '--base BASE --delta THIS_DIRECTORY --output NEW_COMBINED_DIRECTORY.\n'
            'The original results remain unchanged. New partial qualification is\n'
            'under work/added-qualification-20261012 with both verifier and inputs.\n'
            'No model inference, training, paid compute or physical accuracy claim.\n'
            'Keep private and retain original public-source attribution.\n')
        files = manifest(root, 'Input cache and partial qualification; no model improvement result')
        with a.archive.open('xb') as stream:
            with gzip.GzipFile(filename='', fileobj=stream, mode='wb', compresslevel=6, mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode='w') as tar:
                    for path in sorted(root.rglob('*')):
                        if not path.is_file():
                            continue
                        info = tar.gettarinfo(str(path), arcname=str(path.relative_to(root)))
                        info.uid = info.gid = info.mtime = 0
                        info.uname = info.gname = ''
                        info.mode = 0o644
                        with path.open('rb') as source:
                            tar.addfile(info, source)
        receipt = dict(status='PACKAGED_PENDING_RESTORE', archive_name=a.archive.name,
            **identity(a.archive), manifest_files=len(files),
            manifest_sha256=sha(root / 'MANIFEST.json'), payload_bytes=sum(r['bytes'] for r in files.values()),
            new_paths=sum(r['before'] is None for r in entries.values()),
            updated_operational_paths=sum(r['before'] is not None for r in entries.values()),
            base_archive_sha256=BASE_ARCHIVE_SHA256, implementation_sha256=sha(Path(__file__)),
            full_population_admitted=False, new_model_forwards=0, paid_compute=False)
        write(a.receipt, receipt)
        print(json.dumps(receipt))


def apply(a):
    if a.output.exists():
        raise FileExistsError('Use a new combined output directory')
    for source in [a.base, a.delta]:
        if a.output.resolve().is_relative_to(source.resolve()):
            raise ValueError('Output must be outside both source trees')
    baseline = verify(a.base, BASE_MANIFEST_SHA256)
    verify(a.delta)
    index = json.loads((a.delta / 'INDEX.json').read_text())
    if (index['base_manifest_sha256'], index['base_archive_sha256']) != (BASE_MANIFEST_SHA256, BASE_ARCHIVE_SHA256):
        raise ValueError('Delta belongs to a different checkpoint')
    for name, row in index['entries'].items():
        relative(name)
        if not name.startswith('work/') or baseline.get(name) != row['before']:
            raise ValueError('Unexpected baseline path: ' + name)
        if row['before'] is not None and name not in MUTABLE:
            raise ValueError('Attempt to change frozen source: ' + name)
        if identity(a.delta / 'overlay' / name) != row['after']:
            raise ValueError('Delta identity mismatch: ' + name)
    shutil.copytree(a.base, a.output, ignore=shutil.ignore_patterns('__pycache__'))
    history = a.output / 'retry-2318-history'
    history.mkdir()
    for name in ['MANIFEST.json', 'SHA256SUMS']:
        shutil.move(a.output / name, history / ('BASE_' + name))
    for name, row in index['entries'].items():
        target = a.output / name
        if row['before'] is not None:
            old = history / name
            old.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(target, old)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(a.delta / 'overlay' / name, target)
    shutil.copytree(a.delta / 'evidence', a.output / 'retry-2318-evidence')
    shutil.copyfile(a.delta / 'INDEX.json', a.output / 'retry-2318-evidence/INDEX.json')
    result = dict(status='PASS', base_manifest_sha256=BASE_MANIFEST_SHA256,
        delta_manifest_sha256=sha(a.delta / 'MANIFEST.json'), applied_paths=len(index['entries']),
        original_restore_unchanged=True, historical_qualification_unchanged=True,
        historical_qualification_scope='Both dated original qualifications unchanged; partial added qualification stored separately under work/added-qualification-20261012.',
        full_population_admitted=False, new_model_forwards=0, paid_compute=False)
    write(a.output / 'retry-2318-evidence/APPLIED.json', result)
    manifest(a.output, 'Combined cache; each qualification retains its dated input scope')
    verify(a.output)
    print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='mode', required=True)
    p = sub.add_parser('package')
    for arg in ['base', 'work', 'packet', 'archive', 'receipt']:
        p.add_argument('--' + arg, type=Path, required=True)
    p = sub.add_parser('apply')
    for arg in ['base', 'delta', 'output']:
        p.add_argument('--' + arg, type=Path, required=True)
    args = parser.parse_args()
    (package if args.mode == 'package' else apply)(args)


if __name__ == '__main__':
    main()
