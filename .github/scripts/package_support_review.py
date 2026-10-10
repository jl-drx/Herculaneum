"""Package the pinned, compact PHerc0800 geometry handoff; no inference or LFS."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil

EXPECTED_MANIFEST = '16249b33163d391e8d4d7abe73998fa9dd8864f1b4d5cf7937e2fc36af2ac12f'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def package(source, destination, commit):
    manifest = source / 'PACKET_MANIFEST.json'
    if digest(manifest) != EXPECTED_MANIFEST:
        raise ValueError('Packet manifest differs from the frozen handoff')
    records = json.loads(manifest.read_text())['files']
    seen = set()
    for row in records:
        name = row['path']; relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or name in seen:
            raise ValueError('Unsafe or duplicated packet member')
        seen.add(name); path = source / name
        if any(p.is_symlink() for p in [path, *path.parents]):
            raise ValueError('Symlinks are not packet members')
        if not path.is_file() or path.stat().st_size != row['bytes'] or digest(path) != row['sha256']:
            raise ValueError('Packet member checksum or length differs: ' + name)
    actual = {str(p.relative_to(source)) for p in source.rglob('*') if p.is_file()}
    if actual != seen | {'PACKET_MANIFEST.json'}:
        raise ValueError('Packet contains unexpected or missing files')
    if destination.exists():
        raise ValueError('Output must be a new directory')
    destination.mkdir(parents=True)
    for name in sorted(actual):
        out = destination / name; out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, out)
        if digest(out) != digest(source / name):
            raise ValueError('Copied member differs')
    (destination / 'PROVENANCE.json').write_text(json.dumps(dict(
        repository='jl-drx/Herculaneum', workflow_commit=commit,
        packet_manifest_sha256=EXPECTED_MANIFEST, input_files=len(records),
        inference_run=False, lfs_objects_fetched=0, credentials_included=False,
        scope='Private CPU-only geometry review; assignment is START_HERE.md.'), indent=2)+'\n')
    files = sorted(p for p in destination.rglob('*') if p.is_file())
    (destination / 'SHA256SUMS').write_text(''.join(f'{digest(p)}  {p.relative_to(destination).as_posix()}\n' for p in files))
    return dict(status='PASS', packet_files=len(actual), lfs_objects_fetched=0, inference_run=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.source, args.output, args.commit)))
