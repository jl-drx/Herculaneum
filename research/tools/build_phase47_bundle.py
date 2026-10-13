"""Package the sealed bounded grid study; preserve all original artifact bytes."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import tarfile

from export_snapshot import scan


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lab', required=True, type=Path)
    parser.add_argument('--expected-manifest-sha256', required=True)
    args = parser.parse_args()
    lab = args.lab.resolve()
    phase = lab / 'reports/fortyseventh-experiment'
    seal = phase / 'final-verification.json'
    assert sha(seal) == args.expected_manifest_sha256, 'Seal differs from external anchor'
    manifest = json.loads(seal.read_text())
    pins = {r['path']: r for r in manifest['records']}
    files = sorted(p for p in phase.rglob('*') if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts)
    files += [lab / 'reports/research-register-v43.json',
              lab / 'reports/fortysixth-experiment/work/valid.npy',
              lab / 'reports/fortysixth-experiment/final-verification.json']
    scan([p for p in files if p.suffix in {'.py', '.md', '.json', '.log', '.txt'}])
    research = Path(__file__).resolve().parents[1]
    name = 'pherc343-grid-20261010-001.tar.gz'
    archive = research / 'workspace/bundles' / name
    index = research / 'workspace/PHASE47_BUNDLE.json'
    assert not archive.exists() and not index.exists(), 'Immutable bundle already exists'
    records = []
    with archive.open('xb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', compresslevel=1, mtime=0) as zipped, tarfile.open(fileobj=zipped, mode='w|') as tar:
        for file in files:
            checksum = sha(file)
            if file != seal:
                expected = pins[str(file)]
                assert checksum == expected['sha256'] and file.stat().st_size == expected['size_bytes'], str(file)
            else:
                assert checksum == args.expected_manifest_sha256
            name = file.relative_to(lab).as_posix()
            info = tar.gettarinfo(str(file), arcname=name)
            info.uid = info.gid = 0
            info.uname = info.gname = ''
            info.mtime = 0
            info.mode = 0o644
            with file.open('rb') as stream:
                tar.addfile(info, stream)
            assert sha(file) == checksum, 'Source changed while packaging'
            records.append(dict(path=name, bytes=info.size, sha256=checksum))
    result = dict(schema_version=1, scope='Saved-output replay, no CT/model replay or physical-independence certificate',
                  seal_sha256=args.expected_manifest_sha256,
                  bundle=dict(name=archive.name, bytes=archive.stat().st_size, sha256=sha(archive)),
                  files=len(records), expanded_bytes=sum(r['bytes'] for r in records), members=records)
    with index.open('x') as output:
        json.dump(result, output, indent=2, sort_keys=True)
        output.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'members'}))


if __name__ == '__main__':
    main()
