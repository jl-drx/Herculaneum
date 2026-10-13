"""Build a sealed study's immutable LFS supplement and member-checksum index."""
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
    parser.add_argument('--lab', type=Path, required=True)
    parser.add_argument('--study', required=True)
    parser.add_argument('--register', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--index', required=True)
    parser.add_argument('--expected-seal', required=True)
    args = parser.parse_args()
    for value in [args.study, args.register, args.name, args.index]:
        if Path(value).name != value or value in {'.', '..'}:
            raise ValueError('Arguments must be single path components')
    lab = args.lab.resolve(); phase = lab / 'reports' / args.study
    seal = phase / 'final-verification.json'
    if sha(seal) != args.expected_seal:
        raise ValueError('Seal differs from independent anchor')
    manifest = json.loads(seal.read_text())
    records = {r['path']: r for r in manifest['records']}
    sources = sorted(p for p in phase.rglob('*') if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts)
    sources += [lab / 'reports' / args.register, Path(manifest['prior_manifest']['path'])]
    if len(set(sources)) != len(sources):
        raise ValueError('Duplicate source')
    scan([p for p in sources if p.suffix in {'.py', '.md', '.json', '.log', '.txt'}])
    root = Path(__file__).resolve().parents[1] / 'workspace'
    bundle = root / 'bundles' / args.name; index = root / args.index
    if bundle.exists() or index.exists():
        raise ValueError('Immutable supplement already exists')
    members = []
    with bundle.open('xb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', compresslevel=1, mtime=0) as zipped, tarfile.open(fileobj=zipped, mode='w|') as tar:
        for source in sources:
            digest = sha(source)
            expected = records.get(str(source))
            if source == seal:
                if digest != args.expected_seal:
                    raise ValueError('Seal changed')
            elif expected is None or digest != expected['sha256'] or source.stat().st_size != expected['size_bytes']:
                raise ValueError('Source is absent from seal or changed: ' + str(source))
            info = tar.gettarinfo(str(source), arcname=source.relative_to(lab).as_posix())
            info.uid = info.gid = 0; info.uname = info.gname = ''; info.mtime = 0; info.mode = 0o644
            with source.open('rb') as stream:
                tar.addfile(info, stream)
            if sha(source) != digest:
                raise ValueError('Source changed while packaging')
            members.append(dict(path=info.name, bytes=info.size, sha256=digest))
    meta = dict(schema_version=1, study=args.study, seal_sha256=args.expected_seal,
                bundle=dict(name=bundle.name, bytes=bundle.stat().st_size, sha256=sha(bundle)),
                files=len(members), expanded_bytes=sum(m['bytes'] for m in members), members=members)
    with index.open('x') as output:
        json.dump(meta, output, indent=2, sort_keys=True); output.write('\n')
    print(json.dumps({k: v for k, v in meta.items() if k != 'members'}))


if __name__ == '__main__':
    main()
