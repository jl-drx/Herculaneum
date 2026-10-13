"""Build immutable, selectively restorable Git LFS bundles of a research lab.

Installed software, credentials, operational state and the current raw CT cache
are deliberately excluded. Inputs and archived evidence retain original bytes.
"""
import argparse
import collections
import datetime
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import tarfile

TEXT = {'.py', '.md', '.json', '.jsonl', '.csv', '.tsv', '.txt', '.toml', '.yaml',
        '.yml', '.sh', '.lock', '.html', '.body', '.ipynb', '.xml', '.source'}
PATTERNS = {
    'token': re.compile(rb'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|(?:AKIA|ASIA)[A-Z0-9]{16})\b'),
    'private-key': re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    'signed-url': re.compile(rb'[?&](?:X-Amz-Signature|X-Amz-Security-Token|access_token)=', re.I),
    'authenticated-url': re.compile(rb'https?://[^\s/:]+:[^\s/@]+@'),
}


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def exclude(rel):
    parts = rel.parts
    low = rel.as_posix().lower()
    if any(x in parts for x in ['.git', '.private', '__pycache__', 'plot-cache', 'squashfs-root']):
        return 'private, generated cache, or installed binary'
    if any(x.startswith('portability-') for x in parts) or parts[0] in {'tools', 'artifacts-infrastructure'}:
        return 'publication/operational tooling outside research snapshot'
    if any(x in low for x in ['runpod-setup', 'cloudflare', 'credential', 'oauth', 'setup-validation', 'r2-readiness']):
        return 'credential or infrastructure setup scope'
    if rel.name.startswith(('STARTUP', '.env')) or 'startup-save' in low or rel.suffix in {'.pyc', '.partial'}:
        return 'startup/private/temporary state'
    if 'fortysixth-experiment' in parts:
        if 'renderer-cache' in parts or 'ct-cache' in low or 'pipeline-' in low or rel.name.startswith('PIPELINE_'):
            return 'live process state or reacquirable current raw CT cache'
        if rel.suffix.lower() in {'.appimage', '.zip', '.log'} or 'cached-transport' in low or rel.name == 'fixed-pipeline.lock':
            return 'renderer download/transport/runtime state'
    return None


def group(rel):
    low = rel.as_posix()
    if low in {'data/models/ink9um-seed42-step075000.pth', 'data/models/ink9um-seed43-step075000.pth'}:
        return 'pherc343-models'
    if 'fortysixth-experiment' in rel.parts:
        if rel.suffix == '.pth':
            return 'pherc343-models'
        if any(x.endswith('.zarr') for x in rel.parts) or rel.name == 'valid.npy':
            return 'pherc343-inputs'
        if rel.suffix in {'.npy', '.tif', '.png'} or 'maps_plain' in rel.parts or 'maps_padded' in rel.parts:
            return 'pherc343-results'
    if ('fortyfifth-experiment' in rel.parts and
            ('ph343-mesh-replay' in rel.parts or 'pherc343' in rel.parts or rel.name.startswith('concat_w047'))
            and rel.suffix not in TEXT):
        return 'pherc343-reference'
    if rel.suffix in TEXT or rel.name.upper().startswith(('LICENSE', 'NOTICE', 'COPYING')):
        return 'core'
    return 'history'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--lab', type=Path, required=True)
    ap.add_argument('--destination', type=Path, required=True)
    ap.add_argument('--plan-only', action='store_true')
    args = ap.parse_args()
    lab = args.lab.resolve(); dest = args.destination.resolve(); dest.mkdir(parents=True, exist_ok=True)
    chosen = collections.defaultdict(list); omitted = []; symlinks = []
    for f in sorted(lab.rglob('*')):
        rel = f.relative_to(lab)
        if f.is_symlink():
            symlinks.append(rel.as_posix()); continue
        if not f.is_file():
            continue
        reason = exclude(rel)
        if reason:
            omitted.append({'path': rel.as_posix(), 'bytes': f.stat().st_size, 'reason': reason}); continue
        if f.suffix in TEXT:
            data = f.read_bytes()
            hits = [name for name, rx in PATTERNS.items() if rx.search(data)]
            if hits:
                # Preserve originals locally; never log the matching values.
                omitted.append({'path': rel.as_posix(), 'bytes': len(data),
                                'reason': 'credential-like pattern requires review: ' + ', '.join(hits)})
                continue
        chosen[group(rel)].append(f)
    summary = {k: {'files': len(v), 'bytes': sum(f.stat().st_size for f in v)} for k, v in chosen.items()}
    print(json.dumps({'groups': summary, 'omitted_files': len(omitted), 'symlinks': symlinks}), flush=True)
    if args.plan_only:
        return
    assert not (dest / 'MANIFEST.json').exists(), 'Snapshot already exists'
    bundles = []; files = []; maximum = 600 * 1024**2
    for kind, sources in sorted(chosen.items()):
        partitions = []; part = []; size = 0
        for f in sources:
            if part and size + f.stat().st_size > maximum:
                partitions.append(part); part = []; size = 0
            part.append(f); size += f.stat().st_size
        if part:
            partitions.append(part)
        for number, members in enumerate(partitions, 1):
            name = f'{kind}-{number:03}.tar.gz'; path = dest / 'bundles' / name
            path.parent.mkdir(exist_ok=True); records = []
            with path.open('xb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', compresslevel=1, mtime=0) as zipped, tarfile.open(fileobj=zipped, mode='w|') as archive:
                for f in members:
                    before = f.stat(); identity = sha(f); relative = f.relative_to(lab).as_posix()
                    info = archive.gettarinfo(str(f), arcname=relative)
                    info.uid = info.gid = 0; info.uname = info.gname = ''; info.mtime = 0; info.mode = 0o644
                    with f.open('rb') as stream:
                        archive.addfile(info, stream)
                    after = f.stat()
                    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), f'Mutating source: {relative}'
                    records.append({'path': relative, 'bytes': before.st_size, 'sha256': identity,
                                    'bundle': name, 'group': kind})
            bundle = {'name': name, 'group': kind, 'bytes': path.stat().st_size,
                      'sha256': sha(path), 'files': len(records), 'expanded_bytes': sum(x['bytes'] for x in records)}
            bundles.append(bundle); files.extend(records)
            print(json.dumps({'bundle_complete': bundle}), flush=True)
    manifest = {'schema_version': 1, 'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'source_lab': str(lab), 'scope': 'Retained research files; excludes listed operational/private state and current raw CT cache. Dated snapshot, not a running worker or full filesystem backup.',
                'groups': summary, 'bundles': bundles, 'files': files, 'excluded': omitted,
                'symlinks_not_archived': symlinks, 'archive_bytes': sum(b['bytes'] for b in bundles),
                'expanded_bytes': sum(f['bytes'] for f in files)}
    (dest / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'status': 'BUILT', 'files': len(files), 'bundles': len(bundles),
                      'archive_bytes': manifest['archive_bytes'], 'expanded_bytes': manifest['expanded_bytes']}), flush=True)


if __name__ == '__main__':
    main()
