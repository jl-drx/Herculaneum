"""Bind completed local evidence while preserving every prior twentyeighth identity."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
LAB = HERE.parent.parent
PRIOR = LAB / 'reports/twentyeighth-experiment/final-verification.json'
PRIOR_SHA = '594ab84d0efc41522dfa6e5b5397bf5e56ee7bfc0e10ae39eeaf14019888c78b'


def record(path):
    path = Path(path).resolve(strict=True)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(path), size_bytes=path.stat().st_size, sha256=digest)


assert record(PRIOR)['sha256'] == PRIOR_SHA
prior = json.loads(PRIOR.read_text())
records = {}
for item in prior['records']:
    assert record(item['path']) == item, 'Prior frozen evidence changed: ' + item['path']
    assert item['path'] not in records
    records[item['path']] = item
destination = HERE / 'final-verification.json'
assert not destination.exists(), 'Final manifest must be new'
new_paths = [PRIOR, LAB / 'reports/research-register-v25.json']
new_paths += [p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts
              and p != destination]
phase_bytes = sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())
assert phase_bytes < 32*1024**2, 'Aggregate phase29 output quota exceeded'
for path in new_paths:
    item = record(path)
    assert item['path'] not in records or records[item['path']] == item
    records[item['path']] = item
for path in (LAB / 'reports/frag6-public-reference-recheck-20261009').rglob('*'):
    if path.is_file() and '__pycache__' not in path.parts:
        item = record(path)
        assert item['path'] not in records or records[item['path']] == item
        records[item['path']] = item
checkout = Path('/workspace/Herculaneum')
head = subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip()
status = subprocess.check_output(['git', '-C', str(checkout), 'status', '--short'], text=True)
assert head == '49fabf3e890d5b880f391b27d581f1e7a1ebdbcc'
assert status == '?? .codex/\n', 'Checkout changed beyond preserved untracked .codex/'
ordered = [records[k] for k in sorted(records)]
manifest = dict(schema_version=1, status='PASS', created_at_utc=datetime.now(timezone.utc).isoformat(),
                prior_manifest=record(PRIOR), prior_records_preserved=len(prior['records']),
                records=ordered, record_count=len(ordered),
                retained_bytes=sum(r['size_bytes'] for r in ordered),
                phase_bytes_before_final_manifest=phase_bytes,
                checkout=dict(path=str(checkout), commit=head, status=status),
                scope='Local byte-integrity retention; no external backup/publication/restoration claim',
                gpu_spend_GBP=0, total_first_GPU_pilot_cap_GBP=10)
destination.write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
assert phase_bytes + destination.stat().st_size < 32*1024**2
print(json.dumps(dict(status='PASS', manifest=record(destination),
                     record_count=manifest['record_count'], retained_bytes=manifest['retained_bytes'],
                     prior_records_preserved=manifest['prior_records_preserved'])))
