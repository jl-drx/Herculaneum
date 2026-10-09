"""Create the additive local integrity manifest; no model/network operations."""
import datetime
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
lab = root.parents[1]
previous = root.parent / 'eighth-experiment/final-verification.json'
expected_previous = '8bf317faf2e0c95b7b23522cd5f1681c002054f3cc887ab7e6f3cc063aae04f1'
output = root / 'final-verification.json'
assert not output.exists(), 'Never replace a frozen manifest'
def record(path):
    path = Path(path).resolve(strict=True)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(path), size_bytes=path.stat().st_size, sha256=digest)
assert record(previous)['sha256'] == expected_previous
old = json.loads(previous.read_text())
expected = {}
def collect(value):
    if isinstance(value, dict):
        if {'path', 'sha256'} <= value.keys() and ('size_bytes' in value or 'byte_length' in value):
            item = dict(path=value['path'], sha256=value['sha256'], size_bytes=value.get('size_bytes', value.get('byte_length')))
            assert Path(item['path']).is_absolute(), item['path']
            assert item['path'] not in expected or expected[item['path']] == item, 'Conflicting retained identities'
            expected[item['path']] = item
        for child in value.values():
            collect(child)
    elif isinstance(value, list):
        for child in value:
            collect(child)
collect(old['records'])
files = [p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.cache' not in p.parts]
# Validate references from the final execution/review authorities. An early
# fixture receipt describes source before final freezing; retain its literal
# bytes as history without asserting those superseded source versions still
# occupy the final paths. Do not silently repin an executed plan or result.
authorities = [
    'implementation/final-plan/control-plan.json', 'implementation/code-freeze.json',
    'implementation/run/A_smooth/result.json', 'implementation/run/B_smooth/result.json',
    'implementation/analysis/metrics.json', 'source-audit/SOURCE_RECEIPTS.json',
    'source-audit/ENVIRONMENT_READONLY.json', 'review/code-review.json',
    'review/smoothing-fixture-receipt-v2.json', 'review/result/independent-verification.json',
    'review/review-manifest.json', 'visualization/render/figure-receipt.json',
]
for relative in authorities:
    collect(json.loads((root/relative).read_text()))
records = {}
for path, wanted in expected.items():
    actual = record(path)
    assert actual == wanted, 'Changed prior/frozen artifact: ' + path
    records[path] = actual
for path in files + [previous, lab/'STARTUP_AFTER_NINTH.md', lab/'reports/research-register-v5.json']:
    actual = record(path)
    assert actual['path'] not in records or records[actual['path']] == actual
    records[actual['path']] = actual
rows = sorted(records.values(), key=lambda value: value['path'])
manifest = dict(schema_version=1, created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    status='completed_mixed_negative_smoothing_result_independently_verified',
    scope='Ninth frozen CPU control and all previous eighth bindings; local integrity only, not backup/publication/restoration, independent accuracy, reading or novelty. Bytecode/caches/self excluded.',
    previous_manifest=record(previous), previous_record_count=len(old['records']),
    active_reference_authorities=authorities, historical_pre_freeze_receipt='review/smoothing-fixture-receipt.json', record_count=len(rows),
    retained_bytes=sum(row['size_bytes'] for row in rows), records=rows)
with output.open('x') as stream:
    json.dump(manifest, stream, indent=2)
    stream.write('\n')
print(json.dumps(dict(status='PASS', manifest=record(output), record_count=len(rows), retained_bytes=manifest['retained_bytes'])))
