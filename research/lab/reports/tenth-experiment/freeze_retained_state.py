"""Freeze additive local evidence without model calls or network access."""
import datetime
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parent
lab=root.parents[1]
previous=root.parent/'ninth-experiment/final-verification.json'
expected_previous='4cbb13062270200aceb802b3691bee679175dca97810061ca293a18211c01b1e'
output=root/'final-verification.json'
assert not output.exists(), 'Never replace a frozen manifest'
def record(path):
    path=Path(path).resolve(strict=True)
    with path.open('rb') as stream: digest=hashlib.file_digest(stream,'sha256').hexdigest()
    return dict(path=str(path),size_bytes=path.stat().st_size,sha256=digest)
assert record(previous)['sha256']==expected_previous
old=json.loads(previous.read_text())
expected={}
def collect(value):
    if isinstance(value,dict):
        if {'path','sha256'}<=value.keys() and ('size_bytes' in value or 'byte_length' in value):
            item=dict(path=value['path'],sha256=value['sha256'],size_bytes=value.get('size_bytes',value.get('byte_length')))
            assert Path(item['path']).is_absolute(),item['path']
            assert item['path'] not in expected or expected[item['path']]==item, 'Conflicting active identities: '+item['path']
            expected[item['path']]=item
        for child in value.values():collect(child)
    elif isinstance(value,list):
        for child in value:collect(child)
collect(old['records'])
# Historical/superseded review receipts remain literal retained files; only
# these final authorities define currently active source dependencies.
authorities=[
    'implementation/final-plan/control-plan.json','implementation/code-freeze.json',
    'implementation/architecture-check.json','implementation/run/A_preserve/result.json',
    'implementation/run/B_preserve/result.json','implementation/analysis/metrics.json',
    'reference-audit/SOURCE_RECEIPTS.json','reference-audit/MANIFEST.json',
    'diagnostic-audit/audit-manifest.json','review/code-review.json',
    'review/result/independent-verification.json','review/review-manifest.json',
    'visualization/render/figure-receipt.json','visualization/visualization-manifest.json',
]
for relative in authorities:collect(json.loads((root/relative).read_text()))
records={}
for path,wanted in expected.items():
    actual=record(path)
    assert actual==wanted,'Changed prior/frozen artifact: '+path
    records[path]=actual
files=[p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.cache' not in p.parts]
for path in files+[previous,lab/'STARTUP_AFTER_TENTH.md',lab/'reports/research-register-v6.json']:
    actual=record(path)
    assert actual['path'] not in records or records[actual['path']]==actual
    records[actual['path']]=actual
rows=sorted(records.values(),key=lambda value:value['path'])
manifest=dict(schema_version=1,created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    status='completed_mixed_preservation_result_independently_verified',
    scope='Tenth CPU preservation study, lineage/score audits and all ninth bindings; local integrity only, not backup/publication/restoration, independent accuracy, reading or novelty. Caches/bytecode/self excluded.',
    previous_manifest=record(previous),previous_record_count=len(old['records']),active_reference_authorities=authorities,
    record_count=len(rows),retained_bytes=sum(row['size_bytes'] for row in rows),records=rows)
with output.open('x') as stream:json.dump(manifest,stream,indent=2);stream.write('\n')
print(json.dumps(dict(status='PASS',manifest=record(output),record_count=len(rows),retained_bytes=manifest['retained_bytes'])))
