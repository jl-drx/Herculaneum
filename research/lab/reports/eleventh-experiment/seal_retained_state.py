"""Append current evidence to unchanged tenth bindings; never rewrite old receipts."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

ROOT = Path(__file__).resolve().parent
LAB = ROOT.parents[1]
PRIOR = ROOT.parent / 'tenth-experiment/final-verification.json'
PRIOR_SHA = 'e2310ea95fb9a797617201852127ae37b018bc19f63e05345206afebdaec3c88'


def record(path):
    path = Path(path).resolve(strict=True)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(path), size_bytes=path.stat().st_size, sha256=digest)


assert record(PRIOR)['sha256'] == PRIOR_SHA
previous = json.loads(PRIOR.read_text())
assert previous['record_count'] == 1222
records = {}
for bound in previous['records']:
    assert record(bound['path']) == bound, 'Prior frozen evidence changed: ' + bound['path']
    records[bound['path']] = bound
assert len(records) == 1222
paths = [PRIOR, LAB / 'STARTUP_AFTER_ELEVENTH.md', ROOT.parent / 'research-register-v7.json']
paths += [path for path in ROOT.rglob('*') if path.is_file()
          and '__pycache__' not in path.parts and path.suffix != '.pyc'
          and path.name != 'final-verification.json']
runtime = json.loads((ROOT / 'runtime-receipt.json').read_text())
paths += [Path(v['metadata']['path']) for v in runtime['packages'].values()]
plan = json.loads((ROOT / 'comparison/final-plan/comparison-plan.json').read_text())
for bound in plan['verified_inputs']:
    actual = record(bound['path'])
    assert actual == bound
    paths.append(Path(bound['path']))
for path in paths:
    bound = record(path)
    assert bound['path'] not in records or records[bound['path']] == bound
    records[bound['path']] = bound
metrics = json.loads((ROOT / 'comparison/analysis/metrics.json').read_text())
assert metrics['status'] == 'PASS' and metrics['verified_run_count'] == 20
assert metrics['verified_tile_forward_count'] == 500
for name in ['review/comparison/completed-comparison-verification.json',
             'result-verification/final-review/independent-verification.json']:
    assert json.loads((ROOT / name).read_text())['status'] == 'PASS'
output = dict(schema_version=1, created_at_utc=datetime.now(timezone.utc).isoformat(),
    status='completed_optical_reference_comparison_independently_verified_no_consistent_continuation_gain',
    scope='All tenth bindings unchanged plus eleventh retained evidence. Local integrity, not backup/publication/restoration, independent historical holdout, reading or novelty. Caches/bytecode/self excluded. Historical superseded receipts retained literally rather than repinned.',
    previous_manifest=record(PRIOR), previous_record_count=1222,
    active_reference_authorities=['RESEARCH_DESIGN.md','INFERENCE_REVIEW.md',
        'ct-acquisition/acquisition-plan.json','ct-acquisition/run/provenance.json',
        'pooled-reference-descriptor.json','comparison/code-freeze.json',
        'comparison/final-plan/comparison-plan.json','comparison/run/results.json',
        'comparison/analysis/metrics.json','review/comparison/review-manifest.json',
        'result-verification/final-review/independent-verification.json',
        'visualization/MANIFEST.json'],
    record_count=len(records),retained_bytes=sum(r['size_bytes'] for r in records.values()),
    records=sorted(records.values(),key=lambda r:r['path']))
target = ROOT / 'final-verification.json'
with target.open('x') as stream:
    json.dump(output,stream,indent=2);stream.write('\n')
print(json.dumps(dict(manifest=record(target),record_count=output['record_count'],retained_bytes=output['retained_bytes'])))
