"""Read-only byte verification; an additive Git commit does not invalidate old evidence."""
import argparse
import hashlib
import json
from pathlib import Path

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--expected-manifest-sha256',required=True)
parser.add_argument('--lab-root',type=Path,default=Path(__file__).resolve().parents[2])
args=parser.parse_args();path=Path(__file__).resolve().parent/'final-verification.json'
raw=path.read_bytes()
if hashlib.sha256(raw).hexdigest()!=args.expected_manifest_sha256:raise SystemExit('Manifest differs from external anchor')
m=json.loads(raw);errors=[];seen=set();total=0
for item in m['records']:
    original=Path(item['path'])
    local=args.lab_root/original.relative_to('/workspace/herculaneum-lab') if original.is_relative_to('/workspace/herculaneum-lab') else original
    if item['path'] in seen:errors.append('Duplicate path: '+item['path'])
    seen.add(item['path'])
    try:
        with local.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        size=local.stat().st_size;total+=size
        if digest!=item['sha256'] or size!=item['size_bytes']:errors.append(item['path'])
    except OSError:errors.append(item['path'])
if len(seen)!=m['record_count'] or total!=m['retained_bytes']:errors.append('Manifest totals')
print(json.dumps(dict(status='FAIL' if errors else 'PASS',manifest_sha256=hashlib.sha256(raw).hexdigest(),records=len(seen),retained_bytes=total,failures=errors)))
raise SystemExit(bool(errors))
