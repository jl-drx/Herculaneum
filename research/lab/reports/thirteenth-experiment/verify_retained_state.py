"""Read-only local evidence verification; no model, network, or write operations."""
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--expected-manifest-sha256', required=True)
args = parser.parse_args()
manifest = Path(__file__).resolve().parent / 'final-verification.json'
payload = manifest.read_bytes()
digest = hashlib.sha256(payload).hexdigest()
if digest != args.expected_manifest_sha256:
    raise SystemExit('Manifest differs from the saved external anchor')
data = json.loads(payload)
failures, seen, total = [], set(), 0
for item in data['records']:
    path = Path(item['path'])
    if str(path) in seen:
        failures.append('Duplicate path: ' + str(path))
    seen.add(str(path))
    try:
        with path.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        size = path.stat().st_size
        if size != item['size_bytes'] or actual != item['sha256']:
            failures.append(str(path))
        total += size
    except OSError:
        failures.append(str(path))
if len(data['records']) != data['record_count'] or total != data['retained_bytes']:
    failures.append('Manifest count or byte total differs')
print(json.dumps(dict(status='FAIL' if failures else 'PASS', manifest_sha256=digest,
                     record_count=len(data['records']), retained_bytes=total,
                     failures=failures), indent=2))
raise SystemExit(bool(failures))
