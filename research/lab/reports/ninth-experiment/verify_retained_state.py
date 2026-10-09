"""Read-only retained-evidence verification; no model calls, network or writes."""
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--expected-manifest-sha256')
args = parser.parse_args()
manifest = Path(__file__).resolve().parent / 'final-verification.json'
manifest_bytes = manifest.read_bytes()
digest = hashlib.sha256(manifest_bytes).hexdigest()
if args.expected_manifest_sha256 and digest != args.expected_manifest_sha256:
    raise SystemExit('Manifest differs from the supplied saved anchor')
data = json.loads(manifest_bytes)
failures = []
seen = set()
total = 0
for record in data['records']:
    path = Path(record['path'])
    if str(path) in seen:
        failures.append('Duplicate path: ' + str(path))
    seen.add(str(path))
    try:
        with path.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        size = path.stat().st_size
        if size != record['size_bytes'] or actual != record['sha256']:
            failures.append(str(path))
        total += size
    except OSError:
        failures.append(str(path))
if len(data['records']) != data['record_count'] or total != data['retained_bytes']:
    failures.append('Manifest count or byte total differs')
print(json.dumps(dict(status='FAIL' if failures else 'PASS',
    manifest_sha256=digest, record_count=len(data['records']),
    retained_bytes=total, failures=failures), indent=2))
raise SystemExit(bool(failures))
