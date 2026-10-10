"""Read-only integrity check; never reruns completed research."""
import hashlib
import json
from pathlib import Path

manifest = Path(__file__).resolve().parent / 'final-verification.json'
data = json.loads(manifest.read_text())
failures = []
for record in data['records']:
    path = Path(record['path'])
    try:
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if path.stat().st_size != record['size_bytes'] or digest != record['sha256']:
            failures.append(str(path))
    except OSError:
        failures.append(str(path))
print(json.dumps({'status': 'FAIL' if failures else 'PASS',
                  'record_count': len(data['records']), 'failures': failures}, indent=2))
raise SystemExit(bool(failures))
