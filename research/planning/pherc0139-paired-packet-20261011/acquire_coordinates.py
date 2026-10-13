"""Fetch only the frozen fifteen native coordinate rasters; fail closed."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent

def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def acquire(item):
    output = ROOT / 'coordinates' / (item['id'] + '.tif')
    receipt = output.with_suffix('.json')
    assert sha(ROOT / 'receipts' / (item['id'] + '.json')) == item['head_receipt_sha256']
    if receipt.exists():
        r = json.loads(receipt.read_text())
        assert r['url'] == item['url'] and r['etag'] == item['etag']
        assert output.stat().st_size == item['bytes'] == r['bytes'] and sha(output) == r['sha256']
        return r
    assert not output.exists(), 'Unreceipted output requires investigation'
    partial = output.with_suffix('.partial')
    with requests.get(item['url'], headers={'Accept-Encoding': 'identity', 'If-Match': item['etag']},
                      stream=True, timeout=(15, 60), allow_redirects=False) as response:
        assert response.status_code == 200
        assert int(response.headers['Content-Length']) == item['bytes']
        assert response.headers.get('ETag') == item['etag']
        assert response.headers.get('Content-Encoding', 'identity') == 'identity'
        n = 0
        with partial.open('xb') as f:
            for block in response.iter_content(65536):
                n += len(block)
                assert n <= item['bytes']
                f.write(block)
        assert n == item['bytes']
    os.replace(partial, output)
    r = dict(item, sha256=sha(output), retrieved_utc=datetime.now(timezone.utc).isoformat())
    receipt.write_text(json.dumps(r, indent=2) + '\n')
    return r

def main():
    plan = json.loads((ROOT / 'COORDINATE_PLAN.json').read_text())
    (ROOT / 'coordinates').mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=plan['max_workers']) as pool:
        rows = list(pool.map(acquire, plan['requests']))
    assert sum(r['bytes'] for r in rows) == plan['max_body_bytes']
    result = dict(status='COMPLETE', plan_sha256=sha(ROOT / 'COORDINATE_PLAN.json'),
                  code_sha256=sha(Path(__file__)), requests=rows,
                  coordinate_bytes=sum(r['bytes'] for r in rows), CT_teacher_annotation_bytes=0)
    (ROOT / 'COORDINATE_LEDGER.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'requests'}))

if __name__ == '__main__':
    main()
