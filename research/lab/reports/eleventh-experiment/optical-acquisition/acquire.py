"""Execute only the frozen optical-quality acquisition; no model or CT pixel reads."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import re
import struct

import requests

ROOT = Path(__file__).resolve().parent
PLAN = json.loads((ROOT / 'quality-plan.json').read_text())
PLAN_SHA = hashlib.sha256((ROOT / 'quality-plan.json').read_bytes()).hexdigest()
LIMITS = PLAN['limits']


def save_ledger(ledger):
    temp = ROOT / 'request-ledger.partial'
    temp.write_text(json.dumps(ledger, indent=2) + '\n')
    temp.replace(ROOT / 'request-ledger.json')


def fetch(name, url, cap, byte_range=None):
    ledger_path = ROOT / 'request-ledger.json'
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {
        'plan_sha256': PLAN_SHA, 'requests': [], 'body_bytes': 0}
    assert ledger['plan_sha256'] == PLAN_SHA
    assert url in [a['url'] for a in PLAN['assets']] + list(PLAN['header_assets'].values())
    assert not (ROOT / name).exists() and Path(name).name == name
    assert len(ledger['requests']) < LIMITS['http_attempts']
    cap = min(cap, LIMITS['total_response_bytes'] - ledger['body_bytes'])
    assert cap > 0
    headers = {'Accept-Encoding': 'identity'}
    previous = [r for r in ledger['requests'] if r['url'] == url and r.get('success')]
    if byte_range:
        start, end = byte_range
        assert 0 <= start <= end and end - start + 1 <= cap
        headers['Range'] = f'bytes={start}-{end}'
        if previous:
            headers['If-Match'] = previous[0]['etag']
    entry = {'name': name, 'url': url, 'requested_range': byte_range,
             'started_at_utc': datetime.now(timezone.utc).isoformat(), 'success': False}
    ledger['requests'].append(entry)
    save_ledger(ledger)
    body = bytearray()
    try:
        with requests.Session() as session:
            with session.get(url, headers=headers, stream=True, timeout=(15, 30),
                             allow_redirects=False, verify=True) as response:
                entry.update({'status': response.status_code,
                              'headers': {k: v for k, v in response.headers.items() if k.lower() in
                                          ('content-length', 'content-range', 'content-encoding',
                                           'etag', 'last-modified', 'content-type')}})
                assert response.status_code == (206 if byte_range else 200), 'Unexpected HTTP status'
                assert response.headers.get('Content-Encoding', 'identity').lower() in ('', 'identity')
                length = int(response.headers['Content-Length'])
                assert 0 < length <= cap, 'Declared body exceeds cap'
                etag = response.headers.get('ETag')
                entry['etag'] = etag
                if byte_range:
                    match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers['Content-Range'])
                    assert match
                    got_start, got_end, total = map(int, match.groups())
                    assert (got_start, got_end) == byte_range and total > got_end
                    assert length == got_end - got_start + 1
                    assert etag and not etag.startswith('W/'), 'Need strong validator for ranges'
                    entry['object_total_bytes'] = total
                    if previous:
                        assert etag == previous[0]['etag'] and total == previous[0]['object_total_bytes']
                while len(body) < length:
                    block = response.raw.read(min(65536, length - len(body)), decode_content=False)
                    assert block, 'Truncated body'
                    body.extend(block)
                assert len(body) == length
                entry['success'] = True
    except Exception as error:
        entry['failure'] = type(error).__name__ + ': ' + str(error)
        raise
    finally:
        entry['body_bytes'] = len(body)
        entry['body_sha256'] = hashlib.sha256(body).hexdigest()
        entry['completed_at_utc'] = datetime.now(timezone.utc).isoformat()
        ledger['body_bytes'] += len(body)
        if body:
            path = ROOT / (name if entry['success'] else name + '.partial')
            path.write_bytes(body)
            entry['retained_path'] = str(path)
        save_ledger(ledger)
    print(json.dumps({k: entry[k] for k in ['name', 'status', 'body_bytes', 'body_sha256', 'success']}))
    return bytes(body)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['initial', 'ifd'])
    args = parser.parse_args()
    if args.stage == 'initial':
        for asset in PLAN['assets']:
            fetch(asset['name'], asset['url'], asset['max_encoded_bytes'])
        fetch('slice32-header.bin', PLAN['header_assets']['slice32'], 16, (0, 15))
        fetch('surface-ppm-prefix.bin', PLAN['header_assets']['surface_ppm'], 256, (0, 255))
    else:
        header = (ROOT / 'slice32-header.bin').read_bytes()
        assert header[:4] == b'II*\x00', 'Only classic little-endian TIFF is supported here'
        offset = struct.unpack_from('<I', header, 4)[0]
        count_raw = fetch('slice32-ifd-count.bin', PLAN['header_assets']['slice32'], 2, (offset, offset + 1))
        count = struct.unpack('<H', count_raw)[0]
        assert 0 < count <= 128
        fetch('slice32-ifd.bin', PLAN['header_assets']['slice32'], LIMITS['header_range_max_bytes'],
              (offset, offset + 2 + 12 * count + 4 - 1))
