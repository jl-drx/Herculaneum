"""Retain bounded public research source bodies under one shared10MiB budget."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urlparse

import requests

ROOT = Path(__file__).resolve().parent
LIMIT = 10 * 1024 * 1024


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('url')
    ap.add_argument('--label', required=True)
    ap.add_argument('--subdir', required=True)
    ap.add_argument('--max-bytes', type=int, default=1024 * 1024)
    args = ap.parse_args()
    parsed = urlparse(args.url)
    if parsed.scheme != 'https' or parsed.username or parsed.password or parsed.fragment:
        raise ValueError('Use a public HTTPS source URL without embedded credentials or fragment.')
    if not re.fullmatch(r'[a-zA-Z0-9_.-]+', args.label) or not re.fullmatch(r'[a-zA-Z0-9_-]+', args.subdir):
        raise ValueError('Use safe source labels and subdirectory names.')
    if not 1 <= args.max_bytes <= 2 * 1024 * 1024:
        raise ValueError('Per-request cap must be within1byte–2MiB.')
    folder = ROOT / 'raw-sources' / args.subdir
    folder.mkdir(parents=True, exist_ok=True)
    ledger = ROOT / 'network-ledger.json'
    lock = ROOT / 'network-budget.lock'
    with lock.open('a') as hold:
        fcntl.flock(hold, fcntl.LOCK_EX)
        log = json.loads(ledger.read_text()) if ledger.exists() else dict(
            maximum_response_body_bytes=LIMIT, actual_response_body_bytes=0, requests=[])
        remaining = LIMIT - log['actual_response_body_bytes']
        if remaining <= 0:
            raise ValueError('Shared research body budget is exhausted.')
        cap = min(remaining, args.max_bytes)
        filename = args.label + '-' + hashlib.sha256(args.url.encode()).hexdigest()[:12] + '.body'
        destination = folder / filename
        if destination.exists():
            raise ValueError('Retained source already exists; read it instead of refetching.')
        rec = dict(url=args.url, retrieved_at_utc=datetime.now(timezone.utc).isoformat(),
                   local_path=str(destination), request_body_cap_bytes=cap,
                   automatic_retries=0, redirects_followed=False)
        data = bytearray()
        try:
            with requests.get(args.url, stream=True, allow_redirects=False, timeout=(15, 45),
                              headers={'User-Agent': 'Herculaneum-methods-audit/1.0',
                                       'Accept-Encoding': 'identity'}) as response:
                rec.update(http_status=response.status_code, content_type=response.headers.get('Content-Type'),
                           content_length=response.headers.get('Content-Length'), etag=response.headers.get('ETag'),
                           last_modified=response.headers.get('Last-Modified'), location=response.headers.get('Location'))
                while len(data) < cap:
                    block = response.raw.read(min(8192, cap - len(data)), decode_content=False)
                    if not block:
                        break
                    data.extend(block)
                declared = rec['content_length']
                rec['complete_body'] = (len(data) < cap or (declared is not None and int(declared) == len(data)))
                rec['content_encoding'] = response.headers.get('Content-Encoding')
        except requests.RequestException as error:
            rec.update(error_type=type(error).__name__, complete_body=False)
        destination.write_bytes(data)
        rec.update(size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        metadata = destination.with_suffix('.metadata.json')
        metadata.write_text(json.dumps(rec, indent=2) + '\n')
        log['actual_response_body_bytes'] += len(data)
        log['requests'].append(rec)
        ledger.write_text(json.dumps(log, indent=2) + '\n')
        destination.chmod(0o444)
        metadata.chmod(0o444)
        print(json.dumps(rec, indent=2))
        print('shared_actual_response_body_bytes', log['actual_response_body_bytes'])


if __name__ == '__main__':
    main()
