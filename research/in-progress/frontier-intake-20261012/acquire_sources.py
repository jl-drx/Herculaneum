"""Bounded public source acquisition; no inference and no implicit retries."""
from __future__ import annotations
import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
from pathlib import Path
import urllib.error
import urllib.parse
import urllib.request


def clean_url(url):
    p = urllib.parse.urlsplit(url)
    return urllib.parse.urlunsplit((p.scheme, p.netloc, p.path, '', ''))


def acquire(item, output):
    name = item['id']
    if not name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in name):
        raise ValueError('Unsafe source identifier')
    target = output / (name + '.raw')
    receipt_path = output / (name + '.receipt.json')
    if target.exists() or receipt_path.exists():
        raise FileExistsError(name)
    receipt = {'id': name, 'url': item['url'], 'utc': dt.datetime.now(dt.timezone.utc).isoformat()}
    try:
        req = urllib.request.Request(item['url'], headers={'User-Agent': 'Plinius-intake/1.0'})
        with urllib.request.urlopen(req, timeout=40) as response:
            cap = item.get('max_bytes', 4 * 1024 * 1024)
            data = response.read(cap + 1)
            if len(data) > cap:
                raise ValueError('Source exceeds frozen byte cap')
            if response.status != 200:
                raise ValueError('Unexpected HTTP status')
            receipt.update(status=response.status, final_url=clean_url(response.url),
                           etag=response.headers.get('ETag'),
                           content_type=response.headers.get('Content-Type'),
                           link=response.headers.get('Link'))
        digest = hashlib.sha256(data).hexdigest()
        if item.get('sha256') and digest != item['sha256']:
            raise ValueError('Source checksum mismatch')
        target.write_bytes(data)
        receipt.update(file=target.name, bytes=len(data), sha256=digest)
    except Exception as error:
        receipt.update(error_type=type(error).__name__)
        if isinstance(error, urllib.error.HTTPError):
            receipt['http_status'] = error.code
        # Raw exceptions/redirect query strings may contain signed URL material.
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plan = json.loads(args.plan.read_text())
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(lambda row: acquire(row, args.output), plan))
    print(json.dumps([{k: v for k, v in row.items() if k in ('id', 'bytes', 'status', 'error_type', 'http_status')} for row in rows], indent=2))
    if any('error_type' in row for row in rows):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
