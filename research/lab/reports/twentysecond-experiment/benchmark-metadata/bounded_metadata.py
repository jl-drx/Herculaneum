"""Exact six public text requests. Absolute deadline, durable receipts, no retries."""
import contextlib
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

BASE = Path(__file__).resolve().parent
PLAN = BASE.parent / 'benchmark-inventory/METADATA_QUERY_PROPOSAL.json'
GO = BASE.parent / 'PHERC0841_METADATA_GO.json'
PLAN_SHA = '6087fb9c5e82b5418d6113fa17575d58d865cedc7d8b0a1ec76e3829c492201a'
GO_SHA = '225c2d47aaecc23d80a1ee165f86a5b9d92e8a4cba6f9404db308d2e962f8297'
HOST = 'vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com'

class Stop(RuntimeError):
    pass

def require(condition, reason):
    if not condition:
        raise Stop(reason)

def digest(data):
    return hashlib.sha256(data).hexdigest()

def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)

def validate_url(url):
    u = urllib.parse.urlsplit(url)
    require(u.scheme == 'https' and u.hostname == HOST and u.port in (None, 443)
            and u.username is None and u.password is None and not u.fragment, 'URL outside exact HTTPS scope')

def unique_json_pairs(pairs):
    out = {}
    for k, v in pairs:
        require(k not in out, 'Duplicate JSON key')
        out[k] = v
    return out

def parse_body(query, body):
    text = body.decode('utf-8')
    if query['id'].endswith('package-inventory'):
        require('<!DOCTYPE' not in text.upper() and '<!ENTITY' not in text.upper(), 'XML declarations forbidden')
        root = ET.fromstring(text)
        ns = '{http://s3.amazonaws.com/doc/2006-03-01/}'
        require(root.tag == ns + 'ListBucketResult', 'Unexpected XML root')
        require(root.findtext(ns + 'IsTruncated') == 'false', 'Truncated/unknown inventory')
        prefix = urllib.parse.parse_qs(urllib.parse.urlsplit(query['url']).query)['prefix'][0]
        require(root.findtext(ns + 'Prefix') == prefix, 'Wrong inventory prefix')
        require(root.findtext(ns + 'Name') == 'vesuvius-challenge-open-data', 'Wrong bucket')
        contents = root.findall(ns + 'Contents')
        require(len(contents) <= 100 and int(root.findtext(ns + 'KeyCount')) == len(contents), 'Inventory count mismatch')
        objects = []
        for item in contents:
            key = item.findtext(ns + 'Key')
            require(key and key.startswith(prefix), 'Out-of-prefix key')
            objects.append({'key': key, 'size': int(item.findtext(ns + 'Size')), 'etag': item.findtext(ns + 'ETag')})
        require(len({x['key'] for x in objects}) == len(objects), 'Duplicate object key')
        return {'kind': 'complete_prefix_inventory', 'prefix': prefix, 'is_truncated': False, 'objects': objects}
    out = json.loads(text, object_pairs_hook=unique_json_pairs)
    require(isinstance(out, dict) and out.get('zarr_format') == 3 and out.get('node_type') == 'group', 'Unexpected Zarr root schema')
    return {'kind': 'zarr_root_metadata', 'metadata': out}

def check_headers(response, query):
    require(response.geturl() == query['url'], 'Redirect/final URL changed')
    require(response.status == 200, 'HTTP status is not 200')
    h = response.headers
    transfer = h.get('Transfer-Encoding', '').strip().lower()
    require(transfer in ('', 'chunked'), 'Unexpected Transfer-Encoding')
    require(h.get('Content-Encoding', 'identity').strip().lower() == 'identity', 'Content-Encoding forbidden')
    values = h.get_all('Content-Length', [])
    require(len(values) <= 1 and not (transfer and values), 'Ambiguous response framing')
    if values:
        require(values[0].isascii() and values[0].isdigit(), 'Invalid Content-Length')
        size = int(values[0])
        require(0 < size <= query['max_body_bytes'], 'Response body cap')
    else:
        size = None
    mime = h.get('Content-Type', '').split(';')[0].strip().lower()
    expected = {'application/xml', 'text/xml'} if query['id'].endswith('package-inventory') else {'application/json', 'text/plain'}
    require(mime in expected, 'Unexpected text Content-Type')
    return size

def copy_body(response, size, stream, cap=None):
    count = 0
    limit = size if size is not None else cap
    require(limit is not None and limit > 0, 'Missing read bound')
    while count < limit:
        chunk = response.read1(min(4096, limit - count))
        if not chunk:
            require(size is None, 'Short body')
            return count
        require(len(chunk) <= limit - count, 'Reader exceeded requested length')
        stream.write(chunk)
        stream.flush()
        os.fsync(stream.fileno())
        count += len(chunk)
    require(size is not None, 'Unframed body reached cap; completeness unverified')
    return count

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None

@contextlib.contextmanager
def absolute_deadline(seconds):
    def alarm(*_):
        raise Stop('Absolute request deadline exceeded')
    old = signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)

def run(out):
    require(not out.exists(), 'Output already exists; no retry/resume')
    require(digest(PLAN.read_bytes()) == PLAN_SHA and digest(GO.read_bytes()) == GO_SHA, 'Plan/GO pin mismatch')
    freeze = json.loads((BASE / 'SOURCE_FREEZE.json').read_text())
    for source in freeze['sources']:
        data = Path(source['path']).read_bytes()
        require(len(data) == source['byte_length'] and digest(data) == source['sha256'], 'Source changed after freeze')
    require(freeze['fixtures_status'] == 'PASS', 'Fixtures not passed')
    plan = json.loads(PLAN.read_text())
    require(len(plan['queries']) == 6 and sum(q['max_body_bytes'] for q in plan['queries']) == 294912, 'Fixed budget mismatch')
    for q in plan['queries']:
        validate_url(q['url'])
    out.mkdir()
    ledger = {'started_utc': utc(), 'plan_sha256': PLAN_SHA, 'go_sha256': GO_SHA,
              'source_freeze_sha256': digest((BASE / 'SOURCE_FREEZE.json').read_bytes()),
              'maximum_attempts': 6, 'maximum_observed_body_bytes': 294912, 'requests': []}
    ledger_path = out / 'request-ledger.json'
    atomic_json(ledger_path, ledger)
    context = ssl.create_default_context()
    require(context.check_hostname and context.verify_mode == ssl.CERT_REQUIRED, 'TLS verification disabled')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect(), urllib.request.HTTPSHandler(context=context))
    status = 'PASS'
    started = time.monotonic()
    for query in plan['queries']:
        rec = {'id': query['id'], 'url': query['url'], 'started_utc': utc(),
               'reserved_body_bytes': query['max_body_bytes'], 'status': 'STARTED', 'observed_body_bytes': 0}
        ledger['requests'].append(rec)
        require(len(ledger['requests']) <= 6, 'Attempt budget')
        require(sum(r['reserved_body_bytes'] for r in ledger['requests']) <= 294912, 'Reserved budget')
        atomic_json(ledger_path, ledger)
        body_path = out / (query['id'] + '.body')
        body_path.touch(exist_ok=False)
        try:
            with absolute_deadline(30):
                req = urllib.request.Request(query['url'], headers={'User-Agent': 'Herculaneum-phase22-metadata/1.0', 'Accept-Encoding': 'identity'})
                try:
                    response = opener.open(req, timeout=30)
                except urllib.error.HTTPError as error:
                    response = error
                with response:
                    rec['http_status'] = response.status
                    rec['headers'] = {k: v for k, v in response.headers.items() if k.lower() in ['content-type', 'content-length', 'etag', 'last-modified', 'content-encoding', 'transfer-encoding']}
                    atomic_json(ledger_path, ledger)
                    size = check_headers(response, query)
                    with body_path.open('wb') as stream:
                        copy_body(response, size, stream, query['max_body_bytes'])
                parsed = parse_body(query, body_path.read_bytes())
                atomic_json(out / (query['id'] + '.parsed.json'), parsed)
                rec['status'] = 'PASS'
        except Exception as error:
            rec['status'] = 'STOP'
            rec['error_type'] = type(error).__name__
            rec['error'] = str(error) if isinstance(error, Stop) else 'Transport or parser exception; no automatic retry'
            status = 'STOP'
        finally:
            body = body_path.read_bytes()
            rec.update(observed_body_bytes=len(body), body_sha256=digest(body), body_path=str(body_path), completed_utc=utc())
            atomic_json(ledger_path, ledger)
        print(json.dumps({'id': query['id'], 'status': rec['status'], 'http_status': rec.get('http_status'), 'bytes': rec['observed_body_bytes']}), flush=True)
        if status != 'PASS':
            break
    ledger.update(completed_utc=utc(), status=status, total_observed_body_bytes=sum(r['observed_body_bytes'] for r in ledger['requests']), wall_seconds=time.monotonic()-started)
    require(ledger['total_observed_body_bytes'] <= 294912, 'Observed budget')
    atomic_json(ledger_path, ledger)
    return 0 if status == 'PASS' else 1

if __name__ == '__main__':
    sys.exit(run(Path(sys.argv[1])))
