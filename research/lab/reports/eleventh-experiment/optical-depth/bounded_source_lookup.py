"""Bounded public HTTPS source-code lookup; no redirects or retries."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import ssl
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parent
LEDGER = ROOT / 'request-ledger.json'
MAX_ATTEMPTS = 12
MAX_TOTAL = 1024 * 1024
MAX_BODY = 256 * 1024


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def fetch(name, url, discovered_in):
    parsed = urllib.parse.urlsplit(url)
    if (parsed.scheme != 'https' or parsed.username or parsed.password
            or parsed.hostname not in {'api.github.com', 'raw.githubusercontent.com', 'github.com'}):
        raise ValueError('Public official GitHub HTTPS source only')
    if '/' in name or name.startswith('.') or (ROOT / name).exists():
        raise ValueError('Fresh ordinary evidence filename required')
    evidence = Path(discovered_in).resolve(strict=True)
    ledger = json.loads(LEDGER.read_text()) if LEDGER.exists() else {
        'attempt_limit': MAX_ATTEMPTS, 'total_response_body_limit_bytes': MAX_TOTAL,
        'per_response_body_limit_bytes': MAX_BODY, 'redirects': False,
        'automatic_retries': 0, 'verified_tls': True, 'attempts': [],
        'response_body_bytes': 0,
        'scope': 'Text metadata/source only; no arrays/images/checkpoints/models',
    }
    if len(ledger['attempts']) >= MAX_ATTEMPTS:
        raise ValueError('Attempt budget exhausted')
    cap = min(MAX_BODY, MAX_TOTAL - ledger['response_body_bytes'])
    if cap <= 0:
        raise ValueError('Body budget exhausted')
    record = {'attempt': len(ledger['attempts']) + 1, 'url': url,
              'discovered_in': str(evidence),
              'discovered_in_sha256': hashlib.sha256(evidence.read_bytes()).hexdigest(),
              'started_at_utc': datetime.now(timezone.utc).isoformat(),
              'response_body_bytes': 0, 'status': 'started'}
    ledger['attempts'].append(record)
    LEDGER.write_text(json.dumps(ledger, indent=2) + '\n')
    payload = bytearray()
    response = None
    try:
        opener = urllib.request.build_opener(
            NoRedirect(), urllib.request.HTTPSHandler(context=ssl.create_default_context()))
        request = urllib.request.Request(url, headers={
            'Accept-Encoding': 'identity', 'User-Agent': 'Herculaneum-bounded-source-audit',
            'Accept': 'application/vnd.github+json' if parsed.hostname == 'api.github.com' else 'text/plain',
        })
        try:
            response = opener.open(request, timeout=30)
        except urllib.error.HTTPError as error:
            response = error
        record.update({'http_status': response.status,
                       'content_type': response.headers.get('Content-Type'),
                       'declared_content_length': response.headers.get('Content-Length'),
                       'location': response.headers.get('Location'),
                       'etag': response.headers.get('ETag')})
        kind = (record['content_type'] or '').lower()
        declared = record['declared_content_length']
        if kind.startswith(('image/', 'video/', 'audio/')):
            record['status'] = 'rejected_non_text_media'
        elif response.headers.get('Content-Encoding', 'identity') != 'identity':
            record['status'] = 'rejected_encoded_response'
        elif declared and declared.isdigit() and int(declared) > cap:
            record['status'] = 'rejected_declared_size_exceeds_cap'
        else:
            complete = False
            while len(payload) < cap:
                block = response.read(min(8192, cap - len(payload)))
                if not block:
                    complete = True
                    break
                payload.extend(block)
            if declared and declared.isdigit() and len(payload) == int(declared):
                complete = True
            record['body_complete'] = complete
            record['status'] = 'complete' if complete else 'bounded_partial_body'
            if 300 <= response.status < 400:
                record['status'] = 'redirect_not_followed'
            elif response.status >= 400:
                record['status'] = 'http_failure_preserved'
    except Exception as error:
        record['status'] = 'request_failed'
        record['error_type'] = type(error).__name__
    finally:
        if response is not None:
            response.close()
        record['response_body_bytes'] = len(payload)
        if payload:
            target = ROOT / name
            target.write_bytes(payload)
            record.update({'saved_path': str(target),
                           'sha256': hashlib.sha256(payload).hexdigest()})
        record['completed_at_utc'] = datetime.now(timezone.utc).isoformat()
        ledger['response_body_bytes'] += len(payload)
        LEDGER.write_text(json.dumps(ledger, indent=2) + '\n')
    print(json.dumps(record))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name')
    parser.add_argument('url')
    parser.add_argument('--discovered-in', required=True)
    args = parser.parse_args()
    fetch(args.name, args.url, args.discovered_in)
