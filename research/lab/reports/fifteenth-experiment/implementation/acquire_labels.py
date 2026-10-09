"""Root-executed six-object acquisition; no image decoding, redirects or retries."""
from pathlib import Path
import argparse
import datetime
import hashlib
import http.client
import json
import signal
import ssl
import urllib.error
import urllib.parse
import urllib.request
from label_common import *


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def fetch(obj, root, ledger, opener):
    url = obj['url']
    parsed = urllib.parse.urlsplit(url)
    require(parsed.scheme == 'https' and parsed.netloc == 'dl.ash2txt.org'
            and not parsed.query and not parsed.fragment and parsed.path.endswith('.png'), 'Invalid source URL')
    require(len(ledger['requests']) < LIMITS['http_attempts'], 'Attempt budget exhausted')
    cap = min(LIMITS['maximum_response_bytes'], LIMITS['total_response_bytes'] - ledger['charged_response_bytes'])
    require(cap > 0, 'Body budget exhausted')
    entry = dict(object=obj, attempt=len(ledger['requests'])+1, success=False, body_bytes=0,
                 reserved_cap_bytes=cap, charged_response_bytes=cap,
                 started_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    ledger['requests'].append(entry)
    ledger['charged_response_bytes'] += cap
    save(root / 'request-ledger.json', ledger)
    partial = root / (obj['fragment'] + '-' + obj['kind'] + '.png.partial')
    digest, retained, response = hashlib.sha256(), 0, None
    try:
        try:
            response = opener.open(urllib.request.Request(url, headers={'Accept-Encoding': 'identity'}), timeout=25)
        except urllib.error.HTTPError as error:
            response = error
        entry['http_status'] = response.status
        entry['http_identity'] = {k.lower(): v for k, v in response.headers.items() if k.lower() in
                                  ('content-length', 'content-type', 'content-encoding', 'etag', 'last-modified')}
        require(response.headers.get('Content-Encoding', 'identity').lower() in ('', 'identity'), 'Transport encoding rejected')
        declared = response.headers.get('Content-Length')
        require(declared is None or declared.isdigit(), 'Invalid declared length')
        length = int(declared) if declared is not None else None
        require(length is None or 0 < length <= cap, 'Declared response exceeds cap')
        with partial.open('xb') as stream:
            while retained < (length if length is not None else cap):
                remaining = (length if length is not None else cap)-retained
                try:
                    block = response.read(min(65536, remaining))
                except http.client.IncompleteRead as error:
                    block = error.partial
                    require(len(block) <= remaining, 'Truncated response partial exceeds cap')
                    stream.write(block)
                    digest.update(block)
                    retained += len(block)
                    raise ValueError('Truncated response; retained partial bytes') from None
                if not block:
                    entry['complete_eof'] = True
                    break
                require(len(block) <= remaining, 'Returned block exceeds bounded read')
                stream.write(block)
                digest.update(block)
                retained += len(block)
        require(length is not None and retained == length or length is None and entry.get('complete_eof'),
                'Truncated body or unknown-length response reached cap')
        require(response.status == 200, 'Only HTTP200 is allowed; no fill/redirect/retry')
        # CRC and dimensions are structural checks, with no image pixel decoding.
        entry['png_structure'] = png_structure(partial.read_bytes(), obj['shape_yx'])
        target = partial.with_suffix('')
        partial.replace(target)
        entry['file'] = record(target)
        entry['success'] = True
        ledger['charged_response_bytes'] -= cap-retained
        entry['charged_response_bytes'] = retained
        return entry['file']
    except Exception as error:
        entry['failure_type'] = type(error).__name__
        raise
    finally:
        if response is not None:
            response.close()
        entry.update(body_bytes=retained, body_sha256=digest.hexdigest(),
                     completed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        if partial.exists():
            entry['partial_file'] = record(partial)
        ledger['response_body_bytes'] += retained
        save(root / 'request-ledger.json', ledger)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    plan = load_plan(args.plan)
    root = Path(args.output_dir).resolve()
    require(not root.exists(), 'Fresh acquisition directory required')
    root.mkdir(parents=True)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Acquisition wall cap')))
    signal.alarm(LIMITS['acquisition_wall_seconds'])
    ledger = dict(schema_version=1, plan=record(args.plan), requests=[], response_body_bytes=0,
                  charged_response_bytes=0,
                  maximums=LIMITS, redirects_followed=0, automatic_retries=0,
                  verified_tls=True, accounting='Successful bodies charge actual retained bytes; failed attempts charge full reserved cap; retained partial bytes separate. Not total wire traffic')
    result = dict(schema_version=1, status='STARTED', plan=record(args.plan), objects=[],
                  image_pixel_decodes=0, model_calls=0)
    opener = urllib.request.build_opener(NoRedirect(), urllib.request.HTTPSHandler(context=ssl.create_default_context()))
    try:
        for obj in plan['objects']:
            result['objects'].append(dict(**obj, file=fetch(obj, root, ledger, opener)))
        require(len(result['objects']) == 6, 'All six objects must complete')
        load_plan(args.plan)
        result['status'] = 'PASS'
    except Exception as error:
        result.update(status='STOP', failure_type=type(error).__name__)
        raise
    finally:
        signal.alarm(0)
        result.update(http_attempts=len(ledger['requests']), response_body_bytes=ledger['response_body_bytes'],
                      charged_response_bytes=ledger['charged_response_bytes'],
                      request_ledger=record(root / 'request-ledger.json'), output_bytes=directory_bytes(root))
        save(root / 'provenance.json', result)
    print(json.dumps({'status': 'PASS', 'provenance': record(root / 'provenance.json')}))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps({'status': 'STOP', 'failure_type': type(error).__name__}))
        raise SystemExit(1)
