"""Small strict Range transport with raw-body retention and conservative failure charges."""
from pathlib import Path
import datetime
import hashlib
import http.client
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from probe_common import *


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class RangeClient:
    def __init__(self, root, urls, limits, plan_record, pins=None, opener=None):
        self.root, self.urls, self.limits = Path(root), set(urls), limits
        require(not self.root.exists(), 'Fresh stage directory required')
        self.root.mkdir(parents=True)
        (self.root/'raw').mkdir()
        self.pins = pins or {}
        self.opener = opener or urllib.request.build_opener(NoRedirect(), urllib.request.HTTPSHandler(context=ssl.create_default_context()))
        self.ledger = dict(schema_version=1, plan=plan_record, limits=limits, requests=[],
                           charged_response_bytes=0, retained_body_bytes=0, redirects_followed=0,
                           automatic_retries=0, verified_tls=True,
                           accounting='Reserve bounded response cap before attempt; successful complete bodies charge actual bytes, failed attempts retain full reservation. Partial stored bytes separate; not total wire traffic.')
        save(self.root/'request-ledger.json', self.ledger)

    def get(self, url, ranges, name, mode='single'):
        require(url in self.urls, 'URL outside frozen inventory')
        parsed = urllib.parse.urlsplit(url)
        require(parsed.scheme == 'https' and parsed.netloc == 'dl.ash2txt.org' and not parsed.query
                and not parsed.fragment and parsed.path.endswith('/07.tif'), 'Exact public plane07 HTTPS source required')
        require(mode in ('single','multipart') and len(ranges) == (1 if mode == 'single' else 8), 'Frozen range count changed')
        require(all(type(a) is int and type(b) is int and 0 <= a <= b for a,b in ranges), 'Invalid Range')
        require(Path(name).name == name and not name.startswith('.'), 'Invalid artifact name')
        useful = sum(b-a+1 for a,b in ranges)
        cap = useful+(1 if mode == 'single' else LIMITS['multipart_framing_bytes'])
        require(cap <= self.limits['per_response'] and len(self.ledger['requests']) < self.limits['attempts']
                and self.ledger['charged_response_bytes']+cap <= self.limits['body_bytes'], 'Frozen response/attempt budget exhausted')
        pin = self.pins.get(url)
        if pin:
            require(all(b < pin['object_total_bytes'] for _,b in ranges), 'Range exceeds pinned object')
        headers = {'Range': 'bytes='+','.join(f'{a}-{b}' for a,b in ranges), 'Accept-Encoding': 'identity'}
        if pin:
            headers['If-Match'] = pin['etag']
        started = time.monotonic()
        entry = dict(url=url, ranges_inclusive=ranges, mode=mode, request_headers=headers,
                     started_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), success=False,
                     reserved_cap_bytes=cap, charged_response_bytes=cap, retained_body_bytes=0)
        self.ledger['requests'].append(entry)
        self.ledger['charged_response_bytes'] += cap
        save(self.root/'request-ledger.json', self.ledger)
        path, partial = self.root/'raw'/name, self.root/'raw'/(name+'.partial')
        require(not path.exists() and not partial.exists(), 'Artifact already exists')
        digest, retained, response = hashlib.sha256(), 0, None
        try:
            try:
                response = self.opener.open(urllib.request.Request(url, headers=headers), timeout=25)
            except urllib.error.HTTPError as error:
                response = error
            entry['http_status'] = response.status
            entry['http_identity'] = {k.lower():v for k,v in response.headers.items() if k.lower() in
                                      ('content-length','content-range','content-type','content-encoding','etag','last-modified')}
            for key in ('Content-Length','Content-Range','Content-Type','Content-Encoding','ETag'):
                values = response.headers.get_all(key) if hasattr(response.headers,'get_all') else (
                    [response.headers[key]] if key in response.headers else [])
                require(len(values or []) <= 1, 'Duplicate singleton HTTP identity/framing header')
            require(response.status == 206, 'HTTP200/redirect/error rejected before any body read; no fallback')
            require(response.headers.get('Content-Encoding','identity').lower() in ('','identity'), 'Response encoding rejected')
            etag = response.headers.get('ETag','')
            require(re.fullmatch(r'"[\x21\x23-\x7e]+"', etag) is not None, 'Quoted strong ETag required')
            if pin:
                require(etag == pin['etag'], 'ETag changed')
            if mode == 'single':
                start,end,total = content_range(response.headers.get('Content-Range'))
                require([start,end] == list(ranges[0]), 'Range mismatch')
                if pin:
                    require(total == pin['object_total_bytes'], 'Object size changed')
            else:
                require(pin is not None and response.headers.get('Content-Range') is None, 'Multipart must use audited object identity')
                total = pin['object_total_bytes']
            declared = response.headers.get('Content-Length')
            require(declared is None or declared.isdigit(), 'Invalid response length')
            length = int(declared) if declared is not None else None
            require(length is None or 0 < length <= cap, 'Declared body exceeds bounded cap')
            if mode == 'single' and length is not None:
                require(length == useful, 'Single-range length mismatch')
            with partial.open('xb') as stream:
                while retained < (length if length is not None else cap):
                    remaining = (length if length is not None else cap)-retained
                    try:
                        block = response.read(min(65536,remaining))
                    except http.client.IncompleteRead as error:
                        block = error.partial
                        require(len(block) <= remaining, 'Partial read exceeds cap')
                        stream.write(block); digest.update(block); retained += len(block)
                        raise ContractError('Truncated transfer; partial preserved') from None
                    if not block:
                        entry['complete_eof'] = True
                        break
                    require(len(block) <= remaining, 'Bounded read overflow')
                    stream.write(block); digest.update(block); retained += len(block)
            require(length is not None and retained == length or length is None and entry.get('complete_eof'), 'Incomplete bounded response')
            if mode == 'single':
                require(retained == useful, 'Single range actual length mismatch')
            else:
                entry['parts'] = parse_multipart(partial.read_bytes(), response.headers.get('Content-Type'), ranges, total)
            partial.replace(path)
            entry.update(success=True, etag=etag, object_total_bytes=total, file=record(path),
                         charged_response_bytes=retained)
            self.ledger['charged_response_bytes'] -= cap-retained
            self.pins.setdefault(url, dict(etag=etag, object_total_bytes=total))
            return path, entry
        except Exception as error:
            entry['failure_type'] = type(error).__name__
            if isinstance(error,ContractError):
                entry['stop_reason'] = str(error)
            raise
        finally:
            if response is not None:
                response.close()
            entry.update(retained_body_bytes=retained, body_sha256=digest.hexdigest(),
                         elapsed_seconds=time.monotonic()-started,
                         completed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            if partial.exists():
                entry['partial_file'] = record(partial)
            self.ledger['retained_body_bytes'] += retained
            save(self.root/'request-ledger.json', self.ledger)
