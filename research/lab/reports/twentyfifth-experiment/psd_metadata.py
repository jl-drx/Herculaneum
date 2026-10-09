"""Read only PSD framing and bounded XMP/IPTC resources; no image decoder."""
import argparse
import contextlib
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import signal
import ssl
import struct
import time
import urllib.error
import urllib.request

URL = 'https://dl.ash2txt.org/fragments/Frag6/PHerc51Cr4Fr8.volpkg/working/PHerc0051Cr04Fr08_53keV_3.24um/surface_processing/inklabels.psd'
SELECTED = {1028: 'IPTC', 1060: 'XMP'}
MAX_ATTEMPTS = 104
MAX_RESERVED = 262144
MAX_METADATA = 65536
MAX_RESOURCES = 48


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def pin(path):
    path = Path(path)
    return dict(path=str(path.resolve()), size_bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def save(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.writing')
    with temporary.open('w') as stream:
        json.dump(data, stream, indent=2, sort_keys=True)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


@contextlib.contextmanager
def deadline():
    def expire(*_):
        raise TimeoutError('30-second absolute request deadline')
    old = signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, 30)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class Client:
    def __init__(self, root, opener=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.opener = opener or urllib.request.build_opener(
            NoRedirect(), urllib.request.HTTPSHandler(context=ssl.create_default_context()))
        self.etag = self.total = None
        self.log = dict(url=URL, requests=[], reserved_body_bytes=0, retained_body_bytes=0,
                        retries=0, redirects_followed=0, verified_tls=True)
        self.flush()

    def flush(self):
        save(self.root/'request-ledger.json', self.log)
        require(sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file()) <= 8*1024**2,
                'Artifact acceptance bound')

    def request(self, start=None, length=None, category='identity'):
        head = start is None
        require(head == (length is None), 'Invalid request')
        if head:
            require(not self.log['requests'], 'HEAD must be first and unique')
        else:
            require(self.total is not None and self.etag is not None, 'No object identity')
            require(type(start) is int and type(length) is int and 0 <= start and
                    0 < length <= MAX_METADATA and start + length <= self.total, 'Range bounds')
        reserve = 0 if head else length + 1
        require(len(self.log['requests']) < MAX_ATTEMPTS, 'Attempt budget')
        require(self.log['reserved_body_bytes'] + reserve <= MAX_RESERVED, 'Byte budget')
        entry = dict(method='HEAD' if head else 'GET', start=start, length=length,
                     category=category, reserved_bytes=reserve, returned_bytes=0, success=False)
        self.log['requests'].append(entry)
        self.log['reserved_body_bytes'] += reserve
        self.flush()
        headers = {'Accept-Encoding': 'identity', 'User-Agent': 'Herculaneum-Annotation-Metadata-Audit'}
        if not head:
            headers.update(Range=f'bytes={start}-{start+length-1}', **{'If-Match': self.etag})
        response = None
        body = bytearray()
        tick = time.monotonic()
        try:
            with deadline():
                try:
                    response = self.opener.open(urllib.request.Request(URL, headers=headers,
                                                method=entry['method']), timeout=15)
                except urllib.error.HTTPError as error:
                    response = error
                h = response.headers
                entry['status'] = response.status
                require(sum(len(k)+len(v)+4 for k,v in h.raw_items()) <= 16384, 'Header cap')
                keys = ['Content-Length','Content-Range','Content-Type','Content-Encoding',
                        'Transfer-Encoding','ETag','Last-Modified']
                entry['headers'] = {k:h.get_all(k,[]) for k in keys}
                require(all(len(h.get_all(k,[])) <= 1 for k in keys), 'Duplicate HTTP header')
                require(response.status == (200 if head else 206), 'Unexpected HTTP status; body unread')
                require(h.get('Content-Encoding','identity').lower() in ('','identity'), 'Encoding')
                require(h.get('Transfer-Encoding') is None, 'Transfer encoding')
                require(h.get('Content-Length','').isdigit(), 'Missing exact length')
                etag = h.get('ETag','')
                require(re.fullmatch(r'"[\x21\x23-\x7e]+"', etag) is not None, 'Strong ETag required')
                mime = h.get('Content-Type','').split(';')[0].strip().lower()
                require(mime in ('image/vnd.adobe.photoshop','image/x-photoshop',
                                 'application/octet-stream','application/x-photoshop'), 'Unexpected MIME')
                if head:
                    self.total = int(h['Content-Length']); self.etag = etag
                    require(34 <= self.total <= 128*1024**2, 'Object length outside PSD bound')
                    entry.update(success=True, object_bytes=self.total, etag=etag)
                    return
                require(etag == self.etag, 'Object ETag changed')
                require(int(h['Content-Length']) == length, 'Exact range length mismatch')
                expected = f'bytes {start}-{start+length-1}/{self.total}'
                require(h.get('Content-Range') == expected, 'Exact range/total mismatch')
                while len(body) < length:
                    try:
                        block = response.read(length-len(body))
                    except http.client.IncompleteRead as error:
                        body.extend(error.partial)
                        raise ValueError('Partial response retained') from None
                    require(block and len(block) <= length-len(body), 'Truncated/excess response')
                    body.extend(block)
                extra = response.read(1)
                body.extend(extra)
                require(not extra, 'Excess response retained')
                entry['success'] = True
                return bytes(body)
        except Exception as error:
            entry['error'] = type(error).__name__ + ': ' + str(error)
            raise
        finally:
            if response is not None:
                response.close()
            if body:
                suffix = '.bin' if entry['success'] else '.partial'
                target = self.root/f'{len(self.log["requests"]):03d}-{category}{suffix}'
                target.write_bytes(body)
                entry['body'] = pin(target)
            entry['returned_bytes'] = len(body)
            entry['elapsed_seconds'] = time.monotonic()-tick
            self.log['retained_body_bytes'] += len(body)
            self.flush()


def inspect(get, total):
    header = get(0, 30, 'header')
    require(len(header) == 30 and header[:4] == b'8BPS', 'PSD signature/header')
    version = struct.unpack('>H', header[4:6])[0]
    require(version == 1 and header[6:12] == b'\0'*6, 'Unsupported PSD version/reserved')
    channels, height, width, depth, color = struct.unpack('>HIIHH', header[12:26])
    require(1 <= channels <= 56 and 1 <= height <= 30000 and 1 <= width <= 30000,
            'PSD dimension/channel bound')
    require(depth in (1,8,16,32) and color in (0,1,2,3,4,7,8,9), 'PSD depth/color')
    color_length = struct.unpack('>I', header[26:30])[0]
    cursor = 30 + color_length
    require(cursor + 4 <= total, 'Color section outside object')
    resources_length = struct.unpack('>I', get(cursor,4,'resources-length'))[0]
    cursor += 4
    resource_start, end = cursor, cursor + resources_length
    require(end + 4 <= total, 'Resources outside object/layer length field')
    result = dict(header=dict(version=version,channels=channels,height=height,width=width,
                              depth=depth,color_mode=color,color_data_bytes=color_length),
                  resources_start=resource_start, resources_end=end, resources=[],
                  selected_metadata={}, image_or_layer_payloads_decoded=False,
                  layer_section_or_composite_payloads_requested=False)
    while cursor < end:
        require(len(result['resources']) < MAX_RESOURCES, 'Resource count bound')
        require(cursor+7 <= end, 'Resource prefix outside section')
        prefix = get(cursor,7,'resource-prefix')
        require(prefix[:4] == b'8BIM', 'Resource signature')
        ident = struct.unpack('>H',prefix[4:6])[0]
        name_length = prefix[6]
        name_field_length = (1 + name_length + 1)//2*2
        rest_length = name_field_length-1+4
        require(cursor+7+rest_length <= end, 'Resource name/size outside section')
        rest = get(cursor+7,rest_length,'resource-name-size')
        size = struct.unpack('>I',rest[-4:])[0]
        payload = cursor+7+rest_length
        next_cursor = payload+size+(size%2)
        require(next_cursor <= end, 'Resource payload outside section')
        item = dict(id=ident,name_hex=rest[:name_length].hex(),header_start=cursor,
                    payload_start=payload,payload_bytes=size,next_resource=next_cursor,
                    payload_read=ident in SELECTED)
        if ident in SELECTED:
            key = SELECTED[ident]
            require(key not in result['selected_metadata'], 'Duplicate selected metadata')
            require(size <= MAX_METADATA, 'Metadata payload cap')
            data = get(payload,size,key.lower()) if size else b''
            result['selected_metadata'][key] = dict(size_bytes=size,
                sha256=hashlib.sha256(data).hexdigest(),text_utf8=data.decode('utf-8',errors='replace'))
        result['resources'].append(item)
        cursor = next_cursor
    require(cursor == end, 'Resource extent mismatch')
    result['status'] = 'PASS'
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--go',required=True)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    go = json.loads(Path(args.go).read_text())
    require(go['status']=='GO' and go['url']==URL, 'Missing exact metadata-only GO')
    for item in go['frozen_files']:
        require(pin(item['path']) == item, 'Frozen source/fixture/protocol changed')
    client = Client(args.output)
    try:
        client.request()
        result = inspect(lambda start,length,kind:client.request(start,length,kind),client.total)
        for item in go['frozen_files']:
            require(pin(item['path']) == item, 'Source changed during acquisition')
        result.update(url=URL,etag=client.etag,object_bytes=client.total,go=pin(args.go),
                      http_attempts=len(client.log['requests']),
                      returned_body_bytes=client.log['retained_body_bytes'])
        save(client.root/'PSD_METADATA.json',result)
        print(json.dumps({k:result[k] for k in ['status','http_attempts','returned_body_bytes','header']}))
    except Exception as error:
        save(client.root/'STOP.json',dict(status='STOP',reason=type(error).__name__+': '+str(error)))
        raise


if __name__ == '__main__':
    main()
