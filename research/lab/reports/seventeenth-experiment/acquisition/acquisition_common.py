"""Phase17 immutable-input accounting and strict243-part parser; no scientific decoder.
Multipart parser adapted additively from phase16 production_common.py SHA256
2b18a2e229bc773d580daadf98d9ce2870cfadc6e6e4fec804a2b3042f1ba74a.
"""
from pathlib import Path
import hashlib
import json
import os
import re

ROOT=Path(__file__).resolve().parent
LIMITS={'part_header_bytes':512,'framing_bytes_per_part':768,'framing_base_bytes':1024,'rows':[243]}
STAGE_LIMITS={'headers-ir':{'attempts':9,'body_bytes':96*1024**2+6*1025,'per_response':48*1024**2,'wall_seconds':600},
              'samples':{'attempts':12,'body_bytes':2671680,'per_response':222640,'wall_seconds':600}}
LATTICE=[32,72,112,152,192,232,272,312,351]
EXPECTED_CASES=[('frag1-context1',[2304,1152,1152,1152]),('frag1-context2',[0,6912,1152,1152]),
('frag2-context1',[1152,1152,1152,1152]),('frag2-context2',[8064,12672,1152,1152]),
('frag3-context1',[2304,1152,1152,1152]),('frag3-context2',[3456,5760,1152,1152])]
CODE_NAMES=['acquisition_common.py','regional_http.py','prepare_acquisition.py','run_acquisition.py','check_protocol_fixtures.py']

class ContractError(ValueError):
    """Authored nonsecret validation text may be retained; transport exceptions remain redacted."""
    pass


def require(value, message):
    if not value:
        raise ContractError(message)


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def record(path):
    path = Path(path).resolve()
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha_file(path)}


def verify(item):
    path = Path(item['path']).resolve(strict=True)
    require(path.stat().st_size == item.get('bytes', item.get('size_bytes')), 'Pinned input size changed')
    require(sha_file(path) == item['sha256'], 'Pinned input hash changed')
    return path


def save(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.writing')
    with temp.open('w') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
        stream.flush(); os.fsync(stream.fileno())
    temp.replace(path)
    fd=os.open(path.parent,os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def nested_records(value):
    if isinstance(value, dict):
        if {'path', 'sha256'} <= value.keys() and ('bytes' in value or 'size_bytes' in value):
            yield value
        for v in value.values():
            yield from nested_records(v)
    elif isinstance(value, list):
        for v in value:
            yield from nested_records(v)


def content_range(value):
    match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', value or '')
    require(match is not None, 'Exact Content-Range required')
    start, end, total = map(int, match.groups())
    require(0 <= start <= end < total, 'Invalid Content-Range bounds')
    return start, end, total


def parse_multipart(body, content_type, ranges, total):
    """Length-based binary parsing; delimiter-like payloads are ordinary bytes."""
    require(len(body) <= sum(b-a+1 for a,b in ranges)+LIMITS['framing_bytes_per_part']*len(ranges)+LIMITS['framing_base_bytes'], 'Multipart body exceeds framing allowance')
    boundary_match = re.fullmatch(r'multipart/byteranges\s*;\s*boundary=(?:"([A-Za-z0-9\x27()+_,./:=?-]{1,70})"|([A-Za-z0-9\x27+_.-]{1,70}))\s*', content_type or '', re.I)
    require(boundary_match is not None, 'One bounded multipart boundary required')
    boundary = (boundary_match[1] or boundary_match[2]).encode('ascii')
    require(len(ranges) in LIMITS['rows'] and len(set(map(tuple,ranges))) == len(ranges), 'Exactly243 unique ranges required')
    for i, span in enumerate(sorted(ranges)):
        require(0 <= span[0] <= span[1] < total, 'Requested multipart span invalid')
        if i:
            require(sorted(ranges)[i-1][1] < span[0], 'Requested multipart spans overlap')
    marker = b'--'+boundary
    prefix = 2 if body.startswith(b'\r\n') else 0
    require(body[prefix:].startswith(marker+b'\r\n'), 'Multipart preamble/framing rejected')
    cursor, seen, parts = prefix+len(marker)+2, set(), []
    while True:
        end_headers = body.find(b'\r\n\r\n', cursor, cursor+LIMITS['part_header_bytes']+4)
        require(end_headers >= 0 and end_headers-cursor <= LIMITS['part_header_bytes'], 'Multipart header exceeds bound')
        raw = body[cursor:end_headers]
        try:
            lines = raw.decode('ascii').split('\r\n')
        except UnicodeDecodeError:
            raise ContractError('Non-ASCII multipart header') from None
        headers = {}
        for line in lines:
            require(re.fullmatch(r'[A-Za-z0-9-]+:[ \t]*[^\r\n]+', line) is not None, 'Invalid multipart header')
            name, value = line.split(':', 1)
            name, value = name.lower(), value.strip()
            require(all(32 <= ord(c) <= 126 or c == '\t' for c in value), 'Control character in MIME header')
            require(name not in headers and name in ('content-type','content-range','content-length'),
                    'Duplicate/unsupported MIME header or transfer encoding')
            headers[name] = value
        start, end, observed_total = content_range(headers.get('content-range'))
        span = (start, end)
        require(observed_total == total and span in set(map(tuple, ranges)) and span not in seen,
                'Unrequested/duplicate/coalesced multipart range or changed object total')
        size = end-start+1
        if 'content-length' in headers:
            require(headers['content-length'].isdigit() and int(headers['content-length']) == size, 'Part length mismatch')
        payload_offset = end_headers+4
        require(payload_offset+size <= len(body), 'Truncated multipart payload')
        payload = body[payload_offset:payload_offset+size]
        parts.append(dict(range_inclusive=[start,end], payload_offset=payload_offset, payload_bytes=size,
                          payload_sha256=hashlib.sha256(payload).hexdigest(), headers=headers))
        seen.add(span)
        cursor = payload_offset+size
        require(body[cursor:cursor+2+len(marker)] == b'\r\n'+marker, 'Invalid multipart CRLF/delimiter')
        cursor += 2+len(marker)
        if body[cursor:cursor+4] == b'--\r\n':
            cursor += 4
            require(cursor == len(body) and len(parts) == len(ranges), 'Missing parts or trailing multipart data')
            return parts
        require(body[cursor:cursor+2] == b'\r\n' and len(parts) < len(ranges), 'Extra part or bad framing')
        cursor += 2
