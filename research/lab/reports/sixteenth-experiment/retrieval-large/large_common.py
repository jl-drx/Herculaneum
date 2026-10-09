"""Fixed 256-row probe contracts and binary multipart parsing; no pixel decoder."""
from pathlib import Path
import hashlib
import json
import re
import struct
import sys

ROOT = Path(__file__).resolve().parent
DESIGN_SHA = '3f9b8e0f1efdc848572547833664ceea836afb7190e1a9944b4013509520d850'
HEADER_RESULT_SHA = 'dde73506ee5de6b40979125303c5956a52e42052b1b340a74201b3314487ed1b'
SERVING_RESULT_SHA = '82fe4e347c3a41ed934e9b14571675a8dad9fff245e453c448ede226d16519da'
V2_DESIGN_SHA = 'b19be5cc4de3622a38cea5c586f55559c7211333438408bfcdcaebd54f7d22bd'
CODE_NAMES = ['large_common.py', 'range_http.py', 'prepare_plan.py', 'run_large_probe.py', 'check_fixtures.py']
LIMITS = {'http': {'attempts': 6, 'body_bytes': 16777216, 'per_response': 6291456},
          'wall_seconds': 240, 'output_bytes': 33554432, 'address_space_bytes': 2147483648,
          'part_header_bytes': 512, 'framing_bytes_per_part': 768,
          'framing_base_bytes': 1024, 'rows': 256}
MAXIMUM_RESERVATION = 13157891



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
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


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


def disjoint(a, b):
    return a[1] <= b[0] or b[1] <= a[0]


def parse_multipart(body, content_type, ranges, total):
    """Length-based binary parsing; delimiter-like payloads are ordinary bytes."""
    require(len(body) <= sum(b-a+1 for a,b in ranges)+LIMITS['framing_bytes_per_part']*LIMITS['rows']+LIMITS['framing_base_bytes'], 'Multipart body exceeds framing allowance')
    boundary_match = re.fullmatch(r'multipart/byteranges\s*;\s*boundary=(?:"([A-Za-z0-9\x27()+_,./:=?-]{1,70})"|([A-Za-z0-9\x27+_.-]{1,70}))\s*', content_type or '', re.I)
    require(boundary_match is not None, 'One bounded multipart boundary required')
    boundary = (boundary_match[1] or boundary_match[2]).encode('ascii')
    require(len(ranges) == LIMITS['rows'] and len(set(map(tuple, ranges))) == LIMITS['rows'], 'Exactly 256 unique ranges required')
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
            require(cursor == len(body) and len(parts) == LIMITS['rows'], 'Missing parts or trailing multipart data')
            return parts
        require(body[cursor:cursor+2] == b'\r\n' and len(parts) < LIMITS['rows'], 'Extra part or bad framing')
        cursor += 2
