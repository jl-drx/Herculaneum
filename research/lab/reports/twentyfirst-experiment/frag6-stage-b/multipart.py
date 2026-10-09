"""Strict parser adapted from pinned phase17 acquisition_common.py; no networking."""
import hashlib
import re
from common import LIMITS, ContractError, require

def content_range(value):
    match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', value or '')
    require(match is not None, 'Exact Content-Range required')
    start, end, total = map(int, match.groups())
    require(0 <= start <= end < total, 'Invalid Content-Range bounds')
    return start, end, total


def parse_multipart(body, content_type, ranges, total):
    """Length-based binary parsing; delimiter-like payloads are ordinary bytes."""
    require(len(body) <= sum(b-a+1 for a,b in ranges)+LIMITS['maximum_part_framing_bytes']*len(ranges)+LIMITS['maximum_response_framing_bytes'], 'Multipart body exceeds framing allowance')
    boundary_match = re.fullmatch(r'multipart/byteranges\s*;\s*boundary=(?:"([A-Za-z0-9\x27()+_,./:=?-]{1,70})"|([A-Za-z0-9\x27+_.-]{1,70}))\s*', content_type or '', re.I)
    require(boundary_match is not None, 'One bounded multipart boundary required')
    boundary = (boundary_match[1] or boundary_match[2]).encode('ascii')
    require(len(ranges) == LIMITS['expected_parts'] and len(set(map(tuple,ranges))) == len(ranges), 'Exactly243 unique ranges required')
    for i, span in enumerate(sorted(ranges)):
        require(0 <= span[0] <= span[1] < total, 'Requested multipart span invalid')
        if i:
            require(sorted(ranges)[i-1][1] < span[0], 'Requested multipart spans overlap')
    marker = b'--'+boundary
    prefix = 2 if body.startswith(b'\r\n') else 0
    require(body[prefix:].startswith(marker+b'\r\n'), 'Multipart preamble/framing rejected')
    cursor, seen, parts = prefix+len(marker)+2, set(), []
    while True:
        part_header_start = cursor
        end_headers = body.find(b'\r\n\r\n', cursor, cursor+LIMITS['maximum_part_header_bytes']+4)
        require(end_headers >= 0 and end_headers-cursor <= LIMITS['maximum_part_header_bytes'], 'Multipart header exceeds bound')
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
        if 'content-type' in headers:
            require(headers['content-type'].lower() == 'application/octet-stream', 'Unexpected PPM part MIME')
        payload_offset = end_headers+4
        require(payload_offset+size <= len(body), 'Truncated multipart payload')
        payload = body[payload_offset:payload_offset+size]
        parts.append(dict(range_inclusive=[start,end], payload_offset=payload_offset, payload_bytes=size,
                          payload_sha256=hashlib.sha256(payload).hexdigest(), headers=headers,
                          framing_bytes=end_headers+4-part_header_start+2+len(marker)+2))
        seen.add(span)
        cursor = payload_offset+size
        require(parts[-1]['framing_bytes'] <= LIMITS['maximum_part_framing_bytes'], 'Per-part framing allowance exceeded')
        require(body[cursor:cursor+2+len(marker)] == b'\r\n'+marker, 'Invalid multipart CRLF/delimiter')
        cursor += 2+len(marker)
        if body[cursor:cursor+4] == b'--\r\n':
            cursor += 4
            require(cursor == len(body) and len(parts) == len(ranges), 'Missing parts or trailing multipart data')
            response_framing = len(body)-sum(p['payload_bytes']+p['framing_bytes'] for p in parts)
            require(0 <= response_framing <= LIMITS['maximum_response_framing_bytes'], 'Response framing allowance exceeded')
            return parts
        require(body[cursor:cursor+2] == b'\r\n' and len(parts) < len(ranges), 'Extra part or bad framing')
        cursor += 2
