"""Pinned metadata contracts and classic TIFF/multipart parsers; no numeric pixel decoder."""
from pathlib import Path
import hashlib
import json
import re
import struct
import sys

ROOT = Path(__file__).resolve().parent
DESIGN_SHA = '95d7879c4efafb6b8cc54e6887cfef0e9ad9d4800e05ff90a38ed74a74c8466c'
SOURCE_SHA = '53ff9fd5cccb2604ebcd8b116b69b1f3c4ba25ed79fa624aaedf21bdaa709890'
BENCHMARK_SHA = 'a4026cce7ca55d5aaeeaa0736061ab6b7c8aa13875155682d24e74c139fe5735'
CODE_NAMES = ['probe_common.py', 'range_http.py', 'prepare_probe_plan.py', 'run_probe.py', 'check_fixtures.py']
LIMITS = {'headers': {'attempts': 12, 'body_bytes': 65536, 'per_response': 16384},
          'serving': {'attempts': 6, 'body_bytes': 524288, 'per_response': 262144},
          'wall_seconds': 180, 'output_bytes': 8388608, 'part_header_bytes': 512,
          'multipart_framing_bytes': 8192, 'rows': 8}


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


def parse_tiff(header, ifd, total, shape_yx):
    require(len(header) == 8 and header[:4] == b'II*\x00', 'Only classic little-endian TIFF supported')
    offset = struct.unpack_from('<I', header, 4)[0]
    require(offset >= 8 and offset+2 <= total, 'IFD offset exceeds object')
    require(len(ifd) >= 6, 'Incomplete IFD')
    count = struct.unpack_from('<H', ifd)[0]
    require(0 < count <= 128 and len(ifd) == 2+12*count+4 and offset+len(ifd) <= total,
            'IFD count/extent exceeds frozen bounds')
    require(struct.unpack_from('<I', ifd, 2+12*count)[0] == 0, 'Multiple TIFF pages unsupported')
    tags = {}
    sizes = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8}
    for i in range(count):
        index = 2+12*i
        tag, kind, number, value = struct.unpack_from('<HHII', ifd, index)
        require(tag not in tags and kind in sizes and number > 0, 'Duplicate/unsupported/empty TIFF field')
        tags[tag] = {'type': kind, 'count': number, 'value_or_offset': value,
                     'value_bytes_hex': ifd[index+8:index+12].hex(), 'storage_bytes': number*sizes[kind]}
    defaults = {}
    def scalar(tag, default=None):
        if tag not in tags:
            require(default is not None, f'Required TIFF tag absent:{tag}')
            defaults[str(tag)] = default
            return default
        value = tags[tag]
        require(value['type'] in (3, 4) and value['count'] == 1, f'TIFF scalar encoding unsupported:{tag}')
        return value['value_or_offset'] & (65535 if value['type'] == 3 else 4294967295)
    height, width = shape_yx
    require(width > 0 and height > 0 and scalar(256) == width and scalar(257) == height, 'TIFF canvas mismatch')
    for tag, expected, default in [(258,16,None),(259,1,1),(262,1,None),(266,1,1),
                                   (274,1,1),(277,1,1),(284,1,1),(317,1,1),(339,1,1)]:
        require(scalar(tag, default) == expected, f'Unsupported TIFF pixel convention:{tag}')
    require(not ({320,322,323,324,325,330,338,347,513,514} & set(tags)), 'Tiles/subIFDs/color/JPEG unsupported')
    require(scalar(278, 4294967295) >= height, 'Multiple TIFF strips unsupported')
    strip_offset, strip_bytes = scalar(273), scalar(279)
    strip = [strip_offset, strip_offset+strip_bytes]
    require(strip_bytes == height*width*2 and strip_offset >= 8 and strip[1] <= total,
            'Strip extent/dtype mismatch')
    table = [offset, offset+len(ifd)]
    require(disjoint(strip, table), 'Pixel strip overlaps IFD')
    auxiliary = []
    for tag, value in tags.items():
        if value['storage_bytes'] > 4:
            extent = [value['value_or_offset'], value['value_or_offset']+value['storage_bytes']]
            require(extent[0] >= 8 and extent[1] <= total and disjoint(extent, strip)
                    and disjoint(extent, table), 'Auxiliary TIFF metadata extent overlaps pixels/header/IFD or exceeds object')
            auxiliary.append({'tag': tag, 'extent_half_open': extent, 'dereferenced': False})
    return dict(width=width, height=height, dtype='<u2', ifd_offset=offset, ifd_bytes=len(ifd),
                strip_offset=strip_offset, strip_bytes=strip_bytes, tags=tags,
                auxiliary_metadata_extents=auxiliary, specification_defaults_used=defaults,
                default_source='TIFF6.0 defaults: Compression/FillOrder/Orientation/SamplesPerPixel/PlanarConfiguration=1; RowsPerStrip=2^32-1; Predictor/SampleFormat=1',
                orientation=1, photometric='min-is-black', numeric_pixel_decodes=0)


def parse_multipart(body, content_type, ranges, total):
    """Length-based binary parsing; delimiter-like payloads are ordinary bytes."""
    boundary_match = re.fullmatch(r'multipart/byteranges\s*;\s*boundary=(?:"([A-Za-z0-9\x27()+_,./:=?-]{1,70})"|([A-Za-z0-9\x27+_.-]{1,70}))\s*', content_type or '', re.I)
    require(boundary_match is not None, 'One bounded multipart boundary required')
    boundary = (boundary_match[1] or boundary_match[2]).encode('ascii')
    require(len(ranges) == 8 and len(set(map(tuple, ranges))) == 8, 'Exactly eight unique ranges required')
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
            require(cursor == len(body) and len(parts) == 8, 'Missing parts or trailing multipart data')
            return parts
        require(body[cursor:cursor+2] == b'\r\n' and len(parts) < 8, 'Extra part or bad framing')
        cursor += 2
