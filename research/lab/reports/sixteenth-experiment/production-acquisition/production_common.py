"""Metadata and128/256-row production HTTP contracts; no numeric pixel decoder."""
from pathlib import Path
import hashlib
import json
import re
import struct
import sys

ROOT = Path(__file__).resolve().parent
DESIGN_SHA = '5c942839223b69b8667da841db82babc15dbf62eaed0bf37005d76130e5315d8'
SCHEMA_SHA = '70c810cbfc6bc1ed174f9c3d7c5654c6e3e8f4223ec6ea101aea7dd373be2f5c'
LARGE_RESULT_SHA = 'de9d67d9c3b091e73246178de5621b2db4d4a27ef7d1e295d211fef4b33439c6'
HEADER_RESULT_SHA = 'dde73506ee5de6b40979125303c5956a52e42052b1b340a74201b3314487ed1b'
SERVING_RESULT_SHA = '82fe4e347c3a41ed934e9b14571675a8dad9fff245e453c448ede226d16519da'
BENCHMARK_SHA = 'a4026cce7ca55d5aaeeaa0736061ab6b7c8aa13875155682d24e74c139fe5735'
SOURCE_SHA = '53ff9fd5cccb2604ebcd8b116b69b1f3c4ba25ed79fa624aaedf21bdaa709890'
CODE_NAMES = ['production_common.py','range_http.py','prepare_plan.py','fetch_worker.py','run_acquisition.py','check_fixtures.py']
LIMITS = {'metadata': {'attempts':450,'body_bytes':524288,'per_response':16384,'wall_seconds':300},
          'pixels': {'attempts':1530,'body_bytes':1342177280,'per_response':1048576,'wall_seconds':900},
          'concurrency':4,'request_timeout_seconds':25,'child_timeout_seconds':30,
          'cleanup_reserve_seconds':10,'output_bytes':2147483648,'address_space_bytes':2147483648,
          'part_header_bytes':512,'framing_bytes_per_part':768,'framing_base_bytes':1024,
          'rows':[256,128]}
EXPECTED_USEFUL_BYTES = 812187648
EXPECTED_RESERVED_BYTES = 1084483584



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


def tiff_header(header):
    require(len(header) == 8 and header[:2] in (b'II',b'MM'), 'Classic TIFF byte-order marker required')
    order = '<' if header[:2] == b'II' else '>'
    require(struct.unpack_from(order+'H',header,2)[0] == 42, 'Only classic TIFF magic42 supported; no BigTIFF')
    return dict(struct_endian=order,byte_order='little' if order == '<' else 'big',
                dtype=order+'u2',ifd_offset=struct.unpack_from(order+'I',header,4)[0])


def parse_tiff(header, ifd, total, shape_yx):
    layout = tiff_header(header)
    order,offset = layout['struct_endian'],layout['ifd_offset']
    require(offset >= 8 and offset+2 <= total, 'IFD offset exceeds object')
    require(len(ifd) >= 6, 'Incomplete IFD')
    count = struct.unpack_from(order+'H', ifd)[0]
    require(0 < count <= 128 and len(ifd) == 2+12*count+4 and offset+len(ifd) <= total,
            'IFD count/extent exceeds frozen bounds')
    require(struct.unpack_from(order+'I', ifd, 2+12*count)[0] == 0, 'Multiple TIFF pages unsupported')
    tags = {}
    sizes = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8}
    for i in range(count):
        index = 2+12*i
        tag, kind, number, value = struct.unpack_from(order+'HHII', ifd, index)
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
        raw = bytes.fromhex(value['value_bytes_hex'])
        return struct.unpack(order+('H' if value['type'] == 3 else 'I'),raw[:2] if value['type'] == 3 else raw)[0]
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
    return dict(width=width, height=height, dtype=layout['dtype'], byte_order=layout['byte_order'],
                struct_endian=order,ifd_offset=offset, ifd_bytes=len(ifd),
                strip_offset=strip_offset, strip_bytes=strip_bytes, tags=tags,
                auxiliary_metadata_extents=auxiliary, specification_defaults_used=defaults,
                default_source='TIFF6.0 defaults: Compression/FillOrder/Orientation/SamplesPerPixel/PlanarConfiguration=1; RowsPerStrip=2^32-1; Predictor/SampleFormat=1',
                orientation=1, photometric='min-is-black', numeric_pixel_decodes=0)


def parse_multipart(body, content_type, ranges, total):
    """Length-based binary parsing; delimiter-like payloads are ordinary bytes."""
    require(len(body) <= sum(b-a+1 for a,b in ranges)+LIMITS['framing_bytes_per_part']*len(ranges)+LIMITS['framing_base_bytes'], 'Multipart body exceeds framing allowance')
    boundary_match = re.fullmatch(r'multipart/byteranges\s*;\s*boundary=(?:"([A-Za-z0-9\x27()+_,./:=?-]{1,70})"|([A-Za-z0-9\x27+_.-]{1,70}))\s*', content_type or '', re.I)
    require(boundary_match is not None, 'One bounded multipart boundary required')
    boundary = (boundary_match[1] or boundary_match[2]).encode('ascii')
    require(len(ranges) in LIMITS['rows'] and len(set(map(tuple,ranges))) == len(ranges), 'Exactly128 or256 unique ranges required')
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
