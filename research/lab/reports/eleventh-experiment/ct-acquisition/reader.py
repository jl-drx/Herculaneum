"""Pinned classic-TIFF ranges for predeclared Frag4 crops; no full-image decoding."""
import argparse
from datetime import datetime, timezone
import hashlib
import http.client
import json
from pathlib import Path
import re
import resource
import ssl
import struct
import urllib.error
import urllib.parse
import urllib.request


class ContractError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise ContractError(message)


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def file_record(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha_file(path), 'bytes': path.stat().st_size}


def verify_record(record):
    path = Path(record['path']).resolve(strict=True)
    require(path.stat().st_size == record['bytes'], 'Pinned input size changed')
    require(sha_file(path) == record['sha256'], 'Pinned input hash changed')
    return path


def save_json(path, value):
    temporary = path.with_name(path.name + '.writing')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class RangeClient:
    """Each attempted range reserves its entire cap, including failed requests."""
    def __init__(self, output_dir, urls, limits, plan_sha256, opener=None):
        self.root = Path(output_dir)
        require(not self.root.exists(), 'A fresh output directory is required')
        self.root.mkdir(parents=True)
        (self.root / 'ranges').mkdir()
        self.urls = set(urls)
        self.limits = limits
        self.pins = {}
        self.opener = opener or urllib.request.build_opener(
            NoRedirect(), urllib.request.HTTPSHandler(context=ssl.create_default_context()))
        self.ledger = {'schema_version': 1, 'plan_sha256': plan_sha256,
                       'maximums': limits, 'automatic_retries': 0,
                       'redirects_followed': 0, 'verified_tls': True,
                       'charged_response_bytes': 0, 'retained_body_bytes': 0,
                       'accounting': 'A started attempt charges the full frozen request cap; '
                                     'partial stored bytes are separate. Not total wire traffic.',
                       'requests': []}
        self._save()

    def _save(self):
        save_json(self.root / 'request-ledger.json', self.ledger)

    def get(self, url, start, end, name):
        require(url in self.urls, 'URL was not declared in the frozen source inventory')
        parsed = urllib.parse.urlsplit(url)
        require(parsed.scheme == 'https' and parsed.hostname == 'dl.ash2txt.org'
                and not parsed.username and not parsed.password and not parsed.query
                and not parsed.fragment, 'Only the exact public source HTTPS URL is allowed')
        require(isinstance(start, int) and isinstance(end, int) and 0 <= start <= end,
                'Invalid byte range')
        cap = end - start + 1
        require(cap <= self.limits['maximum_per_range_bytes'], 'Range exceeds request cap')
        require(len(self.ledger['requests']) < self.limits['maximum_http_attempts'],
                'HTTP attempt budget exhausted')
        require(self.ledger['charged_response_bytes'] + cap
                <= self.limits['maximum_charged_response_bytes'], 'Response budget exhausted')
        require(Path(name).name == name and not name.startswith('.'), 'Invalid range filename')
        target = self.root / 'ranges' / name
        partial = target.with_name(target.name + '.partial')
        require(not target.exists() and not partial.exists(), 'Range output already exists')
        headers = {'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity',
                   'User-Agent': 'Herculaneum-pinned-TIFF-reader'}
        previous = self.pins.get(url)
        if previous:
            require(end < previous['object_total_bytes'], 'Range lies outside pinned object')
            headers['If-Match'] = previous['etag']
        record = {'url': url, 'range_inclusive': [start, end], 'request_cap_bytes': cap,
                  'if_match': previous['etag'] if previous else None,
                  'started_at_utc': datetime.now(timezone.utc).isoformat(),
                  'charged_response_bytes': cap, 'retained_body_bytes': 0, 'success': False}
        self.ledger['requests'].append(record)
        self.ledger['charged_response_bytes'] += cap
        self._save()
        response = None
        retained = 0
        digest = hashlib.sha256()
        try:
            request = urllib.request.Request(url, headers=headers)
            try:
                response = self.opener.open(request, timeout=30)
            except urllib.error.HTTPError as error:
                response = error
            record.update({'http_status': response.status,
                           'headers': {k.lower(): v for k, v in response.headers.items()
                                       if k.lower() in {'etag', 'content-length', 'content-range',
                                                        'content-encoding', 'content-type'}}})
            require(response.status == 206, 'Expected HTTP206; redirects/errors/full objects stop')
            require(response.headers.get('Content-Encoding', 'identity').lower() in ('', 'identity'),
                    'Encoded ranges are not supported')
            length = response.headers.get('Content-Length')
            require(length is not None and length.isdigit() and int(length) == cap,
                    'Content-Length must equal the exact requested range length')
            record['declared_response_body_bytes'] = int(length)
            match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',
                                 response.headers.get('Content-Range', ''))
            require(match is not None, 'Exact Content-Range required')
            got_start, got_end, total = map(int, match.groups())
            require((got_start, got_end) == (start, end) and total > end,
                    'Returned range bounds or object size are invalid')
            etag = response.headers.get('ETag', '')
            require(re.fullmatch(r'"[\x21\x23-\x7e]+"', etag) is not None,
                    'A quoted strong ETag is required')
            if previous:
                require(etag == previous['etag'] and total == previous['object_total_bytes'],
                        'Pinned source ETag or object size changed')
            record.update({'etag': etag, 'object_total_bytes': total})
            with partial.open('xb') as stream:
                while retained < cap:
                    try:
                        block = response.read(min(65536, cap - retained))
                    except http.client.IncompleteRead as error:
                        block = error.partial
                        require(len(block) <= cap - retained, 'Partial response exceeds range cap')
                        stream.write(block)
                        digest.update(block)
                        retained += len(block)
                        raise ContractError('Truncated range body') from None
                    require(block, 'Truncated range body')
                    require(len(block) <= cap - retained, 'Response body exceeds frozen range cap')
                    stream.write(block)
                    digest.update(block)
                    retained += len(block)
            require(retained == cap, 'Range body length mismatch')
            partial.replace(target)
            record['success'] = True
            self.pins.setdefault(url, {'etag': etag, 'object_total_bytes': total})
        except Exception as error:
            record['failure_type'] = type(error).__name__
            if isinstance(error, ContractError):
                record['stop_reason'] = str(error)
            raise
        finally:
            if response is not None:
                response.close()
            record.update({'retained_body_bytes': retained,
                           'body_sha256': digest.hexdigest(),
                           'completed_at_utc': datetime.now(timezone.utc).isoformat()})
            stored = target if record['success'] else partial
            if stored.exists():
                record['retained_path'] = str(stored.resolve())
            self.ledger['retained_body_bytes'] += retained
            self._save()
        return target, record


def inspect_tiff(client, url, layer_index, expected_width, expected_height):
    prefix = f'layer-{layer_index:02d}'
    header_path, header_receipt = client.get(url, 0, 7, prefix + '-header.bin')
    header = header_path.read_bytes()
    require(header[:4] == b'II*\x00', 'Only classic little-endian TIFF is accepted')
    ifd_offset = struct.unpack_from('<I', header, 4)[0]
    total = header_receipt['object_total_bytes']
    require(8 <= ifd_offset and ifd_offset + 2 <= total, 'Invalid TIFF IFD location')
    count_path, _ = client.get(url, ifd_offset, ifd_offset + 1, prefix + '-ifd-count.bin')
    count = struct.unpack('<H', count_path.read_bytes())[0]
    require(0 < count <= 128, 'Unsupported TIFF IFD entry count')
    length = 2 + 12 * count + 4
    require(ifd_offset + length <= total, 'TIFF IFD exceeds object size')
    ifd_path, _ = client.get(url, ifd_offset, ifd_offset + length - 1, prefix + '-ifd.bin')
    data = ifd_path.read_bytes()
    require(struct.unpack_from('<H', data)[0] == count, 'IFD count changed between reads')
    tags = {}
    for index in range(count):
        offset = 2 + index * 12
        tag, kind, number = struct.unpack_from('<HHI', data, offset)
        require(tag not in tags, 'Duplicate TIFF tag')
        tags[tag] = {'type': kind, 'count': number,
                     'value_or_offset': struct.unpack_from('<I', data, offset + 8)[0],
                     'value_bytes_hex': data[offset + 8:offset + 12].hex()}
    require(struct.unpack_from('<I', data, 2 + count * 12)[0] == 0,
            'Only a single-image TIFF IFD is accepted')

    def scalar(tag, default=None):
        if tag not in tags:
            require(default is not None, f'Required TIFF tag absent:{tag}')
            return default
        value = tags[tag]
        require(value['count'] == 1 and value['type'] in (3, 4),
                f'TIFF tag must be an inline SHORT or LONG scalar:{tag}')
        raw = bytes.fromhex(value['value_bytes_hex'])
        return struct.unpack('<H' if value['type'] == 3 else '<I',
                             raw[:2] if value['type'] == 3 else raw)[0]

    require(scalar(256) == expected_width and scalar(257) == expected_height,
            'TIFF dimensions differ from the frozen plan')
    checks = {258: (16, None), 259: (1, None), 262: (1, None),
              266: (1, 1), 274: (1, 1), 277: (1, None), 284: (1, 1),
              317: (1, 1), 339: (1, 1)}
    for tag, (expected, default) in checks.items():
        require(scalar(tag, default) == expected, f'Unsupported TIFF pixel convention:{tag}')
    require(not ({320, 322, 323, 324, 325, 330, 338} & set(tags)),
            'Color maps, tiles, SubIFDs and extra samples are unsupported')
    require(scalar(278) >= expected_height, 'Multiple row strips are unsupported')
    strip_offset, strip_length = scalar(273), scalar(279)
    require(strip_length == expected_width * expected_height * 2,
            'Single strip must contain exactly width*height uint16 pixels')
    require(strip_offset >= 8 and strip_offset + strip_length <= total,
            'TIFF strip lies outside object')
    require(strip_offset + strip_length <= ifd_offset
            or strip_offset >= ifd_offset + length, 'Strip overlaps TIFF IFD')
    return {'layer_index': layer_index, 'url': url, 'etag': header_receipt['etag'],
            'object_total_bytes': total, 'ifd_offset': ifd_offset, 'ifd_length': length,
            'width': expected_width, 'height': expected_height, 'dtype': '<u2',
            'strip_offset': strip_offset, 'strip_bytes': strip_length,
            'tags': tags, 'orientation': 1, 'photometric': 'min-is-black',
            'normal_or_depth_order_change': False}


def merged_row_bands(crops):
    spans = sorted((c['bbox_source_xywh'][1],
                    c['bbox_source_xywh'][1] + c['bbox_source_xywh'][3]) for c in crops)
    merged = []
    for start, stop in spans:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(stop, merged[-1][1])
        else:
            merged.append([start, stop])
    return merged


def decode_crop_rows(path, contract, band_y_start, bbox):
    """Read/decode only selected X pixels, retaining the full-width raw band separately."""
    import numpy as np
    x, y, width, height = bbox
    require(0 <= x and 0 <= y and x + width <= contract['width']
            and y + height <= contract['height'], 'Crop exceeds TIFF canvas')
    result = np.empty((height, width), dtype=np.uint16)
    with Path(path).open('rb') as stream:
        for row in range(height):
            offset = ((y - band_y_start + row) * contract['width'] + x) * 2
            require(offset >= 0, 'Crop is outside retained row band')
            stream.seek(offset)
            data = stream.read(width * 2)
            require(len(data) == width * 2, 'Retained band does not contain crop')
            result[row] = np.frombuffer(data, dtype='<u2')
    return result


def xy_sum3(native):
    import numpy as np
    height, width = native.shape
    require(native.dtype == np.uint16 and height % 3 == 0 and width % 3 == 0,
            'Native crop must be uint16 and divisible by3')
    return native.reshape(height // 3, 3, width // 3, 3).sum(axis=(1, 3), dtype=np.uint32)


def pooled_float32(sum27):
    import numpy as np
    require(sum27.dtype == np.uint32, 'Exact uint32 pooling sum required')
    return (sum27.astype(np.float64) / (27 * 257)).astype(np.float32)


def memory_check(limits):
    # Linux ru_maxrss is KiB; this execution environment is Linux.
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    require(peak < limits['maximum_peak_rss_bytes'], 'Peak RSS exceeded the frozen limit')
    return peak


def load_plan(path, expected_sha256):
    path = Path(path).resolve(strict=True)
    require(sha_file(path) == expected_sha256, 'Frozen plan hash changed')
    plan = json.loads(path.read_text())
    require(plan['schema_version'] == 1 and plan['ready'] is True, 'Plan is not ready')
    pins = plan['input_pins']
    for pin in pins:
        verify_record(pin)
    require(any(Path(p['path']).resolve() == Path(__file__).resolve()
                and p['sha256'] == sha_file(__file__) for p in pins), 'Executed reader must be pinned')
    inventory_path = verify_record(plan['source_inventory'])
    inventory = json.loads(inventory_path.read_text())
    by_name = {a['name']: a['url'] for a in inventory['assets']
               if re.fullmatch(r'\d\d\.tif', a['name'])}
    require(set(by_name) == {f'{i:02d}.tif' for i in range(65)}, 'Exact65-layer inventory required')
    layers = plan['source_layers']
    require([a['index'] for a in layers] == list(range(7, 58)), 'Exactly source layers7..57 required')
    require(all(a['url'] == by_name[f"{a['index']:02d}.tif"] for a in layers),
            'Layer URL differs from retained inventory')
    require(plan['pooling'] == {'factor_zyx': [3, 3, 3],
                               'formula': 'float32(sum_uint32_27/(27*257))',
                               'quantization': 'none'}, 'Pooling contract changed')
    crops = plan['crops']
    require(len(crops) == 2 and len({c['case_id'] for c in crops}) == 2, 'Exactly two named crops required')
    full_width, full_height = plan['expected_tiff']['width'], plan['expected_tiff']['height']
    for c in crops:
        require(re.fullmatch(r'[A-Za-z0-9_-]+', c['case_id']) is not None, 'Invalid case identifier')
        x, y, width, height = c['bbox_source_xywh']
        require(all(type(v) is int for v in (x, y, width, height))
                and 0 <= x and 0 <= y and 0 < width <= 1152 and 0 < height <= 1152
                and width % 3 == 0 and height % 3 == 0
                and x + width <= full_width and y + height <= full_height, 'Invalid frozen crop')
    limits = plan['limits']
    require(0 < limits['maximum_http_attempts'] <= 260
            and 0 < limits['maximum_charged_response_bytes'] <= 1717986918
            and 0 < limits['maximum_per_range_bytes'] <= 14680064
            and 0 < limits['maximum_peak_rss_bytes'] <= 1073741824, 'Frozen limits exceed authorization')
    bands = merged_row_bands(crops)
    require(all((stop - start) * full_width * 2 <= limits['maximum_per_range_bytes']
                for start, stop in bands), 'Merged band exceeds frozen cap')
    require(51 * (3 + len(bands)) <= limits['maximum_http_attempts'], 'Request cap cannot cover plan')
    payload = 51 * sum((stop - start) * full_width * 2 for start, stop in bands)
    # Header8 + count2 + maximum128-entry IFD1542 on each layer.
    require(payload + 51 * (8 + 2 + 1542) <= limits['maximum_charged_response_bytes'],
            'Plan response budget cannot cover payload plus bounded metadata')
    return path, plan


def acquire(plan_path, plan_sha256, output_dir, opener=None):
    plan_path, plan = load_plan(plan_path, plan_sha256)
    client = RangeClient(output_dir, [x['url'] for x in plan['source_layers']],
                         plan['limits'], plan_sha256, opener=opener)
    result = {'schema_version': 1, 'status': 'STARTED', 'plan': file_record(plan_path),
              'reader': file_record(__file__), 'input_pins': plan['input_pins'],
              'crops': plan['crops'], 'source_layers': [], 'outputs': [],
              'pooling': plan['pooling'], 'normal_order_preserved': True,
              'declared_source_center': 32, 'source_groups': [list(range(i, i + 3)) for i in range(7, 58, 3)],
              'pooled_source_center_indices': list(range(8, 58, 3)),
              'quantization': 'none', 'pixel_validation': 'No ink or optical accuracy claim'}
    try:
        import numpy as np
        memory_check(plan['limits'])
        outputs = {c['case_id']: np.empty((17, c['bbox_source_xywh'][3] // 3,
                                         c['bbox_source_xywh'][2] // 3), dtype=np.float32)
                   for c in plan['crops']}
        sums = {name: np.zeros(arr.shape[1:], dtype=np.uint32) for name, arr in outputs.items()}
        bands = merged_row_bands(plan['crops'])
        for ordinal, layer in enumerate(plan['source_layers']):
            contract = inspect_tiff(client, layer['url'], layer['index'],
                                    plan['expected_tiff']['width'], plan['expected_tiff']['height'])
            contract['row_bands'] = []
            result['source_layers'].append(contract)
            for band_index, (start_y, stop_y) in enumerate(bands):
                start = contract['strip_offset'] + start_y * contract['width'] * 2
                end = contract['strip_offset'] + stop_y * contract['width'] * 2 - 1
                path, receipt = client.get(layer['url'], start, end,
                                           f"layer-{layer['index']:02d}-band-{band_index}.bin")
                contract['row_bands'].append({'y_start': start_y, 'y_stop_exclusive': stop_y,
                                              'range': receipt, 'file': file_record(path)})
                for crop in plan['crops']:
                    bbox = crop['bbox_source_xywh']
                    if start_y <= bbox[1] and bbox[1] + bbox[3] <= stop_y:
                        native = decode_crop_rows(path, contract, start_y, bbox)
                        sums[crop['case_id']] += xy_sum3(native)
                        del native
            if ordinal % 3 == 2:
                for name in outputs:
                    outputs[name][ordinal // 3] = pooled_float32(sums[name])
                    sums[name].fill(0)
            memory_check(plan['limits'])
        for crop in plan['crops']:
            name = crop['case_id']
            directory = client.root / name
            directory.mkdir()
            path = directory / 'surface.npy'
            np.save(path, outputs[name], allow_pickle=False)
            x, y, width, height = crop['bbox_source_xywh']
            result['outputs'].append({'case_id': name, 'file': file_record(path),
                                      'shape_zyx': list(outputs[name].shape), 'dtype': 'float32',
                                      'bbox_source_xywh': [x, y, width, height],
                                      'pixel_center_mapping': {'x_native': f'{x}+1+3*x_output',
                                                               'y_native': f'{y}+1+3*y_output',
                                                               'z_native': '8+3*z_output'},
                                      'declared_spacing_um_zyx': [9.72, 9.72, 9.72],
                                      'scale_status': 'dataset-declared; native reproduction not performed'})
        for pin in plan['input_pins'] + [plan['source_inventory']]:
            verify_record(pin)
        require(sha_file(plan_path) == plan_sha256, 'Plan changed during acquisition')
        result.update({'status': 'PASS', 'peak_rss_bytes': memory_check(plan['limits']),
                       'charged_response_bytes': client.ledger['charged_response_bytes'],
                       'retained_body_bytes': client.ledger['retained_body_bytes'],
                       'http_attempts': len(client.ledger['requests']),
                       'request_ledger': file_record(client.root / 'request-ledger.json')})
    except Exception as error:
        result.update({'status': 'STOP', 'failure_type': type(error).__name__,
                       'charged_response_bytes': client.ledger['charged_response_bytes'],
                       'retained_body_bytes': client.ledger['retained_body_bytes'],
                       'http_attempts': len(client.ledger['requests'])})
        if isinstance(error, ContractError):
            result['stop_reason'] = str(error)
        save_json(client.root / 'provenance.json', result)
        raise
    save_json(client.root / 'provenance.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    try:
        result = acquire(args.plan, args.plan_sha256, args.output_dir)
        print(json.dumps({'status': result['status'], 'http_attempts': result['http_attempts'],
                          'charged_response_bytes': result['charged_response_bytes'],
                          'peak_rss_bytes': result['peak_rss_bytes']}))
    except Exception as error:
        # Never print transport exception strings or potentially signed URL text.
        print(json.dumps({'status': 'STOP', 'failure_type': type(error).__name__}))
        raise SystemExit(1)
