"""Offline synthetic TIFF/range tests; no real source arrays or network calls."""
import argparse
import hashlib
import http.client
import importlib.util
import io
import json
from pathlib import Path
import socket
import struct
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('frag4_reader', ROOT / 'reader.py')
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)
SOURCE_PREFIX = ('https://dl.ash2txt.org/fragments/Frag4/PHercParis1Fr39.volpkg/working/'
                 '54keV_exposed_surface/PHercParis1Fr39_54keV_surface_volume/')
LIMITS = {'maximum_http_attempts': 260, 'maximum_charged_response_bytes': 1717986918,
          'maximum_per_range_bytes': 14680064, 'maximum_peak_rss_bytes': 1073741824}


def synthetic_value(z, y, x):
    # Unequal axes, nonlinear Y, and non-integral means distinguish indexing errors.
    return 100 * z + 11 * y * y + 3 * x + (z + y + x) % 5


def synthetic_tiff(z=7, width=9, height=12, overrides=None, extra=None, next_ifd=0):
    pixels = b''.join(struct.pack('<H', synthetic_value(z, y, x))
                      for y in range(height) for x in range(width))
    tags = {256: (3, 1, width), 257: (3, 1, height), 258: (3, 1, 16),
            259: (3, 1, 1), 262: (3, 1, 1), 273: (4, 1, 8),
            277: (3, 1, 1), 278: (3, 1, height), 279: (4, 1, len(pixels)),
            284: (3, 1, 1), 339: (3, 1, 1)}
    tags.update(overrides or {})
    tags.update(extra or {})
    offset = 8 + len(pixels)
    ifd = struct.pack('<H', len(tags)) + b''.join(
        struct.pack('<HHII', tag, *value) for tag, value in sorted(tags.items()))
    ifd += struct.pack('<I', next_ifd)
    return b'II*\x00' + struct.pack('<I', offset) + pixels + ifd


class Response(io.BytesIO):
    def __init__(self, payload, status, headers):
        super().__init__(payload)
        self.status = status
        self.headers = headers


class FakeServer:
    def __init__(self, objects, mutate=None):
        self.objects = objects
        self.mutate = mutate
        self.calls = []

    def open(self, request, timeout):
        headers = {k.lower(): v for k, v in request.header_items()}
        self.calls.append({'url': request.full_url, 'headers': headers})
        source = self.objects[request.full_url]
        low, high = map(int, headers['range'].split('=')[1].split('-'))
        etag = '"synthetic-' + hashlib.sha256(source).hexdigest() + '"'
        if len([x for x in self.calls if x['url'] == request.full_url]) > 1:
            assert headers['if-match'] == etag
        response = Response(source[low:high + 1], 206,
                            {'Content-Length': str(high - low + 1),
                             'Content-Range': f'bytes {low}-{high}/{len(source)}',
                             'ETag': etag, 'Content-Encoding': 'identity'})
        return self.mutate(response, len(self.calls)) if self.mutate else response


class Fixtures(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='frag4-reader-fixture-')
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def client(self, server, name='ranges', limits=None):
        return reader.RangeClient(self.root / name, list(server.objects), limits or LIMITS,
                                  'synthetic-plan', opener=server)

    def test_51_layers_asymmetric_two_crops_end_to_end(self):
        import numpy as np
        inventory = self.root / 'inventory.json'
        inventory.write_text(json.dumps({'assets': [{'name': f'{z:02d}.tif',
                                                     'url': SOURCE_PREFIX + f'{z:02d}.tif'}
                                                    for z in range(65)]}))
        objects = {SOURCE_PREFIX + f'{z:02d}.tif': synthetic_tiff(z) for z in range(7, 58)}
        server = FakeServer(objects)
        crops = [{'case_id': 'upper', 'bbox_source_xywh': [0, 0, 6, 6]},
                 {'case_id': 'lower', 'bbox_source_xywh': [3, 6, 6, 6]}]
        plan = {'schema_version': 1, 'ready': True,
                'input_pins': [reader.file_record(ROOT / 'reader.py')],
                'source_inventory': reader.file_record(inventory),
                'source_layers': [{'index': z, 'url': SOURCE_PREFIX + f'{z:02d}.tif'}
                                  for z in range(7, 58)],
                'expected_tiff': {'width': 9, 'height': 12}, 'crops': crops,
                'pooling': {'factor_zyx': [3, 3, 3],
                            'formula': 'float32(sum_uint32_27/(27*257))', 'quantization': 'none'},
                'limits': LIMITS}
        plan_path = self.root / 'plan.json'
        plan_path.write_text(json.dumps(plan))
        result = reader.acquire(plan_path, reader.sha_file(plan_path), self.root / 'output', opener=server)
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['http_attempts'], 204)  # adjacent spans coalesce into one band
        self.assertEqual(result['source_groups'], [list(range(z, z + 3)) for z in range(7, 58, 3)])
        self.assertEqual(result['pooled_source_center_indices'][8], 32)
        for crop in crops:
            x0, y0, width, height = crop['bbox_source_xywh']
            array = np.load(self.root / 'output' / crop['case_id'] / 'surface.npy', allow_pickle=False)
            self.assertEqual(array.dtype, np.float32)
            self.assertEqual(array.shape, (17, 2, 2))
            for group in range(17):
                for oy in range(2):
                    for ox in range(2):
                        total = sum(synthetic_value(7 + 3 * group + dz,
                                                    y0 + 3 * oy + dy, x0 + 3 * ox + dx)
                                    for dz in range(3) for dy in range(3) for dx in range(3))
                        self.assertEqual(array[group, oy, ox], np.float32(total / (27 * 257)))
        ledger = json.loads((self.root / 'output' / 'request-ledger.json').read_text())
        self.assertEqual(ledger['charged_response_bytes'], ledger['retained_body_bytes'])
        for request in ledger['requests']:
            raw = Path(request['retained_path']).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), request['body_sha256'])
            self.assertEqual(len(raw), request['request_cap_bytes'])
        self.assertTrue(all('if-match' in x['headers'] for i, x in enumerate(server.calls) if i % 4))

    def test_each_layer_structure_rejects_pixel_convention_changes(self):
        variants = [{'overrides': {258: (3, 1, 8)}}, {'overrides': {259: (3, 1, 5)}},
                    {'overrides': {273: (4, 2, 8)}}, {'extra': {274: (3, 1, 2)}},
                    {'overrides': {339: (3, 1, 3)}}, {'next_ifd': 20},
                    {'overrides': {279: (4, 1, 218)}}, {'extra': {317: (3, 1, 2)}}]
        url = SOURCE_PREFIX + '07.tif'
        for i, variant in enumerate(variants):
            with self.subTest(variant=variant):
                server = FakeServer({url: synthetic_tiff(**variant)})
                client = self.client(server, name=f'structure-{i}')
                with self.assertRaises(reader.ContractError):
                    reader.inspect_tiff(client, url, 7, 9, 12)
                self.assertEqual(len(server.calls), 3)
                self.assertTrue(all('band' not in r.get('retained_path', '')
                                    for r in client.ledger['requests']))

    def test_big_endian_and_bigtiff_stop_after_eight_bytes(self):
        url = SOURCE_PREFIX + '07.tif'
        for i, signature in enumerate([b'MM\x00*', b'II+\x00']):
            with self.subTest(signature=signature):
                source = signature + synthetic_tiff()[4:]
                server = FakeServer({url: source})
                with self.assertRaises(reader.ContractError):
                    reader.inspect_tiff(self.client(server, f'header-{i}'), url, 7, 9, 12)
                self.assertEqual(len(server.calls), 1)

    def test_pinned_etag_change_stops_without_retry(self):
        url = SOURCE_PREFIX + '07.tif'
        def mutate(response, number):
            if number == 2:
                response.headers['ETag'] = '"changed"'
            return response
        server = FakeServer({url: synthetic_tiff()}, mutate)
        client = self.client(server)
        with self.assertRaises(reader.ContractError):
            reader.inspect_tiff(client, url, 7, 9, 12)
        self.assertEqual(len(server.calls), 2)
        self.assertEqual(client.ledger['charged_response_bytes'], 10)
        self.assertEqual(client.ledger['retained_body_bytes'], 8)

    def test_bad_range_headers_and_status_stop_before_body(self):
        url = SOURCE_PREFIX + '07.tif'
        changes = [('status', 200), ('Content-Length', '9'), ('Content-Range', 'bytes 1-8/362'),
                   ('ETag', 'W/"weak"'), ('Content-Encoding', 'gzip'), ('Content-Length', None)]
        for i, (key, value) in enumerate(changes):
            with self.subTest(change=(key, value)):
                def mutate(response, number):
                    if key == 'status':
                        response.status = value
                    elif value is None:
                        response.headers.pop(key)
                    else:
                        response.headers[key] = value
                    return response
                server = FakeServer({url: synthetic_tiff()}, mutate)
                client = self.client(server, f'bad-header-{i}')
                with self.assertRaises(reader.ContractError):
                    client.get(url, 0, 7, 'header.bin')
                self.assertEqual(client.ledger['charged_response_bytes'], 8)
                self.assertEqual(client.ledger['retained_body_bytes'], 0)
                self.assertEqual(len(server.calls), 1)

    def test_partial_incomplete_read_is_preserved_and_fully_charged(self):
        url = SOURCE_PREFIX + '07.tif'
        class BrokenResponse(Response):
            def read(self, count):
                raise http.client.IncompleteRead(b'abcde', count - 5)
        server = FakeServer({url: synthetic_tiff()},
                            lambda response, number: BrokenResponse(b'', response.status, response.headers))
        client = self.client(server)
        with self.assertRaises(reader.ContractError):
            client.get(url, 0, 7, 'header.bin')
        self.assertEqual(client.ledger['charged_response_bytes'], 8)
        self.assertEqual(client.ledger['retained_body_bytes'], 5)
        self.assertEqual((client.root / 'ranges' / 'header.bin.partial').read_bytes(), b'abcde')
        self.assertEqual(client.ledger['requests'][0]['body_sha256'], hashlib.sha256(b'abcde').hexdigest())
        self.assertEqual(len(server.calls), 1)

    def test_budget_and_undeclared_urls_reject_before_request(self):
        url = SOURCE_PREFIX + '07.tif'
        server = FakeServer({url: synthetic_tiff()})
        limits = dict(LIMITS, maximum_charged_response_bytes=7)
        client = self.client(server, limits=limits)
        with self.assertRaises(reader.ContractError):
            client.get(url, 0, 7, 'header.bin')
        with self.assertRaises(reader.ContractError):
            client.get(SOURCE_PREFIX + '08.tif', 0, 1, 'other.bin')
        self.assertEqual(server.calls, [])
        self.assertEqual(client.ledger['charged_response_bytes'], 0)

    def test_declared_large_bands_budget_and_float32_without_quantization(self):
        import numpy as np
        crops = [{'bbox_source_xywh': [1152, 0, 1152, 1152]},
                 {'bbox_source_xywh': [3456, 8064, 1152, 1152]}]
        self.assertEqual(reader.merged_row_bands(crops), [[0, 1152], [8064, 9216]])
        self.assertEqual(1152 * 6064 * 2, 13971456)
        self.assertEqual(51 * 2 * 1152 * 6064 * 2, 1425088512)
        values = reader.pooled_float32(np.array([0, 27 * 65535, 12345], dtype=np.uint32))
        self.assertEqual(values.dtype, np.float32)
        np.testing.assert_array_equal(values[:2], np.array([0, 255], dtype=np.float32))
        self.assertEqual(values[2], np.float32(12345 / (27 * 257)))
        self.assertNotEqual(float(values[2]), round(float(values[2])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', required=True)
    args = parser.parse_args()
    target = Path(args.receipt)
    if target.exists():
        raise SystemExit('Use a fresh fixture receipt')
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Fixtures)
    names = [t.id() for t in suite]
    # Tests use an in-process response fixture; any accidental socket connect stops.
    with patch.object(socket.socket, 'connect', side_effect=AssertionError('Network forbidden')), \
         patch.object(socket.socket, 'connect_ex', side_effect=AssertionError('Network forbidden')):
        outcome = unittest.TextTestRunner(verbosity=2).run(suite)
    receipt = {'schema_version': 1, 'status': 'PASS' if outcome.wasSuccessful() else 'FAIL',
               'reader': reader.file_record(ROOT / 'reader.py'),
               'fixture': reader.file_record(__file__), 'tests': names,
               'tests_run': outcome.testsRun, 'failures': len(outcome.failures),
               'errors': len(outcome.errors), 'network_calls': 0,
               'real_source_arrays_read': False, 'model_calls': 0,
               'synthetic_full_pipeline': '51 layers7..57,17 depth triplets,two asymmetric crops',
               'peak_rss_bytes': resource_peak if (resource_peak := reader.memory_check(LIMITS)) else 0}
    target.write_text(json.dumps(receipt, indent=2) + '\n')
    raise SystemExit(0 if outcome.wasSuccessful() else 1)
