"""Pre-output fixtures: synthetic coordinates only, immutable attempt receipts."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import struct
import numpy as np
from independent_geometry import decode, ensure, mapping, native_bounds, reconstruct
from scalar_oracle import point
from verify_runtime import PLAN, check_plan, check_transport, pin

ROOT = Path(__file__).resolve().parent

def main(output):
    ensure(output.parent == ROOT and not output.exists(), 'fresh synthetic output directory')
    output.mkdir()
    tests = []
    result = {'status': 'RUNNING', 'actual_XYZ_reads': 0, 'network_requests': 0, 'checks': tests}
    def passed(name):
        tests.append({'name': name, 'status': 'PASS'})
    def rejects(name, action):
        try:
            action()
        except (ValueError, KeyError):
            passed(name)
        else:
            raise ValueError('failed to reject: ' + name)
    try:
        plan, freeze = check_plan()
        passed('all producer and proposal source pins; independent structural ranges')
        scale = [.05000000074505806, .05000000074505806]
        dims = [26511, 26511, 76953]
        for context in plan['contexts']:
            for axis, canvas, mesh in [('x', 28880, 1444), ('y', 28080, 1404)]:
                index = 0 if axis == 'x' else 1
                declared = context['axis_' + axis]
                values, floor, nearest, extent = mapping(context['fine_raster_bbox_xywh'][index], canvas, mesh, scale[index])
                ensure(extent == declared['dependency_inclusive']
                       and float(values.min()) == declared['coordinate_min']
                       and float(values.max()) == declared['coordinate_max'], 'axis proposal equality')
                passed(context['id'] + ' axis ' + axis + ' staged mapping and dependency')
        # Opaque gaps contain deliberately different bit patterns than selected coordinates.
        bits = np.arange(6400, dtype=np.uint32).reshape(80, 80) + np.uint32(0x3f000000)
        body = bytearray(b'\xff' * 456624)
        for row in range(80):
            body[row * 5776:row * 5776 + 320] = struct.pack('<80I', *bits[row])
        decoded = decode(body)
        ensure(np.array_equal(decoded.view(np.uint32), bits), 'selected row bits')
        alternate = bytearray(body)
        for row in range(79):
            alternate[row * 5776 + 320:(row + 1) * 5776] = b'\x00' * 5456
        ensure(np.array_equal(decoded.view(np.uint32), decode(alternate).view(np.uint32)), 'gap invariance')
        passed('strided selected-row decode and poisoned-gap opacity')
        rejects('truncated raw body rejected', lambda: decode(body[:-1]))
        rejects('oversized raw body rejected', lambda: decode(body + b'x'))
        # All coordinates are generated here; no actual acquisition data exists in this fixture.
        yy, xx = np.indices((80, 80), dtype=np.float32)
        for index, context in enumerate(plan['contexts']):
            xyz = np.stack([xx + np.float32(1000), yy + np.float32(1000),
                            np.full((80, 80), np.float32(1000))], axis=-1)
            if index:
                xyz[..., 2] += xx * np.float32(.25) - yy * np.float32(.5)
            mesh, bounds, counts, coverage = reconstruct(xyz, context, scale, dims, point)
            ensure(mesh.all() and bounds.all() and counts['mesh_and_bounds_valid'] == 2359296
                   and coverage['scalar_points_checked'] == 169, 'analytic plane reconstruction')
            passed(context['id'] + ' complete plane masks and 169 scalar point/all68 checks')
        context = plan['contexts'][0]
        xyz = np.stack([xx + np.float32(1000), yy + np.float32(1000),
                        np.full((80, 80), np.float32(10))], axis=-1)
        mesh, bounds, counts, _ = reconstruct(xyz, context, scale, dims, point)
        ensure(mesh.all() and not bounds.any() and counts['outside_declared_native_interpolation_bounds'] == 2359296,
               'valid mesh wholly outside depth support')
        passed('mesh/bounds separation and lower-depth rejection for complete context')
        xyz[:] = np.float32(1000)
        mesh, bounds, counts, _ = reconstruct(xyz, context, scale, dims, point)
        ensure(not mesh.any() and not bounds.any() and counts['invalid_normal_stencil_or_direction'] == 2359296,
               'degenerate normals')
        passed('complete degenerate-normal rejection with finite positions')
        xyz[..., 0], xyz[..., 1], xyz[..., 2] = xx + 1000, yy + 1000, np.float32(1000)
        xyz[38, 44, 0] = np.float32(-1)
        mesh, bounds, counts, _ = reconstruct(xyz, context, scale, dims, point)
        ensure(counts['invalid_position_stencil_or_base'] > 0 and counts['invalid_normal_stencil_or_direction'] > 0
               and 0 < mesh.sum() < 147456 and np.array_equal(mesh, bounds), 'sentinel rejection partition')
        passed('sentinel causes disjoint position and normal rejection, complete all16 reduction')
        ensure(native_bounds(np.array([[0, 0, 0]], dtype=np.float32), np.zeros((1, 3), dtype=np.float32), dims)[0], 'zero inclusive lower bound')
        ensure(not native_bounds(np.array([[26510, 0, 0]], dtype=np.float32), np.zeros((1, 3), dtype=np.float32), dims)[0], 'strict upper bound')
        ensure(native_bounds(np.array([[np.nextafter(np.float32(26510), np.float32(0)), 0, 0]], dtype=np.float32), np.zeros((1, 3), dtype=np.float32), dims)[0], 'adjacent lower float')
        passed('native exact and adjacent-FP32 boundary cases')
        # Retained fake transport ledger is independent of any HTTP code.
        run = output / 'synthetic-range-ledger'
        run.mkdir()
        ledger = {'status': 'PASS', 'reserved_bytes_all_attempts': 2739750, 'charged_bytes': 2739744, 'requests': []}
        for request in plan['requests']:
            path = run / (request['id'] + '.body')
            path.write_bytes(body)
            lo, hi = request['range_inclusive']
            ledger['requests'].append({'id': request['id'], 'url': request['source']['url'],
                                      'range_inclusive': [lo, hi], 'if_match': request['source']['strong_etag'],
                                      'status': 'PASS', 'http_status': 206, 'reserved_body_bytes': 456625,
                                      'retained_body_bytes': 456624, 'body': pin(path),
                                      'response_headers': {'Content-Range': f'bytes {lo}-{hi}/8109638',
                                                           'ETag': request['source']['strong_etag'], 'Content-Length': '456624'}})
        path = run / 'transport-ledger.json'
        def write_and_check(value):
            path.write_text(json.dumps(value))
            return check_transport(plan, {'transport_ledger': pin(path)}, run)
        write_and_check(ledger)
        passed('exact six retained range receipt/body checks')
        changes = [('status', 200), ('etag', '"changed"'), ('range', 'bytes 0-456623/8109638'),
                   ('encoding', 'gzip'), ('length', '456623'), ('reservation', 456624), ('hash', '0' * 64), ('url', 'https://invalid.example/')]
        for case, value in changes:
            altered = copy.deepcopy(ledger)
            row = altered['requests'][0]
            if case == 'status': row['http_status'] = value
            elif case == 'etag': row['response_headers']['ETag'] = value
            elif case == 'range': row['response_headers']['Content-Range'] = value
            elif case == 'encoding': row['response_headers']['Content-Encoding'] = value
            elif case == 'length': row['response_headers']['Content-Length'] = value
            elif case == 'reservation': row['reserved_body_bytes'] = value
            elif case == 'hash': row['body']['sha256'] = value
            elif case == 'url': row['url'] = value
            rejects('retained transport mutation rejected: ' + case, lambda a=altered: write_and_check(a))
        write_and_check(ledger)
        result['status'] = 'PASS'
    except BaseException as exc:
        result.update(status='STOP', error_type=type(exc).__name__, reason=str(exc))
    result['check_count'] = len(tests)
    destination = output / 'verification.json'
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'status': result['status'], 'checks': len(tests), 'receipt': pin(destination), 'reason': result.get('reason')}))
    return 0 if result['status'] == 'PASS' else 2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    raise SystemExit(main(parser.parse_args().output))
