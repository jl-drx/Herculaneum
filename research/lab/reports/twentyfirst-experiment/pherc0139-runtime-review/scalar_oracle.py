"""Scalar PHerc0139 geometric-support oracle, with struct-rounded FP32 stages.

No NumPy, producer imports, I/O, or actual-coordinate reads occur at import.
The CLI writes synthetic fixture evidence only to the explicitly given path.
Python binary64 evaluates an operation on FP32 operands before struct rounds
the result; no multiply-add is fused. Indexing a supplied ndarray is permitted.
"""
import math
import struct


def f(value):
    """Round to IEEE binary32, including signed zero and overflow to infinity."""
    value = float(value)
    try:
        return struct.unpack('<f', struct.pack('<f', value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


def add(a, b):
    return f(float(a) + float(b))


def sub(a, b):
    return f(float(a) - float(b))


def mul(a, b):
    return f(float(a) * float(b))


def dot_self(v):
    return add(add(mul(v[0], v[0]), mul(v[1], v[1])), mul(v[2], v[2]))


def raster_coordinate(mesh_count, canvas_count, scale, first, local):
    """One retained fine pixel in its historical aligned 128-pixel tile."""
    if first % 128 or local < 0 or not isinstance(local, int):
        raise ValueError('unaligned raster origin or invalid local index')
    s = f(scale)
    if not math.isfinite(s) or s <= 0:
        raise ValueError('invalid scale')
    tile, retained = divmod(local, 128)
    center = f(mesh_count / 2.0 / s)
    origin = mul(-0.5, sub(f(canvas_count), 1.0))
    offset = add(add(origin, 0.0), f(first + 128 * tile))
    upper_left = mul(add(f(offset / 1.0), center), s)
    scratch_origin = f(float(upper_left) - 4.0 * float(s))
    position = add(scratch_origin, mul(f(retained + 4), s))
    if not math.isfinite(position):
        raise ValueError('nonfinite raster coordinate')
    lower = math.floor(position)
    # All authorized mapped positions are positive, so lround == floor(p+.5).
    nearest = math.floor(float(position) + 0.5)
    if lower < 0 or lower + 1 >= mesh_count or not 1 <= nearest < mesh_count - 1:
        raise ValueError('global mesh stencil boundary')
    return position, lower, nearest, sub(position, f(lower))


def valid_vertex(v):
    return all(math.isfinite(c) for c in v) and v[2] > 0.0 and v[0] != -1.0


def depth_position(base, direction, offset):
    return tuple(add(b, mul(d, f(offset))) for b, d in zip(base, direction))


def native_bounds(base, direction, dims, all_depths=False):
    """Pure coordinate-bound decision; separate from mesh/stencil validity."""
    if len(base) != 3 or len(direction) != 3 or len(dims) != 3:
        raise ValueError('three dimensions required')
    if not all(math.isfinite(float(n)) and float(n) > 1 for n in dims):
        raise ValueError('invalid native dimensions')
    depths = range(-33, 35) if all_depths else (-33, 34)
    for offset in depths:
        for component, size in zip(depth_position(base, direction, offset), dims):
            if not math.isfinite(component) or not 0.0 <= component < float(size) - 1.0:
                return False
    return True


def point(xyz, context, scale, dims, local_x, local_y):
    """Evaluate a single fine point; bounds_ok includes mesh_ok.

    base/direction are tuples of FP32-representable Python floats. Invalid
    normal stencils return zero direction, matching the declared cache output.
    An escaped mesh window raises instead of inventing additional support.
    """
    if len(xyz) != 80 or any(len(row) != 80 for row in xyz):
        raise ValueError('expected 80 by 80 supplied XYZ window')
    if context.get('selected_source_depth_half_open') != [21, 89] or context.get('normal_offsets_inclusive') != [-33, 34] or context.get('all_depth_count') != 68:
        raise ValueError('depth contract mismatch')
    fx, fy, fw, fh = context['fine_raster_bbox_xywh']
    ox, oy, width, height = context['mesh_window_xywh']
    if (fw, fh, width, height) != (1536, 1536, 80, 80):
        raise ValueError('extent mismatch')
    if not isinstance(local_x, int) or not isinstance(local_y, int) or not 0 <= local_x < fw or not 0 <= local_y < fh:
        raise ValueError('point outside frozen fine extent')
    px, xfloor, xn, wx = raster_coordinate(1444, 28880, scale[0], fx, local_x)
    py, yfloor, yn, wy = raster_coordinate(1404, 28080, scale[1], fy, local_y)
    x, y, nx, ny = xfloor - ox, yfloor - oy, xn - ox, yn - oy
    if not (0 <= x < width - 1 and 0 <= y < height - 1 and 1 <= nx < width - 1 and 1 <= ny < height - 1):
        raise ValueError('stencil escaped frozen coordinate window')

    def vertex(col, row):
        value = tuple(f(c) for c in xyz[row][col])
        if len(value) != 3:
            raise ValueError('XYZ triplet required')
        return value

    corners = [vertex(x, y), vertex(x + 1, y), vertex(x, y + 1), vertex(x + 1, y + 1)]
    one_x, one_y = sub(1.0, wx), sub(1.0, wy)
    base = []
    for k in range(3):
        top = add(mul(corners[0][k], one_x), mul(corners[1][k], wx))
        bottom = add(mul(corners[2][k], one_x), mul(corners[3][k], wx))
        base.append(add(mul(top, one_y), mul(bottom, wy)))
    base = tuple(base)
    position_ok = all(valid_vertex(v) for v in corners) and all(math.isfinite(b) for b in base)
    center, left, right, above, below = [vertex(a, b) for a, b in ((nx, ny), (nx - 1, ny), (nx + 1, ny), (nx, ny - 1), (nx, ny + 1))]
    direction = (0.0, 0.0, 0.0)
    normal_ok = all(valid_vertex(v) for v in (center, left, right, above, below))
    if normal_ok:
        horizontal = tuple(sub(right[k], left[k]) for k in range(3))
        vertical = tuple(sub(below[k], above[k]) for k in range(3))
        cross = tuple(sub(mul(horizontal[(k + 1) % 3], vertical[(k + 2) % 3]), mul(horizontal[(k + 2) % 3], vertical[(k + 1) % 3])) for k in range(3))
        length2 = dot_self(cross)
        normal_ok = all(math.isfinite(c) for c in cross) and math.isfinite(length2) and length2 > 0.0
        if normal_ok:
            reciprocal_length = f(1.0 / f(math.sqrt(length2)))
            flipped = tuple(f(-mul(c, reciprocal_length)) for c in cross)
            second2 = dot_self(flipped)
            normal_ok = all(math.isfinite(c) for c in flipped) and math.isfinite(second2) and second2 > 0.0
            if normal_ok:
                second_length = f(math.sqrt(second2))
                direction = tuple(f(c / second_length) for c in flipped)
                normal_ok = all(math.isfinite(c) for c in direction)
                if not normal_ok:
                    direction = (0.0, 0.0, 0.0)
    mesh_ok = position_ok and normal_ok
    bounds = native_bounds(base, direction, dims)
    return {'base': base, 'direction': direction, 'position_ok': bool(position_ok),
            'mesh_ok': bool(mesh_ok), 'bounds_ok': bool(mesh_ok and bounds),
            'native_bounds_ok': bool(bounds), 'normal_ok': bool(normal_ok),
            'mesh_position': (px, py), 'bilinear_lower_global': (xfloor, yfloor),
            'nearest_global': (xn, yn), 'endpoint_positions': [depth_position(base, direction, -33), depth_position(base, direction, 34)]}


def synthetic_fixtures():
    """All coordinates below are constructed in memory, never file-loaded."""
    import random
    scale = (f(0.05), f(0.05))
    dims = (26511, 26511, 76953)
    context = {'fine_raster_bbox_xywh': [7168, 19456, 1536, 1536],
               'mesh_window_xywh': [357, 972, 80, 80],
               'selected_source_depth_half_open': [21, 89],
               'normal_offsets_inclusive': [-33, 34], 'all_depth_count': 68}
    checks = []

    def record(name, condition, expected, observed):
        if not condition:
            raise AssertionError((name, expected, observed))
        checks.append({'name': name, 'expected': expected, 'observed': observed, 'pass': True})

    def plane(tilt=False, reverse=False, height=1000.0):
        return [[[f(200 - j if reverse else 100 + j), f(200 + i), f(height + j + 2 * i if tilt else height)] for j in range(80)] for i in range(80)]

    flat = point(plane(), context, scale, dims, 0, 0)
    expected_base = (101.42501831054688, 200.82501220703125, 1000.0)
    record('flat_interpolation_and_flipped_normal', flat['base'] == expected_base and flat['direction'] == (0.0, 0.0, -1.0) and flat['bounds_ok'], {'base': expected_base, 'direction': [0, 0, -1], 'accept': True}, flat)
    tilted = point(plane(tilt=True), context, scale, dims, 0, 0)
    expected_direction = (1 / math.sqrt(6), 2 / math.sqrt(6), -1 / math.sqrt(6))
    # Plane z=1000+u+2v; FP32 interpolation stays within one ulp here.
    expected_tilt_z = 1000 + 1.425018310546875 + 2 * 0.82501220703125
    record('tilted_analytic_plane', max(abs(a - b) for a, b in zip(tilted['direction'], expected_direction)) < 1e-7 and abs(tilted['base'][2] - expected_tilt_z) <= 0.0001220703125 and tilted['bounds_ok'], {'direction_analytic': expected_direction, 'base_z_analytic': expected_tilt_z, 'normal_tolerance': 1e-7, 'position_tolerance': 0.0001220703125}, tilted)
    reversed_plane = point(plane(reverse=True), context, scale, dims, 0, 0)
    record('reversed_parameterization_reverses_direction', reversed_plane['direction'] == (0.0, 0.0, 1.0) and reversed_plane['bounds_ok'], [0, 0, 1], reversed_plane['direction'])
    for name, component, value in [('x_sentinel', 0, -1.0), ('z_zero', 2, 0.0), ('z_negative', 2, -2.0), ('nan', 1, math.nan), ('positive_inf', 0, math.inf), ('negative_inf', 2, -math.inf)]:
        mesh = plane()
        mesh[0][1][component] = value
        result = point(mesh, context, scale, dims, 0, 0)
        record('bilinear_' + name, not result['position_ok'] and not result['mesh_ok'] and not result['bounds_ok'], [False, False, False], [result['position_ok'], result['mesh_ok'], result['bounds_ok']])
    mesh = plane()
    mesh[1][0][0] = -1.0  # Left normal neighbor, outside bilinear corners.
    neighbor = point(mesh, context, scale, dims, 0, 0)
    record('invalid_normal_neighbor_outside_bilinear_cell', neighbor['position_ok'] and not neighbor['mesh_ok'] and neighbor['direction'] == (0.0, 0.0, 0.0), [True, False, [0, 0, 0]], [neighbor['position_ok'], neighbor['mesh_ok'], neighbor['direction']])
    degenerate = point([[[100.0, 200.0, 1000.0] for j in range(80)] for i in range(80)], context, scale, dims, 0, 0)
    record('degenerate_normal_rejected', degenerate['position_ok'] and not degenerate['mesh_ok'], [True, False], [degenerate['position_ok'], degenerate['mesh_ok']])
    for height, expected in [(34.0, True), (33.0, False)]:
        result = point(plane(height=height), context, scale, dims, 0, 0)
        record('flat_depth_lower_bound_' + str(height), result['mesh_ok'] and result['bounds_ok'] is expected, expected, result['bounds_ok'])
    escaped = dict(context, mesh_window_xywh=[358, 972, 80, 80])
    try:
        point(plane(), escaped, scale, dims, 0, 0)
        raised = False
    except ValueError:
        raised = True
    record('normal_window_border_aborts', raised, True, raised)
    # Adjacent binary32 values grounded by bit patterns around strict bounds.
    prior_99 = struct.unpack('<f', struct.pack('<I', 0x42c60000 - 1))[0]
    neg_subnormal = struct.unpack('<f', struct.pack('<I', 0x80000001))[0]
    cases = [((50.0, 50.0, 0.0), (0.0, 0.0, -0.0), True),
             ((50.0, 50.0, neg_subnormal), (0.0, 0.0, 0.0), False),
             ((50.0, 50.0, prior_99), (0.0, 0.0, 0.0), True),
             ((50.0, 50.0, 99.0), (0.0, 0.0, 0.0), False),
             ((50.0, 50.0, 34.0), (0.0, 0.0, -1.0), True),
             ((50.0, 50.0, 66.0), (0.0, 0.0, -1.0), False),
             ((50.0, 50.0, math.inf), (0.0, 0.0, 0.0), False),
             ((50.0, 50.0, math.nan), (0.0, 0.0, 0.0), False)]
    for i, (base, direction, expected) in enumerate(cases):
        a, b = native_bounds(base, direction, (100, 100, 100)), native_bounds(base, direction, (100, 100, 100), True)
        record('hand_grounded_depth_boundary_' + str(i), a is expected and b is expected, expected, {'endpoints': a, 'all68': b})
    generator = random.Random(13092026)
    accepted = 0
    for i in range(2048):
        direction = tuple(f(generator.uniform(-1.0, 1.0)) for k in range(3))
        base = tuple(f(generator.uniform(-40.0, 140.0)) for k in range(3))
        a, b = native_bounds(base, direction, (100, 100, 100)), native_bounds(base, direction, (100, 100, 100), True)
        if a != b:
            raise AssertionError(('endpoint_all68_disagreement', i, base, direction))
        accepted += a
    record('2048_endpoint_vs_all68', True, 'identical decisions', {'identical': 2048, 'accepted': accepted, 'rejected': 2048 - accepted, 'seed': 13092026})
    return {'status': 'PASS', 'synthetic_only': True, 'actual_coordinate_reads': 0,
            'numpy_imported_or_used': False, 'producer_imported': False,
            'checks': checks, 'check_count': len(checks),
            'arithmetic': 'scalar Python operation on FP32-representable values then struct <f round; no FMA; no NumPy arithmetic',
            'bounds_ok_definition': 'mesh_ok AND native_bounds_ok',
            'scope': 'Scalar geometric/computational-support oracle only; no CT observation or historical renderer equivalence claim'}


if __name__ == '__main__':
    import argparse
    import hashlib
    import json
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    receipt = synthetic_fixtures()
    receipt['source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    destination = Path(args.output)
    with destination.open('x') as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'status': receipt['status'], 'checks': receipt['check_count'], 'source_sha256': receipt['source_sha256']}))
