"""Independent bounded-support reconstruction. No producer imports or data access."""
import math
import struct
import numpy as np

SAMPLES = (0, 1, 3, 4, 127, 128, 255, 512, 767, 1024, 1407, 1534, 1535)

def ensure(condition, message):
    if not condition:
        raise ValueError(message)

def f32(value):
    try:
        return struct.unpack('<f', struct.pack('<f', value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)

def decode(body):
    """Integer bits preserve every selected float32 bit pattern, including NaNs."""
    ensure(len(body) == 456624, 'exact retained range length')
    bits = np.empty((80, 80), dtype=np.uint32)
    for row in range(80):
        selected = body[row * 5776:row * 5776 + 320]
        ensure(len(selected) == 320, 'selected row length')
        bits[row] = struct.unpack('<80I', selected)
    return bits.view(np.float32)

def mapping(first, canvas, mesh, scale):
    scale = f32(scale)
    origin = f32(-0.5 * f32(canvas - 1.0))
    center = f32(mesh / 2.0 / scale)
    coordinates = []
    fused = []
    for index in range(1536):
        tile = first + index // 128 * 128
        ul = f32(f32(f32(origin + tile) + center) * scale)
        scratch = f32(ul - 4.0 * scale)
        local = float(index % 128 + 4)
        coordinates.append(f32(scratch + f32(local * scale)))
        fused.append(f32(scratch + local * scale))
    c = np.asarray(coordinates, dtype=np.float32)
    floor = np.asarray([math.floor(v) for v in coordinates], dtype=np.int64)
    nearest = np.asarray([math.floor(v + .5) for v in coordinates], dtype=np.int64)
    ensure(all(math.floor(v) == a and math.floor(v + .5) == b
               for v, a, b in zip(fused, floor, nearest)), 'FMA dependency differs')
    return c, floor, nearest, [min(int(floor.min()), int(nearest.min()) - 1),
                                max(int(floor.max()) + 1, int(nearest.max()) + 1)]

def norm2(v):
    squares = np.multiply(v, v)
    return np.add(np.add(squares[..., 0], squares[..., 1]), squares[..., 2])

def normal_grid(xyz):
    vertices = np.isfinite(xyz).all(-1) & (xyz[..., 2] > 0) & (xyz[..., 0] != -1)
    centers = np.arange(1, 79)
    cy, cx = np.meshgrid(centers, centers, indexing='ij')
    support = np.logical_and.reduce([vertices[cy, cx], vertices[cy - 1, cx],
                                    vertices[cy + 1, cx], vertices[cy, cx - 1], vertices[cy, cx + 1]])
    across = np.subtract(xyz[cy, cx + 1], xyz[cy, cx - 1])
    down = np.subtract(xyz[cy + 1, cx], xyz[cy - 1, cx])
    cross = np.cross(across, down)
    length_squared = norm2(cross)
    valid = support & np.isfinite(cross).all(-1) & np.isfinite(length_squared) & (length_squared > 0)
    inverse = np.divide(np.float32(1), np.sqrt(np.where(valid, length_squared, np.float32(1))))
    direction = np.negative(np.multiply(cross, inverse[..., None]))
    second_length_squared = norm2(direction)
    valid &= np.isfinite(direction).all(-1) & np.isfinite(second_length_squared) & (second_length_squared > 0)
    direction = np.divide(direction, np.sqrt(np.where(valid, second_length_squared, np.float32(1)))[..., None])
    valid &= np.isfinite(direction).all(-1)
    result = np.zeros((80, 80, 3), dtype=np.float32)
    result[cy, cx] = np.where(valid[..., None], direction, np.float32(0))
    accepted = np.zeros((80, 80), dtype=bool)
    accepted[cy, cx] = valid
    return vertices, result, accepted

def native_bounds(base, direction, dims, offsets=(-33, 34)):
    accepted = np.ones(base.shape[:-1], dtype=bool)
    upper = np.asarray(dims, dtype=np.float64) - 1
    for depth in offsets:
        coordinates = np.add(base, np.multiply(direction, np.float32(depth)))
        accepted &= np.isfinite(coordinates).all(-1)
        accepted &= ((coordinates >= 0) & (coordinates.astype(np.float64) < upper)).all(-1)
    return accepted

def reconstruct(xyz, context, scale, dims, scalar_point=None):
    ensure(xyz.shape == (80, 80, 3) and xyz.dtype == np.float32, 'XYZ shape/dtype')
    ensure(context['selected_source_depth_half_open'] == [21, 89], 'depth extent')
    ensure(context['normal_offsets_inclusive'] == [-33, 34] and context['all_depth_count'] == 68, 'depth offsets')
    x, y, width, height = context['mesh_window_xywh']
    ensure(width == height == 80, 'window extent')
    fine = context['fine_raster_bbox_xywh']
    ax, ix, nx, dx = mapping(fine[0], 28880, 1444, scale[0])
    ay, iy, ny, dy = mapping(fine[1], 28080, 1404, scale[1])
    ensure(dx == [x, x + 79] and dy == [y, y + 79], 'complete dependency window')
    ensure(np.all((nx - x >= 1) & (nx - x <= 78)) and np.all((ny - y >= 1) & (ny - y <= 78)), 'normal indices')
    wx = np.subtract(ax, ix.astype(np.float32))[None, :, None]
    wy = np.subtract(ay, iy.astype(np.float32))
    ix, iy, nx, ny = ix - x, iy - y, nx - x, ny - y
    ensure(ix.min() >= 0 and ix.max() <= 78 and iy.min() >= 0 and iy.max() <= 78, 'bilinear indices')
    vertices, normals, normal_ok = normal_grid(xyz)
    masks = [np.zeros((384, 384), dtype=bool), np.zeros((384, 384), dtype=bool)]
    partition = np.zeros(4, dtype=np.int64)
    samples_checked = 0
    sample_bytes = bytearray()
    for start in range(0, 1536, 64):
        end = start + 64
        rows = iy[start:end, None]
        columns = ix[None, :]
        valid_position = np.logical_and.reduce([vertices[rows, columns], vertices[rows, columns + 1],
                                               vertices[rows + 1, columns], vertices[rows + 1, columns + 1]])
        top = np.add(np.multiply(xyz[rows, columns], np.subtract(np.float32(1), wx)),
                     np.multiply(xyz[rows, columns + 1], wx))
        bottom = np.add(np.multiply(xyz[rows + 1, columns], np.subtract(np.float32(1), wx)),
                        np.multiply(xyz[rows + 1, columns + 1], wx))
        weights_y = wy[start:end, None, None]
        base = np.add(np.multiply(top, np.subtract(np.float32(1), weights_y)), np.multiply(bottom, weights_y))
        valid_position &= np.isfinite(base).all(-1)
        direction = normals[ny[start:end, None], nx[None, :]]
        mesh = valid_position & normal_ok[ny[start:end, None], nx[None, :]] & np.isfinite(direction).all(-1)
        bounds = mesh & native_bounds(base, direction, dims)
        partition += [np.count_nonzero(~valid_position), np.count_nonzero(valid_position & ~mesh),
                      np.count_nonzero(mesh & ~bounds), np.count_nonzero(bounds)]
        for target, fine_mask in zip(masks, (mesh, bounds)):
            # Separate horizontal/vertical passes, unlike producer tile reshape.
            horizontal = fine_mask[:, 0::4] & fine_mask[:, 1::4] & fine_mask[:, 2::4] & fine_mask[:, 3::4]
            target[start // 4:end // 4] = horizontal[0::4] & horizontal[1::4] & horizontal[2::4] & horizontal[3::4]
        if scalar_point is not None:
            for sy in SAMPLES:
                if start <= sy < end:
                    for sx in SAMPLES:
                        ref = scalar_point(xyz, context, scale, dims, sx, sy)
                        actual_base = base[sy - start, sx]
                        if np.isfinite(actual_base).all():
                            ensure(struct.pack('<3f', *actual_base) == struct.pack('<3f', *ref['base']), 'scalar base bits')
                        actual_direction = direction[sy - start, sx]
                        if ref['mesh_ok']:
                            ensure(struct.pack('<3f', *actual_direction) == struct.pack('<3f', *ref['direction']), 'scalar direction bits')
                        ensure(bool(valid_position[sy - start, sx]) == ref['position_ok'], 'scalar position acceptance')
                        ensure(bool(mesh[sy - start, sx]) == ref['mesh_ok'], 'scalar mesh acceptance')
                        ensure(bool(bounds[sy - start, sx]) == ref['bounds_ok'], 'scalar bounds acceptance')
                        all_depths = bool(native_bounds(actual_base[None], actual_direction[None], dims, range(-33, 35))[0])
                        endpoint = bool(native_bounds(actual_base[None], actual_direction[None], dims)[0])
                        ensure(all_depths == endpoint, 'all68 versus endpoints')
                        sample_bytes.extend(struct.pack('<6f', *actual_base, *actual_direction))
                        samples_checked += 1
    counts = {'fine_total': 2359296, 'invalid_position_stencil_or_base': int(partition[0]),
              'invalid_normal_stencil_or_direction': int(partition[1]),
              'outside_declared_native_interpolation_bounds': int(partition[2]),
              'mesh_and_bounds_valid': int(partition[3]), 'mesh_supported_level2': int(masks[0].sum()),
              'mesh_and_bounds_supported_level2': int(masks[1].sum()), 'level2_total': 147456,
              'tiles': 144, 'depth_bound_method': 'two endpoints, staged FP32 monotonicity, all68 depths',
              'native_voxels_observed': 0}
    ensure(int(partition.sum()) == 2359296 and np.all(~masks[1] | masks[0]), 'partition/subset')
    return *masks, counts, {'scalar_points_checked': samples_checked, 'sample_bytes': bytes(sample_bytes)}
