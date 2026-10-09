#!/usr/bin/env python3
"""Frozen, bounded source-render check; plan is offline, run is explicitly separate.

Adapted without importing the original (which downloads at module import).
This diagnostic validates mapping at ten points, never ink or surface placement.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import itertools
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
import requests
import tifffile

LAB = Path('/workspace/herculaneum-lab')
HERE = Path(__file__).resolve().parent
GEOMETRY = LAB / 'data/experiments/pherc0800-larger-geometry/20251028220955'
SURFACE = LAB / 'data/experiments/pherc0800-screen384-20261008'
DATA = LAB / 'data/experiments/pherc0800-fifth-native-check'
COMMIT = '530584554c031c982ba3aa243e1ea39c4bef9818'
URL = ('https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0800/'
       'volumes/20250521135224-8.640um-1.2m-116keV-masked.zarr/0/')
METADATA_SHA = '0dc78ab465bc614b960cfb86b235e6405ed8e54efc450eb7d1e7d213f5409d18'
ACQUISITION_SHA = '81c340861d41ba9c36d935eecae72b89f09277d6acad61742af3769a8abacedb'
SURFACE_SHA = 'f01e5936447d896af0336841f70667d2a8807c54ed83885ba2e7599d413ef423'
CHUNK_BYTES = 128 ** 3
NATIVE_CAP = 16 * 1024 * 1024
METADATA_CAP = 256 * 1024
POINTS = [(f'{group}.{direction}', x + dx, y + dy)
          for group, x, y in [('A', 1278, 504), ('B', 1100, 628)]
          for direction, dx, dy in [('peak', 0, 0), ('left', -1, 0),
                                    ('right', 1, 0), ('up', 0, -1), ('down', 0, 1)]]
OFFSETS = list(range(-15, 16))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text())


def write_json_new(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def file_record(path, expected=None):
    path = Path(path).resolve()
    digest = sha(path)
    require(expected is None or digest == expected, f'Input SHA mismatch: {path.name}')
    return {'path': str(path), 'sha256': digest, 'size_bytes': path.stat().st_size}


def verify_records(records):
    for rec in records:
        require(Path(rec['path']).stat().st_size == rec['size_bytes'], 'Input size changed')
        require(sha(rec['path']) == rec['sha256'], f"Input SHA changed: {Path(rec['path']).name}")


def f32(value):
    return np.asarray(value, dtype=np.float32)


def fma32(a, b, c):
    # float64 holds the exact product/sum of these float32 operands before rounding.
    return np.float32(float(a) * float(b) + float(c))


def unit(value):
    value = np.asarray(value, dtype=np.float64)
    length = np.linalg.norm(value)
    require(np.isfinite(length) and length > 0, 'Degenerate visualization tangent')
    return (value / length).tolist()


def verify_render(render):
    expected = {'commit': COMMIT, 'binary-path': 'vc_render_tifxyz',
                'group-idx': '0', 'scale': '1', 'num-slices': '31',
                'rotate': '0', 'flip': '-1', 'flip-normals': 'true',
                'composite': 'false', 'voxel-size': '8.64', 'voxel-unit': 'micrometer',
                'zarr-compressor': 'none', 'zarr-separator': '/'}
    for name, value in expected.items():
        require(render.get(name) == value, f'Render convention mismatch: {name}')
    for name in ['affine-transform', 'affines', 'affine-matrix', 'scale-segmentation']:
        require(render.get(name) == '', f'Unexpected transform: {name}')
    for name in ['crop-x', 'crop-y', 'crop-width', 'crop-height']:
        require(render.get(name) == '0', f'Unexpected render crop: {name}')
    # Missing slice-step is the pinned CLI default (1.0); no fitted adjustment.
    require(render.get('slice-step', '1') in ['1', '1.0'], 'Unexpected depth step')
    require('20251028220955-auto_grown_20251028220955262' in render['segmentation-path'],
            'Wrong render segment')


def verify_native_metadata(metadata):
    expected = {'shape': [24298, 9867, 9867], 'chunks': [128, 128, 128],
                'dtype': '|u1', 'compressor': None, 'filters': None,
                'fill_value': 0, 'order': 'C', 'dimension_separator': '/', 'zarr_format': 2}
    for name, value in expected.items():
        require(metadata.get(name) == value, f'Native metadata mismatch: {name}')


def point_geometry(name, x, y, coords, scale, canvas_shape, full_map, local_map,
                   geometry_mask, observed_mask, bbox, native_metadata):
    canvas_h, canvas_w = canvas_shape
    c0, r0 = (x // 128) * 128, (y // 128) * 128
    center = f32([float(coords.shape[1]) / 2 / float(scale[0]),
                  float(coords.shape[0]) / 2 / float(scale[1])])
    offset = f32([-0.5 * (canvas_w - 1) + c0, -0.5 * (canvas_h - 1) + r0])
    ul = f32(f32(offset + center) * scale)
    origin = np.array([float(ul[k]) - 4 * float(scale[k]) for k in [0, 1]], dtype=np.float32)
    gx = np.float32(origin[0] + np.float32(x - c0 + 4) * scale[0])
    gy = np.float32(origin[1] + np.float32(y - r0 + 4) * scale[1])
    ix, iy = int(gx), int(gy)
    require(0 <= ix < coords.shape[1] - 1 and 0 <= iy < coords.shape[0] - 1,
            'Bilinear geometry outside mesh')
    wx, wy = np.float32(gx - ix), np.float32(gy - iy)
    p00, p01, p10, p11 = coords[iy, ix], coords[iy, ix+1], coords[iy+1, ix], coords[iy+1, ix+1]
    require(np.isfinite([p00, p01, p10, p11]).all() and
            (np.array([p00, p01, p10, p11]) > 0).all(), 'Invalid bilinear geometry')
    base = f32(f32(f32(p00 * np.float32(1-wx)) + f32(p01 * wx)) * np.float32(1-wy)
               + f32(f32(p10 * np.float32(1-wx)) + f32(p11 * wx)) * wy)
    nx, ny = int(np.floor(float(gx) + .5)), int(np.floor(float(gy) + .5))
    require(1 <= nx < coords.shape[1] - 1 and 1 <= ny < coords.shape[0] - 1,
            'Normal stencil outside mesh')
    xl, xr, yu, yd = coords[ny, nx-1], coords[ny, nx+1], coords[ny-1, nx], coords[ny+1, nx]
    stencil = np.array([coords[ny, nx], xl, xr, yu, yd])
    require(np.isfinite(stencil).all() and (stencil > 0).all(), 'Invalid normal stencil')
    xv, yv = f32(xr-xl), f32(yd-yu)
    normal = f32([xv[1]*yv[2]-xv[2]*yv[1], xv[2]*yv[0]-xv[0]*yv[2],
                  xv[0]*yv[1]-xv[1]*yv[0]])
    len2 = np.float32(normal[0]*normal[0] + normal[1]*normal[1] + normal[2]*normal[2])
    require(np.isfinite(len2) and len2 > 0, 'Degenerate renderer normal')
    normal = f32(normal * np.float32(1 / np.sqrt(len2)))
    unflipped = normal.copy()
    normal = f32(-normal)
    len2 = np.float32(normal[0]*normal[0] + normal[1]*normal[1] + normal[2]*normal[2])
    normal = f32(normal / np.sqrt(len2))
    require(abs(float(np.linalg.norm(normal)) - 1) <= 1e-5, 'Normal is not unit length')
    lx, ly = x - bbox[0], y - bbox[1]
    require(0 <= lx < bbox[2] and 0 <= ly < bbox[3], 'Point outside acquired surface')
    require(bool(geometry_mask[ly, lx]) and bool(observed_mask[ly, lx]),
            'Point geometry or observed surface support invalid')
    require(np.array_equal(full_map[y, x], local_map[ly, lx]), 'Local/global map mismatch')
    error = float(np.abs(base - full_map[y, x]).max())
    require(float(np.linalg.norm(base - full_map[y, x])) < .01,
            'Archived/cached geometry differs by >=.01 native voxel')
    samples = []
    for layer, depth in enumerate(OFFSETS):
        xyz = f32([fma32(normal[k], depth, base[k]) for k in [0, 1, 2]])
        corner0 = np.floor(xyz[::-1]).astype(np.int64)
        corners = [corner0 + np.array(delta) for delta in itertools.product([0, 1], repeat=3)]
        require(all((v >= 0).all() and (v < native_metadata['shape']).all() for v in corners),
                'Native interpolation corner outside volume')
        samples.append({'layer': layer, 'offset_voxels': depth, 'position_ct_xyz': xyz.tolist(),
                        'corners_zyx': [v.tolist() for v in corners],
                        'chunk_indices_zyx': sorted({tuple((v // 128).tolist()) for v in corners})})
    tangent_u, tangent_v = unit(xv), unit(yv)
    # Visualization-only orthogonal frame; it never changes renderer sampling.
    orthogonal_v = unit(np.cross(np.asarray(normal, dtype=float), tangent_u))
    return {'name': name, 'source_xy': [x, y], 'surface_local_roi_xy': [lx, ly],
            'render_tile_origin_xy': [c0, r0], 'mesh_grid_xy': [float(gx), float(gy)],
            'bilinear_mesh_cells_xy': [[ix, iy], [ix+1, iy], [ix, iy+1], [ix+1, iy+1]],
            'normal_grid_xy': [nx, ny], 'normal_stencil_cells_xy':
            [[nx, ny], [nx-1, ny], [nx+1, ny], [nx, ny-1], [nx, ny+1]],
            'base_ct_xyz': base.tolist(), 'cached_map_ct_xyz': full_map[y, x].tolist(),
            'cached_vs_archived_base_max_abs_error_voxels': error,
            'normal_xyz_unflipped': unflipped.tolist(), 'normal_xyz_flipped': normal.tolist(),
            'normal_length': float(np.linalg.norm(normal)),
            'tangent_u_source_mesh_xyz': tangent_u, 'tangent_v_source_mesh_xyz': tangent_v,
            'visualization_orthogonal_frame_xyz': {'u': tangent_u, 'v': orthogonal_v,
                                                   'n': unit(normal)},
            'samples': samples, 'letter': None, 'transcription': None, 'translation': None}


def build_plan(plan_path):
    require(not plan_path.exists(), 'Plan already exists; do not replace a frozen plan')
    provenance_path = SURFACE / 'provenance.json'
    provenance = load_json(provenance_path)
    require(provenance['sample_id'] == 'PHerc0800' and
            provenance['target_volume_id'] == '20250521135224' and
            provenance['segment_id'] == '20251028220955', 'Wrong surface identity')
    bbox = provenance['bbox_source_xywh']
    require(bbox == [1024, 384, 384, 384], 'Source crop changed')
    require(provenance['array_sha256'] == SURFACE_SHA, 'Wrong acquired surface SHA')
    acquisition = load_json(provenance['frozen_plan_path'])
    records = [file_record(provenance_path), file_record(provenance['frozen_plan_path'], ACQUISITION_SHA),
               file_record(__file__)]
    require(provenance['frozen_plan_sha256'] == ACQUISITION_SHA, 'Wrong acquisition plan pin')
    require(acquisition['bbox_source_xywh'] == bbox, 'Acquisition crop mismatch')
    expected_inputs = {r['path']: r for r in acquisition['verified_inputs']}
    geometry_paths = [GEOMETRY / n for n in ['mesh.meta.json', 'mesh.x.tif', 'mesh.y.tif',
                      'mesh.z.tif', 'surface.level0.zarray.json', 'source-render-export-metadata.json',
                      'full-surface-ct-coordinates.xyz.npy', 'full-surface-valid-mask.npy']]
    # Mesh TIFF bytes are separately pinned now; they were retained with verified acquisition inputs.
    for path in geometry_paths:
        expected = expected_inputs.get(str(path), {}).get('sha256')
        require(expected is not None or path.name in ['mesh.x.tif', 'mesh.y.tif', 'mesh.z.tif'],
                'Geometry not linked to frozen acquisition')
        records.append(file_record(path, expected))
    for name, rec in provenance['derived_outputs'].items():
        records.append(file_record(SURFACE / name, rec['sha256']))
        array = np.load(SURFACE / name, mmap_mode='r', allow_pickle=False)
        require(list(array.shape) == rec['shape'] and str(array.dtype) == rec['dtype'],
                f'Acquired output schema mismatch: {name}')
    for rec in provenance['metadata_sources'] + provenance['source_chunks']:
        require(rec['http_status'] == 200, 'This frozen comparison requires observed surface chunks')
        records.append(file_record(SURFACE / rec['local_file'], rec['sha256']))
    require(provenance['derived_outputs']['surface.npy']['sha256'] == SURFACE_SHA,
            'Surface output pin mismatch')
    export = load_json(GEOMETRY / 'source-render-export-metadata.json')
    require(export['parameters']['target_volume'] == '20250521135224', 'Wrong render target volume')
    render = export['creation_info']['provenance']['parameters']
    verify_render(render)
    require(render == acquisition['render_provenance']['parameters'], 'Frozen render metadata differs')
    canvas = load_json(GEOMETRY / 'surface.level0.zarray.json')
    require(canvas['shape'] == [31, 2100, 2100], 'Unexpected source canvas shape')
    coords = np.stack([tifffile.imread(GEOMETRY / f'mesh.{axis}.tif') for axis in 'xyz'], axis=-1)
    require(coords.dtype == np.float32 and coords.shape == (105, 105, 3), 'Wrong mesh shape/dtype')
    scale = f32(load_json(GEOMETRY / 'mesh.meta.json')['scale'])
    require(np.array_equal(scale, f32([.05, .05])), 'Unexpected mesh scale')
    full_map = np.load(GEOMETRY / 'full-surface-ct-coordinates.xyz.npy', mmap_mode='r')
    require(full_map.shape == (2100, 2100, 3) and full_map.dtype == np.float32, 'Wrong full map')
    local_map = np.load(SURFACE / 'ct-coordinates.xyz.npy', mmap_mode='r')
    mask = np.load(SURFACE / 'geometry-mask.npy', mmap_mode='r')
    observed = np.load(SURFACE / 'observed-mask.npy', mmap_mode='r')
    native_meta_path = LAB / 'data/experiments/pherc0800-larger-native-consistency/ct.level0.zarray.json'
    records.append(file_record(native_meta_path, METADATA_SHA))
    native_meta = load_json(native_meta_path)
    verify_native_metadata(native_meta)
    source_dir = LAB / 'data/experiments/pherc0800-native-ct-consistency/export-source'
    for name in ['vc_render_tifxyz.cpp', 'Slicing.cpp', 'QuadSurface.cpp', 'Geometry.cpp']:
        records.append(file_record(source_dir / name))
    prior_helper = LAB / 'data/experiments/pherc0800-native-ct-consistency/check_native_ct.py'
    records.append(file_record(prior_helper))
    points = [point_geometry(name, x, y, coords, scale, canvas['shape'][1:], full_map,
                             local_map, mask, observed, bbox, native_meta) for name, x, y in POINTS]
    required = sorted({tuple(index) for point in points for sample in point['samples']
                       for index in sample['chunk_indices_zyx']})
    require(len(required) <= 8 and len(required) * CHUNK_BYTES <= NATIVE_CAP, 'Native plan exceeds cap')
    # Reuse only exact source URL + report SHA + actual SHA, never merely matching filenames.
    reusable = {}
    for report_path, cache_dir in [
        (LAB / 'reports/first-experiment/native-ct-consistency.json',
         LAB / 'data/experiments/pherc0800-native-ct-consistency'),
        (LAB / 'reports/third-experiment/native-ct-consistency.json',
         LAB / 'data/experiments/pherc0800-larger-native-consistency')]:
        report = load_json(report_path)
        records.append(file_record(report_path))
        require(report['target_volume_id'] == '20250521135224', 'Cache report wrong volume')
        for rec in report['native_chunk_sources']:
            index = tuple(rec['chunk_index_zyx'])
            if index not in required or rec.get('missing', False):
                continue
            require(rec['url'] == URL + '/'.join(map(str, index)), 'Cache source URL mismatch')
            path = cache_dir / rec['file']
            record = file_record(path, rec['sha256'])
            require(record['size_bytes'] == CHUNK_BYTES, 'Cache chunk wrong size')
            records.append(record)
            reusable[index] = {'path': str(path), 'sha256': rec['sha256'], 'source_record': rec}
    chunks = [{'chunk_index_zyx': list(index), 'url': URL + '/'.join(map(str, index)),
               'size_bytes': CHUNK_BYTES, 'local_path': str(DATA / ('ct-chunk-' +
               '-'.join(map(str, index)) + '.bin')), 'verified_reuse': reusable.get(index)}
              for index in required]
    plan = {'schema_version': 1, 'phase': 'frozen_before_native_network_or_model_read',
            'created_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
            'sample_id': 'PHerc0800', 'target_volume_id': '20250521135224',
            'segment_id': '20251028220955', 'bbox_source_xywh': bbox,
            'surface_shape_zyx': [31, 384, 384], 'canvas_shape_yx': [2100, 2100],
            'point_selection': 'Two parent-supplied fourth-experiment diagnostic score peaks plus fixed cardinal +/-1 source pixel. Selected after model inspection; no independent discovery or accuracy claim.',
            'points': points, 'offsets_voxels': OFFSETS, 'voxel_size_um': 8.64,
            'native_metadata_path': str(native_meta_path), 'native_metadata_sha256': METADATA_SHA,
            'native_metadata': native_meta, 'native_metadata_url': URL + '.zarray',
            'render_parameters': render, 'source_commit': COMMIT,
            'sampling_convention': 'Archived half-pixel tile/halo float32 bilinear base; nearest mesh-grid cached normal, recorded flip=true and renormalization; layer offset -15..15; float32 position and fused trilinear operations; clamp [0,255] then uint8 truncation. No fitting or parameter search.',
            'precision_limitation': 'Python emulates archived float32/FMA arithmetic; compiled ARM contraction can differ at quantization boundaries. Exact and <=1 counts are separately reported.',
            'chunks': chunks, 'required_distinct_native_chunks': len(chunks),
            'forecast_download_native_chunks': sum(c['verified_reuse'] is None for c in chunks),
            'forecast_download_native_bytes': sum(c['size_bytes'] for c in chunks if c['verified_reuse'] is None),
            'forecast_distinct_native_bytes': len(chunks) * CHUNK_BYTES,
            'budget': {'max_distinct_native_chunks': 8, 'max_native_chunk_bytes': NATIVE_CAP,
                       'max_metadata_response_bytes': METADATA_CAP, 'max_attempts_per_object': 1,
                       'metadata_requests': 1, 'allow_redirects': False, 'allow_missing_fill': False},
            'preregistered_checks': {'sample_count': 310, 'exact_intensity_matches_reported': True,
                'within_one_intensity_matches_reported': True, 'max_abs_uint8_error_to_continue': 1,
                'max_base_coordinate_error_norm_voxels_exclusive': .01,
                'max_unit_normal_length_error': 1e-5,
                'metadata_must_match_frozen_byte_sha_and_schema': True},
            'stop_rules': ['Metadata, code, input, convention or frozen sample mismatch.',
                           'Native budget exceeded, missing object or unsupported response.',
                           'Any of 310 rendered uint8 intensities differs by >1; no automatic expansion.'],
            'local_cross_section_contract': 'Use retained base XYZ, unit normals, mesh U/V and visualization orthogonal frame. Only these planned native chunks may be used. Trilinear plane pixels lacking any of eight corners are unknown; no extra requests for figures.',
            'limitations': ['The ten nearby points are dependent; five points in each group share the nearest normal cache grid.',
                'Technical render consistency does not establish papyrus surface placement, ink, letters or layer identity.',
                'Physical independence from previously inspected papyrus remains unresolved.',
                'The public CT is compared to an export from an internal source path; equality is checked locally, not assumed globally.'],
            'letter': None, 'transcription': None, 'translation': None,
            'input_files': sorted({r['path']: r for r in records}.values(), key=lambda r: r['path'])}
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_new(plan_path, plan)
    print(json.dumps({'plan_path': str(plan_path), 'plan_sha256': sha(plan_path),
                      'points': len(points), 'samples': 310, 'chunks': required,
                      'download_bytes': plan['forecast_download_native_bytes']}))


class NativeStop(Exception):
    pass


def fetch_once(session, url, cap, path, records, counters, category):
    # Exact public HTTPS URLs only; no custom credential headers, redirects or retries.
    require(url.startswith(URL), 'Unexpected native source URL')
    raw = bytearray()
    record = {'url': url, 'local_path': str(path), 'retrieved_at_utc': dt.datetime.now(dt.timezone.utc).isoformat()}
    records.append(record)
    try:
        with session.get(url, stream=True, timeout=(15, 60), allow_redirects=False,
                         headers={'Accept-Encoding': 'identity'}) as response:
            record.update(http_status=response.status_code, etag=response.headers.get('ETag'),
                          last_modified=response.headers.get('Last-Modified'),
                          content_encoding=response.headers.get('Content-Encoding'))
            require(response.headers.get('Content-Encoding', 'identity') in ['', 'identity'],
                    'Unsupported content encoding')
            # HTTP error bodies are retained under the same fixed per-object budget.
            while True:
                block = response.raw.read(min(65536, cap - len(raw) + 1), decode_content=False)
                if not block:
                    break
                raw.extend(block)
                counters[category] += len(block)
                require(len(raw) <= cap, 'HTTP object exceeds planned byte cap')
                require(counters['native'] <= NATIVE_CAP and counters['metadata'] <= METADATA_CAP,
                        'Total HTTP response budget exceeded')
            record.update(size_bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
            retained_path = path if response.status_code == 200 else path.with_name(path.name + '.http-error')
            retained_path.write_bytes(raw)
            record['local_path'] = str(retained_path)
            if response.status_code != 200:
                code = None
                try:
                    code = ET.fromstring(raw).findtext('Code')
                except ET.ParseError:
                    pass
                record['s3_error_code'] = code
                record['diagnosis'] = 'S3 NoSuchKey; no observed native data' if response.status_code == 404 and code == 'NoSuchKey' else 'HTTP failure; no observed native data'
                raise NativeStop(record['diagnosis'])
    except requests.RequestException as exc:
        record['diagnosis'] = 'Transport failure; no retry or source substitution'
        # Do not serialize exception messages which might contain proxy/auth details.
        record['exception_type'] = type(exc).__name__
        raise NativeStop(record['diagnosis']) from None
    return bytes(raw), record


def trilinear(xyz, arrays):
    def at(z, y, x):
        index = (z // 128, y // 128, x // 128)
        require(index in arrays, 'Interpolation support not retained')
        return np.float32(arrays[index][z % 128, y % 128, x % 128])
    x, y, z = map(np.float32, xyz)
    iz, iy, ix = int(z), int(y), int(x)
    c000, c100 = at(iz, iy, ix), at(iz+1, iy, ix)
    c010, c110 = at(iz, iy+1, ix), at(iz+1, iy+1, ix)
    c001, c101 = at(iz, iy, ix+1), at(iz+1, iy, ix+1)
    c011, c111 = at(iz, iy+1, ix+1), at(iz+1, iy+1, ix+1)
    fz, fy, fx = np.float32(z-iz), np.float32(y-iy), np.float32(x-ix)
    c00 = fma32(fx, np.float32(c001-c000), c000)
    c01 = fma32(fx, np.float32(c011-c010), c010)
    c10 = fma32(fx, np.float32(c101-c100), c100)
    c11 = fma32(fx, np.float32(c111-c110), c110)
    c0 = fma32(fy, np.float32(c01-c00), c00)
    c1 = fma32(fy, np.float32(c11-c10), c10)
    return fma32(fz, np.float32(c1-c0), c0)


def run_plan(plan_path, expected_plan_sha):
    require(sha(plan_path) == expected_plan_sha, 'Frozen plan SHA mismatch')
    plan = load_json(plan_path)
    verify_records(plan['input_files'])
    require(plan['source_commit'] == COMMIT and plan['offsets_voxels'] == OFFSETS, 'Wrong frozen convention')
    verify_render(plan['render_parameters'])
    verify_native_metadata(plan['native_metadata'])
    require([(p['name'], *p['source_xy']) for p in plan['points']] == POINTS, 'Frozen points differ')
    require(len(plan['chunks']) <= 8 and len(plan['chunks']) * CHUNK_BYTES <= NATIVE_CAP, 'Chunk cap exceeded')
    require(plan['budget'] == {'max_distinct_native_chunks': 8, 'max_native_chunk_bytes': NATIVE_CAP,
            'max_metadata_response_bytes': METADATA_CAP, 'max_attempts_per_object': 1,
            'metadata_requests': 1, 'allow_redirects': False, 'allow_missing_fill': False}, 'Budget changed')
    DATA.mkdir(parents=True, exist_ok=False)
    counters = {'native': 0, 'metadata': 0}
    responses, chunk_sources, arrays = [], [], {}
    result = {'created_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'frozen_plan_path': str(plan_path), 'frozen_plan_sha256': expected_plan_sha,
              'status': 'stopped_pending_diagnosis', 'expansion_gate_passed': False,
              'scope': 'Source-render mapping at fixed points; no ink validation or papyrus layer identity.',
              'sample_id': 'PHerc0800', 'target_volume_id': '20250521135224',
              'segment_id': '20251028220955', 'native_metadata': plan['native_metadata'],
              'bbox_source_xywh': plan['bbox_source_xywh'], 'responses': responses,
              'native_chunk_sources': chunk_sources, 'points': [], 'response_bytes': counters,
              'input_files': plan['input_files'], 'limitations': plan['limitations'],
              'local_cross_section_contract': plan['local_cross_section_contract'],
              'letter': None, 'transcription': None, 'translation': None}
    try:
        with requests.Session() as session:
            raw, _ = fetch_once(session, plan['native_metadata_url'], METADATA_CAP,
                                DATA / 'ct.level0.zarray.json', responses, counters, 'metadata')
            require(hashlib.sha256(raw).hexdigest() == plan['native_metadata_sha256'],
                    'Remote metadata bytes differ from frozen source')
            verify_native_metadata(json.loads(raw))
            for chunk in plan['chunks']:
                index = tuple(chunk['chunk_index_zyx'])
                require(chunk['url'] == URL + '/'.join(map(str, index)), 'Chunk URL differs')
                path = DATA / ('ct-chunk-' + '-'.join(map(str, index)) + '.bin')
                require(str(path) == chunk['local_path'], 'Chunk destination differs')
                reuse = chunk['verified_reuse']
                if reuse is not None:
                    require(sha(reuse['path']) == reuse['sha256'], 'Verified cache changed')
                    raw = Path(reuse['path']).read_bytes()
                    require(len(raw) == CHUNK_BYTES, 'Reused chunk size mismatch')
                    path.write_bytes(raw)
                    record = {'url': chunk['url'], 'local_path': str(path), 'sha256': sha(path),
                              'size_bytes': len(raw), 'reused_from': reuse['path'],
                              'original_source_record': reuse['source_record'], 'acquisition': 'verified_cache_copy'}
                else:
                    raw, record = fetch_once(session, chunk['url'], CHUNK_BYTES, path,
                                             responses, counters, 'native')
                    require(len(raw) == CHUNK_BYTES, 'Native chunk size differs')
                    record['acquisition'] = 'new_http200_payload'
                record['chunk_index_zyx'] = list(index)
                chunk_sources.append(record)
                arrays[index] = np.frombuffer(raw, dtype=np.uint8).reshape((128,) * 3)
        surface = np.load(SURFACE / 'surface.npy', mmap_mode='r', allow_pickle=False)
        require(surface.shape == (31, 384, 384) and surface.dtype == np.uint8, 'Surface schema differs')
        for point in plan['points']:
            pred = np.array([trilinear(s['position_ct_xyz'], arrays) for s in point['samples']], dtype=np.float32)
            pred_uint = np.clip(pred, 0, 255).astype(np.uint8)
            lx, ly = point['surface_local_roi_xy']
            observed = np.array(surface[:, ly, lx], dtype=np.uint8)
            difference = np.abs(observed.astype(np.int16) - pred_uint.astype(np.int16))
            result['points'].append({**point, 'observed_profile_uint8': observed.tolist(),
                    'native_ct_profile_float32': pred.tolist(), 'rendered_profile_uint8': pred_uint.tolist(),
                    'absolute_intensity_errors': difference.tolist(),
                    'exact_matches': int((difference == 0).sum()),
                    'within_one_intensity_count': int((difference <= 1).sum()),
                    'max_abs_error': int(difference.max()), 'mean_abs_error': float(difference.mean())})
        differences = [v for p in result['points'] for v in p['absolute_intensity_errors']]
        result['aggregate'] = {'samples': len(differences), 'exact_matches': sum(v == 0 for v in differences),
                'within_one_intensity_count': sum(v <= 1 for v in differences),
                'max_abs_error': max(differences), 'mean_abs_error': float(np.mean(differences))}
        require(len(differences) == 310, 'Wrong comparison sample count')
        verify_records(plan['input_files'])
        require(max(differences) <= 1, 'Render intensity error >1: stop expansion pending diagnosis')
        result.update(status='source_render_mapping_reproduced_within_one_at_fixed_points',
                      expansion_gate_passed=True)
    except (NativeStop, ValueError, OSError, json.JSONDecodeError) as exc:
        result['stop_reason'] = str(exc) if not isinstance(exc, OSError) else type(exc).__name__
    result['retained_native_bytes'] = len(arrays) * CHUNK_BYTES
    result['decoded_distinct_native_chunks'] = len(arrays)
    # Manifest excludes itself and includes retained raw HTTP-error bytes if any.
    write_json_new(HERE / 'result.json', result)
    output_paths = [HERE / 'result.json', *sorted(p for p in DATA.iterdir() if p.is_file())]
    write_json_new(HERE / 'manifest.json', {'frozen_plan': file_record(plan_path),
                   'inputs': plan['input_files'], 'outputs': [file_record(p) for p in output_paths]})
    print(json.dumps({'status': result['status'], 'expansion_gate_passed': result['expansion_gate_passed'],
                      'response_bytes': counters, 'aggregate': result.get('aggregate'),
                      'result_path': str(HERE / 'result.json')}))
    return 0 if result['expansion_gate_passed'] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)
    offline = subparsers.add_parser('plan', help='Offline geometry-only freeze; no CT or model reads')
    offline.add_argument('--plan', type=Path, default=HERE / 'frozen-plan.json')
    online = subparsers.add_parser('run', help='Run approved frozen plan; bounded native CT downloads')
    online.add_argument('--plan', type=Path, default=HERE / 'frozen-plan.json')
    online.add_argument('--plan-sha256', required=True)
    args = parser.parse_args()
    if args.command == 'plan':
        build_plan(args.plan.resolve())
        return 0
    return run_plan(args.plan.resolve(), args.plan_sha256)


if __name__ == '__main__':
    sys.exit(main())
