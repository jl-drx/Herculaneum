"""Acquire one frozen chunk-aligned PHerc0800 crop with bounded provenance."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import requests

from fetch_zarr_chunk import fetch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError('Preserve existing acquisition; choose a fresh output directory.')
    plan = json.loads(args.plan.read_text())
    for item in plan['verified_inputs']:
        if sha(item['path']) != item['sha256']:
            raise ValueError('A frozen selection/geometry/metadata input changed.')
    meta = json.loads(Path(plan['array_metadata_path']).read_text())
    attrs = json.loads(Path(plan['attrs_path']).read_text())
    if meta['shape'][0] != 31 or meta['chunks'] != [31, 128, 128] or meta['dtype'] != '|u1' or meta['order'] != 'C' or meta.get('compressor') is not None or meta.get('filters') or meta['fill_value'] != 0 or meta.get('dimension_separator') != '/':
        raise ValueError('Unexpected native surface contract.')
    x, y, width, height = plan['bbox_source_xywh']
    if width != height or width not in (1536, 2048) or any(v % 128 for v in (x, y, width, height)) or x < 0 or y < 0 or x + width > meta['shape'][2] or y + height > meta['shape'][1]:
        raise ValueError('Region must be a bounded chunk-aligned square inside its source.')
    keys = [(row, col) for row in range(y // 128, (y + height) // 128)
            for col in range(x // 128, (x + width) // 128)]
    chunk_bytes = 31 * 128 * 128
    maximum_bytes = len(keys) * chunk_bytes
    if maximum_bytes != plan['maximum_chunk_payload_bytes'] or maximum_bytes > 128 * 1024 * 1024:
        raise ValueError('Plan exceeds the declared 128 MiB chunk payload cap.')
    base = plan['surface_url'].rstrip('/')
    prefix = 'https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0800/segments/'
    if not base.startswith(prefix):
        raise ValueError('Only the declared public PHerc0800 surface source is permitted.')
    current_meta, meta_record = fetch(base + '/0/.zarray', 64 * 1024)
    current_attrs, attrs_record = fetch(base + '/.zattrs', 64 * 1024)
    if json.loads(current_meta) != meta or json.loads(current_attrs) != attrs:
        raise ValueError('Remote metadata changed after the geometry selection was frozen.')
    full_mask = np.load(plan['geometry_mask_path'], mmap_mode='r', allow_pickle=False)
    full_xyz = np.load(plan['coordinate_map_path'], mmap_mode='r', allow_pickle=False)
    if full_mask.dtype != np.bool_ or full_mask.shape != tuple(meta['shape'][1:]) or full_xyz.shape != (*full_mask.shape, 3) or full_xyz.dtype != np.float32:
        raise ValueError('Geometry arrays do not align with native surface shape.')

    def download(key):
        row, col = key
        url = base + f'/0/0/{row}/{col}'
        try:
            data, record = fetch(url, chunk_bytes)
        except requests.HTTPError as error:
            if error.response.status_code != 404 or not getattr(error, 'zarr_missing_key', False):
                raise
            data = bytes(chunk_bytes)
            record = {'url': url, 'http_status': 404, 'source_absence': 'Explicit S3 NoSuchKey',
                      'size_bytes': 0, 'reconstruction': 'Declared Zarr v2 fill_value=0',
                      'reconstructed_sha256': hashlib.sha256(data).hexdigest()}
        if len(data) != chunk_bytes:
            raise ValueError('Source chunk length differs from native metadata.')
        return key, data, record

    with ThreadPoolExecutor(max_workers=4) as pool:
        downloaded = list(pool.map(download, keys))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    surface = np.empty((31, height, width), dtype=np.uint8)
    records = []
    for (row, col), data, record in downloaded:
        block = np.frombuffer(data, dtype=np.uint8).reshape(31, 128, 128)
        dy, dx = row * 128 - y, col * 128 - x
        surface[:, dy:dy + 128, dx:dx + 128] = block
        filename = f'source-chunk-0-{row}-{col}.bin'
        (args.output_dir / filename).write_bytes(data)
        record.update({'chunk_index_zyx': [0, row, col], 'local_file': filename,
                       'local_payload_role': 'reconstructed fill' if record.get('http_status') == 404 else 'observed source'})
        etag = record.get('etag', '') or ''
        etag = etag.strip('"')
        if len(etag) == 32 and '-' not in etag:
            record['etag_matches_payload_md5'] = hashlib.md5(data).hexdigest() == etag
            if not record['etag_matches_payload_md5']:
                raise ValueError('Source ETag does not match its retained chunk payload.')
        records.append(record)
    roi_mask = np.asarray(full_mask[y:y + height, x:x + width])
    roi_xyz = np.asarray(full_xyz[y:y + height, x:x + width])
    if not roi_mask.any() or not np.isfinite(roi_xyz[roi_mask]).all():
        raise ValueError('Selected crop lacks finite valid geometry.')
    outputs = {'surface.npy': surface, 'geometry-mask.npy': roi_mask,
               'ct-coordinates.xyz.npy': roi_xyz,
               'selected-depth-nonzero-mask.npy': np.any(surface[7:24] != 0, axis=0)}
    for filename, values in outputs.items():
        np.save(args.output_dir / filename, values, allow_pickle=False)
    (args.output_dir / 'source.zarray.json').write_bytes(current_meta)
    (args.output_dir / 'source.zattrs.json').write_bytes(current_attrs)
    provenance = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
                  'sample_id': 'PHerc0800', 'target_volume_id': '20250521135224',
                  'segment_id': plan['segment_id'], 'surface_url': base,
                  'pyramid_level': 0, 'bbox_source_xywh': [x, y, width, height],
                  'shape_zyx': list(surface.shape), 'dtype': 'uint8',
                  'voxel_spacing_zyx_um': [8.64, 8.64, 8.64],
                  'nominal_canvas_size_mm_xy': [width * .00864, height * .00864],
                  'array_sha256': sha(args.output_dir / 'surface.npy'),
                  'frozen_plan_path': str(args.plan.resolve()), 'frozen_plan_sha256': sha(args.plan),
                  'verified_inputs': plan['verified_inputs'], 'render_provenance': plan['render_provenance'],
                  'metadata_sources': [meta_record, attrs_record], 'source_chunks': records,
                  'maximum_chunk_payload_bytes': maximum_bytes,
                  'downloaded_chunk_payload_bytes': sum(item['size_bytes'] for item in records),
                  'source_chunk_count': len(records),
                  'missing_chunks_reconstructed_as_declared_fill': sum(item.get('http_status') == 404 for item in records),
                  'geometry_valid_pixels': int(roi_mask.sum()),
                  'geometry_valid_fraction': float(roi_mask.mean()),
                  'selected_depth_nonzero_geometry_pixels': int((roi_mask & outputs['selected-depth-nonzero-mask.npy']).sum()),
                  'derived_outputs': {name: {'sha256': sha(args.output_dir / name), 'shape': list(values.shape), 'dtype': str(values.dtype)} for name, values in outputs.items()},
                  'acquisition_code_sha256': sha(Path(__file__)),
                  'status': 'Frozen geometry-selected broader context; no ink or text claim',
                  'limitations': ['Segment ID does not establish a separate physical surface.',
                                  'Geometry and intensity support are not ink labels.',
                                  'Canvas scale is nominal; flattening distortion remains unmeasured.']}
    (args.output_dir / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    for item in plan['verified_inputs']:
        if sha(item['path']) != item['sha256']:
            raise ValueError('Frozen selection input changed during acquisition.')
    print(json.dumps({key: provenance[key] for key in ['segment_id', 'bbox_source_xywh', 'shape_zyx', 'downloaded_chunk_payload_bytes', 'missing_chunks_reconstructed_as_declared_fill', 'geometry_valid_fraction', 'selected_depth_nonzero_geometry_pixels']}, indent=2))


if __name__ == '__main__':
    main()
