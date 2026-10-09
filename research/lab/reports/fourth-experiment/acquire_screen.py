"""Acquire one frozen 384-square PHerc0800 screen; preserve observed/fill support."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import requests

PREFIX = 'https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0800/segments/'
CHUNK_BYTES = 31 * 128 * 128
TOTAL_LIMIT = 6 * 1024 * 1024


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('plan', type=Path)
    ap.add_argument('--output-dir', type=Path, required=True)
    args = ap.parse_args()
    plan = json.loads(args.plan.read_text())
    if args.output_dir.exists():
        raise ValueError('Output exists: preserve prior or partial acquisition.')
    plan_hash = sha(args.plan)
    for item in plan['verified_inputs']:
        if sha(item['path']) != item['sha256']:
            raise ValueError('Frozen geometry/selection input hash mismatch.')
    meta = json.loads(Path(plan['array_metadata_path']).read_text())
    attrs = json.loads(Path(plan['attrs_path']).read_text())
    if (meta['zarr_format'] != 2 or meta['shape'][0] != 31 or
        meta['chunks'] != [31, 128, 128] or meta['dtype'] != '|u1' or
        meta['order'] != 'C' or meta.get('compressor') is not None or
        meta.get('filters') or meta['fill_value'] != 0 or
        meta.get('dimension_separator') != '/'):
        raise ValueError('Unexpected uncompressed native source contract.')
    x, y, width, height = plan['bbox_source_xywh']
    if (width != 384 or height != 384 or min(x, y) < 0 or x % 128 or y % 128 or
        x + width > meta['shape'][2] or y + height > meta['shape'][1]):
        raise ValueError('Crop must be chunk-aligned 384-square inside source.')
    if plan['maximum_chunk_payload_bytes'] != 9 * CHUNK_BYTES:
        raise ValueError('Payload bound changed.')
    base = plan['surface_url'].rstrip('/')
    if not base.startswith(PREFIX) or '?' in base or '#' in base or '/..' in base:
        raise ValueError('Unexpected source URL.')
    mask = np.load(plan['geometry_mask_path'], mmap_mode='r', allow_pickle=False)
    xyz = np.load(plan['coordinate_map_path'], mmap_mode='r', allow_pickle=False)
    if mask.dtype != np.bool_ or mask.shape != tuple(meta['shape'][1:]):
        raise ValueError('Geometry mask does not align with source.')
    if xyz.dtype != np.float32 or xyz.shape != (*mask.shape, 3):
        raise ValueError('XYZ map does not align with source.')
    roi_mask = np.asarray(mask[y:y + height, x:x + width])
    roi_xyz = np.asarray(xyz[y:y + height, x:x + width])
    if not roi_mask.all() or not np.isfinite(roi_xyz).all():
        raise ValueError('Frozen crop lacks complete finite valid geometry.')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / 'frozen-plan.json').write_bytes(args.plan.read_bytes())
    session = requests.Session()
    session.mount('https://', requests.adapters.HTTPAdapter(max_retries=0))
    consumed = 0
    records = []

    def fetch(url, cap, filename, allow_missing=False):
        nonlocal consumed
        with session.get(url, stream=True, allow_redirects=False, timeout=(10, 45)) as r:
            body = bytearray()
            read_cap = min(cap if r.status_code == 200 else 65536, TOTAL_LIMIT - consumed)
            for block in r.iter_content(16384):
                consumed += len(block)
                body.extend(block)
                if len(body) > read_cap or consumed > TOTAL_LIMIT:
                    raise ValueError('Response exceeded bounded response-body budget.')
            data = bytes(body)
            rec = dict(url=url, retrieved_at_utc=datetime.now(timezone.utc).isoformat(),
                       http_status=r.status_code, size_bytes=len(data),
                       sha256=hashlib.sha256(data).hexdigest(), etag=r.headers.get('ETag'),
                       last_modified=r.headers.get('Last-Modified'), local_file=filename)
            if r.status_code == 404 and allow_missing:
                root = ET.fromstring(data)
                if root.findtext('Code') != 'NoSuchKey':
                    raise ValueError('404 was not explicit S3 NoSuchKey.')
                (args.output_dir / (filename + '.missing.xml')).write_bytes(data)
                rec['error_record_file'] = filename + '.missing.xml'
                rec['source_absence'] = 'Explicit S3 NoSuchKey'
                rec['reconstruction'] = 'Declared Zarr v2 fill_value=0'
                rec['local_payload_role'] = 'reconstructed fill'
                rec['observed_payload_bytes'] = 0
                data = bytes(CHUNK_BYTES)
                rec['reconstructed_sha256'] = hashlib.sha256(data).hexdigest()
            elif r.status_code != 200:
                raise ValueError(f'Public source HTTP {r.status_code}; no fallback or automatic retry.')
            else:
                rec['local_payload_role'] = 'observed source'
                rec['observed_payload_bytes'] = len(data)
                etag = (rec['etag'] or '').strip('"')
                if len(etag) == 32 and all(c in '0123456789abcdefABCDEF' for c in etag):
                    if hashlib.md5(data).hexdigest() != etag.lower():
                        raise ValueError('ETag MD5 and retained source payload differ.')
                    rec['etag_matches_payload_md5'] = True
            (args.output_dir / filename).write_bytes(data)
            return data, rec

    metadata_records = []
    for suffix, expected, filename in [('/0/.zarray', meta, 'source.zarray.json'),
                                        ('/.zattrs', attrs, 'source.zattrs.json')]:
        data, rec = fetch(base + suffix, 65536, filename)
        metadata_records.append(rec)
        if json.loads(data) != expected:
            raise ValueError('Remote source metadata changed after selection.')
    surface = np.empty((31, 384, 384), dtype=np.uint8)
    observed = np.zeros((384, 384), dtype=bool)
    for row in range(y // 128, y // 128 + 3):
        for col in range(x // 128, x // 128 + 3):
            filename = f'source-chunk-0-{row}-{col}.bin'
            data, rec = fetch(base + f'/0/0/{row}/{col}', CHUNK_BYTES, filename, True)
            if len(data) != CHUNK_BYTES:
                raise ValueError('Source chunk length differs from metadata.')
            dy, dx = row * 128 - y, col * 128 - x
            surface[:, dy:dy + 128, dx:dx + 128] = np.frombuffer(data, dtype=np.uint8).reshape(31, 128, 128)
            observed[dy:dy + 128, dx:dx + 128] = rec['http_status'] == 200
            rec['chunk_index_zyx'] = [0, row, col]
            records.append(rec)
    outputs = {'surface.npy': surface, 'geometry-mask.npy': roi_mask,
               'ct-coordinates.xyz.npy': roi_xyz, 'observed-mask.npy': observed}
    for filename, values in outputs.items():
        np.save(args.output_dir / filename, values, allow_pickle=False)
    for item in plan['verified_inputs']:
        if sha(item['path']) != item['sha256']:
            raise ValueError('Frozen input changed during acquisition.')
    if sha(args.plan) != plan_hash:
        raise ValueError('Plan changed during acquisition.')
    provenance = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        sample_id='PHerc0800', target_volume_id='20250521135224', segment_id=plan['segment_id'],
        surface_url=base, pyramid_level=0, bbox_source_xywh=[x, y, width, height],
        shape_zyx=list(surface.shape), dtype='uint8', voxel_spacing_zyx_um=[8.64] * 3,
        nominal_canvas_size_mm_xy=[384 * .00864] * 2, array_sha256=sha(args.output_dir / 'surface.npy'),
        frozen_plan_path=str(args.plan.resolve()), frozen_plan_sha256=plan_hash,
        verified_inputs=plan['verified_inputs'], render_provenance=plan['render_provenance'],
        metadata_sources=metadata_records, source_chunks=records,
        maximum_chunk_payload_bytes=9 * CHUNK_BYTES,
        downloaded_chunk_payload_bytes=sum(r['observed_payload_bytes'] for r in records),
        total_response_body_bytes=consumed, maximum_response_body_bytes=TOTAL_LIMIT,
        automatic_retries=0, request_count=11,
        missing_chunks_reconstructed_as_declared_fill=sum(r['http_status'] == 404 for r in records),
        geometry_valid_pixels=int(roi_mask.sum()), observed_source_pixels=int(observed.sum()),
        derived_outputs={n:dict(sha256=sha(args.output_dir / n),shape=list(a.shape),dtype=str(a.dtype)) for n,a in outputs.items()},
        acquisition_code_path=str(Path(__file__).resolve()), acquisition_code_sha256=sha(__file__),
        status='Frozen geometry-selected source acquisition; no ink or text claim',
        physical_independence='Unresolved: base-coordinate separation does not reconstruct full sampling support',
        transcription=None, translation=None)
    (args.output_dir / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps({k:provenance[k] for k in ['segment_id','bbox_source_xywh','downloaded_chunk_payload_bytes','total_response_body_bytes','missing_chunks_reconstructed_as_declared_fill','observed_source_pixels']}, indent=2))


if __name__ == '__main__':
    main()
