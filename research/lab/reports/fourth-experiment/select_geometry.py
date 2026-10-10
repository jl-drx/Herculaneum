#!/usr/bin/env python3
"""Exact geometry-only PHerc0800 selection; never opens intensity/model data."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree


LAB = Path('/workspace/herculaneum-lab')
GEOMETRY = LAB / 'data/experiments/pherc0800-larger-geometry'
SIZE = 384
ALIGNMENT = 128
THRESHOLD = 40.0
QUERY_BATCH = 131072


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def file_record(path):
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'sha256': sha256(path), 'size_bytes': path.stat().st_size}


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def summary(values):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return {'count': int(values.size), 'finite_count': int(finite.size), 'min': float(finite.min()) if finite.size else None, 'max': float(finite.max()) if finite.size else None, 'quantiles_0_1_50_95_99_100': np.percentile(finite, [0, 1, 50, 95, 99, 100]).tolist() if finite.size else None}


def intersects(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


def load_map(mask_path, xyz_path, expected_shape):
    mask = np.load(mask_path, mmap_mode='r', allow_pickle=False)
    xyz = np.load(xyz_path, mmap_mode='r', allow_pickle=False)
    if mask.shape != tuple(expected_shape) or mask.dtype != np.bool_:
        raise ValueError('Unexpected Boolean geometry-mask contract.')
    if xyz.shape != tuple(expected_shape) + (3,) or xyz.dtype != np.float32:
        raise ValueError('Unexpected float32 XYZ coordinate-map contract.')
    valid_xyz = xyz[mask]
    if not np.isfinite(valid_xyz).all() or not (valid_xyz > 0).all():
        raise ValueError('Geometry-valid coordinates must be finite and positive.')
    return mask, xyz


def geometry_diagnostics(xyz, global_mask, x, y):
    points = np.asarray(xyz, dtype=np.float64)
    dx = np.linalg.norm(points[:, 1:] - points[:, :-1], axis=2)
    dy = np.linalg.norm(points[1:] - points[:-1], axis=2)
    tx = (points[1:-1, 2:] - points[1:-1, :-2]) / 2
    ty = (points[2:, 1:-1] - points[:-2, 1:-1]) / 2
    cross_length = np.linalg.norm(np.cross(tx, ty), axis=2)
    distance = distance_transform_edt(np.pad(global_mask, 1, constant_values=False))[1:-1, 1:-1]
    crop_distance = distance[y:y + SIZE, x:x + SIZE]
    return {
        'horizontal_neighbor_jump_voxels': summary(dx),
        'vertical_neighbor_jump_voxels': summary(dy),
        'horizontal_zero_or_nearzero_count_at_1e_minus12': int((dx <= 1e-12).sum()),
        'vertical_zero_or_nearzero_count_at_1e_minus12': int((dy <= 1e-12).sum()),
        'central_finite_difference_cross_norm_voxels_squared': summary(cross_length),
        'finite_difference_degenerate_count_at_1e_minus8': int((cross_length <= 1e-8).sum()),
        'finite_difference_nonfinite_count': int((~np.isfinite(cross_length)).sum()),
        'original_geometry_boundary_distance_pixels': summary(crop_distance),
        'boundary_metric': 'Euclidean distance to an invalid pixel in the original global mask, padded with one invalid border.',
        'renderer_normals_reconstructed': False,
        'normal_proxy_limit': 'Central finite-difference cross products are a base-map diagnostic, not the archived renderer normals, recto identification, or through-depth support reconstruction.'
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--methodology', type=Path, required=True)
    parser.add_argument('--methodology-sha256', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    methodology_path = args.methodology.resolve(strict=True)
    if sha256(methodology_path) != args.methodology_sha256:
        raise ValueError('Frozen methodology hash mismatch.')
    method = json.loads(methodology_path.read_text())
    if method['selector_sha256'] != sha256(Path(__file__)):
        raise ValueError('Selector source differs from the frozen methodology.')
    if method['crop_size_yx'] != [SIZE, SIZE] or method['chunk_alignment_pixels'] != ALIGNMENT or method['minimum_nearest_base_coordinate_distance_voxels_strictly_greater_than'] != THRESHOLD:
        raise ValueError('Frozen rule differs from the implemented constants.')
    output = args.output_dir.resolve()
    if not output.is_relative_to(Path(__file__).resolve().parent):
        raise ValueError('Output must stay within the new fourth-experiment directory.')
    output.mkdir(parents=True, exist_ok=False)
    verified = {}

    def verify(path, expected_sha, expected_size=None, role='geometry evidence'):
        record = file_record(path)
        if record['sha256'] != expected_sha or (expected_size is not None and record['size_bytes'] != expected_size):
            raise ValueError(f'Retained input checksum/size mismatch: {path}')
        record['role'] = role
        previous = verified.get(record['path'])
        if previous and previous['sha256'] != record['sha256']:
            raise ValueError('Retained input changed during verification.')
        verified[record['path']] = record
        return record

    for item in method['frozen_metadata_inputs']:
        verify(item['path'], item['sha256'], role='Frozen provenance/metadata snapshot')
    selection = json.loads(Path(method['geometry_selection_path']).read_text())
    source_records = json.loads(Path(method['source_records_path']).read_text())
    source_lookup = {item['local_file']: item for item in source_records}
    if selection['sample_id'] != 'PHerc0800' or selection['target_volume_id'] != '20250521135224':
        raise ValueError('Geometry selection belongs to another target volume.')

    prior_sources, prior_points = [], []
    small_prov_path = LAB / 'data/experiments/pherc0800-surface980/provenance.json'
    small_prov = json.loads(small_prov_path.read_text())
    small_summary_path = LAB / 'data/experiments/pherc0800-surface-geometry/full-surface-geometry-summary.json'
    small_summary = json.loads(small_summary_path.read_text())
    small_outputs = {e['file']: e for e in small_summary['derived_outputs']}
    small_base = small_summary_path.parent
    small_mask_path = small_base / 'full-surface-valid-mask.npy'
    small_xyz_path = small_base / 'full-surface-ct-coordinates.xyz.npy'
    for p in (small_mask_path, small_xyz_path):
        e = small_outputs[p.name]
        verify(p, e['sha256'], e['byte_count'], 'Prior acquired geometry')
    small_mask, small_xyz = load_map(small_mask_path, small_xyz_path, (980, 980))
    if small_prov['bbox_source_xywh'] != [0, 0, 980, 980] or small_prov['segment_id'] != small_summary['segment_id']:
        raise ValueError('Initial acquired extent/provenance does not align to full geometry.')
    prior_points.append(np.asarray(small_xyz[small_mask], dtype=np.float64))
    prior_sources.append({'segment_id': small_prov['segment_id'], 'target_volume_id': small_prov['target_volume_id'], 'bbox_source_xywh': small_prov['bbox_source_xywh'], 'geometry_valid_acquired_pixel_count': int(small_mask.sum()), 'mask': verified[str(small_mask_path.resolve())], 'coordinates': verified[str(small_xyz_path.resolve())], 'provenance': verified[str(small_prov_path.resolve())], 'intensity_array_sha256_recorded_in_provenance': small_prov['array_sha256'], 'intensity_array_opened': False})

    larger_base = LAB / 'data/experiments/pherc0800-larger1536'
    larger_prov_path = larger_base / 'provenance.json'
    larger_prov = json.loads(larger_prov_path.read_text())
    larger_mask_path, larger_xyz_path = larger_base / 'geometry-mask.npy', larger_base / 'ct-coordinates.xyz.npy'
    for p in (larger_mask_path, larger_xyz_path):
        e = larger_prov['derived_outputs'][p.name]
        verify(p, e['sha256'], role='Prior acquired geometry')
    larger_mask, larger_xyz = load_map(larger_mask_path, larger_xyz_path, (1536, 1536))
    if larger_prov['bbox_source_xywh'] != [512, 512, 1536, 1536] or larger_prov['segment_id'] != '20251028222030':
        raise ValueError('Unexpected larger acquired extent.')
    prior_points.append(np.asarray(larger_xyz[larger_mask], dtype=np.float64))
    prior_sources.append({'segment_id': larger_prov['segment_id'], 'target_volume_id': larger_prov['target_volume_id'], 'bbox_source_xywh': larger_prov['bbox_source_xywh'], 'geometry_valid_acquired_pixel_count': int(larger_mask.sum()), 'mask': verified[str(larger_mask_path.resolve())], 'coordinates': verified[str(larger_xyz_path.resolve())], 'provenance': verified[str(larger_prov_path.resolve())], 'intensity_array_sha256_recorded_in_provenance': larger_prov['array_sha256'], 'intensity_array_opened': False})
    if any(p['target_volume_id'] != selection['target_volume_id'] for p in prior_sources):
        raise ValueError('Prior coordinates are not in the same native CT volume.')
    points = np.concatenate(prior_points, axis=0)
    del prior_points
    tree_started = time.perf_counter()
    tree = cKDTree(points, leafsize=16, compact_nodes=True, balanced_tree=True, copy_data=True)
    tree_seconds = time.perf_counter() - tree_started
    print(f'Exact prior tree contains {len(points):,} acquired geometry-valid centers; built in {tree_seconds:.2f} s.', flush=True)
    write_json(output / 'prior-acquisition-inventory.json', {'sources': prior_sources, 'total_prior_points': len(points), 'duplicates_retained': True, 'derivative_crops': 'All retained pilot/candidate/independent/control crops are subsets of these acquired extents; no derivative substitutes for the full extents.'})

    records, segment_summaries, segment_data = [], [], {}
    for segment in sorted(selection['segments'], key=lambda s: s['segment_id']):
        sid = segment['segment_id']
        meta_records = {}
        for name in ('surface.level0.zarray.json', 'surface.zattrs.json', 'mesh.meta.json'):
            entry = source_lookup[f'{sid}/{name}']
            path = GEOMETRY / entry['local_file']
            rec = verify(path, entry['sha256'], entry['size_bytes'], 'Recorded remote metadata')
            rec['source_url'] = entry['url']
            rec['source_etag'] = entry.get('etag')
            rec['source_last_modified'] = entry.get('last_modified')
            meta_records[name] = rec
        zarray = json.loads(Path(meta_records['surface.level0.zarray.json']['path']).read_text())
        if zarray['shape'] != segment['surface_shape_zyx'] or zarray['chunks'] != [31, 128, 128] or zarray['dtype'] != '|u1' or zarray['compressor'] is not None or zarray['order'] != 'C' or zarray.get('dimension_separator', '.') != '/':
            raise ValueError('Source Zarr contract differs from recorded native uint8 uncompressed chunks.')
        render_path = GEOMETRY / sid / 'source-render-export-metadata.json'
        # The extracted render file is frozen by methodology; parameters must also match the earlier recorded selection.
        render = json.loads(render_path.read_text())
        if render['parameters']['target_volume'] != selection['target_volume_id'] or render['creation_info']['provenance']['parameters'] != segment['render_parameters']:
            raise ValueError('Extracted render provenance disagrees with frozen geometry metadata.')
        mask_entry = segment['outputs']['full-surface-valid-mask.npy']
        xyz_entry = segment['outputs']['full-surface-ct-coordinates.xyz.npy']
        mask_path, xyz_path = Path(mask_entry['path']), Path(xyz_entry['path'])
        verify(mask_path, mask_entry['sha256'], mask_entry['size_bytes'], 'Candidate global geometry mask')
        verify(xyz_path, xyz_entry['sha256'], xyz_entry['size_bytes'], 'Candidate global base CT coordinates')
        height, width = segment['surface_shape_zyx'][1:]
        mask, xyz = load_map(mask_path, xyz_path, (height, width))
        counts = {'total_chunk_aligned_candidates': 0, 'excluded_same_source_extent': 0, 'excluded_not_fully_valid': 0, 'geometry_eligible': 0, 'passes_distance_rule': 0}
        eligible = []
        query_mask = np.zeros(mask.shape, dtype=bool)
        for y in range(0, height - SIZE + 1, ALIGNMENT):
            for x in range(0, width - SIZE + 1, ALIGNMENT):
                bbox = [x, y, SIZE, SIZE]
                counts['total_chunk_aligned_candidates'] += 1
                record = {'segment_id': sid, 'bbox_source_xywh': bbox, 'geometry_valid_pixels': int(mask[y:y + SIZE, x:x + SIZE].sum()), 'minimum_nearest_prior_base_distance_voxels': None}
                if any(p['segment_id'] == sid and intersects(bbox, p['bbox_source_xywh']) for p in prior_sources):
                    record['status'] = 'excluded_same_source_extent'
                    counts['excluded_same_source_extent'] += 1
                elif record['geometry_valid_pixels'] != SIZE * SIZE:
                    record['status'] = 'excluded_not_fully_valid'
                    counts['excluded_not_fully_valid'] += 1
                else:
                    record['status'] = 'pending_exact_distance'
                    eligible.append(record)
                    query_mask[y:y + SIZE, x:x + SIZE] = True
                    counts['geometry_eligible'] += 1
                records.append(record)
        query_started = time.perf_counter()
        flat_positions = np.flatnonzero(query_mask)
        distance_map = np.full(mask.shape, np.nan, dtype=np.float64)
        flat_xyz = xyz.reshape(-1, 3)
        flat_distances = distance_map.reshape(-1)
        print(f'{sid}: {counts["geometry_eligible"]} eligible crops; querying {len(flat_positions):,} unique centers exactly.', flush=True)
        for start in range(0, len(flat_positions), QUERY_BATCH):
            positions = flat_positions[start:start + QUERY_BATCH]
            values, _ = tree.query(np.asarray(flat_xyz[positions], dtype=np.float64), k=1, eps=0, p=2, workers=4)
            flat_distances[positions] = values
        query_seconds = time.perf_counter() - query_started
        for record in eligible:
            x, y, _, _ = record['bbox_source_xywh']
            values = distance_map[y:y + SIZE, x:x + SIZE]
            if not np.isfinite(values).all():
                raise ValueError('Exact-query coverage is incomplete for an eligible crop.')
            minimum = float(values.min())
            record['minimum_nearest_prior_base_distance_voxels'] = minimum
            record['maximum_nearest_prior_base_distance_voxels'] = float(values.max())
            record['median_nearest_prior_base_distance_voxels'] = float(np.median(values))
            record['status'] = 'passes_distance_rule' if minimum > THRESHOLD else 'fails_distance_rule'
            counts['passes_distance_rule'] += int(minimum > THRESHOLD)
        if eligible:
            path = output / f'{sid}-queried-nearest-prior-distances.npy'
            np.save(path, distance_map, allow_pickle=False)
            counts['distance_map'] = file_record(path)
        counts.update({'segment_id': sid, 'unique_centers_queried': len(flat_positions), 'query_seconds': query_seconds})
        segment_summaries.append(counts)
        segment_data[sid] = (segment, mask, xyz, meta_records, render_path)
        print(f'{sid}: {counts["passes_distance_rule"]} pass >40 voxels; queries took {query_seconds:.2f} s.', flush=True)

    passing = [r for r in records if r['status'] == 'passes_distance_rule']
    passing.sort(key=lambda r: (-r['minimum_nearest_prior_base_distance_voxels'], r['segment_id'], r['bbox_source_xywh'][1], r['bbox_source_xywh'][0]))
    write_json(output / 'candidate-scores.json', {'methodology_sha256': args.methodology_sha256, 'ranking': method['ranking'], 'segment_summaries': segment_summaries, 'candidates': records, 'passing_candidate_count': len(passing)})
    selected_plan = None
    if passing:
        chosen = passing[0]
        sid = chosen['segment_id']
        segment, mask, xyz, metadata, render_path = segment_data[sid]
        x, y, _, _ = chosen['bbox_source_xywh']
        crop_xyz = np.ascontiguousarray(xyz[y:y + SIZE, x:x + SIZE])
        crop_mask = np.ascontiguousarray(mask[y:y + SIZE, x:x + SIZE])
        np.save(output / 'selected-geometry-mask.npy', crop_mask, allow_pickle=False)
        np.save(output / 'selected-ct-coordinates.xyz.npy', crop_xyz, allow_pickle=False)
        # Independent exact query of the chosen crop records its nearest witness and complete distance distribution.
        distances, nearest_indices = tree.query(np.asarray(crop_xyz.reshape(-1, 3), dtype=np.float64), k=1, eps=0, p=2, workers=4)
        if float(distances.min()) != chosen['minimum_nearest_prior_base_distance_voxels']:
            raise ValueError('Selected-crop exact-query recheck disagrees with its candidate score.')
        np.save(output / 'selected-nearest-prior-distances.npy', distances.reshape(SIZE, SIZE), allow_pickle=False)
        minimum_flat_index = int(distances.argmin())
        witness = {'candidate_source_xy': [x + minimum_flat_index % SIZE, y + minimum_flat_index // SIZE], 'candidate_base_ct_xyz_voxels': crop_xyz.reshape(-1, 3)[minimum_flat_index].astype(float).tolist(), 'prior_tree_index': int(nearest_indices[minimum_flat_index]), 'prior_base_ct_xyz_voxels': points[nearest_indices[minimum_flat_index]].tolist(), 'distance_voxels': float(distances[minimum_flat_index])}
        write_json(output / 'selected-geometry-diagnostics.json', geometry_diagnostics(crop_xyz, mask, x, y))
        chunks = [{'index_zyx': [0, cy, cx], 'url': segment['surface_url'].rstrip('/') + f'/0/0/{cy}/{cx}', 'expected_uncompressed_payload_bytes': 507904} for cy in range(y // 128, (y + SIZE) // 128) for cx in range(x // 128, (x + SIZE) // 128)]
        selected_plan = {
            'schema_version': 1, 'kind': 'geometry-selected-acquisition-plan', 'created_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'selected-awaiting-independent-review-and-acquisition',
            'sample_id': selection['sample_id'], 'target_volume_id': selection['target_volume_id'], 'segment_id': sid, 'segment_long_id': segment['long_id'], 'bbox_source_xywh': [x, y, SIZE, SIZE], 'shape_zyx': [31, SIZE, SIZE], 'dtype': 'uint8', 'native_spacing_um': 8.64, 'nominal_canvas_mm_xy': [3.31776, 3.31776],
            'surface_url': segment['surface_url'].rstrip('/'), 'source_array_url': segment['surface_url'].rstrip('/') + '/0', 'source_array_metadata': metadata['surface.level0.zarray.json'], 'source_attributes': metadata['surface.zattrs.json'], 'mesh_metadata': metadata['mesh.meta.json'], 'mesh_url': segment['mesh_url'],
            'source_global_geometry_mask': verified[str(Path(segment['outputs']['full-surface-valid-mask.npy']['path']).resolve())], 'source_global_coordinate_map': verified[str(Path(segment['outputs']['full-surface-ct-coordinates.xyz.npy']['path']).resolve())],
            'render_provenance': {'metadata': verified[str(render_path.resolve())], 'commit': segment['render_commit'], 'parameters': segment['render_parameters']},
            'selected_geometry_mask': file_record(output / 'selected-geometry-mask.npy'), 'selected_base_ct_coordinates': file_record(output / 'selected-ct-coordinates.xyz.npy'), 'selected_nearest_prior_distances': file_record(output / 'selected-nearest-prior-distances.npy'), 'geometry_diagnostics': file_record(output / 'selected-geometry-diagnostics.json'),
            'selection': {'methodology': file_record(methodology_path), 'selector': file_record(Path(__file__)), 'candidate_scores': file_record(output / 'candidate-scores.json'), 'prior_inventory': file_record(output / 'prior-acquisition-inventory.json'), 'criterion': chosen, 'distance_summary_voxels': summary(distances), 'nearest_witness': witness, 'prior_tree_point_count': len(points), 'query': {'algorithm': 'scipy.spatial.cKDTree', 'k': 1, 'eps': 0, 'p': 2, 'workers': 4, 'prior_subsampling': False, 'candidate_subsampling': False}},
            'verified_inputs': list(verified.values()),
            'acquisition': {'source_chunks': chunks, 'source_chunk_count': len(chunks), 'maximum_source_payload_bytes': len(chunks) * 507904, 'maximum_all_remote_response_bytes_including_metadata': 6 * 1024 * 1024, 'automatic_retries': 0, 'metadata_revalidation_required_before_chunk_reads': True, 'remote_reads_performed_here': 0},
            'predeclared_model_controls': method['predeclared_model_controls'],
            'limitations': method['limitations'], 'independent_physical_surface_established': False, 'through_depth_support_disjointness_established': False, 'ink_or_text_established': False
        }
        write_json(output / 'selected-plan.json', selected_plan)
    for record in verified.values():
        if sha256(record['path']) != record['sha256']:
            raise ValueError('Retained geometry/metadata input changed during selection.')
    if sha256(methodology_path) != args.methodology_sha256 or sha256(Path(__file__)) != method['selector_sha256']:
        raise ValueError('Frozen selector/methodology changed during execution.')
    write_json(output / 'verified-inputs.json', list(verified.values()))
    outcome = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'selected' if selected_plan else 'no-candidate-stop', 'methodology_sha256': args.methodology_sha256, 'selector_sha256': method['selector_sha256'], 'prior_tree_points': len(points), 'tree_build_seconds': tree_seconds, 'segment_summaries': segment_summaries, 'total_grid_candidates': len(records), 'geometry_eligible_candidates': sum(s['geometry_eligible'] for s in segment_summaries), 'passing_candidate_count': len(passing), 'selected': passing[0] if passing else None, 'selected_plan_path': str(output / 'selected-plan.json') if selected_plan else None, 'verified_input_count': len(verified), 'verified_inputs_unchanged_at_end': True, 'wall_seconds': time.perf_counter() - started, 'runtime_versions': {name: importlib.metadata.version(name) for name in ('numpy', 'scipy')}, 'network_calls': 0, 'intensity_arrays_opened': 0, 'model_predictions_opened': 0, 'model_runs': 0, 'through_depth_support_disjointness_established': False, 'failure_policy': 'Stop without threshold relaxation if no candidate passes.'}
    write_json(output / 'outcome.json', outcome)
    outputs = [file_record(p) for p in sorted(output.iterdir()) if p.is_file()]
    write_json(output / 'artifact-manifest.json', {'methodology': file_record(methodology_path), 'selector': file_record(Path(__file__)), 'files': outputs})
    for p in output.iterdir():
        if p.is_file():
            p.chmod(0o444)
    print(json.dumps({k: outcome[k] for k in ('status', 'total_grid_candidates', 'geometry_eligible_candidates', 'passing_candidate_count', 'selected', 'selected_plan_path', 'verified_input_count', 'wall_seconds')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
