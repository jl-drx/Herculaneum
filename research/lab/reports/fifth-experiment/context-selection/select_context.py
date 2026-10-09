"""Geometry-only post hoc context selector; never load intensities or predictions.

Run with the separately frozen methodology.json. Existing files are immutable
inputs; the output directory must not exist. A failure produces a coverage
outcome, never an acquisition plan with relaxed geometry requirements.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.ndimage import distance_transform_edt


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def record(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path), size_bytes=path.stat().st_size)


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def summary(values):
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    return dict(count=int(values.size), finite_count=int(finite.size),
                min=float(finite.min()) if finite.size else None,
                max=float(finite.max()) if finite.size else None,
                quantiles_0_1_50_95_99_100=(np.quantile(finite, [0, .01, .5, .95, .99, 1]).tolist()
                                           if finite.size else None))


def geometry_diagnostics(mask, xyz, x, y, size, margin):
    roi = np.asarray(xyz[y:y + size, x:x + size], dtype=np.float64)
    valid = np.asarray(mask[y:y + size, x:x + size])
    # Zero-pad so the finite source canvas boundary is also treated as invalid.
    boundary = distance_transform_edt(np.pad(mask, 1, constant_values=False))[1:-1, 1:-1]
    clearance = boundary[y:y + size, x:x + size]
    horizontal = np.linalg.norm(roi[:, 1:] - roi[:, :-1], axis=2)
    vertical = np.linalg.norm(roi[1:] - roi[:-1], axis=2)
    horizontal = horizontal[valid[:, 1:] & valid[:, :-1]]
    vertical = vertical[valid[1:] & valid[:-1]]
    dx = (roi[1:-1, 2:] - roi[1:-1, :-2]) * .5
    dy = (roi[2:, 1:-1] - roi[:-2, 1:-1]) * .5
    stencil_valid = (valid[1:-1, 1:-1] & valid[1:-1, 2:] & valid[1:-1, :-2] &
                     valid[2:, 1:-1] & valid[:-2, 1:-1])
    area = np.linalg.norm(np.cross(dx, dy), axis=2)[stencil_valid]
    return dict(coordinates_finite_on_valid_pixels=bool(np.isfinite(roi[valid]).all()),
                horizontal_neighbor_jump_voxels=summary(horizontal),
                vertical_neighbor_jump_voxels=summary(vertical),
                zero_horizontal_neighbor_jumps=int(np.count_nonzero(horizontal == 0)),
                zero_vertical_neighbor_jumps=int(np.count_nonzero(vertical == 0)),
                central_difference_cross_norm_voxels_squared=summary(area),
                finite_difference_cross_norm_at_most_1e_minus_8=int(np.count_nonzero(area <= 1e-8)),
                full_crop_valid_pixel_global_boundary_distance_pixels=summary(clearance[valid]),
                interior_valid_pixel_global_boundary_distance_pixels=summary(
                    clearance[margin:-margin, margin:-margin][valid[margin:-margin, margin:-margin]]),
                archive_normals_reconstructed=False,
                full_renderer_sampling_support_reconstructed=False,
                interpretation='Neighbor and finite-difference diagnostics only; no placement, support independence, ink or text authentication.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--methodology', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    if args.output_dir.exists():
        raise ValueError('Preserve prior results: output directory already exists.')
    method_hash = sha(args.methodology)
    method = json.loads(args.methodology.read_text())
    if method['selector']['sha256'] != sha(__file__):
        raise ValueError('Selector differs from the frozen methodology.')
    for item in method['verified_inputs']:
        if sha(item['path']) != item['sha256'] or Path(item['path']).stat().st_size != item['size_bytes']:
            raise ValueError('Frozen input hash or byte size differs.')
    if (method['crop_size_pixels'] != 768 or method['origin_grid_pixels'] != 128 or
            method['point_margin_pixels'] != 128 or method['analysis_margin_pixels'] != 128 or
            method['require_full_crop_geometry_valid'] is not True):
        raise ValueError('This selector supports only the declared 768-square rule.')
    old = json.loads(Path(method['fourth_selected_plan']['path']).read_text())
    prior = json.loads(Path(method['fourth_acquisition_provenance']['path']).read_text())
    meta = json.loads(Path(old['source_array_metadata']['path']).read_text())
    if (old['segment_id'] != '20251028220955' or old['bbox_source_xywh'] != [1024, 384, 384, 384] or
            meta['shape'] != [31, 2100, 2100] or meta['chunks'] != [31, 128, 128] or
            meta['dtype'] != '|u1' or meta['compressor'] is not None or meta['filters'] or
            meta['order'] != 'C' or meta['fill_value'] != 0 or meta['dimension_separator'] != '/'):
        raise ValueError('Pinned source contract changed.')
    if (prior['surface_url'] != old['surface_url'] or prior['bbox_source_xywh'] != old['bbox_source_xywh'] or
            prior['segment_id'] != old['segment_id'] or prior['missing_chunks_reconstructed_as_declared_fill'] != 0):
        raise ValueError('Prior observed source is not the declared fourth field.')
    mask = np.load(old['source_global_geometry_mask']['path'], mmap_mode='r', allow_pickle=False)
    xyz = np.load(old['source_global_coordinate_map']['path'], mmap_mode='r', allow_pickle=False)
    if mask.dtype != np.bool_ or mask.shape != (2100, 2100):
        raise ValueError('Global Boolean geometry mask shape differs.')
    if xyz.dtype != np.float32 or xyz.shape != (2100, 2100, 3):
        raise ValueError('Global float32 XYZ geometry map shape differs.')
    size, grid, margin = 768, 128, 128
    old_x, old_y, old_w, old_h = old['bbox_source_xywh']
    old_cx, old_cy = old_x + old_w / 2, old_y + old_h / 2
    points = method['fixed_followup_points_source_xy']
    candidates = []
    grid_count = 0
    for y in range(0, mask.shape[0] - size + 1, grid):
        for x in range(0, mask.shape[1] - size + 1, grid):
            grid_count += 1
            contains_old = (x <= old_x and y <= old_y and x + size >= old_x + old_w and
                            y + size >= old_y + old_h)
            point_margins = [[px - x, py - y, x + size - px, y + size - py] for px, py in points]
            if not contains_old or any(min(margins) < margin for margins in point_margins):
                continue
            roi_mask = mask[y:y + size, x:x + size]
            valid_count = int(roi_mask.sum())
            interior_valid = int(roi_mask[margin:-margin, margin:-margin].sum())
            fully_valid = valid_count == size * size
            if fully_valid and not np.isfinite(xyz[y:y + size, x:x + size]).all():
                raise ValueError('Valid geometry mask contains nonfinite coordinates.')
            center_distance_squared = ((x + size / 2 - old_cx) ** 2 + (y + size / 2 - old_cy) ** 2)
            candidates.append(dict(bbox_source_xywh=[x, y, size, size], geometry_valid_pixels=valid_count,
                geometry_invalid_pixels=size * size - valid_count,
                valid_interior_128_margin_pixels=interior_valid,
                full_geometry_valid=fully_valid, center_distance_squared_pixels=center_distance_squared,
                center_distance_pixels=float(center_distance_squared ** .5),
                fixed_point_edge_margins_left_top_right_bottom_pixels=point_margins,
                fourth_bbox_edge_margins_left_top_right_bottom_pixels=[
                    old_x - x, old_y - y, x + size - old_x - old_w, y + size - old_y - old_h]))
    eligible = [c for c in candidates if c['full_geometry_valid']]
    eligible.sort(key=lambda c: (-c['valid_interior_128_margin_pixels'],
                                c['center_distance_squared_pixels'], c['bbox_source_xywh'][1],
                                c['bbox_source_xywh'][0]))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    candidate_path = args.output_dir / 'candidate-scores.json'
    write_json(candidate_path, dict(source_grid_origin_count=grid_count,
        containment_and_point_margin_candidate_count=len(candidates),
        fully_valid_candidate_count=len(eligible), candidates=candidates,
        eligible_ranked_bbox_source_xywh=[c['bbox_source_xywh'] for c in eligible]))
    base_inputs = method['verified_inputs'] + [record(args.methodology), record(__file__), record(candidate_path)]
    outcome = dict(schema_version=1, kind='posthoc-context-geometry-coverage-outcome',
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        methodology_path=str(args.methodology.resolve()), methodology_sha256=method_hash,
        selector_sha256=sha(__file__), segment_id=old['segment_id'],
        prior_bbox_source_xywh=old['bbox_source_xywh'], requested_crop_size_pixels=size,
        source_grid_origin_count=grid_count, containment_and_point_margin_candidate_count=len(candidates),
        fully_valid_candidate_count=len(eligible), acquisition_performed=False,
        intensities_loaded=False, predictions_loaded=False, network_requests=0,
        selection_basis='Post hoc surrounding context of the fourth field; eligibility/ranking use geometry only.',
        held_out_validation=False, new_independent_coverage_claimed=False)
    if not eligible:
        # This diagnostic ranking was predeclared and cannot produce an acquisition plan.
        best = sorted(candidates, key=lambda c: (-c['geometry_valid_pixels'],
            -c['valid_interior_128_margin_pixels'], c['center_distance_squared_pixels'],
            c['bbox_source_xywh'][1], c['bbox_source_xywh'][0]))[0] if candidates else None
        outcome.update(status='STOP: no fully geometry-valid crop satisfies the frozen rule',
                       selected_bbox_source_xywh=None, acquisition_plan_path=None,
                       best_coverage_candidate_diagnostic_only=best, criterion_relaxed=False)
        if best is not None:
            x, y, _, _ = best['bbox_source_xywh']
            diagnostic_path = args.output_dir / 'best-coverage-geometry-diagnostics.json'
            write_json(diagnostic_path, geometry_diagnostics(mask, xyz, x, y, size, margin))
            outcome['best_coverage_geometry_diagnostics'] = record(diagnostic_path)
    else:
        winner = eligible[0]
        x, y, _, _ = winner['bbox_source_xywh']
        crop_mask_path = args.output_dir / 'selected-geometry-mask.npy'
        crop_xyz_path = args.output_dir / 'selected-ct-coordinates.xyz.npy'
        np.save(crop_mask_path, np.asarray(mask[y:y + size, x:x + size]), allow_pickle=False)
        np.save(crop_xyz_path, np.asarray(xyz[y:y + size, x:x + size]), allow_pickle=False)
        diagnostic_path = args.output_dir / 'selected-geometry-diagnostics.json'
        write_json(diagnostic_path, geometry_diagnostics(mask, xyz, x, y, size, margin))
        prior_chunks = {}
        for chunk in prior['source_chunks']:
            index = tuple(chunk['chunk_index_zyx'])
            if (index in prior_chunks or chunk['http_status'] != 200 or
                    chunk['local_payload_role'] != 'observed source' or chunk['size_bytes'] != 507904):
                raise ValueError('Prior cache source record is not one unique observed full chunk.')
            prior_chunks[index] = chunk
        chunks = []
        for row in range(y // grid, y // grid + size // grid):
            for col in range(x // grid, x // grid + size // grid):
                index = (0, row, col)
                url = old['surface_url'].rstrip('/') + f'/0/0/{row}/{col}'
                chunk = dict(index_zyx=list(index), chunk_index_zyx=list(index), url=url,
                    expected_uncompressed_payload_bytes=507904, expected_payload_bytes=507904,
                    acquisition_action='download_new_observed_or_explicit_NoSuchKey_fill')
                if index in prior_chunks:
                    cached = prior_chunks[index]
                    local_path = Path(method['fourth_acquisition_provenance']['path']).parent / cached['local_file']
                    if cached['url'] != url or sha(local_path) != cached['sha256']:
                        raise ValueError('Prior cached chunk identity or hash differs.')
                    chunk.update(acquisition_action='reuse_verified_observed_source',
                        reuse=dict(path=str(local_path.resolve()), sha256=cached['sha256'],
                            size_bytes=cached['size_bytes'], original_http_status=cached['http_status'],
                            original_url=cached['url'], original_etag=cached['etag'],
                            original_last_modified=cached['last_modified'],
                            original_retrieved_at_utc=cached['retrieved_at_utc'],
                            provenance=method['fourth_acquisition_provenance']))
                chunks.append(chunk)
        reuse_count = sum(c['acquisition_action'] == 'reuse_verified_observed_source' for c in chunks)
        if len(chunks) != 36 or reuse_count != 9:
            raise ValueError('Containment must produce 36 chunks with all nine prior chunks reusable.')
        selected_records = [record(crop_mask_path), record(crop_xyz_path), record(diagnostic_path)]
        plan = dict(schema_version=1, kind='posthoc-context-acquisition-plan',
            status='Geometry selected; await native CT mapping gate and frozen 768/cache-aware acquisition adapter',
            sample_id=old['sample_id'], target_volume_id=old['target_volume_id'],
            segment_id=old['segment_id'], segment_long_id=old['segment_long_id'],
            bbox_source_xywh=winner['bbox_source_xywh'], shape_zyx=[31, size, size], dtype='uint8',
            native_spacing_um=old['native_spacing_um'], nominal_canvas_mm_xy=[size * .00864] * 2,
            surface_url=old['surface_url'], source_array_url=old['source_array_url'],
            array_metadata_path=old['source_array_metadata']['path'], attrs_path=old['source_attributes']['path'],
            geometry_mask_path=old['source_global_geometry_mask']['path'],
            coordinate_map_path=old['source_global_coordinate_map']['path'],
            render_provenance=old['render_provenance'], mesh_url=old['mesh_url'],
            selected_geometry_mask=selected_records[0], selected_base_ct_coordinates=selected_records[1],
            geometry_diagnostics=selected_records[2],
            maximum_chunk_payload_bytes=36 * 507904,
            maximum_new_chunk_payload_bytes=27 * 507904,
            source_chunk_count=36, reused_observed_source_chunk_count=9, new_source_chunk_count=27,
            source_chunks=chunks, automatic_retries=0,
            maximum_all_remote_response_bytes_including_metadata=method['maximum_all_remote_response_bytes_including_metadata'],
            verified_inputs=base_inputs + selected_records,
            selection=dict(methodology=record(args.methodology), selector=record(__file__),
                           candidate_scores=record(candidate_path), criterion=winner,
                           fourth_selected_plan=method['fourth_selected_plan'],
                           fourth_acquisition_provenance=method['fourth_acquisition_provenance'],
                           analysis_margin_pixels=margin, posthoc=True),
            physical_independence='Not established; context includes the entire previously acquired fourth field.',
            full_sampling_support_disjointness_established=False, held_out_validation=False,
            ink_or_text_established=False, transcription=None, translation=None,
            acquisition_adapter_compatibility=dict(flat_plan_fields_match_fourth_acquire_screen=True,
                original_fourth_acquire_screen_can_execute=False,
                reason='Original adapter hard-codes384-square,9chunks and6MiB; freeze a separate768/cache-aware adapter before acquisition.'),
            native_ct_mapping_gate_passed=None, acquisition_adapter_sha256=None)
        plan_path = args.output_dir / 'acquisition-plan.json'
        write_json(plan_path, plan)
        outcome.update(status='Selected full-valid post hoc context; no acquisition performed',
            selected_bbox_source_xywh=winner['bbox_source_xywh'], selected_criterion=winner,
            acquisition_plan=record(plan_path), source_chunks=36, verified_reuse_chunks=9, new_source_chunks=27,
            total_chunk_payload_bytes=36 * 507904, maximum_new_chunk_payload_bytes=27 * 507904)
    for item in method['verified_inputs']:
        if sha(item['path']) != item['sha256']:
            raise ValueError('Frozen input changed during selection.')
    if sha(args.methodology) != method_hash:
        raise ValueError('Methodology changed during selection.')
    outcome['elapsed_seconds'] = time.monotonic() - started
    outcome['verified_inputs'] = base_inputs
    write_json(args.output_dir / 'coverage-outcome.json', outcome)
    print(json.dumps({k: outcome[k] for k in ['status', 'source_grid_origin_count',
        'containment_and_point_margin_candidate_count', 'fully_valid_candidate_count',
        'selected_bbox_source_xywh', 'elapsed_seconds']}, indent=2))
    for output in args.output_dir.iterdir():
        output.chmod(0o444)


if __name__ == '__main__':
    main()
