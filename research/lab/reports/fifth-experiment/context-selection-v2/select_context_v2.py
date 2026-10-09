"""Explicit geometry-informed v2 context selector; no intensity/prediction reads.

The immutable v1 selector provides hash/JSON and geometry-diagnostic helpers.
The complete v1 stop outcome remains untouched. This selector emits a separate
acquisition plan only when the frozen partial-geometry and strict-support gates
pass. Acquisition still requires native CT mapping checks and a separate adapter.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def record(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path), size_bytes=path.stat().st_size)


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def strict_geometry_support(crop_mask):
    """Union FULL footprints of all bad scheduled tiles, without data-dependent skips."""
    if crop_mask.dtype != np.bool_ or crop_mask.shape != (768, 768):
        raise ValueError('Expected Boolean768-square geometry.')
    bad_support = np.zeros_like(crop_mask)
    bad_tiles = []
    for y in range(0, 641, 64):
        for x in range(0, 641, 64):
            if not crop_mask[y:y + 128, x:x + 128].all():
                bad_support[y:y + 128, x:x + 128] = True
                bad_tiles.append([x, y])
    interior = np.zeros_like(crop_mask)
    interior[128:640, 128:640] = True
    strict = interior & crop_mask & ~bad_support
    return strict, bad_support, bad_tiles


def rank(candidate):
    return (-candidate['strict_geometry_interior_pixels'], -candidate['geometry_valid_pixels'],
            candidate['center_distance_squared_pixels'], candidate['bbox_source_xywh'][1],
            candidate['bbox_source_xywh'][0])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--methodology', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    args = ap.parse_args()
    started = time.monotonic()
    if args.output_dir.exists():
        raise ValueError('Output exists: preserve prior results.')
    method_hash = sha(args.methodology)
    method = json.loads(args.methodology.read_text())
    if method['selector']['sha256'] != sha(__file__):
        raise ValueError('Selector differs from frozen methodology.')
    for item in method['verified_inputs']:
        if sha(item['path']) != item['sha256'] or Path(item['path']).stat().st_size != item['size_bytes']:
            raise ValueError('Frozen input identity changed.')
    helper_record = method['parent_v1_selector']
    if sha(helper_record['path']) != helper_record['sha256']:
        raise ValueError('Immutable v1 diagnostic helper identity changed.')
    spec = importlib.util.spec_from_file_location('immutable_v1_context_helpers', helper_record['path'])
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)  # its main guard does not execute selection
    if (method['crop_size_pixels'] != 768 or method['origin_grid_pixels'] != 128 or
            method['point_margin_pixels'] != 128 or method['analysis_margin_pixels'] != 128 or
            method['minimum_whole_valid_pixels'] != 560333 or
            method['minimum_strict_geometry_interior_pixels'] != 249037):
        raise ValueError('Declared v2 geometry contract changed.')
    old = json.loads(Path(method['fourth_selected_plan']['path']).read_text())
    prior = json.loads(Path(method['fourth_acquisition_provenance']['path']).read_text())
    meta = json.loads(Path(old['source_array_metadata']['path']).read_text())
    if (old['segment_id'] != '20251028220955' or old['bbox_source_xywh'] != [1024, 384, 384, 384] or
            meta['shape'] != [31, 2100, 2100] or meta['chunks'] != [31, 128, 128] or
            meta['dtype'] != '|u1' or meta['compressor'] is not None or meta['filters'] or
            meta['order'] != 'C' or meta['fill_value'] != 0 or meta['dimension_separator'] != '/'):
        raise ValueError('Source contract changed.')
    if (prior['surface_url'] != old['surface_url'] or prior['bbox_source_xywh'] != old['bbox_source_xywh'] or
            prior['segment_id'] != old['segment_id'] or prior['missing_chunks_reconstructed_as_declared_fill'] != 0):
        raise ValueError('Prior cached source is not the declared fourth field.')
    mask = np.load(old['source_global_geometry_mask']['path'], mmap_mode='r', allow_pickle=False)
    xyz = np.load(old['source_global_coordinate_map']['path'], mmap_mode='r', allow_pickle=False)
    if mask.dtype != np.bool_ or mask.shape != (2100, 2100):
        raise ValueError('Global geometry mask shape or dtype differs.')
    if xyz.dtype != np.float32 or xyz.shape != (2100, 2100, 3):
        raise ValueError('Global XYZ shape or dtype differs.')
    old_x, old_y, old_w, old_h = old['bbox_source_xywh']
    old_cx, old_cy = old_x + old_w / 2, old_y + old_h / 2
    points = method['fixed_followup_points_source_xy']
    if any(not isinstance(v, int) for xy in points for v in xy):
        raise ValueError('Frozen points must be integer source XY; no implicit rounding.')
    candidates = []
    grid_count = 0
    for y in range(0, 2100 - 768 + 1, 128):
        for x in range(0, 2100 - 768 + 1, 128):
            grid_count += 1
            margins = [[px - x, py - y, x + 768 - px, y + 768 - py] for px, py in points]
            if not (x <= old_x and y <= old_y and x + 768 >= old_x + old_w and
                    y + 768 >= old_y + old_h) or any(min(m) < 128 for m in margins):
                continue
            crop_mask = np.asarray(mask[y:y + 768, x:x + 768])
            if not np.isfinite(xyz[y:y + 768, x:x + 768][crop_mask]).all():
                raise ValueError('A geometry-valid selected-source coordinate is nonfinite.')
            strict, bad_support, bad_tiles = strict_geometry_support(crop_mask)
            valid_count = int(crop_mask.sum())
            strict_count = int(strict.sum())
            point_checks = []
            for px, py in points:
                checks = []
                for dx, dy in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]:
                    local_x, local_y = px + dx - x, py + dy - y
                    checks.append(dict(source_xy=[px + dx, py + dy], local_xy=[local_x, local_y],
                        strict_geometry=bool(0 <= local_x < 768 and 0 <= local_y < 768 and strict[local_y, local_x])))
                point_checks.append(dict(point_source_xy=[px, py], point_and_cardinal_neighbors=checks,
                                         all_five_inside_strict_geometry=all(c['strict_geometry'] for c in checks)))
            whole_passes = valid_count * 20 >= 768 * 768 * 19
            strict_passes = strict_count * 20 >= 512 * 512 * 19
            points_pass = all(c['all_five_inside_strict_geometry'] for c in point_checks)
            center_squared = ((x + 384 - old_cx) ** 2 + (y + 384 - old_cy) ** 2)
            candidates.append(dict(bbox_source_xywh=[x, y, 768, 768], geometry_valid_pixels=valid_count,
                geometry_invalid_pixels=768 * 768 - valid_count, geometry_valid_fraction=valid_count / (768 * 768),
                strict_geometry_interior_pixels=strict_count, strict_geometry_interior_fraction=strict_count / (512 * 512),
                all_scheduled_tile_count=121, invalid_geometry_tile_count=len(bad_tiles),
                invalid_geometry_tile_origins_local_xy=bad_tiles, bad_tile_full_support_pixels=int(bad_support.sum()),
                whole_crop_95_percent_valid_passes=whole_passes, strict_interior_95_percent_passes=strict_passes,
                points_and_cardinal_neighbors_pass=points_pass, fixed_point_checks=point_checks,
                fixed_point_edge_margins_left_top_right_bottom_pixels=margins,
                fourth_bbox_edge_margins_left_top_right_bottom_pixels=[
                    old_x - x, old_y - y, x + 768 - old_x - old_w, y + 768 - old_y - old_h],
                center_distance_squared_pixels=center_squared, center_distance_pixels=float(center_squared ** .5),
                feasible=bool(whole_passes and strict_passes and points_pass)))
    feasible = sorted([c for c in candidates if c['feasible']], key=rank)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    candidate_path = args.output_dir / 'candidate-scores.json'
    write_json(candidate_path, dict(source_grid_origin_count=grid_count,
        containment_and_point_margin_candidate_count=len(candidates), feasible_candidate_count=len(feasible),
        ranked_feasible_bbox_source_xywh=[c['bbox_source_xywh'] for c in feasible], candidates=candidates))
    base_inputs = method['verified_inputs'] + [record(args.methodology), record(__file__), record(candidate_path)]
    outcome = dict(schema_version=2, kind='geometry-informed-posthoc-context-v2-outcome',
        created_at_utc=datetime.now(timezone.utc).isoformat(), method=record(args.methodology), selector=record(__file__),
        source_grid_origin_count=grid_count, containment_and_point_margin_candidate_count=len(candidates),
        feasible_candidate_count=len(feasible), criterion_relaxed_after_evaluation=False,
        intensities_loaded=False, predictions_loaded=False, network_requests=0, acquisition_performed=False,
        v1_stop_preserved=True, v2_informed_by_v1_geometry_outcomes=True,
        held_out_validation=False, new_independent_coverage_claimed=False, selected_bbox_source_xywh=None)
    best = feasible[0] if feasible else sorted(candidates, key=rank)[0] if candidates else None
    if best is not None:
        x, y, _, _ = best['bbox_source_xywh']
        crop_mask = np.asarray(mask[y:y + 768, x:x + 768])
        crop_xyz = np.asarray(xyz[y:y + 768, x:x + 768])
        strict, bad_support, bad_tiles = strict_geometry_support(crop_mask)
        prefix = 'selected' if feasible else 'best-failed-diagnostic'
        records = {}
        for label, values in [('geometry-mask', crop_mask), ('strict-geometry-interior-mask', strict),
                              ('bad-tile-support-mask', bad_support), ('ct-coordinates.xyz', crop_xyz)]:
            path = args.output_dir / f'{prefix}-{label}.npy'
            np.save(path, values, allow_pickle=False)
            records[label] = record(path)
        diagnostic_path = args.output_dir / f'{prefix}-geometry-diagnostics.json'
        write_json(diagnostic_path, helpers.geometry_diagnostics(mask, xyz, x, y, 768, 128))
        records['geometry-diagnostics'] = record(diagnostic_path)
        outcome['geometry_artifacts'] = records
        if feasible:
            prior_chunks = {}
            for chunk in prior['source_chunks']:
                index = tuple(chunk['chunk_index_zyx'])
                if (index in prior_chunks or chunk['http_status'] != 200 or chunk['size_bytes'] != 507904 or
                        chunk['local_payload_role'] != 'observed source'):
                    raise ValueError('Prior cache is not nine unique observed full chunks.')
                prior_chunks[index] = chunk
            chunks = []
            for row in range(y // 128, y // 128 + 6):
                for col in range(x // 128, x // 128 + 6):
                    index = (0, row, col)
                    url = old['surface_url'].rstrip('/') + f'/0/0/{row}/{col}'
                    chunk = dict(chunk_index_zyx=list(index), index_zyx=list(index), url=url,
                        expected_payload_bytes=507904, expected_uncompressed_payload_bytes=507904,
                        acquisition_action='download_new_observed_or_explicit_NoSuchKey_fill')
                    if index in prior_chunks:
                        cached = prior_chunks[index]
                        local_path = Path(method['fourth_acquisition_provenance']['path']).parent / cached['local_file']
                        if cached['url'] != url or sha(local_path) != cached['sha256']:
                            raise ValueError('Reusable observed chunk identity changed.')
                        chunk.update(acquisition_action='reuse_verified_observed_source', reuse=dict(
                            path=str(local_path.resolve()), sha256=cached['sha256'], size_bytes=cached['size_bytes'],
                            original_url=cached['url'], original_http_status=cached['http_status'],
                            original_etag=cached['etag'], original_last_modified=cached['last_modified'],
                            original_retrieved_at_utc=cached['retrieved_at_utc'],
                            provenance=method['fourth_acquisition_provenance']))
                    chunks.append(chunk)
            if len(chunks) != 36 or sum(c['acquisition_action'] == 'reuse_verified_observed_source' for c in chunks) != 9:
                raise ValueError('Containing fourth chunk-aligned bbox must reuse exactly nine of36chunks.')
            selection = dict(methodology=record(args.methodology), selector=record(__file__),
                candidate_scores=record(candidate_path), criterion=best,
                fourth_selected_plan=method['fourth_selected_plan'],
                fourth_acquisition_provenance=method['fourth_acquisition_provenance'],
                parent_v1_stop=method['parent_v1_stop'], geometry_informed_revision=True,
                posthoc_fixed_points_and_neighborhood=True, strict_support_is_geometry_only=True)
            plan = dict(schema_version=2, kind='geometry-informed-posthoc-context-acquisition-plan',
                status='Feasible geometry plan; acquisition conditional on native CT mapping and frozen768partial-geometry/cache-aware adapter',
                sample_id=old['sample_id'], target_volume_id=old['target_volume_id'], segment_id=old['segment_id'],
                segment_long_id=old['segment_long_id'], bbox_source_xywh=best['bbox_source_xywh'],
                shape_zyx=[31, 768, 768], dtype='uint8', native_spacing_um=8.64, nominal_canvas_mm_xy=[6.63552] * 2,
                surface_url=old['surface_url'], source_array_url=old['source_array_url'],
                array_metadata_path=old['source_array_metadata']['path'], attrs_path=old['source_attributes']['path'],
                geometry_mask_path=old['source_global_geometry_mask']['path'],
                coordinate_map_path=old['source_global_coordinate_map']['path'],
                reuse_source_provenance_path=method['fourth_acquisition_provenance']['path'],
                reuse_source_provenance_sha256=method['fourth_acquisition_provenance']['sha256'],
                render_provenance=old['render_provenance'], mesh_url=old['mesh_url'],
                selected_geometry_mask=records['geometry-mask'], selected_base_ct_coordinates=records['ct-coordinates.xyz'],
                strict_geometry_interior_mask=records['strict-geometry-interior-mask'],
                bad_tile_support_mask=records['bad-tile-support-mask'], geometry_diagnostics=records['geometry-diagnostics'],
                source_chunks=chunks, source_chunk_count=36, reused_observed_source_chunk_count=9, new_source_chunk_count=27,
                maximum_chunk_payload_bytes=18284544, maximum_new_chunk_payload_bytes=13713408,
                maximum_all_remote_response_bytes_including_metadata=13844480, maximum_remote_request_count=29,
                automatic_retries=0, metadata_revalidation_required=True,
                verified_inputs=base_inputs + list(records.values()), selection=selection,
                geometry_policy=dict(full_crop_geometry_valid_required=False, minimum_whole_valid_fraction=.95,
                    minimum_strict_geometry_interior_fraction=.95, tile_size_pixels=128, tile_stride_pixels=64,
                    all121tiles_treated_as_potentially_scored=True, analysis_margin_pixels=128,
                    both_fixed_points_and_cardinal_neighbors_required_inside_strict_geometry=True,
                    any_missing_or_unobserved_selected_depth_support_must_be_excluded_from_actual_strict_analysis=True),
                physical_independence='Not established; this is a post hoc context extension containing the previously observed fourth field.',
                through_depth_sampling_support_independence_established=False,
                full_renderer_sampling_support_reconstructed=False, held_out_validation=False,
                native_ct_mapping_gate_passed=None, acquisition_adapter_sha256=None,
                original_fourth_acquisition_adapter_can_execute=False,
                ink_or_text_established=False, transcription=None, translation=None)
            selected_path = args.output_dir / 'selected-plan.json'
            write_json(selected_path, plan)
            acquisition_plan = dict(plan)
            acquisition_plan.update(selected_plan_path=str(selected_path.resolve()),
                selected_plan_sha256=sha(selected_path), verified_inputs=plan['verified_inputs'] + [record(selected_path)])
            acquisition_path = args.output_dir / 'acquisition-plan.json'
            write_json(acquisition_path, acquisition_plan)
            outcome.update(status='Feasible geometry-informed post hoc v2context; no acquisition performed',
                selected_bbox_source_xywh=best['bbox_source_xywh'], selected_criterion=best,
                selected_plan=record(selected_path), acquisition_plan=record(acquisition_path))
        else:
            outcome.update(status='STOP v2: no candidate satisfies all frozen geometry/support/point gates',
                best_failed_candidate_diagnostic_only=best, acquisition_plan=None)
    else:
        outcome.update(status='STOP v2: no spatially eligible candidates', acquisition_plan=None)
    for item in method['verified_inputs']:
        if sha(item['path']) != item['sha256']:
            raise ValueError('Frozen input changed during evaluation.')
    if sha(args.methodology) != method_hash:
        raise ValueError('Frozen methodology changed during evaluation.')
    outcome['verified_inputs'] = base_inputs
    outcome['elapsed_seconds'] = time.monotonic() - started
    write_json(args.output_dir / 'coverage-outcome.json', outcome)
    print(json.dumps({k: outcome[k] for k in ['status', 'source_grid_origin_count',
        'containment_and_point_margin_candidate_count', 'feasible_candidate_count',
        'selected_bbox_source_xywh', 'elapsed_seconds']}, indent=2))
    for path in args.output_dir.iterdir():
        path.chmod(0o444)


if __name__ == '__main__':
    main()
