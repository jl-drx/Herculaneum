"""Fixed offline input-domain summaries; no labels, predictions or models."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import resource
import signal
import socket
import sys
import time
import numpy as np


def offline(*args, **kwargs):
    raise RuntimeError('Network disabled in input-domain audit')


socket.create_connection = offline
socket.socket.connect = offline
socket.socket.connect_ex = offline
HERE = Path(__file__).resolve().parent
QUANTILES = [0, 1, 5, 50, 95, 99, 100]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def record(path):
    path = Path(path).resolve(strict=True)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(path), size_bytes=path.stat().st_size, sha256=digest)


def verify(item):
    require(record(item['path']) == item, 'Frozen file identity differs: ' + Path(item['path']).name)
    return Path(item['path'])


def rss():
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def summary(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    require(np.isfinite(values).all(), 'Nonfinite source/normalized statistic input')
    if not values.size:
        return dict(count=0, zero_count=0, zero_fraction=None, fractional_count=0,
                    fractional_fraction=None, quantiles=None, mean=None, std=None,
                    median=None, mad=None)
    q = np.percentile(values, QUANTILES, method='linear')
    median = float(q[3])
    zero_count = int(np.count_nonzero(values == 0))
    fractional = int(np.count_nonzero(values != np.rint(values)))
    return dict(count=int(values.size), zero_count=zero_count, zero_fraction=zero_count / values.size,
                fractional_count=fractional, fractional_fraction=fractional / values.size,
                quantiles={str(k): float(v) for k, v in zip(QUANTILES, q)},
                mean=float(values.mean()), std=float(values.std(ddof=0)), median=median,
                mad=float(np.median(np.abs(values - median))))


def pair_summary(first, second, spacing):
    first = np.asarray(first, dtype=np.float64).reshape(-1)
    second = np.asarray(second, dtype=np.float64).reshape(-1)
    require(first.shape == second.shape and np.isfinite(first).all() and np.isfinite(second).all(),
            'Pairwise statistic support differs or is nonfinite')
    if not first.size:
        return dict(pair_count=0, mean_absolute_difference=None,
                    mean_absolute_difference_per_nominal_um=None,
                    percentile95_absolute_difference=None, pearson_correlation=None)
    difference = np.abs(first - second)
    a = first - first.mean()
    b = second - second.mean()
    aa = float(np.sum(a * a, dtype=np.float64))
    bb = float(np.sum(b * b, dtype=np.float64))
    correlation = None if aa == 0 or bb == 0 else float(
        np.clip(np.sum(a * b, dtype=np.float64) / math.sqrt(aa * bb), -1, 1))
    mean_difference = float(difference.mean())
    return dict(pair_count=int(first.size), mean_absolute_difference=mean_difference,
                mean_absolute_difference_per_nominal_um=mean_difference / spacing,
                percentile95_absolute_difference=float(np.percentile(difference, 95, method='linear')),
                pearson_correlation=correlation)


def traced_normalizer(raw, normalizer):
    """Trace the pinned default full-patch recipe, then compare actual output."""
    arr = np.asarray(raw).astype(np.float32, copy=True)
    require(arr.shape == (17, 128, 128) and np.isfinite(arr).all(), 'Full patch contract differs')
    pre_std = float(np.std(arr))
    low = float(np.percentile(arr, 1))
    high = float(np.percentile(arr, 99))
    # Report strict mathematical percentile membership in float64; separately
    # report the effective FP32 clipping bounds and ties after actual clipping.
    comparison = arr.astype(np.float64)
    below = int(np.count_nonzero(comparison < low))
    above = int(np.count_nonzero(comparison > high))
    del comparison
    np.clip(arr, low, high, out=arr)
    effective_low, effective_high = float(np.float32(low)), float(np.float32(high))
    ties_low = int(np.count_nonzero(arr == np.float32(low)))
    ties_high = int(np.count_nonzero(arr == np.float32(high)))
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    scale = 1.4826 * mad
    branch = 'scaled_mad'
    if not math.isfinite(scale) or scale < 1e-6:
        candidates = [(abs(pre_std), 'preclip_std'), (abs(high - low) / 2, 'half_percentile_span')]
        valid = [(value, name) for value, name in candidates if math.isfinite(value) and value >= 1e-6]
        scale, branch = valid[0] if valid else (1.0, 'unit_scale')
    arr -= median
    arr /= scale
    np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    actual = normalizer(raw)
    require(actual.dtype == np.float32 and np.isfinite(actual).all()
            and np.array_equal(actual, arr), 'Normalizer trace differs from actual pinned implementation')
    count = int(arr.size)
    diagnostic = dict(preclip_std_fp32=pre_std, percentile_lower=low, percentile_upper=high,
        effective_lower_fp32=effective_low, effective_upper_fp32=effective_high,
        strictly_below_percentile_count=below, strictly_above_percentile_count=above,
        strictly_below_percentile_fraction=below/count, strictly_above_percentile_fraction=above/count,
        clipped_lower_tie_count=ties_low, clipped_upper_tie_count=ties_high,
        clipped_lower_tie_fraction=ties_low/count, clipped_upper_tie_fraction=ties_high/count,
        clipped_median_fp32=median, clipped_mad_fp32=mad, denominator=scale,
        denominator_branch=branch, raw_zero_count=int(np.count_nonzero(raw == 0)),
        raw_zero_fraction=float(np.count_nonzero(raw == 0)/count),
        normalized=summary(actual), trace_matches_pinned_normalizer_exactly=True)
    return actual, diagnostic


def domain_stats(selected, mask, centers, spacing):
    volume = selected[:, mask]
    per_depth = [dict(model_plane=k, source_group_center=centers[k], raw=summary(selected[k][mask]))
                 for k in range(17)]
    paired_depth = [dict(first_model_plane=k, second_model_plane=k+1,
        first_source_group_center=centers[k], second_source_group_center=centers[k+1],
        **pair_summary(selected[k][mask], selected[k+1][mask], spacing)) for k in range(16)]
    x_pairs = mask[:, :-1] & mask[:, 1:]
    y_pairs = mask[:-1, :] & mask[1:, :]
    spatial = dict(
        X=pair_summary(selected[:, :, :-1][:, x_pairs], selected[:, :, 1:][:, x_pairs], spacing),
        Y=pair_summary(selected[:, :-1, :][:, y_pairs], selected[:, 1:, :][:, y_pairs], spacing))
    return dict(spatial_pixel_count=int(mask.sum()), raw=summary(volume), per_depth=per_depth,
        adjacent_depth_aggregate=pair_summary(selected[:-1, mask], selected[1:, mask], spacing),
        adjacent_depth=paired_depth, adjacent_XY=spatial)


def scalar_aggregate(rows, key):
    values = [row[key] for row in rows]
    return dict(count=len(values), minimum=min(values) if values else None,
                median=float(np.median(values)) if values else None,
                maximum=max(values) if values else None, mean=float(np.mean(values)) if values else None)


def audit_case(case, normalizer, limits):
    arrays = {}
    for role, item in case['arrays'].items():
        require(role not in ['labels', 'inklabels', 'scores', 'logits'], 'Forbidden array role')
        arrays[role] = np.load(verify(item), allow_pickle=False)
    surface = arrays.pop('surface')
    shape = tuple(case['shape_zyx'])
    expected_dtype = np.float32 if case['retained_dtype'] == 'float32' else np.uint8
    require(surface.shape == shape and surface.dtype == expected_dtype and np.isfinite(surface).all()
            and np.all((surface >= 0) & (surface <= 255)), 'Retained source tensor differs')
    height, width = shape[1:]
    for name, mask in arrays.items():
        require(mask.shape == (height, width) and mask.dtype in [np.dtype('bool'), np.dtype('uint8')]
                and np.all((mask == 0) | (mask == 1)), 'Support/membership mask differs: ' + name)
        arrays[name] = mask.astype(bool)
    start, stop = case['model_layer_slice']
    selected = surface[start:stop]
    require(selected.shape == (17, height, width), 'Fixed selected model depth differs')
    full = np.ones((height, width), dtype=bool)
    interior = np.zeros_like(full)
    interior[64:height-64, 64:width-64] = True
    support = arrays['surface_mask'] if 'surface_mask' in arrays else arrays['acquisition_observed_mask']
    masks = dict(whole_context=full, fixed64_interior=interior, separately_named_published_support=support)
    counts = {name: int(mask.sum()) for name, mask in arrays.items()}
    counts['whole_context_pixels'] = height*width
    counts['fixed64_interior_pixels'] = int(interior.sum())
    counts['selected17_nonzero_profile_pixels'] = int(np.any(selected != 0, axis=0).sum())
    counts['selected17_allzero_profile_pixels'] = int(np.all(selected == 0, axis=0).sum())
    if 'supervision_mask' in arrays:
        counts['supervision_validation_overlap'] = int((arrays['supervision_mask'] & arrays['validation_mask']).sum())
        counts['supervision_minus_validation'] = int((arrays['supervision_mask'] & ~arrays['validation_mask']).sum())
    domains = {name: domain_stats(selected, mask, case['selected_source_group_centers'], case['pixel_spacing_um'])
               for name, mask in masks.items()}
    tiles = []
    actual_calls = 0
    for y in range(0, height-127, 64):
        for x in range(0, width-127, 64):
            require(rss() <= limits['rss_bytes'], 'Input audit RSS cap exceeded')
            raw = selected[:, y:y+128, x:x+128]
            row = dict(bbox_roi_xywh=[x, y, 128, 128], raw_zero_count=int(np.count_nonzero(raw == 0)),
                raw_zero_fraction=float(np.count_nonzero(raw == 0)/raw.size),
                separately_named_support_pixel_count=int(support[y:y+128, x:x+128].sum()))
            if not np.any(raw):
                row['status'] = 'skipped_allzero_raw_patch'
                row['normalizer_called'] = False
            else:
                row['status'] = 'normalized_nonempty_patch'
                row['normalizer_called'] = True
                normalized, detail = traced_normalizer(raw, normalizer)
                actual_calls += 1
                row['normalizer'] = detail
                if case['physical_group'] == 'PHercParis1Fr39':
                    scaled = np.multiply(raw.astype(np.float32), np.float32(257), dtype=np.float32)
                    other = normalizer(scaled)
                    actual_calls += 1
                    require(other.dtype == np.float32 and other.shape == normalized.shape and np.isfinite(other).all(),
                            'Affine-scale normalizer output differs')
                    difference = other.astype(np.float64) - normalized.astype(np.float64)
                    absolute = np.abs(difference)
                    row['affine_scale_sanity'] = dict(factor=257, multiplication_dtype='float32',
                        compared_elements=int(normalized.size),
                        exactly_equal_elements=int(np.count_nonzero(other == normalized)),
                        maximum_absolute_difference=float(absolute.max()),
                        mean_absolute_difference=float(absolute.mean()),
                        percentile99_absolute_difference=float(np.percentile(absolute, 99, method='linear')),
                        root_mean_square_difference=float(np.sqrt(np.mean(difference*difference))),
                        no_model_or_score_claim=True)
            tiles.append(row)
    nonempty = [row['normalizer'] for row in tiles if row['normalizer_called']]
    branches = {}
    for row in nonempty:
        name = row['denominator_branch']
        branches[name] = branches.get(name, 0)+1
    aggregates = {name: scalar_aggregate(nonempty, name) for name in ['preclip_std_fp32',
        'percentile_lower', 'percentile_upper', 'clipped_median_fp32', 'clipped_mad_fp32',
        'denominator', 'raw_zero_fraction', 'strictly_below_percentile_fraction',
        'strictly_above_percentile_fraction', 'clipped_lower_tie_fraction', 'clipped_upper_tie_fraction']}
    coverage = np.zeros((height, width), dtype=np.uint8)
    for row in tiles:
        x, y, w, h = row['bbox_roi_xywh']
        coverage[y:y+h, x:x+w] += 1
    return dict(id=case['id'], role=case['role'], physical_group=case['physical_group'],
        source_contract=case, support_membership_counts=counts, domains=domains,
        tiles=tiles, tile_summary=dict(tile_count=len(tiles), normalized_tile_count=len(nonempty),
            skipped_allzero_tile_count=len(tiles)-len(nonempty), denominator_branches=branches,
            aggregates=aggregates, actual_normalizer_calls=actual_calls,
            pixel_tile_coverage_histogram={str(int(k)): int(v) for k,v in zip(*np.unique(coverage, return_counts=True))}),
        dependence='Stride64 tiles overlap by64 pixels along each adjacent axis; tile summaries are descriptive, not independent replicates.')


def training_subset(plan, results):
    subsets = []
    indexed = {case['id']: case for case in results}
    for item in plan['training_tile_subset']:
        lookup = {tuple(tile['bbox_roi_xywh']): tile for tile in indexed[item['id']]['tiles']}
        rows = []
        for tile in item['tiles']:
            original = lookup[tuple(tile['bbox_roi_xywh'])]
            require(original['normalizer_called'], 'A frozen historical training tile is all zero')
            rows.append(dict(bbox_roi_xywh=tile['bbox_roi_xywh'], scheduled_exposures=tile['scheduled_exposures'],
                             normalizer=original['normalizer']))
        require(len(rows) == 6 and sum(row['scheduled_exposures'] for row in rows) == 32,
                'Unique/repeated historical schedule counts differ')
        subsets.append(dict(id=item['id'], unique_tiles=6, scheduled_exposures=32, tiles=rows,
            dependence='Unique tiles are a subset of the full grid; repeated 32-step exposure is not additional independent support. Scalar statistics are invariant under the scheduled XY D4 permutation.'))
    return subsets


def execute(plan_path):
    started = time.perf_counter()
    before = record(plan_path)
    plan = json.loads(Path(plan_path).read_text())
    require(plan['status'] == 'READY_FOR_ROOT_EXECUTION'
            and plan['kind'] == 'Fixed-seven-context-input-domain-audit-v1', 'Audit plan is not ready')
    require(plan['limits'] == dict(wall_seconds=120, rss_bytes=512*1024**2,
        maximum_surface_array_loads=7, maximum_tiles=159, maximum_normalizer_calls=209,
        network_calls=0, model_calls=0, checkpoint_reads=0), 'Audit scope/limits differ')
    require(plan['runtime']['numpy'] == np.__version__ and plan['runtime']['python'] == sys.version.split()[0],
            'Audit runtime package version differs')
    require(plan['code']['path'] == str(Path(__file__).resolve()), 'Audit executable path differs')
    require(len(plan['cases']) == 7 and plan['statistics'] == STATISTICS, 'Fixed diagnostic specification differs')
    for item in plan['input_files']:
        verify(item)
    verify(plan['code'])
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Input audit wall cap exceeded')))
    signal.alarm(plan['limits']['wall_seconds'])
    source = verify(plan['normalizer'])
    spec = importlib.util.spec_from_file_location('fixed_input_audit_normalizer', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require('torch' not in sys.modules, 'Unexpected model/tensor import')
    results = [audit_case(case, module.normalize_robust, plan['limits']) for case in plan['cases']]
    tiles = sum(case['tile_summary']['tile_count'] for case in results)
    calls = sum(case['tile_summary']['actual_normalizer_calls'] for case in results)
    require(tiles == 159 and calls <= 209 and rss() <= plan['limits']['rss_bytes']
            and time.perf_counter()-started <= plan['limits']['wall_seconds'], 'Final execution budget differs')
    subsets = training_subset(plan, results)
    for item in plan['input_files']:
        verify(item)
    require(record(plan_path) == before and 'torch' not in sys.modules, 'Plan changed or unexpected tensor module loaded')
    signal.alarm(0)
    return dict(status='PASS', completed_at_utc=datetime.now(timezone.utc).isoformat(), plan=before,
        contexts=results, adaptation_training_tile_subset=subsets, total_tile_count=tiles,
        actual_normalizer_calls=calls, source_array_loads=7, peak_rss_bytes=rss(),
        wall_seconds=time.perf_counter()-started, verified_inputs=plan['input_files'],
        all_input_files_unchanged_before_after=True, normalizer_source=plan['normalizer'],
        network_calls=0, model_calls=0, checkpoint_reads=0, label_array_reads=0, prediction_or_logit_reads=0,
        no_fitted_preprocessing=True, no_measured_registration_claim=True,
        interpretation='Descriptive input differences do not establish cause of performance changes. Marginal similarity cannot rule out a semantic acquisition/annotation domain shift. Both Frag4 contexts are one specimen; 0139 adaptation contexts and 0814 calibration contexts also share their respective specimens, and stride-64 tiles overlap.')


STATISTICS = dict(
    source_domains=['whole_context', 'fixed64_interior', 'separately_named_published_support'],
    raw_summary=dict(quantile_percent=QUANTILES, quantile_method='linear',
        moments_dtype='float64', std_ddof=0, zero_rule='exact value == 0',
        fractional_rule='float64(value) != rint(float64(value))', mad='median(abs(value-median(value)))'),
    per_depth='All17 recorded source-group centers; forward order only',
    adjacent_pairs=dict(lag=1, axes=['X', 'Y', 'depth'],
        spatial_domain_rule='both endpoints in the fixed same-domain mask',
        depth_domain_rule='same spatial domain for both adjacent model planes',
        moments_dtype='float64', std_ddof=0, constant_correlation=None,
        difference='mean(abs(a-b)); also p95 absolute difference and mean divided by nominal spacing',
        correlation='sum((a-mean(a))*(b-mean(b))) / sqrt(sum((a-mean(a))**2)*sum((b-mean(b))**2)); clip numerical result to[-1,1]'),
    normalizer=dict(patch_shape=[17,128,128], stride_xy=64, mask=False,
        full_patch=True, skip='entire raw selected17 patch == 0',
        clip_percentiles=[1,99], strict_membership_dtype='float64 against calculated percentile bounds',
        clipped_tie_rule='clipped FP32 equals effective FP32 bound',
        trace_agreement='exact array equality to actual pinned normalizer; no additional normalizer call for trace',
        scalar_fields=['preclip std', 'clip bounds', 'clipped median', 'clipped MAD', 'denominator', 'fallback branch', 'zero and clipping fractions'],
        normalized_summary='same full summary as raw marginals'),
    affine_scale_sanity=dict(cases=['frag4-context1','frag4-context2'], factor=257,
        multiplication_dtype='float32', default_normalizer_unchanged=True,
        compare='normalize_robust(FP32raw) versus normalize_robust(FP32(257*FP32raw))',
        statistics=['exact element equality count', 'maximum and mean absolute difference', 'p99 absolute difference', 'RMS difference'],
        decision_gate=None, model_score_claim=False),
    mask_counts_only=True, inklabels_used=False, scores_used=False,
    adaptation_subset='exact12 unique schedule tiles; repeated exposures counted separately; no new normalization calls',
    independent_replication_claim=False, corrections_or_search=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    require(HERE in output.parents and not output.exists(), 'Choose a fresh output directory under input-audit')
    output.mkdir(parents=True)
    try:
        result = execute(args.plan)
    except Exception as error:
        stopped = dict(status='STOPPED', reason=str(error), error_type=type(error).__name__,
            no_partial_statistics_claim=True, network_calls=0, model_calls=0,
            label_array_reads=0, prediction_or_logit_reads=0, checkpoint_reads=0)
        (output/'input-statistics.json').write_text(json.dumps(stopped, indent=2, sort_keys=True)+'\n')
        raise
    path = output/'input-statistics.json'
    path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n')
    print(json.dumps(dict(status='PASS', result=record(path), total_tiles=result['total_tile_count'],
        actual_normalizer_calls=result['actual_normalizer_calls'], peak_rss_bytes=result['peak_rss_bytes'],
        wall_seconds=result['wall_seconds'])))


if __name__ == '__main__':
    main()
