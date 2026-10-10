#!/usr/bin/env python3
"""Analyze frozen larger-surface diagnostics; scores never supply text labels."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

import numpy as np
from scipy.ndimage import distance_transform_edt


ROOT = Path('/workspace/herculaneum-lab')
DATA = ROOT / 'data/experiments/pherc0800-larger-controls1024'
PARENT = ROOT / 'data/experiments/pherc0800-larger1536'
REPORT = ROOT / 'reports/third-experiment'
PLAN = REPORT / 'fixed-controls-plan.json'
EXPECTED_PLAN_SHA = '39c5a2f5cda2051a6c4d173f182f7d62607dd1249739ee3bf50311df05142b75'
DEST = REPORT / 'fixed-control-analysis'


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def statistics(values):
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return {'count': 0}
    return {'count': int(values.size), 'min': float(values.min()), 'max': float(values.max()), 'mean': float(values.mean()), 'std': float(values.std()), 'quantiles_1_50_95_99': [float(v) for v in np.percentile(values, [1, 50, 95, 99])], 'above_0_5_count': int((values > 0.5).sum()), 'above_0_5_fraction': float((values > 0.5).mean())}


def comparison(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if not a.size:
        return {'count': 0}
    difference = b - a
    aa, bb = a > 0.5, b > 0.5
    union = int((aa | bb).sum())
    return {'count': int(a.size), 'exactly_equal_count': int((a == b).sum()), 'unequal_count': int((a != b).sum()), 'absolute_change': statistics(np.abs(difference)), 'signed_change_mean': float(difference.mean()), 'pearson_correlation': float(np.corrcoef(a, b)[0, 1]) if a.std() and b.std() else None, 'both_above_0_5': int((aa & bb).sum()), 'either_above_0_5': union, 'midpoint_jaccard': float((aa & bb).sum() / union) if union else None, 'midpoint_class_changes': int((aa != bb).sum())}


def read_run(directory, label):
    meta_path = directory / 'metadata.json'
    meta = json.loads(meta_path.read_text())
    values_path = directory / f'{label}-probabilities.npy'
    evaluated_path = directory / f'{label}-evaluated-mask.npy'
    for key, path in ((label, values_path), (f'{label}-evaluated-mask', evaluated_path)):
        if sha256(path) != meta['outputs'][key]['sha256']:
            raise ValueError(f'Output hash mismatch: {path}')
    values = np.load(values_path, mmap_mode='r', allow_pickle=False)
    evaluated = np.load(evaluated_path, mmap_mode='r', allow_pickle=False)
    if values.dtype != np.float32 or evaluated.dtype != np.bool_ or values.shape != evaluated.shape:
        raise ValueError('Unexpected score/evaluated array contract.')
    model = next(m for m in meta['models'] if m['label'] == label)
    return values, evaluated, meta, model, {'metadata_path': str(meta_path), 'metadata_sha256': sha256(meta_path), 'score_path': str(values_path), 'score_sha256': sha256(values_path), 'evaluated_path': str(evaluated_path), 'evaluated_sha256': sha256(evaluated_path)}


def strict_tile_support(shape, tiles, valid, observed):
    support = np.ones(shape, dtype=bool)
    bad_tiles = []
    for tile in tiles:
        if tile['status'] != 'scored':
            continue
        x, y, width, height = tile['bbox_roi_xywh']
        view = (slice(y, y + height), slice(x, x + width))
        invalid = int((~valid[view]).sum())
        missing = int((~observed[view]).sum())
        if invalid or missing:
            support[view] = False
            bad_tiles.append({'bbox_roi_xywh': [x, y, width, height], 'invalid_geometry_pixels': invalid, 'missing_selected_depth_pixels': missing})
    return support, bad_tiles


def main():
    if DEST.exists():
        raise ValueError('Analysis outputs already exist; do not overwrite them.')
    if sha256(PLAN) != EXPECTED_PLAN_SHA:
        raise ValueError('Frozen plan hash mismatch.')
    plan = json.loads(PLAN.read_text())
    for item in plan['source_inputs'].values():
        if sha256(Path(item['path'])) != item['sha256']:
            raise ValueError('Frozen control input changed.')
    surface = np.load(DATA / 'surface.npy', mmap_mode='r', allow_pickle=False)
    valid = np.load(DATA / 'geometry-mask.npy', mmap_mode='r', allow_pickle=False)
    parent_prov = json.loads((PARENT / 'provenance.json').read_text())
    for name, entry in parent_prov['derived_outputs'].items():
        if name in ('surface.npy', 'geometry-mask.npy') and sha256(PARENT / name) != entry['sha256']:
            raise ValueError('Retained primary input changed.')
    parent_surface = np.load(PARENT / 'surface.npy', mmap_mode='r', allow_pickle=False)
    parent_valid = np.load(PARENT / 'geometry-mask.npy', mmap_mode='r', allow_pickle=False)
    global_entry = next(e for e in parent_prov['verified_inputs'] if e['path'].endswith('full-surface-valid-mask.npy'))
    global_path = Path(global_entry['path'])
    if sha256(global_path) != global_entry['sha256']:
        raise ValueError('Original global geometry hash mismatch.')
    global_valid = np.load(global_path, mmap_mode='r', allow_pickle=False)
    # Padding makes the outside of the original source canvas explicit invalid geometry.
    global_distance = distance_transform_edt(np.pad(global_valid, 1, constant_values=False))[1:-1, 1:-1]
    geometry_distance = global_distance[768:1792, 768:1792]
    if not np.array_equal(global_valid[768:1792, 768:1792], valid):
        raise ValueError('Control mask does not align to its declared native source origin.')
    crop_interior = np.zeros(valid.shape, dtype=bool)
    crop_interior[64:-64, 64:-64] = True
    observed = {start: np.any(surface[start:start + 17] != 0, axis=0) for start in (5, 7, 9)}
    common = valid & crop_interior & (geometry_distance >= 64)
    for present in observed.values():
        common &= present
    scores, evaluated, records, models, bad_tile_records = {}, {}, {}, {}, {}
    strict_common = common.copy()
    primary_dir = REPORT / 'ink9um-large-forward'
    parent_observed = np.any(parent_surface[7:24] != 0, axis=0)
    for seed in ('seed42', 'seed43'):
        values, ev, meta, model, record = read_run(primary_dir, seed)
        key = f'primary-{seed}'
        scores[key] = values[256:1280, 256:1280]
        evaluated[key] = ev[256:1280, 256:1280]
        records[key], models[key] = record, model
        support, bad_tiles = strict_tile_support(values.shape, model['tiles'], parent_valid, parent_observed)
        strict_common &= support[256:1280, 256:1280]
        bad_tile_records[key] = bad_tiles
    for condition in plan['runs']:
        key = condition['condition']
        values, ev, meta, model, record = read_run(Path(condition['output_dir']), 'seed42')
        start = condition['layer_start_stop'][0]
        if meta['processing']['source_layer_indices'] != condition['source_layer_indices']:
            raise ValueError(f'Control depth/direction mismatch: {key}')
        scores[key], evaluated[key], records[key], models[key] = values, ev, record, model
        support, bad_tiles = strict_tile_support(values.shape, model['tiles'], valid, observed[start])
        strict_common &= support
        bad_tile_records[key] = bad_tiles
    for key, ev in evaluated.items():
        common &= ev & np.isfinite(scores[key])
    strict_common &= common
    equality_domain = valid & crop_interior & observed[7] & evaluated['primary-seed42'] & evaluated['center-forward']
    pairs = [('primary-seed42', 'center-forward'), ('primary-seed42', 'primary-seed43'), ('center-forward', 'minus2-forward'), ('center-forward', 'plus2-forward'), ('center-forward', 'center-reverse')]
    result = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'analysis_code': {'path': str(Path(__file__)), 'sha256': sha256(Path(__file__))},
        'frozen_plan': {'path': str(PLAN), 'sha256': sha256(PLAN)},
        'score_inputs': records,
        'source_native_bbox_xywh': [768, 768, 1024, 1024],
        'domain_definitions': {'primary_common': 'Geometry valid; at least 64 pixels from crop edge and invalid original-global geometry; any nonzero intensity under each selected 17-depth window; every compared model evaluated this pixel; all compared scores finite.', 'strict_common': 'Primary common domain excluding pixels covered by any scored contributing tile with invalid geometry or any XY pixel lacking observed support under that tile\'s selected-depth window.', 'observed_support': 'Any nonzero uint8 intensity across selected depths at an XY location; an operational data-present convention, not an ink label.', 'global_geometry_mask': global_entry, 'geometry_distance_metric': 'Euclidean pixel distance using scipy.ndimage.distance_transform_edt on the original mask padded with one invalid border; retain distance >= 64.', 'crop_edge_margin_pixels': 64, 'global_geometry_margin_pixels': 64, 'midpoint_threshold': 0.5, 'midpoint_interpretation': 'Fixed score-behavior diagnostic; no target accuracy or ink-confidence interpretation.'},
        'counts': {'geometry_valid': int(valid.sum()), 'crop_64_interior': int(crop_interior.sum()), 'global_geometry_64_interior': int((valid & (geometry_distance >= 64)).sum()), 'observed_pixels_by_window_start': {str(start): int(p.sum()) for start, p in observed.items()}, 'primary_common': int(common.sum()), 'strict_common': int(strict_common.sum()), 'equality_check_domain': int(equality_domain.sum())},
        'full_crop_context_consistency': comparison(scores['primary-seed42'][equality_domain], scores['center-forward'][equality_domain]),
        'domains': {},
        'run_timings': {key: {'scheduled_tiles': model['tile_count'], 'scored_tiles': model['scored_tile_count'], 'skipped_tiles': model['skipped_tile_count'], 'inference_seconds': model['inference_seconds']} for key, model in models.items()},
        'strict_support_excluded_tiles': bad_tile_records,
        'limitations': ['This is a fixed sensitivity experiment on shared source material, not independent held-out validation.', 'Neither geometry/intensity support, score magnitude, midpoint counts, nor seed agreement authenticates ink.', 'A 64-pixel margin alone does not guarantee validity of all pixels in contributing tiles.', 'Native surface coordinates have nominal 8.64-micrometer sampling; flattening distortion is not quantified.'],
    }
    for name, domain in (('primary_common', common), ('strict_common', strict_common)):
        result['domains'][name] = {'count': int(domain.sum()), 'scores': {key: statistics(values[domain]) for key, values in scores.items()}, 'comparisons': {f'{a}_vs_{b}': comparison(scores[a][domain], scores[b][domain]) for a, b in pairs}, 'both_primary_seeds_above_0_5': int(((scores['primary-seed42'] > 0.5) & (scores['primary-seed43'] > 0.5) & domain).sum())}
    DEST.mkdir(parents=True, exist_ok=False)
    np.save(DEST / 'primary-common-mask.npy', common, allow_pickle=False)
    np.save(DEST / 'strict-common-mask.npy', strict_common, allow_pickle=False)
    result['domain_masks'] = {name: {'sha256': sha256(DEST / name), 'shape_yx': [1024, 1024], 'dtype': 'bool'} for name in ('primary-common-mask.npy', 'strict-common-mask.npy')}
    (DEST / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 14))
    for ax, depth in zip(axes[0], (11, 15, 19)):
        ax.imshow(surface[depth], cmap='gray', vmin=0, vmax=155)
        ax.set_title(f'Observed CT layer {depth}; fixed display 0–155')
    keys = ['primary-seed42', 'primary-seed43', 'center-forward', 'minus2-forward', 'plus2-forward', 'center-reverse']
    for ax, key in zip(axes[1:].ravel(), keys):
        im = ax.imshow(scores[key], cmap='gray', vmin=0.25, vmax=0.75)
        ax.set_title(f'{key}\nRaw sigmoid score; display 0.25–0.75')
        if np.any(common) and not np.all(common):
            ax.contour(common.astype(np.uint8), levels=[0.5], colors=['cyan'], linewidths=0.35)
        if np.any(strict_common) and not np.all(strict_common):
            ax.contour(strict_common.astype(np.uint8), levels=[0.5], colors=['orange'], linewidths=0.35)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle('Fixed central PHerc0800 control context — no transcription claim\nNative origin (768,768); 1024² context, nominal 8.85 mm; cyan: common domain, orange: strict support', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(DEST / 'fixed-controls-overview.png', dpi=180)
    plt.close(fig)
    print(json.dumps({'analysis_dir': str(DEST), 'counts': result['counts'], 'context_consistency': result['full_crop_context_consistency'], 'primary_midpoint_overlap': result['domains']['primary_common']['both_primary_seeds_above_0_5'], 'strict_midpoint_overlap': result['domains']['strict_common']['both_primary_seeds_above_0_5']}, indent=2))


if __name__ == '__main__':
    main()
