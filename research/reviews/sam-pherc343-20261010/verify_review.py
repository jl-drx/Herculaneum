"""Independent saved-map arithmetic audit; no Torch, inference, or lab writes."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import distance_transform_edt


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


LEVELS = np.arange(256, dtype=np.int64)
ABS = np.abs(LEVELS[:, None] - LEVELS[None, :]).ravel()
PRODUCTS = (LEVELS[:, None] * LEVELS[None, :]).ravel()


def histogram_stats(h):
    n = int(h.sum())
    if n == 0:
        raise ValueError('Empty comparison support')
    cumulative = h.cumsum()
    result = {'n': n, 'mean': int(h @ LEVELS) / n,
              'max': int(np.flatnonzero(h)[-1])}
    for name, q in [('median', .5), ('p95', .95), ('p99', .99)]:
        result[name] = int(np.searchsorted(cumulative, math.ceil(n * q)))
    return result


def compare(a, b, support):
    # Integer joint counts and exact integer moments are independent of the
    # review's floating-vector np.corrcoef calculation.
    codes = a[support].astype(np.int32) * 256 + b[support]
    joint = np.bincount(codes, minlength=65536).reshape(256, 256)
    del codes
    n = int(joint.sum())
    hx, hy = joint.sum(axis=1), joint.sum(axis=0)
    sx, sy = int(hx @ LEVELS), int(hy @ LEVELS)
    sxx, syy = int(hx @ (LEVELS ** 2)), int(hy @ (LEVELS ** 2))
    sxy = int(joint.ravel() @ PRODUCTS)
    variance_product = (n * sxx - sx * sx) * (n * syy - sy * sy)
    if variance_product <= 0:
        raise ValueError('Degenerate correlation support')
    h = np.bincount(ABS, weights=joint.ravel(), minlength=256).astype(np.int64)
    result = histogram_stats(h)
    result.update(pearson=(n * sxy - sx * sy) / math.sqrt(variance_product),
                  identical_fraction=int(np.trace(joint)) / n)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lab', type=Path, required=True)
    parser.add_argument('--review', type=Path, default=Path(__file__).resolve().parent / 'original')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pipeline-pins', type=Path,
                        help='Optional frozen live-pipeline manifest; omitted for a relocated archive replay.')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a fresh verification output')
    lab = args.lab.resolve()
    if args.output.resolve().is_relative_to(lab / 'reports/fortysixth-experiment'):
        raise ValueError('Write the review outside the active study')
    phase = lab / 'reports/fortysixth-experiment'
    old = lab / 'reports/fortyfifth-experiment'
    sources = old / 'source-evidence/pherc343'
    sam = json.loads((args.review / 'metrics.json').read_text())
    checks = []

    def equal(actual, expected, label):
        if isinstance(actual, dict):
            if set(actual) != set(expected):
                raise AssertionError('Key mismatch: ' + label)
            for key in actual:
                equal(actual[key], expected[key], label + '/' + key)
        elif isinstance(actual, float):
            difference = abs(actual - expected)
            if not math.isfinite(actual) or difference > 1e-12:
                raise AssertionError(f'{label}: {actual} != {expected}')
            checks.append({'path': label, 'absolute_error': difference})
        else:
            if actual != expected:
                raise AssertionError(f'{label}: {actual} != {expected}')
            checks.append({'path': label, 'absolute_error': 0})

    valid_path = phase / 'work/valid.npy'
    if sha(valid_path) != '1905c0d05656b4472a533e00b1c140592c7b33ed7a8bb7780873517f747e41fc':
        raise AssertionError('Mask file mismatch')
    valid = np.load(valid_path, allow_pickle=False)
    if valid.dtype != np.bool_ or valid.shape != (2560, 6720):
        raise AssertionError('Invalid mask type/shape')
    area = np.s_[105:2560, 380:6021]
    if hashlib.sha256(np.packbits(valid[area]).tobytes()).hexdigest() != 'f60549218edb9871e5acb8930e176a09fe1aaafef28c10a2261014f70a58dcbc':
        raise AssertionError('Published packed mask mismatch')
    equal(int(valid.sum()), sam['valid_pixels'], 'valid_pixels')
    crop = np.zeros_like(valid); crop[area] = valid[area]
    equal(int(crop.sum()), sam['published_valid_pixels'], 'published_valid_pixels')
    stem = 'concat_w047-w048_R5B2_z9500-11000'
    reference = old / 'source-evidence' / (stem + '.png')
    sidecar = json.loads((sources / 'outputs' / (stem + '.json')).read_text())
    if sha(reference) != sidecar['outputs']['image']['sha256']:
        raise AssertionError('Reference PNG mismatch')
    panel = sidecar['outputs']['image']['data_panel']
    if [panel[k] for k in ('x', 'y', 'width', 'height')] != [173, 182, 5641, 2455]:
        raise AssertionError('Reference panel convention mismatch')
    node_path = old / 'ph343-mesh-replay' / (stem + '_node_source.npz')
    nodes = np.load(node_path, allow_pickle=False)['source']
    if tuple(x * 20 for x in nodes.shape) != valid.shape:
        raise AssertionError('Unexpected ownership extent')
    ownership = np.kron((nodes == 1) | (nodes == 4), np.ones((20, 20), dtype=bool))
    equal({str(v): int((nodes == v).sum()) * 400 for v in np.unique(nodes)},
          sam['source_counts'], 'source_counts')
    horizontal = (ownership[:, :-1] ^ ownership[:, 1:]) & valid[:, :-1] & valid[:, 1:]
    vertical = (ownership[:-1, :] ^ ownership[1:, :]) & valid[:-1, :] & valid[1:, :]
    boundary = np.zeros_like(valid)
    boundary[:, :-1] |= horizontal; boundary[:, 1:] |= horizontal
    boundary[:-1, :] |= vertical; boundary[1:, :] |= vertical
    near = distance_transform_edt(~boundary) <= 20
    extent = {'boundary_endpoint_valid_pixels_full': int(boundary.sum()),
              'boundary_endpoint_valid_fraction_full': int(boundary.sum()) / int(valid.sum()),
              'boundary_endpoint_valid_pixels_published': int((boundary & crop).sum()),
              'boundary_endpoint_valid_fraction_published': int((boundary & crop).sum()) / int(crop.sum()),
              'within20px_published_pixels': int((near & crop).sum()),
              'within20px_published_fraction': int((near & crop).sum()) / int(crop.sum())}
    for k, v in extent.items():
        equal(v, sam['ownership_boundary_extent'][k], 'boundary/' + k)
    full_edges = int(horizontal.sum() + vertical.sum())
    published_edges = int((horizontal & crop[:, :-1] & crop[:, 1:]).sum()
                          + (vertical & crop[:-1, :] & crop[1:, :]).sum())
    if (full_edges, published_edges) != (2244, 2159):
        raise AssertionError('Edge population differs from report')
    boxes = json.loads((sources / 'inputs' / (stem + '_letter_boxes.json')).read_text())['letters']
    if len(boxes) != len(sam['regions']) or len(boxes) != 18:
        raise AssertionError('Region inventory mismatch')
    regions = {}
    for box in boxes:
        x0, y0 = (math.floor(x) for x in box['box'][:2])
        x1, y1 = (math.ceil(x) for x in box['box'][2:])
        sl = np.s_[y0:y1, x0:x1]; support = valid[sl]
        r = next(r for r in sam['regions'] if r['id'] == box['id'])
        for key, actual in [('bounds_xyxy', [x0, y0, x1, y1]),
                            ('valid_pixels', int(support.sum())), ('valid_fraction', float(support.mean()))]:
            equal(actual, r[key], f'region/{box["id"]}/{key}')
        b = next(b for b in sam['ownership_boundary_extent']['regions'] if b['id'] == box['id'])
        for key, actual in [('boundary_endpoint_pixels', int(boundary[sl].sum())),
                            ('within20px_pixels', int((near[sl] & support).sum())),
                            ('within20px_valid_fraction', int((near[sl] & support).sum()) / int(support.sum()))]:
            equal(actual, b[key], f'region/{box["id"]}/boundary/{key}')
        regions[box['id']] = (sl, r)

    merged = {}; phase_summary = {}; receipt_checks = []
    for name, token in [('dnative', 'dnative'), ('seed42', 'hybrid_3d2d-seed42'), ('seed43', 'hybrid_3d2d-seed43')]:
        arrays = {}
        for tile_phase in ['plain', 'padded']:
            path = phase / 'work' / ('maps_' + tile_phase) / (stem + '_66' + ('_padded' if tile_phase == 'padded' else '') + '__' + token + '__L66s26_reverse.tif')
            receipt = json.loads(Path(str(path) + '.receipt.json').read_text())
            original = next(r for r in sam['receipts'] if r['model'] == token and r['phase'] == tile_phase)
            if receipt != original or receipt['status'] != 'PASS':
                raise AssertionError('Receipt mismatch')
            if sha(path) != receipt['output']['sha256'] or path.stat().st_size != receipt['output']['bytes']:
                raise AssertionError('Encoded map mismatch')
            a = tifffile.imread(path)
            if list(a.shape) != receipt['shape'] or a.dtype != np.uint8:
                raise AssertionError('Map shape/dtype mismatch')
            if hashlib.sha256(a.tobytes()).hexdigest() != receipt['decoded_sha256']:
                raise AssertionError('Decoded map mismatch')
            arrays[tile_phase] = a if tile_phase == 'plain' else a[40:2600, 200:6920].copy()
            receipt_checks.append({'model': name, 'phase': tile_phase, 'status': 'PASS'})
        a, b = arrays['plain'], arrays['padded']
        merged[name] = a.copy(); merged[name][ownership] = b[ownership]
        actual = {}
        for key, support in [('full_valid_phase', valid), ('published_valid_phase', crop),
                             ('near_ownership_boundary_phase', crop & near),
                             ('away_ownership_boundary_phase', crop & ~near)]:
            actual[key] = compare(a, b, support)
        gradients = {}
        for kind, image in [('plain', a), ('padded', b), ('merged', merged[name])]:
            x = np.abs(image[:, :-1].astype(np.int16) - image[:, 1:])[horizontal]
            y = np.abs(image[:-1, :].astype(np.int16) - image[1:, :])[vertical]
            gradients[kind] = histogram_stats(np.bincount(x, minlength=256) + np.bincount(y, minlength=256))
        actual['ownership_switch_gradients'] = gradients
        actual['merged_display_on_fraction'] = float((merged[name][crop] > 64).mean())
        equal(actual, sam['phase_results'][name], 'phase/' + name)
        phase_summary[name] = actual
        for region_id, (sl, r) in regions.items():
            support = valid[sl]
            result = {'phase': compare(a[sl], b[sl], support),
                      'merged_mean': float(merged[name][sl][support].mean()),
                      'merged_display_on_fraction': float((merged[name][sl][support] > 64).mean())}
            equal(result, r['models'][name], f'region/{region_id}/model/{name}')
        del arrays, a, b
    agreements = {}
    for a, b in itertools.combinations(merged, 2):
        key = a + ':' + b
        agreements[key] = compare(merged[a], merged[b], crop)
        equal(agreements[key], sam['model_agreement'][key], 'agreement/' + key)
        for region_id, (sl, r) in regions.items():
            equal(compare(merged[a][sl], merged[b][sl], valid[sl]), r['pairwise'][key],
                  f'region/{region_id}/pair/{key}')
    csv_rows = list(csv.DictReader((args.review / 'regions.csv').open()))
    expected_pairs = {(str(region_id), name) for region_id in regions for name in merged}
    if len(csv_rows) != 54 or {(r['region'], r['model']) for r in csv_rows} != expected_pairs:
        raise AssertionError('CSV region/model coverage mismatch')
    for row in csv_rows:
        region_id = int(row['region']); r = regions[region_id][1]; model = r['models'][row['model']]
        for key, expected in [('valid_pixels', r['valid_pixels']),
                              ('phase_mae_uint8', model['phase']['mean']),
                              ('phase_p95_uint8', model['phase']['p95']),
                              ('phase_correlation', model['phase']['pearson']),
                              ('merged_mean_uint8', model['merged_mean']),
                              ('fixed_display_on_fraction', model['merged_display_on_fraction'])]:
            equal(float(row[key]), expected, f'csv/{region_id}/{row["model"]}/{key}')
    pins = None
    if args.pipeline_pins:
        pins = json.loads(args.pipeline_pins.read_text())
        for record in pins.values():
            path = lab / Path(record['path']).relative_to('/workspace/herculaneum-lab')
            if sha(path) != record['sha256'] or path.stat().st_size != record['bytes']:
                raise AssertionError('Frozen pipeline file changed')
    report = {'status': 'PASS', 'checked_at_utc': datetime.now(timezone.utc).isoformat(),
              'method': 'Independent integer joint histograms and exact moments; original saved maps; no execution of supplied analyze.py.',
              'receipt_checks': receipt_checks, 'numeric_values_compared': len(checks),
              'maximum_absolute_numeric_difference': max(c['absolute_error'] for c in checks),
              'phase_results': phase_summary, 'model_agreement': agreements,
              'boundary_extent': extent, 'full_field_boundary_edges': full_edges,
              'published_boundary_edges': published_edges, 'regions_checked': len(regions),
              'csv_rows_checked': len(csv_rows),
              'frozen_pipeline_files_unchanged': len(pins) if pins is not None else None,
              'verifier_sha256': sha(Path(__file__)), 'new_model_forwards': 0,
              'limits': 'Computational agreement on the same retained data, not an independent biological reference, preregistration timestamp verification, or full four-model reproduction.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ['status', 'numeric_values_compared', 'maximum_absolute_numeric_difference', 'regions_checked', 'full_field_boundary_edges', 'published_boundary_edges', 'frozen_pipeline_files_unchanged', 'new_model_forwards']}))


if __name__ == '__main__':
    main()
