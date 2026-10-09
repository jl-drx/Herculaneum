"""Verify retained comparison numerics; no model construction or forward calls."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
NORMALIZATION = Path('/workspace/herculaneum-lab/vendor/villa-ink-baseline/vesuvius/image_proc/intensity/normalization.py')
ROLES = ['frozen_seed42', 'A_smooth', 'B_smooth', 'A_preserve', 'B_preserve']


def record(path):
    path = Path(path).resolve(strict=True)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': str(path), 'size_bytes': path.stat().st_size, 'sha256': digest}


def verify(item):
    actual = record(item['path'])
    assert actual['sha256'] == item['sha256']
    assert actual['size_bytes'] == item.get('size_bytes', item.get('bytes'))
    return Path(item['path'])


def load_json(item):
    return json.loads(verify(item).read_text())


def close(actual, expected):
    assert abs(actual-expected) <= 2e-12, (actual, expected)


def ranking(labels, scores):
    # Ascending score bins, with half credit for positive/negative ties.
    unique, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    positives = np.bincount(inverse, weights=labels.astype(np.int64), minlength=len(counts)).astype(np.int64)
    negatives = counts-positives
    positive, negative = int(labels.sum()), int((~labels).sum())
    below_negative = np.cumsum(negatives)-negatives
    auc = float(np.sum(positives*(below_negative+.5*negatives))/(positive*negative))
    at_or_above_positive = np.cumsum(positives[::-1])[::-1]
    at_or_above_total = np.cumsum(counts[::-1])[::-1]
    ap = float(np.sum((positives/positive)*(at_or_above_positive/at_or_above_total)))
    tp = np.cumsum(positives[::-1])
    fp = np.cumsum(negatives[::-1])
    tpr, fpr = np.r_[0., tp/positive], np.r_[0., fp/negative]
    curves = dict(thresholds_descending=unique[::-1].astype(np.float64),
                  true_positive_count=tp, false_positive_count=fp,
                  tpr=tpr, fpr=fpr, recall=tpr.copy(), precision=np.r_[1., tp/(tp+fp)])
    return auc, ap, len(counts), int(counts[counts > 1].sum()), curves


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics', type=Path, required=True)
    args = parser.parse_args()
    output = json.loads(args.metrics.read_text())
    assert output['status'] == 'PASS' and output['verified_run_count'] == 20
    plan = load_json(output['plan'])
    assert plan['roles'] == ROLES and plan['directions'] == ['forward', 'reverse']
    assert plan['primary_direction'] == 'forward' and plan['secondary_direction'] == 'reverse'
    for item in plan['verified_inputs']:
        verify(item)
    original_metrics = load_json(plan['source_results'])
    for role in ROLES:
        own = load_json(original_metrics['thresholds'][role])
        assert {name: own[name] for name in ['primary', 'secondary']} == plan['models'][role]['thresholds']
        assert plan['models'][role]['threshold_lock'] == original_metrics['thresholds'][role]
    workers = {role: load_json(item) for role, item in output['model_results'].items()}
    spec = importlib.util.spec_from_file_location('review_pure_normalization', NORMALIZATION)
    normalizer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(normalizer)
    torch.set_num_threads(4)
    window = torch.hann_window(128, periodic=False, dtype=torch.float32)
    weights = torch.outer(window, window)
    weights /= torch.clamp(weights.max(), min=torch.finfo(weights.dtype).eps)
    weights = torch.clamp(weights, min=.001).numpy()
    cache = {}
    cases = {}
    for case in plan['cases']:
        cases[case['id']] = {name: np.load(verify(item), allow_pickle=False)
                             for name, item in case['arrays'].items()}
    maximum_map_error = 0.
    runs = tiles = domains = 0
    endpoints = {direction: {} for direction in plan['directions']}
    for direction in plan['directions']:
        for case in plan['cases']:
            case_id = case['id']
            arrays = cases[case_id]
            labels = arrays['labels'].astype(bool)
            source = arrays['surface'] if direction == 'forward' else arrays['surface'][::-1]
            maps, valid = {}, {}
            endpoints[direction][case_id] = {'primary': {}, 'secondary_known': {}}
            for role in ROLES:
                worker = workers[role]
                assert worker['role'] == role and worker['status'] == 'PASS'
                assert worker['plan'] == output['plan']
                assert worker['model_state_unchanged'] and worker['training_calls'] == worker['threshold_fitting_calls'] == 0
                assert worker['state_sha256'] == plan['models'][role]['state_sha256']
                item = worker['runs'][direction][case_id]
                scores = np.load(verify(item['scores']), allow_pickle=False)
                evaluated = np.load(verify(item['evaluated']), allow_pickle=False)
                logits = np.load(verify(item['logits']), allow_pickle=False)
                assert scores.dtype == logits.dtype == np.float32 and evaluated.dtype == np.bool_
                assert scores.shape == evaluated.shape == (384, 384)
                assert logits.shape == (len(item['captured_tiles']), 128, 128)
                probabilities = torch.sigmoid(torch.from_numpy(logits)).numpy()
                sums = np.zeros((384, 384), dtype=np.float32)
                denominator = np.zeros_like(sums)
                scored = [t['bbox_roi_xywh'] for t in item['inference']['tiles'] if t['status'] == 'scored']
                assert scored == [t['bbox_roi_xywh'] for t in item['captured_tiles']]
                assert [t['bbox_roi_xywh'] for t in item['inference']['tiles']] == [[x, y, 128, 128]
                    for y in range(0, 257, 64) for x in range(0, 257, 64)]
                for tile, probability in zip(item['captured_tiles'], probabilities):
                    x, y, w, h = tile['bbox_roi_xywh']
                    assert w == h == 128
                    key = (direction, case_id, x, y)
                    if key not in cache:
                        normalized = np.ascontiguousarray(normalizer.normalize_robust(source[:, y:y+h, x:x+w]), dtype=np.float32)
                        cache[key] = hashlib.sha256(normalized.tobytes()).hexdigest()
                    assert cache[key] == tile['input_sha256']
                    sums[y:y+h, x:x+w] += probability*weights
                    denominator[y:y+h, x:x+w] += weights
                reconstructed = np.zeros_like(sums)
                np.divide(sums, denominator, out=reconstructed, where=denominator > 0)
                assert np.array_equal(evaluated, denominator > 0)
                difference = float(np.max(np.abs(reconstructed-scores)))
                maximum_map_error = max(maximum_map_error, difference)
                # Batched elementwise sigmoid may choose a different vectorized
                # CPU kernel from individual tiles; tolerate one small FP32 ulp.
                assert difference <= 2e-7, difference
                assert evaluated[arrays['primary_mask']].all()
                maps[role], valid[role] = scores, evaluated & np.isfinite(scores)
                runs += 1
                tiles += len(logits)
            for domain, fixed in [('primary', arrays['primary_mask']), ('secondary_known', arrays['known_mask'])]:
                common = fixed.copy()
                for available in valid.values():
                    common &= available
                if domain == 'primary':
                    assert np.array_equal(common, fixed)
                reported = output['per_context'][direction][case_id]['domains'][domain]
                assert np.array_equal(np.load(verify(reported['common_mask']), allow_pickle=False), common)
                yy = labels[common]
                assert reported['fixed_count'] == int(fixed.sum()) and reported['common_count'] == len(yy)
                assert reported['positive'] == int(yy.sum()) and reported['negative'] == int((~yy).sum())
                close(reported['prevalence'], float(yy.mean()))
                for role in ROLES:
                    values = maps[role][common]
                    row = reported['models'][role]
                    auc, ap, groups, tied, curves = ranking(yy, values)
                    close(row['auc'], auc)
                    close(row['ap'], ap)
                    assert row['distinct_score_groups'] == groups and row['tied_score_pixel_count'] == tied
                    ep = {'auc': auc, 'ap': ap}
                    for rule in ['primary', 'secondary']:
                        threshold = plan['models'][role]['thresholds'][rule]
                        predicted = values.astype(np.float64) >= threshold
                        tp, fp = int((predicted & yy).sum()), int((predicted & ~yy).sum())
                        p, n = int(yy.sum()), int((~yy).sum())
                        operating = row['operating'][rule]
                        assert operating['threshold'] == threshold
                        assert operating['confusion'] == dict(tp=tp, fp=fp, tn=n-fp, fn=p-tp)
                        close(operating['sensitivity'], tp/p)
                        close(operating['specificity'], (n-fp)/n)
                        assert operating['pixel_count'] == len(yy)
                        assert operating['supplied_positive_pixels'] == p and operating['supplied_negative_pixels'] == n
                        if tp+fp:
                            close(operating['precision'], tp/(tp+fp))
                        else:
                            assert operating['precision'] is None
                        close(operating['balanced_accuracy'], .5*(tp/p+(n-fp)/n))
                        ep[rule+'_sensitivity'], ep[rule+'_specificity'] = tp/p, (n-fp)/n
                    endpoints[direction][case_id][domain][role] = ep
                    with np.load(verify(row['curves']), allow_pickle=False) as saved_curves:
                        assert set(saved_curves.files) == set(curves)
                        for name, expected in curves.items():
                            assert np.array_equal(saved_curves[name], expected)
                    quantiles = np.quantile(values, [0, .05, .5, .95, 1])
                    for name, value in zip(['min', 'q05', 'median', 'q95', 'max'], quantiles):
                        close(row['score_quantiles'][name], float(value))
                eps = endpoints[direction][case_id][domain]
                for name, first, second in [('A_preserve_minus_A_smooth', 'A_preserve', 'A_smooth'),
                                             ('B_preserve_minus_B_smooth', 'B_preserve', 'B_smooth')]:
                    for key in eps[first]:
                        close(reported['deltas']['matched_preservation'][name][key], eps[first][key]-eps[second][key])
                for role in ROLES:
                    for key in eps[role]:
                        close(reported['deltas']['baseline_relative'][role][key], eps[role][key]-eps['frozen_seed42'][key])
                domains += 1
    for direction in plan['directions']:
        for domain in ['primary', 'secondary_known']:
            means = {role: {key: sum(endpoints[direction][c['id']][domain][role][key] for c in plan['cases'])/2
                           for key in endpoints[direction][plan['cases'][0]['id']][domain][role]} for role in ROLES}
            for role in ROLES:
                for key, value in means[role].items():
                    close(output['equal_context_macro'][direction][domain][role][key], value)
            comparisons = [('A_preserve_minus_A_smooth', 'A_preserve', 'A_smooth'),
                           ('B_preserve_minus_B_smooth', 'B_preserve', 'B_smooth')]
            reported_delta = output['macro_deltas'][direction][domain]
            for name, first, second in comparisons:
                for key in means[first]:
                    close(reported_delta['matched_preservation'][name][key], means[first][key]-means[second][key])
            for role in ROLES:
                for key in means[role]:
                    close(reported_delta['baseline_relative'][role][key], means[role][key]-means['frozen_seed42'][key])
    assert runs == 20 and tiles <= 500 and tiles == output['verified_tile_forward_count']
    for item in plan['verified_inputs']:
        verify(item)
    result = {'status': 'PASS', 'metrics': record(args.metrics), 'plan': output['plan'],
              'checker': record(Path(__file__)), 'pure_normalizer': record(NORMALIZATION),
              'verified_runs': runs, 'verified_raw_logit_tiles': tiles, 'verified_domains': domains,
              'unique_normalized_input_fingerprints_reconstructed': len(cache),
              'maximum_logit_blend_probability_error': maximum_map_error,
              'ranking_method': 'Independent ascending-score-bin Mann-Whitney AUC and grouped AP',
              'network_calls': 0, 'model_constructions': 0, 'model_forward_calls': 0,
              'backward_calls': 0, 'new_checkpoint_loads': 0}
    path = HERE / 'completed-comparison-verification.json'
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': 'PASS', 'receipt': record(path)}))


if __name__ == '__main__':
    main()
