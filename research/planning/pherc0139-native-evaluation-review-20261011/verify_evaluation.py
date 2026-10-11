"""Read-only audit of the frozen native study; never performs model inference.

Checks committed training endpoints before opening any evaluation map. The
scientific metrics and routing are implemented independently of native_learning.
Checkpoint bytes are hashed, never deserialized. No network is used.
"""
from __future__ import annotations

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath

import numpy as np
import zarr

EXPECTED_GO = '2c20b4491a2c94fdc4c479a84a6ab85f309673a113cfdd8088b30fb7df6bf61c'
PLATEAU = 'training_loss_plateau_at_minimum_lr'


class AuditError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    def invalid(value):
        raise AuditError('Nonfinite JSON number: ' + value)
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def checked_path(root, relative):
    name = PurePosixPath(relative)
    require(not name.is_absolute() and '..' not in name.parts and '\\' not in relative,
            'Unsafe artifact path: ' + relative)
    path = root / relative
    require(path.resolve().is_relative_to(root.resolve()), 'Path escapes artifact root')
    require(path.is_file() and not path.is_symlink(), 'Missing or symlinked artifact: ' + relative)
    return path


def verify_pin(root, pin):
    path = checked_path(root, pin['path'])
    require(path.stat().st_size == pin['bytes'] and digest(path) == pin['sha256'],
            'Artifact checksum mismatch: ' + pin['path'])
    return path


def compare(actual, expected, where='', tolerance=1e-10):
    if isinstance(expected, dict):
        require(set(actual) == set(expected), 'Metric keys differ: ' + where)
        for key in expected:
            compare(actual[key], expected[key], where + '/' + key, tolerance)
    elif isinstance(expected, list):
        require(len(actual) == len(expected), 'List lengths differ: ' + where)
        for i, (a, b) in enumerate(zip(actual, expected)):
            compare(a, b, where + '/' + str(i), tolerance)
    elif isinstance(expected, float):
        require(isinstance(actual, (int, float)) and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance),
                'Numeric mismatch: ' + where)
    else:
        require(actual == expected, 'Value mismatch: ' + where)


def metrics(labels, logits):
    """Ascending grouped ranks for AP; Mann–Whitney counts for ROC AUC."""
    y = np.asarray(labels, dtype=bool).reshape(-1)
    s = np.asarray(logits, dtype=np.float64).reshape(-1)
    require(y.size == s.size and y.size > 0 and np.isfinite(s).all(),
            'Empty, mismatched or nonfinite scores')
    positives, negatives = int(y.sum()), int((~y).sum())
    ap = auc = None
    if positives and negatives:
        _, group = np.unique(s, return_inverse=True)
        pos = np.bincount(group[y], minlength=int(group.max()) + 1)
        neg = np.bincount(group[~y], minlength=len(pos))
        lower_negatives = np.cumsum(neg) - neg
        auc = float(np.sum(pos * (lower_negatives + neg / 2)) / (positives * negatives))
        tp = np.cumsum(pos[::-1])
        all_selected = np.cumsum((pos + neg)[::-1])
        ap = float(np.sum(pos[::-1] * tp / all_selected) / positives)
    exp_minus_abs = np.exp(-np.abs(s))
    probability = np.where(s >= 0, 1 / (1 + exp_minus_abs), exp_minus_abs / (1 + exp_minus_abs))
    loss = np.log1p(exp_minus_abs) + np.maximum(s, 0) - y * s
    return dict(ap=ap, auc=auc, positives=positives, negatives=negatives,
                undefined_single_class=not (positives and negatives),
                bce=float(loss.mean()), prevalence=float(y.mean()),
                mean_probability=float(probability.mean()),
                fraction_probability_ge_half=float(np.count_nonzero(s >= 0) / len(s)))


def audit_history(done, plan):
    history = done['history']
    steps = [row['step'] for row in history]
    terminal = done['terminal_step']
    require(steps == list(range(0, terminal + 1, plan['check_every_updates'])),
            'Incomplete or unordered training history')
    require(plan['minimum_updates'] <= terminal <= plan['maximum_updates'], 'Invalid terminal budget')
    require(done['draws'] == terminal * plan['batch_size'], 'Incorrect training exposure count')
    require(all(math.isfinite(r['bce']) and r['bce'] >= 0 for r in history), 'Invalid training BCE')
    selected = min(history, key=lambda row: (row['bce'], row['step']))
    require(selected['step'] == done['endpoint_step'], 'Endpoint is not earliest minimum training BCE')
    rate = plan['learning_rate']
    significant = history[0]['bce']
    bad = 0
    require(history[0]['lr'] == rate, 'Wrong initial learning rate')
    stopped = False
    for row in history[1:]:
        require(row['lr'] == rate, 'Training-history LR differs from frozen schedule')
        if significant - row['bce'] >= plan['plateau_min_delta']:
            significant, bad = row['bce'], 0
        else:
            bad += 1
        if bad >= plan['lr_patience_checks'] and rate > plan['minimum_lr']:
            rate = max(rate * plan['lr_factor'], plan['minimum_lr'])
            bad = 0
        stopped = (row['step'] >= plan['minimum_updates'] and rate == plan['minimum_lr']
                   and bad >= plan['stop_patience_checks'])
        require(not stopped or row['step'] == terminal, 'Training continued past its stopping rule')
    if stopped:
        reason = PLATEAU
    else:
        require(terminal == plan['maximum_updates'], 'Early termination without a confirmed plateau')
        earlier = min(r['bce'] for r in history if r['step'] <= terminal - 1024)
        reason = ('ceiling_still_improving' if earlier - selected['bce'] >= plan['plateau_min_delta']
                  else 'ceiling_without_confirmed_plateau')
    require(done['termination'] == reason, 'Incorrect optimization termination classification')
    return selected


def training_summary(done, plan, seed):
    endpoint = audit_history(done, plan)
    p = endpoint['prevalence']
    require(0 < p < 1, 'Training fold must contain both classes')
    constant_loss = -(p * math.log(p) + (1 - p) * math.log(1 - p))
    reduction = 1 - endpoint['bce'] / constant_loss
    return dict(seed=seed, endpoint_step=endpoint['step'], terminal_step=done['terminal_step'],
                termination=done['termination'], train_ap=endpoint['ap'],
                train_bce_reduction_from_constant=reduction,
                adequate_fit=bool(endpoint['ap'] >= plan['adequate_training_ap']
                                  and reduction >= plan['adequate_training_bce_reduction_from_constant']))


def endpoint_preflight(run, expected_go=EXPECTED_GO):
    require(digest(run / 'GO.json') == expected_go, 'Different frozen study GO')
    go = read(run / 'GO.json')
    for pin in go['code_pins']:
        verify_pin(run, pin)
    compare(read(run / 'PLAN.json'), go['plan'], 'PLAN')
    plan = go['plan']
    completed, missing, summaries = [], [], {}
    for fold in plan['fold_order']:
        for seed in plan['seeds']:
            relative = f'work/{fold}-seed{seed}/COMPLETE.json'
            path = run / relative
            if not path.exists():
                missing.append(relative)
                continue
            require(not path.is_symlink(), 'Symlinked endpoint completion')
            done = read(path)
            norm_path = checked_path(run, f'work/{fold}/NORMALIZATION.json')
            norm = read(norm_path)
            require(isinstance(norm, dict) and norm.get('go_sha256') == expected_go,
                    'Normalization is malformed or belongs to another GO')
            binding = dict(go_sha256=expected_go, fold=fold, seed=seed,
                           normalization_sha256=digest(norm_path))
            compare(done['binding'], binding, relative + '/binding')
            summaries[(fold, seed)] = training_summary(done, plan, seed)
            endpoint = verify_pin(path.parent, done['endpoint_receipt'])
            require(endpoint.name == f'step-{done["endpoint_step"]:06d}.json', 'Wrong endpoint receipt path')
            for step in {done['endpoint_step'], done['terminal_step']}:
                row = read(checked_path(path.parent, f'step-{step:06d}.json'))
                compare(row['binding'], binding, 'checkpoint binding')
                require(row['step'] == step, 'Checkpoint step mismatch')
                require(row['checkpoint']['path'] == f'step-{step:06d}.pt', 'Wrong checkpoint filename')
                verify_pin(path.parent, row['checkpoint'])  # No torch.load or pickle.
                compare(row['train'], next(r for r in done['history'] if r['step'] == step), 'checkpoint train history')
                if step == done['terminal_step']:
                    require(row['best_step'] == done['endpoint_step'] and row['termination'] == done['termination'],
                            'Terminal receipt disagrees with completion')
            completed.append(dict(path=relative, bytes=path.stat().st_size, sha256=digest(path)))
    seal_path = run / 'evaluation/ENDPOINTS.json'
    if missing:
        require(not seal_path.exists(), 'Premature joint endpoint seal')
        require(not any((run / 'evaluation').glob('*.npy')), 'Prediction map exists before all endpoints')
        return dict(status='WAITING_FOR_TRAINING', completed_endpoints=len(completed),
                    required_endpoints=len(completed) + len(missing), missing=missing,
                    held_out_map_bodies_opened=0, completed_histories_verified=True), go, summaries
    if not seal_path.exists():
        require(not any((run / 'evaluation').glob('*.npy')), 'Maps exist without joint endpoint seal')
        return dict(status='WAITING_FOR_ENDPOINT_SEAL', completed_endpoints=len(completed),
                    required_endpoints=len(completed), held_out_map_bodies_opened=0), go, summaries
    seal = read(checked_path(run, 'evaluation/ENDPOINTS.json'))
    compare(seal, dict(go_sha256=expected_go, endpoints=completed), 'joint endpoint seal')
    return dict(status='ENDPOINTS_VERIFIED', completed_endpoints=len(completed),
                required_endpoints=len(completed), held_out_map_bodies_opened=0,
                seal_sha256=digest(seal_path)), go, summaries


def blank_response(row):
    blanks = [core for core in row['cores'] if core['positives'] == 0]
    count = sum(core['negatives'] for core in blanks)
    require(count > 0, 'No annotated-negative support for the frozen guard')
    return sum(core['negatives'] * core['mean_probability'] for core in blanks) / count


def decision(transfer, adequate_fit, blank, budget_limited):
    if transfer and adequate_fit and blank:
        return 'advance_to_broader_paired_supervision_design'
    if budget_limited:
        return 'optimization_not_demonstrated_saturated_no_native_signal_null'
    if adequate_fit and not transfer:
        return 'fit_without_consistent_transfer_prioritize_coverage_or_reference_mismatch'
    if not adequate_fit:
        return 'fit_inadequate_one_training_only_representation_optimizer_diagnosis'
    return 'blank_response_guard_failed_review_calibration_and_transfer'


def summarize(outputs, training, plan):
    folds = []
    for fold in plan['fold_order']:
        selected = {r['model']: r for r in outputs if r['fold'] == fold}
        profile = selected['profile']
        cnns = [selected[f'cnn-seed{seed}'] for seed in plan['seeds']]
        aps = [r['overall']['ap'] for r in cnns]
        require(profile['overall']['ap'] is not None and all(a is not None for a in aps), 'Single-class held-out fold')
        average_ap = float(np.mean(aps))
        profile_blank = blank_response(profile)
        cnn_blank = [blank_response(row) for row in cnns]
        folds.append(dict(fold=fold, profile=profile['overall'], cnn_mean_ap=average_ap,
            cnn_ap_min=min(aps), cnn_ap_max=max(aps), mean_ap_gain=average_ap-profile['overall']['ap'],
            all_seeds_gain=all(a > profile['overall']['ap'] for a in aps),
            training=[training[(fold, seed)] for seed in plan['seeds']],
            profile_blank_mean_probability=profile_blank, cnn_blank_mean_probabilities=cnn_blank,
            blank_response_gate=all(p-profile_blank <= plan['blank_core_max_probability_increase'] for p in cnn_blank)))
    transfer = all(f['all_seeds_gain'] and f['mean_ap_gain'] >= plan['advance_ap_margin'] for f in folds)
    fit = all(t['adequate_fit'] for f in folds for t in f['training'])
    blank = all(f['blank_response_gate'] for f in folds)
    limited = any(t['termination'] != PLATEAU for f in folds for t in f['training'])
    return dict(folds=folds, descriptive_macro_ap_gain=float(np.mean([f['mean_ap_gain'] for f in folds])),
        practical_transfer_gate=transfer, adequate_fit_gate=fit, blank_response_gate=blank,
        decision=decision(transfer, fit, blank, limited))


def load_labels(source, geometry, plan):
    """Native-frame annotations only. No CT, teacher, model weights or inference."""
    inventory = read(source / 'RESULTS.json')
    all_rows = {r['core_id']: dict(r, winding=site['winding'])
                for site in inventory['sites'] for r in site['rows'] if r['eligible']}
    folds = geometry['folds']
    require([f['held_out'] for f in folds] == plan['fold_order'], 'Fold order changed')
    result = {}
    for fold in folds:
        w = fold['held_out']
        ids = fold['original_evaluation_core_ids']
        train = fold['retained_training_core_ids']
        require(fold['fold_geometry_admitted'] and len(set(ids)) == len(ids)
                and len(set(train)) == len(train) and not set(train).intersection(ids), 'Invalid physical fold')
        require(all(all_rows[i]['winding'] != w for i in train), 'Held-out winding in training')
        mask_array = zarr.open_array(str(source / 'arrays' / w / 'supervision'), mode='r')
        label_array = zarr.open_array(str(source / 'arrays' / w / 'inklabels'), mode='r')
        items = []
        for core_id in ids:
            row = all_rows[core_id]
            require(row['winding'] == w and row['width'] == row['height'] == plan['scored_core_width'], 'Wrong core geometry')
            x, y = row['x'], row['y'];width = plan['scored_core_width']
            mask = np.asarray(mask_array[y:y+width, x:x+width]) > 0
            labels = np.asarray(label_array[y:y+width, x:x+width]) > 0
            require(mask.shape == labels.shape == (width, width) and mask.sum() == row['supervised_pixels']
                    and (mask & labels).sum() == row['ink_supervised'], 'Annotation counts changed')
            items.append((core_id, labels, mask))
        result[w] = items
    return result


def verify_evaluation(run, source, expected_go=EXPECTED_GO):
    preflight, go, training = endpoint_preflight(run, expected_go)
    if preflight['status'] != 'ENDPOINTS_VERIFIED':
        return preflight, []
    if not (run / 'RESULTS.json').is_file():
        return dict(preflight, status='WAITING_FOR_COMPLETE_EVALUATION'), []
    plan = go['plan']
    for pin in go['source_pins']:
        verify_pin(source, pin)
    geometry = read(source / 'geometry/RESULTS.json')
    labels = load_labels(source, geometry, plan)
    result = read(checked_path(run, 'RESULTS.json'))
    require(result['go_sha256'] == expected_go, 'Results from a different study')
    names = ['constant', 'profile'] + [f'cnn-seed{s}' for s in plan['seeds']]
    expected_receipts = [f'evaluation/{w}-{n}.json' for w in plan['fold_order'] for n in names]
    require([r['path'] for r in result['outputs']] == expected_receipts, 'Missing, duplicated or reordered result receipts')
    actual_maps = {p.name for p in (run / 'evaluation').glob('*.npy')}
    require(actual_maps == {Path(p).with_suffix('.npy').name for p in expected_receipts}, 'Missing or unexpected maps')
    require(not any((run / 'evaluation').glob('*.partial')), 'Partial evaluation output remains')
    outputs, csv_rows, opened = [], [], 0
    label_counts = {core_id: (int((y & m).sum()), int(m.sum()))
                    for items in labels.values() for core_id, y, m in items}
    for fold in geometry['folds']:
        w = fold['held_out']
        norm_path = checked_path(run, f'work/{w}/NORMALIZATION.json')
        norm = read(norm_path)
        require(norm['training_core_ids'] == fold['retained_training_core_ids'], 'Normalization includes wrong training cores')
        profile = read(checked_path(run, f'work/{w}/PROFILE.json'))
        require(profile['go_sha256'] == expected_go and profile['normalization_sha256'] == digest(norm_path), 'Wrong baseline binding')
        require(profile['fit']['converged'] and profile['fit']['gradient_infinity'] <= 1e-5, 'Unconverged profile reference')
        beta = np.asarray(profile['fit']['beta'])
        require(beta.shape == (plan['input_depth'] + 1,) and np.isfinite(beta).all(), 'Invalid profile coefficients')
        positives = sum(label_counts[i][0] for i in fold['retained_training_core_ids'])
        total = sum(label_counts[i][1] for i in fold['retained_training_core_ids'])
        require(profile['fit']['prevalence'] == positives / total, 'Baseline uses wrong training prevalence')
        for seed in plan['seeds']:
            done = read(run / 'work' / f'{w}-seed{seed}' / 'COMPLETE.json')
            require(all(r['positives'] == positives and r['negatives'] == total - positives
                        and r['prevalence'] == positives / total for r in done['history']),
                    'Training history uses wrong label population')
        normalization = norm['normalization']
        require(normalization['context_occurrences'] == len(fold['retained_training_core_ids'])
                and normalization['pixels'] == len(fold['retained_training_core_ids']) * plan['context_width'] ** 2,
                'Normalization uses wrong context population')
        mean, std = np.asarray(normalization['mean']), np.asarray(normalization['std'])
        require(mean.shape == std.shape == (plan['input_depth'],) and np.isfinite(mean).all()
                and np.isfinite(std).all() and (std >= 1 / 255).all(), 'Invalid normalization coefficients')
    for pin in result['outputs']:
        path = verify_pin(run, pin)
        row = read(path)
        w, model = row['fold'], row['model']
        require(w in plan['fold_order'] and model in names and path.name == f'{w}-{model}.json', 'Wrong fold/model receipt')
        require(row['go_sha256'] == expected_go, 'Wrong map GO binding')
        items = labels[w]
        require(row['core_ids'] == [i[0] for i in items], 'Map/core order mismatch')
        require(row['maps']['path'] == f'{w}-{model}.npy', 'Wrong map filename')
        map_path = verify_pin(run / 'evaluation', row['maps'])
        array = np.load(map_path, allow_pickle=False, mmap_mode='r');opened += 1
        width = plan['scored_core_width']
        dtype = np.dtype('float32') if model.startswith('cnn-') else np.dtype('float64')
        require(array.shape == (len(items), width, width) and array.dtype == dtype,
                'Wrong map shape or dtype')
        require(np.isfinite(array).all(), 'Nonfinite prediction map')
        if model == 'constant':
            p = read(run / 'work' / w / 'PROFILE.json')['fit']['prevalence']
            require(0 < p < 1 and np.all(array == math.log(p/(1-p))), 'Constant differs from training prevalence')
        ys, scores, cores = [], [], []
        for i, (core_id, label, mask) in enumerate(items):
            y, score = label[mask], array[i][mask]
            ys.append(y);scores.append(score)
            computed = dict(core_id=core_id, **metrics(y, score))
            cores.append(computed)
            csv_rows.append(dict(fold=w, model=model, **computed))
        recomputed = dict(overall=metrics(np.concatenate(ys), np.concatenate(scores)), cores=cores)
        compare(recomputed, {k:row[k] for k in ['overall', 'cores']}, f'{w}/{model}')
        outputs.append(dict(fold=w, model=model, **recomputed))
    recomputed = summarize(outputs, training, plan)
    compare(recomputed, {k:result[k] for k in recomputed}, 'final summary')
    return dict(status='PASS', go_sha256=expected_go, endpoints_verified=preflight['completed_endpoints'],
                endpoint_seal_sha256=preflight['seal_sha256'], held_out_map_bodies_opened=opened,
                source_files_verified=len(go['source_pins']), core_metric_rows=len(csv_rows),
                results_sha256=digest(run/'RESULTS.json'), recomputed=recomputed,
                scope='Independent metric arithmetic and receipt/history audit; no neural inference, ink authentication or changed gates'), csv_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    require(not args.output.resolve().is_relative_to(args.run.resolve()), 'Review output must be outside the frozen run')
    require(not args.output.exists(), 'Refusing to overwrite a review receipt')
    if args.preflight_only:
        report, _, _ = endpoint_preflight(args.run)
        rows = []
    else:
        require(args.source is not None, '--source is required for final verification')
        report, rows = verify_evaluation(args.run, args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    if rows:
        with args.output.with_suffix('.cores.csv').open('x', newline='') as stream:
            writer = csv.DictWriter(stream, list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({k:v for k,v in report.items() if k not in ['recomputed', 'missing']}))


if __name__ == '__main__':
    main()
