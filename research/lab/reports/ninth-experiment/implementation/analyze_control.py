"""Descriptive 2x2 smoothing/pseudo retention diagnostics; no held-out screen."""
import argparse
import json
from pathlib import Path
import numpy as np
from control_common import ARMS, load_plan, record, verify, write_json, module, require, evaluation


def artifact_records(value, records):
    if isinstance(value, dict):
        if set(value) == {'path', 'size_bytes', 'sha256'}:
            require(value['path'] not in records or records[value['path']] == value, 'Conflicting artifact identities')
            records[value['path']] = value
        else:
            for item in value.values(): artifact_records(item, records)
    elif isinstance(value, list):
        for item in value: artifact_records(item, records)


def raw_map(item, shape):
    scores = np.load(verify(item['scores']), allow_pickle=False)
    evaluated = np.load(verify(item['evaluated']), allow_pickle=False)
    require(scores.dtype == np.float32 and scores.shape == shape and evaluated.dtype == np.bool_
        and evaluated.shape == shape, 'Raw map contract changed')
    finite = np.isfinite(scores)
    require(np.all((scores[finite] >= 0) & (scores[finite] <= 1)), 'Scores outside raw sigmoid range')
    return scores, evaluated


def endpoints(models):
    return {arm: dict(auc=m['auc'], ap=m['ap'], **{rule+'_'+key: m['operating'][rule][key]
        for rule in ['primary', 'secondary'] for key in ['sensitivity', 'specificity']}) for arm, m in models.items()}


def contrasts(values):
    delta = lambda left, right: {key: values[left][key]-values[right][key] for key in values[left]}
    result = dict(smoothing_with_manual_only=delta('A_smooth', 'A'), smoothing_with_pseudo=delta('B_smooth', 'B'),
        pseudo_at_epsilon_0=delta('B', 'A'), pseudo_at_epsilon_05=delta('B_smooth', 'A_smooth'))
    result['interaction_difference_of_differences'] = {key: result['pseudo_at_epsilon_05'][key]-result['pseudo_at_epsilon_0'][key]
        for key in values['A']}
    return result


def descriptive_screens(results, criteria):
    def paired(candidate, comparator, mitigation):
        rows = {}
        for case_id, case in results.items():
            model = case['models'][candidate]; other = case['models'][comparator]
            delta = dict(sensitivity=model['operating']['primary']['sensitivity']-other['operating']['primary']['sensitivity'],
                auc=model['auc']-other['auc'], ap=model['ap']-other['ap'])
            checks = dict(specificity=model['operating']['primary']['specificity_target_met_exact'],
                sensitivity=delta['sensitivity'] >= criteria['per_context_sensitivity_delta_min'],
                auc=delta['auc'] >= criteria['per_context_auc_delta_min'], ap=delta['ap'] >= criteria['per_context_ap_delta_min'])
            rows[case_id] = dict(deltas=delta, checks=checks, pass_context=all(checks.values()))
        mean = float(np.mean([row['deltas']['sensitivity'] for row in rows.values()]))
        mean_check = mean >= criteria['mitigation_mean_sensitivity_gain_min'] if mitigation else None
        return dict(comparator=comparator, per_context=rows, mean_sensitivity_delta=mean, mean_gain_required=mitigation,
            mean_gain_check=mean_check, pass_descriptive_screen=all(row['pass_context'] for row in rows.values()) and (mean_check if mitigation else True))
    return dict(mitigation={arm: paired(arm, base, True) for arm, base in [('A_smooth', 'A'), ('B_smooth', 'B')]},
        retention={arm: paired(arm, 'frozen_seed42', False) for arm in ['A_smooth', 'B_smooth']})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['plan', 'run-dir', 'output-dir']: parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args(); plan, path = load_plan(args.plan, execute=True); before = record(path)
    require(args.output_dir.resolve() != args.run_dir.resolve() and args.run_dir.resolve() not in args.output_dir.resolve().parents,
        'Analysis must have a separate output directory')
    worker_records = {arm: plan['historical_results'][arm] if arm in plan['historical_results']
        else record(args.run_dir/arm/'result.json') for arm in ARMS}
    workers = {arm: json.loads(verify(item).read_text()) for arm, item in worker_records.items()}
    records = {}; artifact_records(worker_records, records); artifact_records(workers, records)
    locks = {arm: json.loads(verify(w['threshold_lock']).read_text()) for arm, w in workers.items()}
    artifact_records(locks, records)
    for item in records.values(): verify(item)
    for arm, worker in workers.items():
        expected = plan['base_plan'] if arm in plan['historical_results'] else before
        require(worker['status'] == 'PASS' and worker['arm'] == arm and worker['plan'] == expected, 'Worker receipt mismatch')
        require(worker['initial_state_sha256'] == plan['historical_initial_state_sha256'], 'Initializer fingerprint differs')
        require(locks[arm]['arm'] == arm and locks[arm]['plan'] == expected, 'Threshold lock provenance differs')
        if arm not in plan['historical_results']:
            require(worker['training']['smoothing_epsilon'] == .5, 'New worker smoothing differs')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    metric = module(plan['metrics'], 'ninth_metrics'); cutoff = module(plan['cutoff'], 'ninth_operating')
    calibration = {}
    for arm, worker in workers.items():
        calibration[arm] = {}; thresholds = {}
        for case in plan['development']:
            _, labels, primary = evaluation(case); item = worker['development'][case['id']]
            require(locks[arm]['development'][case['id']]['maps'] == item, 'Locked calibration maps differ')
            scores, evaluated = raw_map(item, primary.shape)
            require(evaluated[primary].all() and np.isfinite(scores[primary]).all(), 'Calibration domain changed')
            selection = cutoff.choose_development_threshold(scores[primary & ~labels])
            thresholds[case['id']] = selection['threshold']
            require(locks[arm]['development'][case['id']]['selection']['threshold'] == selection['threshold'], 'Calibration cutoff differs')
            ranking, _ = metric.binary_metrics(labels[primary], scores[primary])
            calibration[arm][case['id']] = dict(auc=ranking['roc_auc'], ap=ranking['average_precision'],
                operating={rule: cutoff.operating_metrics(labels[primary], scores[primary], locks[arm][rule]) for rule in ['primary', 'secondary']})
        require(locks[arm]['primary'] == max(thresholds.values()) and locks[arm]['secondary'] == thresholds['pherc1667-w029'],
            'Fixed primary/secondary calibration rule changed')
    results = {}
    for case in plan['diagnostic']:
        _, labels, primary = evaluation(case); maps = {}; common = primary.copy()
        for arm, worker in workers.items():
            item = worker['fresh' if arm in plan['historical_results'] else 'diagnostic'][case['id']]
            scores, evaluated = raw_map(item, primary.shape); maps[arm] = scores; common &= evaluated & np.isfinite(scores)
        require(int((common & labels).sum()) >= 256 and int((common & ~labels).sum()) >= 256, 'Common class gate failed; no relaxation')
        case_dir = args.output_dir/case['id']; case_dir.mkdir(); np.save(case_dir/'primary-common-mask.npy', common)
        row = dict(role=plan['diagnostic_role'], physical_group=case['physical_group'], bbox_source_xywh=case['data']['bbox_source_xywh'],
            evaluation_participation_role=case['evaluation_participation_role'], fixed_primary_count=int(primary.sum()),
            common_primary_count=int(common.sum()), positive=int((common & labels).sum()), negative=int((common & ~labels).sum()),
            prevalence=float((common & labels).sum()/common.sum()), models={})
        for arm, scores in maps.items():
            yy = labels[common]; values = scores[common]; ranking, curves = metric.binary_metrics(yy, values)
            np.savez_compressed(case_dir/(arm+'-curves.npz'), **curves)
            operating = {rule: cutoff.operating_metrics(yy, values, locks[arm][rule]) for rule in ['primary', 'secondary']}
            for op in operating.values(): op['specificity_target_met_exact'] = (
                op['confusion']['tn']*plan['screens']['specificity_denominator'] >= int((~yy).sum())*plan['screens']['specificity_numerator'])
            row['models'][arm] = dict(auc=ranking['roc_auc'], ap=ranking['average_precision'], operating=operating,
                score_quantiles=dict(zip(['min', 'q05', 'median', 'q95', 'max'], map(float, np.quantile(values, [0, .05, .5, .95, 1])))))
        row['factorial_contrasts'] = contrasts(endpoints(row['models'])); results[case['id']] = row
    values = [endpoints(row['models']) for row in results.values()]
    macro = {arm: {key: float(np.mean([row[arm][key] for row in values])) for key in values[0][arm]} for arm in ARMS}
    for item in records.values(): verify(item)
    load_plan(path, execute=True); require(record(path) == before, 'Plan changed after analysis')
    screens = descriptive_screens(results, plan['screens'])
    output = dict(status='PASS', plan=before, diagnostic=results, calibration=calibration, equal_context_macro=macro,
        factorial_contrasts=contrasts(macro), screens=screens, screen_criteria=plan['screens'],
        consistent_mitigation_both_pairs=all(screen['pass_descriptive_screen'] for screen in screens['mitigation'].values()),
        model_results=worker_records, thresholds={arm: w['threshold_lock'] for arm, w in workers.items()},
        interpretation='Descriptive one-seed smoothing/pseudo interaction and retention diagnostics on two previously inspected same-specimen contexts; passing a screen gives no promotion, new held-out accuracy, significance or reading claim.')
    write_json(args.output_dir/'metrics.json', output)
    size = sum(p.stat().st_size for folder in [args.run_dir, args.output_dir] for p in folder.rglob('*') if p.is_file())
    require(size <= plan['limits']['output_bytes'], 'New run plus analysis output cap failed')
    print(json.dumps(dict(status='PASS', metrics=record(args.output_dir/'metrics.json'))))


if __name__ == '__main__': main()
