"""Fixed-direction, fixed-threshold supplied-reference agreement for five retained states."""
import argparse
import json
from pathlib import Path
import numpy as np
from prepare_reference import load_plan, verify, record, write_json, module, require, load_arrays

ROLES = ['frozen_seed42', 'A_smooth', 'B_smooth', 'A_preserve', 'B_preserve']
DIRECTIONS = ['forward', 'reverse']


def artifact_records(value, records):
    if isinstance(value, dict):
        if set(value) == {'path', 'size_bytes', 'sha256'}:
            require(value['path'] not in records or records[value['path']] == value, 'Conflicting artifact identities')
            records[value['path']] = value
        else:
            for item in value.values(): artifact_records(item, records)
    elif isinstance(value, list):
        for item in value: artifact_records(item, records)


def captured_contract(item):
    captured = item['captured_tiles']; meta = item['inference']
    scored = [tile['bbox_roi_xywh'] for tile in meta['tiles'] if tile['status'] == 'scored']
    require(len(captured) == meta['scored_tile_count'] == len(scored), 'Captured/scored tile counts differ')
    require([tile['bbox_roi_xywh'] for tile in captured] == scored, 'Captured tile placements/order differ')
    for tile in captured:
        x, y, width, height = tile['bbox_roi_xywh']; digest = tile['input_sha256']
        require(width == height == 128 and 0 <= x <= 256 and 0 <= y <= 256, 'Captured tile bounds differ')
        require(isinstance(digest, str) and len(digest) == 64 and all(c in '0123456789abcdef' for c in digest), 'Invalid input fingerprint')
    logits = np.load(verify(item['logits']), mmap_mode='r', allow_pickle=False)
    require(logits.dtype == np.float32 and logits.shape == (len(captured), 128, 128) and np.isfinite(logits).all(), 'Raw tile logit contract differs')
    return [dict(bbox_roi_xywh=tile['bbox_roi_xywh'], input_sha256=tile['input_sha256']) for tile in captured]


def raw_map(item, shape):
    scores = np.load(verify(item['scores']), allow_pickle=False); evaluated = np.load(verify(item['evaluated']), allow_pickle=False)
    require(scores.dtype == np.float32 and scores.shape == shape and evaluated.dtype == np.bool_
        and evaluated.shape == shape, 'Raw probability/evaluated map differs')
    finite = np.isfinite(scores)
    require(np.all((scores[finite] >= 0) & (scores[finite] <= 1)), 'Probability outside raw sigmoid range')
    return scores, evaluated


def endpoints(models):
    return {role: dict(auc=model['auc'], ap=model['ap'], **{rule+'_'+key: model['operating'][rule][key]
        for rule in ['primary', 'secondary'] for key in ['sensitivity', 'specificity']}) for role, model in models.items()}


def deltas(values):
    subtract = lambda first, second: {key: values[first][key]-values[second][key]
        if values[first][key] is not None and values[second][key] is not None else None for key in values[first]}
    return dict(matched_preservation=dict(A_preserve_minus_A_smooth=subtract('A_preserve', 'A_smooth'),
        B_preserve_minus_B_smooth=subtract('B_preserve', 'B_smooth')),
        baseline_relative={role: subtract(role, 'frozen_seed42') for role in ROLES})


def macro(rows):
    values = [endpoints(row['models']) for row in rows]
    return {role: {key: float(np.mean([row[role][key] for row in values]))
        if all(row[role][key] is not None for row in values) else None for key in values[0][role]} for role in ROLES}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['plan', 'run-dir', 'output-dir']: parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args(); plan, path = load_plan(args.plan, execute=True); before = record(path)
    require(set(plan['models']) == set(ROLES) and len(plan['cases']) == 2, 'Fixed model/context scope differs')
    require(args.output_dir.resolve() != args.run_dir.resolve() and args.run_dir.resolve() not in args.output_dir.resolve().parents,
        'Analysis must have a separate output directory')
    index_record = record(args.run_dir/'results.json'); index = json.loads(verify(index_record).read_text())
    require(index['status'] == 'PASS' and index['plan'] == before and set(index['models']) == set(ROLES), 'Incomplete comparison index')
    workers = {role: json.loads(verify(item).read_text()) for role, item in index['models'].items()}
    records = {}; artifact_records(index_record, records); artifact_records(index, records); artifact_records(workers, records)
    for item in records.values(): verify(item)
    for role, worker in workers.items():
        require(worker['status'] == 'PASS' and worker['role'] == role and worker['plan'] == before, 'Model result provenance differs')
        require(set(worker['runs']) == set(DIRECTIONS), 'Fixed directions differ')
        require(all(set(worker['runs'][direction]) == {case['id'] for case in plan['cases']} for direction in DIRECTIONS), 'Run contexts differ')
        require(worker['state_sha256'] == plan['models'][role]['state_sha256'], 'State fingerprint differs')
    metric = module(plan['metrics'], 'eleventh_metrics'); cutoff = module(plan['cutoff'], 'eleventh_operating')
    args.output_dir.mkdir(parents=True, exist_ok=False); results = {}; inputs = {}; verified_count = 0; tile_count = 0
    for direction in DIRECTIONS:
        results[direction] = {}; inputs[direction] = {}; direction_dir = args.output_dir/direction; direction_dir.mkdir()
        for case in plan['cases']:
            data = load_arrays(case); labels = data['labels'].astype(bool); shape = labels.shape
            fixed = dict(primary=data['primary_mask'], secondary_known=data['known_mask'])
            common = {name: mask.copy() for name, mask in fixed.items()}; maps = {}; captured = None
            for role, worker in workers.items():
                item = worker['runs'][direction][case['id']]; evidence = captured_contract(item); verified_count += 1; tile_count += len(evidence)
                if captured is None: captured = evidence
                else: require(evidence == captured, 'Per-tile normalized input/placement differs across models')
                scores, evaluated = raw_map(item, shape); maps[role] = scores
                for mask in common.values(): mask &= evaluated & np.isfinite(scores)
            require(np.array_equal(common['primary'], fixed['primary']), 'Fixed primary contains invalid/unscored pixels; no shrinkage')
            require(int((common['primary'] & labels).sum()) >= 512 and int((common['primary'] & ~labels).sum()) >= 8192,
                'Primary class gate failed; no fallback')
            inputs[direction][case['id']] = dict(scored_tile_count=len(captured), captured_tiles=captured)
            case_dir = direction_dir/case['id']; case_dir.mkdir(); domains = {}
            for domain, mask in common.items():
                mask_path = case_dir/(domain+'-common-mask.npy'); np.save(mask_path, mask)
                row = dict(fixed_count=int(fixed[domain].sum()), common_count=int(mask.sum()), positive=int((mask & labels).sum()),
                    negative=int((mask & ~labels).sum()), prevalence=float(labels[mask].mean()) if mask.any() else None,
                    common_mask=record(mask_path), models={})
                for role, scores in maps.items():
                    yy = labels[mask]; values = scores[mask]; ranking, curves = metric.binary_metrics(yy, values)
                    curve_path = case_dir/(domain+'-'+role+'-curves.npz'); np.savez_compressed(curve_path, **curves)
                    row['models'][role] = dict(auc=ranking['roc_auc'], ap=ranking['average_precision'],
                        distinct_score_groups=ranking['distinct_score_groups'], tied_score_pixel_count=ranking['tied_score_pixel_count'],
                        operating={rule: cutoff.operating_metrics(yy, values, plan['models'][role]['thresholds'][rule]) for rule in ['primary', 'secondary']},
                        curves=record(curve_path), score_quantiles=dict(zip(['min', 'q05', 'median', 'q95', 'max'],
                            map(float, np.quantile(values, [0, .05, .5, .95, 1])))) if values.size else None)
                row['deltas'] = deltas(endpoints(row['models'])); domains[domain] = row
            results[direction][case['id']] = dict(native_bbox_yxhw=case['native_bbox_yxhw'], domains=domains)
    require(verified_count == 20, 'All twenty fixed runs were not retained')
    require(tile_count <= plan['limits']['maximum_tile_forwards'], 'Tile-forward count exceeds frozen limit')
    means = {direction: {domain: macro([case['domains'][domain] for case in results[direction].values()])
        for domain in ['primary', 'secondary_known']} for direction in DIRECTIONS}
    macro_deltas = {direction: {domain: deltas(values) for domain, values in domains.items()} for direction, domains in means.items()}
    for item in records.values(): verify(item)
    load_plan(path, execute=True); require(record(path) == before, 'Plan/code/data changed during analysis')
    output = dict(status='PASS', plan=before, direction_roles=dict(forward='predeclared primary', reverse='predeclared secondary'),
        per_context=results, equal_context_macro=means, macro_deltas=macro_deltas, model_results=index['models'], comparison_index=index_record,
        state_sha256={role: worker['state_sha256'] for role, worker in workers.items()}, captured_input_contract=inputs,
        verified_run_count=verified_count, verified_tile_forward_count=tile_count,
        interpretation='Descriptive supplied-reference agreement on two contexts from one specimen with unresolved historical exposure; directions and frozen thresholds are reported separately, without model selection, success gates, confidence intervals or a reading claim.')
    write_json(args.output_dir/'metrics.json', output)
    size = sum(p.stat().st_size for folder in [args.run_dir, args.output_dir] for p in folder.rglob('*') if p.is_file())
    require(size <= plan['limits']['output_bytes'], 'Comparison run plus analysis output cap failed')
    print(json.dumps(dict(status='PASS', metrics=record(args.output_dir/'metrics.json'))))


if __name__ == '__main__': main()
