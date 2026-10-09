"""Independently verify saved Frag4 scores, participation, pins and metrics.

No model/checkpoint deserialization, experiment metric import or network call.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import socket
import numpy as np
from independent_metrics import compute


def offline(*args, **kwargs):
    raise RuntimeError('Network disabled in independent result verification')


socket.create_connection = offline
socket.socket.connect = offline
socket.socket.connect_ex = offline
HERE = Path(__file__).resolve().parent
LAB = HERE.parents[2]
ROLES = ['frozen_seed42', 'A_smooth', 'B_smooth', 'A_preserve', 'B_preserve']
DIRECTIONS = ['forward', 'reverse']
CASES = ['frag4-context1', 'frag4-context2']
EXPECTED = {
    'frag4-context1': ([0, 1152, 1152, 1152], 3818, 87626),
    'frag4-context2': ([8064, 3456, 1152, 1152], 15252, 62599),
}
SELECTION_SHA = 'c18634b2f64856c514ca7018763ba7fe0aa515640d81043b6d34d0f2dc37b5aa'
DESIGN_SHA = '5c20b5ffc3e00f6b5342c6ec38090e7236109e9535b493028c3fe1ac205a8d4c'
PREPROCESSING = dict(source_planes_inclusive=[7, 57], depth_group_size=3,
    xy_group_size=3, sum_dtype='uint32', division_dtype='float64', output_dtype='float32',
    scale_divisor=257, formula='float32(uint32_sum_of_27_source_uint16_values / float64(27*257))',
    source_group_centers=list(range(8, 57, 3)), center_source_plane=32,
    pixel_spacing_um=9.72, integer_quantization=False)
ENDPOINTS = ['auc', 'ap', 'primary_sensitivity', 'primary_specificity',
             'secondary_sensitivity', 'secondary_specificity']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_record(path):
    path = Path(path).resolve(strict=True)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(path), size_bytes=path.stat().st_size, sha256=digest)


def identity(item):
    sizes = [item[key] for key in ['size_bytes', 'byte_length', 'bytes'] if key in item]
    require(sizes and all(type(n) is int and n >= 0 and n == sizes[0] for n in sizes),
            'Missing or conflicting recorded byte lengths')
    digest = item['sha256']
    require(isinstance(digest, str) and len(digest) == 64
            and all(c in '0123456789abcdef' for c in digest), 'Malformed SHA256')
    path = Path(item['path'])
    require(path.is_absolute(), 'Artifact path is not absolute')
    return dict(path=str(path.resolve(strict=True)), size_bytes=sizes[0], sha256=digest)


class Pins:
    def __init__(self):
        self.records = {}
        self.hashed_bytes = 0

    def check(self, item):
        expected = identity(item)
        path = expected['path']
        if path in self.records:
            require(self.records[path] == expected, 'Conflicting file pins: ' + Path(path).name)
        else:
            require(file_record(path) == expected, 'File identity mismatch: ' + Path(path).name)
            self.records[path] = expected
            self.hashed_bytes += expected['size_bytes']
        return Path(path)

    def collect(self, value):
        if isinstance(value, dict):
            if 'path' in value and 'sha256' in value and any(k in value for k in ['size_bytes', 'byte_length', 'bytes']):
                self.check(value)
            else:
                for child in value.values():
                    self.collect(child)
        elif isinstance(value, list):
            for child in value:
                self.collect(child)

    def json(self, item):
        value = json.loads(self.check(item).read_text())
        self.collect(value)
        return value

    def unchanged(self):
        for expected in self.records.values():
            require(file_record(expected['path']) == expected,
                    'A file changed during independent review: ' + Path(expected['path']).name)


def same_record(first, second, message):
    require(identity(first) == identity(second), message)


def close(first, second, name):
    require(type(first) in [int, float] and type(second) in [int, float]
            and math.isfinite(first) and math.isfinite(second)
            and math.isclose(first, second, rel_tol=0, abs_tol=1e-12),
            'Numeric disagreement: ' + name)


def compare_tree(expected, actual, name):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), 'Endpoint keys differ: ' + name)
        for key, value in expected.items():
            compare_tree(value, actual[key], name + '/' + key)
    else:
        close(expected, actual, name)


def endpoints(models):
    return {role: {key: models[role][key] if key in ['auc', 'ap']
        else models[role]['operating'][key.split('_')[0]][key.split('_')[1]]
        for key in ENDPOINTS} for role in ROLES}


def delta(values):
    def subtract(first, second):
        return {key: values[first][key] - values[second][key] for key in ENDPOINTS}
    return dict(matched_preservation={
        'A_preserve_minus_A_smooth': subtract('A_preserve', 'A_smooth'),
        'B_preserve_minus_B_smooth': subtract('B_preserve', 'B_smooth')},
        baseline_relative={role: subtract(role, 'frozen_seed42') for role in ROLES})


def load_array(pins, record, dtype, shape):
    value = np.load(pins.check(record), allow_pickle=False)
    require(value.dtype == np.dtype(dtype) and value.shape == shape,
            'Array shape/dtype differs: ' + Path(record['path']).name)
    return value


def check_capture(pins, item):
    tiles = item['inference']['tiles']
    grid = [[x, y, 128, 128] for y in range(0, 257, 64) for x in range(0, 257, 64)]
    require([tile['bbox_roi_xywh'] for tile in tiles] == grid
            and item['inference']['tile_count'] == 25, 'Fixed tile grid/order differs')
    require(all(tile['status'] in ['scored', 'skipped'] for tile in tiles), 'Unknown tile status')
    scored = [tile['bbox_roi_xywh'] for tile in tiles if tile['status'] == 'scored']
    captured = item['captured_tiles']
    require([tile['bbox_roi_xywh'] for tile in captured] == scored
            and len(captured) == item['inference']['scored_tile_count'],
            'Captured raw tile placement/count differs')
    for tile in captured:
        digest = tile['input_sha256']
        require(isinstance(digest, str) and len(digest) == 64
                and all(c in '0123456789abcdef' for c in digest), 'Invalid captured normalized input SHA')
    logits = load_array(pins, item['logits'], 'float32', (len(captured), 128, 128))
    require(np.isfinite(logits).all(), 'Raw tile logits are nonfinite')
    # Check blend coverage from placement without approximating Torch's sigmoid
    # or nonperiodic FP32 Hann weights. The archived blend floor is positive.
    coverage = np.zeros((384, 384), dtype=bool)
    for x, y, width, height in scored:
        coverage[y:y+height, x:x+width] = True
    return captured, coverage


def verify(plan_path, result_path, metrics_path):
    pins = Pins()
    own_sources = [file_record(HERE/name) for name in
                   ['verify_reference.py', 'independent_metrics.py', 'check_metric_fixtures.py']]
    for item in own_sources:
        pins.check(item)
    fixtures = pins.json(file_record(HERE/'metric-fixture-receipt.json'))
    require(fixtures['status'] == 'PASS', 'Independent metric fixture receipt did not pass')
    for item in fixtures['code']:
        require(file_record(item['path'])['sha256'] == item['sha256'],
                'Independent metric code changed after its fixture checks')
    plan_record = file_record(plan_path)
    plan = pins.json(plan_record)
    require(plan['kind'] == 'Frag4-five-state-reference-comparison-v1'
            and plan['ready_for_execution'] is True, 'Frozen execution plan is not ready')
    require(plan['roles'] == ROLES and plan['directions'] == DIRECTIONS
            and plan['preprocessing'] == PREPROCESSING, 'Model/direction/preprocessing contract differs')
    require(plan['research_design']['sha256'] == DESIGN_SHA
            and plan['limits']['maximum_tile_forwards'] == 500, 'Scientific design/resource pin differs')
    require(plan['primary_direction'] == 'forward' and plan['secondary_direction'] == 'reverse',
            'Scientific direction roles differ')
    require([case['id'] for case in plan['cases']] == CASES and set(plan['models']) == set(ROLES),
            'Fixed specimen-context/model roles differ')
    require(plan['inference'] == dict(layer_start=0, layer_count=17, patch_shape=[17, 128, 128],
            stride=64, normalization='unchanged robust_mad 1/99 from retained factory',
            mask_passed_to_adapter=False, forward_primary=True, reverse_secondary=True,
            best_orientation_selection=False), 'Frozen inference contract differs')
    source_plan = pins.json(plan['source_plan'])
    source_results = pins.json(plan['source_results'])
    require(source_results['status'] == 'PASS', 'Retained state source study is incomplete')
    same_record(source_results['plan'], plan['source_plan'], 'Original state result/plan mismatch')
    same_record(plan['initializer'], source_plan['initializer'], 'Initializer source mismatch')
    for role in ROLES:
        spec = plan['models'][role]
        require(spec['role'] == role and set(spec['thresholds']) == {'primary', 'secondary'},
                'State/cutoff role mapping differs')
        same_record(spec['result'], source_results['model_results'][role], 'Retained role result pin differs')
        old = pins.json(spec['result'])
        require(old['status'] == 'PASS' and old['arm'] == role
                and old['initial_state_sha256'] == source_plan['historical_initial_state_sha256'],
                'Retained role/state lineage differs')
        same_record(old['threshold_lock'], spec['threshold_lock'], 'Frozen cutoff source differs')
        same_record(spec['threshold_lock'], source_results['thresholds'][role], 'Cutoff/source analysis mismatch')
        lock = pins.json(spec['threshold_lock'])
        require(lock['arm'] == role and lock['plan'] == old['plan'], 'Cutoff lineage differs')
        require(spec['thresholds'] == {key: lock[key] for key in ['primary', 'secondary']},
                'A retained threshold was changed')
        state = source_plan['historical_initial_state_sha256']
        if role == 'frozen_seed42':
            require(spec['checkpoint'] is None, 'Frozen baseline was replaced by continuation checkpoint')
        else:
            same_record(spec['checkpoint'], old['training']['checkpoint'], 'Continuation checkpoint pin differs')
            state = old['training']['final_state_sha256']
        require(spec['state_sha256'] == state, 'Frozen semantic state differs from retained source receipt')
    descriptor = pins.json(plan['descriptor'])
    require(descriptor['status'] == 'pooled_reference_complete_no_inference'
            and all(descriptor['preprocessing'][key] == value for key, value in PREPROCESSING.items()),
            'Prepared source contract differs')
    require([case['id'] for case in descriptor['cases']] == CASES, 'Prepared source cases differ')
    selection_path = LAB/'reports/eleventh-experiment/reference-selection/selection-result.json'
    selection_record = file_record(selection_path)
    require(selection_record['sha256'] == SELECTION_SHA, 'Label-only selection receipt changed')
    selection = pins.json(selection_record)
    arrays = {}
    selected = {case['id']: case for case in selection['selected']}
    for case, original in zip(plan['cases'], descriptor['cases']):
        name = case['id']
        bbox, positive, negative = EXPECTED[name]
        require(case['native_bbox_yxhw'] == original['native_bbox_yxhw'] == selected[name]['native_bbox_yxhw'] == bbox,
                'Source context differs from label-only frozen selection')
        require(set(case['arrays']) == {'surface', 'labels', 'primary_mask', 'known_mask', 'surface_mask'},
                'Prepared input array roles differ')
        for key, item in case['arrays'].items():
            same_record(item, original['arrays'][key], 'Source descriptor array pin differs')
        selected_files = {Path(item['path']).name: item for item in selected[name]['files']}
        for key in ['labels', 'primary_mask', 'known_mask', 'surface_mask']:
            same_record(case['arrays'][key], selected_files[key.replace('_', '-')+'.npy'],
                        'Selection mask/label was replaced')
        label = np.load(pins.check(case['arrays']['labels']), allow_pickle=False)
        require(label.shape == (384, 384) and label.dtype in [np.dtype('uint8'), np.dtype('bool')]
                and np.all((label == 0) | (label == 1)), 'Explicit binary supplied labels differ')
        label = label.astype(bool)
        mask = {key: load_array(pins, case['arrays'][key], 'bool', (384, 384))
                for key in ['primary_mask', 'known_mask', 'surface_mask']}
        border = np.zeros((384, 384), dtype=bool)
        border[32:352, 32:352] = True
        require(np.array_equal(mask['primary_mask'], mask['known_mask'] & border)
                and np.all(~mask['known_mask'] | mask['surface_mask']), 'Fixed participation contract differs')
        require(int((label & mask['primary_mask']).sum()) == positive
                and int((~label & mask['primary_mask']).sum()) == negative, 'Selection class counts differ')
        arrays[name] = (label, mask)
    index_record = file_record(result_path)
    index = pins.json(index_record)
    require(index['status'] == 'PASS' and index['retries'] == 0
            and index['all_fixed_roles_and_directions_retained'] is True
            and set(index['models']) == set(ROLES), 'Incomplete root-owned comparison index')
    same_record(index['plan'], plan_record, 'Comparison ran a different plan')
    require(index['workers'] == [dict(role=role, exit_code=0) for role in ROLES], 'Worker coverage/exit status differs')
    reported_record = file_record(metrics_path)
    reported = pins.json(reported_record)
    require(reported['status'] == 'PASS' and reported['verified_run_count'] == 20, 'Analysis incomplete')
    require(reported['direction_roles'] == dict(forward='predeclared primary', reverse='predeclared secondary')
            and set(reported['per_context']) == set(DIRECTIONS), 'Analysis direction roles differ')
    same_record(reported['plan'], plan_record, 'Analysis used a different plan')
    same_record(reported['comparison_index'], index_record, 'Analysis index pin differs')
    require(reported['model_results'] == index['models'], 'Analysis/model result pins differ')
    workers = {role: pins.json(item) for role, item in index['models'].items()}
    for role, worker in workers.items():
        require(worker['status'] == 'PASS' and worker['role'] == role
                and worker['state_sha256'] == plan['models'][role]['state_sha256']
                and worker['source_model'] == plan['models'][role]
                and worker['training_calls'] == worker['threshold_fitting_calls'] == 0
                and worker['model_state_unchanged'] is True
                and worker['source_file_pins_unchanged_before_after'] is True,
                'Worker state/no-adaptation provenance differs')
        same_record(worker['plan'], plan_record, 'Worker plan pin differs')
        require(set(worker['runs']) == set(DIRECTIONS)
                and all(set(worker['runs'][direction]) == set(CASES) for direction in DIRECTIONS),
                'Worker fixed run coverage differs')
        require(reported['state_sha256'][role] == worker['state_sha256'], 'Analysis state pin differs')
    independent = {}
    total_tiles = 0
    domain_counts = {}
    worker_counts = {role: 0 for role in ROLES}
    for direction in DIRECTIONS:
        independent[direction] = {}
        domain_counts[direction] = {}
        require(set(reported['per_context'][direction]) == set(CASES), 'Analysis context set differs')
        for name in CASES:
            label, masks = arrays[name]
            fixed = dict(primary=masks['primary_mask'], secondary_known=masks['known_mask'])
            common = {key: value.copy() for key, value in fixed.items()}
            maps = {}
            captured_reference = None
            for role in ROLES:
                item = workers[role]['runs'][direction][name]
                require(item['adapter_layer_start'] == 0 and item['adapter_direction'] == direction
                        and item['float32_input_without_uint8_quantization'] is True
                        and item['input_fingerprint_before_model_call'] is True
                        and item['model_input_tensor_unchanged'] is True, 'Adapter/capture contract differs')
                captured, coverage = check_capture(pins, item)
                if captured_reference is None:
                    captured_reference = captured
                else:
                    require(captured_reference == captured, 'Normalized inputs/placements differ across arms')
                total_tiles += len(captured)
                worker_counts[role] += len(captured)
                score = load_array(pins, item['scores'], 'float32', (384, 384))
                evaluated = load_array(pins, item['evaluated'], 'bool', (384, 384))
                finite = np.isfinite(score)
                require(np.array_equal(evaluated, coverage), 'Reported evaluation differs from positive blend support')
                require(np.all((score[finite] >= 0) & (score[finite] <= 1)), 'Saved probability outside [0,1]')
                require(np.all(evaluated[fixed['primary']]) and np.all(finite[fixed['primary']]),
                        'Fixed primary has invalid/unscored pixels; cannot shrink')
                maps[role] = score
                for value in common.values():
                    value &= evaluated & finite
            require(np.array_equal(common['primary'], fixed['primary']), 'Primary participation changed')
            input_report = reported['captured_input_contract'][direction][name]
            require(input_report['captured_tiles'] == captured_reference
                    and input_report['scored_tile_count'] == len(captured_reference), 'Analysis capture receipt differs')
            independent[direction][name] = {}
            domain_counts[direction][name] = {}
            analysis_case = reported['per_context'][direction][name]
            require(analysis_case['native_bbox_yxhw'] == EXPECTED[name][0]
                    and set(analysis_case['domains']) == set(common), 'Analysis domain/context differs')
            for domain, mask in common.items():
                row = analysis_case['domains'][domain]
                primary_count = int((mask & label).sum())
                negative_count = int((mask & ~label).sum())
                require(row['fixed_count'] == int(fixed[domain].sum())
                        and row['common_count'] == primary_count + negative_count
                        and row['positive'] == primary_count and row['negative'] == negative_count,
                        'Reported participating class counts differ')
                saved_mask = load_array(pins, row['common_mask'], 'bool', (384, 384))
                require(np.array_equal(saved_mask, mask), 'Analysis common mask differs from independent reconstruction')
                close(row['prevalence'], primary_count / (primary_count + negative_count), 'prevalence')
                require(set(row['models']) == set(ROLES), 'Analysis domain lacks a fixed arm')
                recalculated = {}
                for role in ROLES:
                    value = compute(label[mask], maps[role][mask], plan['models'][role]['thresholds'])
                    actual = row['models'][role]
                    close(value['auc'], actual['auc'], 'AUC/'+direction+'/'+name+'/'+domain+'/'+role)
                    close(value['ap'], actual['ap'], 'AP/'+direction+'/'+name+'/'+domain+'/'+role)
                    for rule, expected in value['operating'].items():
                        endpoint = actual['operating'][rule]
                        require(endpoint['threshold'] == expected['threshold'], 'Frozen cutoff changed during analysis')
                        require(endpoint['confusion'] == {key: expected[key] for key in ['tp', 'fp', 'tn', 'fn']},
                                'Frozen-threshold confusion counts differ')
                        require(endpoint['pixel_count'] == value['pixels']
                                and endpoint['supplied_positive_pixels'] == value['positive']
                                and endpoint['supplied_negative_pixels'] == value['negative'], 'Operating class counts differ')
                        for key in ['sensitivity', 'specificity']:
                            close(endpoint[key], expected[key], rule+'/'+key)
                    recalculated[role] = value
                calculated_endpoints = endpoints(recalculated)
                compare_tree(delta(calculated_endpoints), row['deltas'], 'Context matched/baseline delta')
                independent[direction][name][domain] = recalculated
                domain_counts[direction][name][domain] = dict(fixed_count=int(fixed[domain].sum()),
                    common_count=int(mask.sum()), positive=primary_count, negative=negative_count,
                    fixed_primary_not_shrunk=domain == 'primary')
    require(total_tiles <= 500 and reported['verified_tile_forward_count'] == total_tiles,
            'Tile-forward count/cap differs')
    for role in ROLES:
        require(worker_counts[role] == workers[role]['tile_forward_count'] <= 100, 'Role tile-forward count differs')
    means = {}
    macro_deltas = {}
    for direction in DIRECTIONS:
        means[direction] = {}
        macro_deltas[direction] = {}
        for domain in ['primary', 'secondary_known']:
            values = [endpoints(independent[direction][name][domain]) for name in CASES]
            mean = {role: {key: (values[0][role][key] + values[1][role][key]) / 2
                          for key in ENDPOINTS} for role in ROLES}
            means[direction][domain] = mean
            macro_deltas[direction][domain] = delta(mean)
    compare_tree(means, reported['equal_context_macro'], 'Equal-context macro')
    compare_tree(macro_deltas, reported['macro_deltas'], 'Macro matched/baseline delta')
    pins.unchanged()
    return dict(status='PASS', completed_at_utc=datetime.now(timezone.utc).isoformat(),
        plan=plan_record, comparison_index=index_record, analysis=reported_record,
        verifier_sources=own_sources, metric_fixtures=file_record(HERE/'metric-fixture-receipt.json'),
        selection=selection_record, verified_file_pins=len(pins.records),
        hashed_unique_input_bytes=pins.hashed_bytes, all_file_pins_unchanged_before_after=True,
        verified_run_count=20, verified_tile_forward_count=total_tiles,
        per_context=independent, domain_counts=domain_counts,
        equal_context_macro=means, macro_deltas=macro_deltas,
        state_sha256={role: plan['models'][role]['state_sha256'] for role in ROLES},
        metrics_numeric_absolute_tolerance=1e-12, confusion_counts_exact=True,
        thresholds_unchanged=True, fixed_primary_pixels_all_evaluated_and_finite=True,
        normalized_input_fingerprints_equal_across_arms=True,
        network_calls=0, model_constructors=0, model_forwards=0, model_backwards=0,
        checkpoint_deserializations=0, CT_pixel_decodes=0,
        limitations=[
            'Semantic state fingerprints checked against frozen worker/source receipts; no independent tensor deserialization or forward reproduction.',
            'Raw logits checked for pins, finite FP32 shape and placement; no independent Torch sigmoid/Hann numeric replay.',
            'Supplied-reference agreement does not authenticate optical registration, historical model holdout, glyphs or translation.',
            'Two contexts are one specimen; no pixel-independent significance or confidence interval claim.'],
        verified_inputs=list(pins.records.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['plan', 'results', 'metrics', 'output-dir']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    require(HERE in output.parents and not output.exists(), 'Receipt must be a new directory under result-verification')
    output.mkdir(parents=True)
    try:
        receipt = verify(args.plan, args.results, args.metrics)
    except Exception as error:
        stopped = dict(status='STOPPED', error_type=type(error).__name__, reason=str(error),
                       model_calls=0, network_calls=0, no_partial_verification_claim=True)
        (output/'independent-verification.json').write_text(json.dumps(stopped, indent=2, sort_keys=True)+'\n')
        raise
    (output/'independent-verification.json').write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False)+'\n')
    text = ('Independent saved-result verification: PASS.\n\n'
        f"All 20 fixed runs and {receipt['verified_tile_forward_count']} captured tile forwards were retained. "
        'Grouped-concordance AUC, suffix-count AP, exact frozen-threshold confusion counts, '
        'context deltas and equal-context macro deltas agree with the primary analysis. '
        'Every fixed primary pixel is evaluated and finite; no primary participation was removed.\n\n'
        f"Verified {receipt['verified_file_pins']} file pins before and after review. "
        'Source contexts, selected annotations, retained state receipts and cutoffs match their frozen records. '
        'All arms have identical captured normalized-input fingerprints and tile placement at each context/direction.\n\n'
        'This is an independent numeric and artifact review, with no models, checkpoint deserialization, '
        'CT pixel decoding or network. Semantic state hashes are checked against worker receipts, '
        'rather than recomputed from tensors. Raw-logit blend values were not independently replayed. '
        'The supplied optical-reference and one-specimen limits remain as stated in the research design.\n')
    (output/'REVIEW.md').write_text(text)
    print(json.dumps(dict(status='PASS', receipt=file_record(output/'independent-verification.json'),
                          review=file_record(output/'REVIEW.md'))))


if __name__ == '__main__':
    main()
