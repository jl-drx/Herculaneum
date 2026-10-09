"""Independent phase20 saved-output verification without Torch or model execution."""
import argparse
import hashlib
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import resource
import signal
import sys

for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
sys.dont_write_bytecode = True
import numpy as np
from review_io import Review, record, fixture_checks as io_fixtures
from review_math import bit_equal, equal_batch_means, fixture_checks as math_fixtures, manual_record, require
from review_features import ARMS, fixture_checks as feature_fixtures
from review_data import development_case, fixture_checks as data_fixtures, training_batches

PROTOCOL_SHA256 = '0c06563607d26d43e8e3e8e83c9debf5bf15bf2dc11722b1d66a2d7028bbb4a1'
UPDATES = [1, 8, 16, 32]
CASES = ['frag3-context1', 'frag3-context2', 'frag1-context1', 'frag1-context2', 'frag2-context1', 'frag2-context2']
TRAINING = dict(steps=32, batch_size=2, seed=20261008, learning_rate=.001, optimizer='AdamW',
                betas=[.9, .999], eps=1e-8, weight_decay=0, gradient_clip_norm=1.,
                loss_smoothing=.5, kl_coefficient=1., foreach=False, fused=False, fixed_updates=UPDATES)


def fixture_checks():
    from review_optimizer import fixture_tests
    for minimum in (1e-4, .005, -.02):
        require(margin_component(minimum, minimum)['passed'] and
                not margin_component(np.nextafter(minimum, -math.inf), minimum)['passed'],
                'Inclusive decision boundary fixture differs')
    fusion_fixture = fusion_fixture_checks()
    return dict(fusion=fusion_fixture, mathematics=math_fixtures(), features=feature_fixtures(), io=io_fixtures(),
                development=data_fixtures(), optimizer=fixture_tests(), inclusive_gate_cases=6)


def fusion_fixture_checks():
    # Finite differences establish the retained manual output derivative independently.
    student = np.asarray([1., -1., 2., 99.], dtype=np.float64)
    teacher = np.asarray([-2., .3, 1., -99.], dtype=np.float64)
    mask = np.asarray([True, True, True, False])
    target = np.asarray([0., 1., 0., -1.])
    expected = np.zeros_like(student)
    expected[mask] = (2/(1+np.exp(-student[mask])) - 1/(1+np.exp(-teacher[mask])) -
                      (.25+.5*target[mask])) / mask.sum()
    for index in range(4):
        plus, minus = student.copy(), student.copy()
        plus[index] += 1e-5; minus[index] -= 1e-5
        actual = (manual_record(plus,teacher,mask,target)['full'] - manual_record(minus,teacher,mask,target)['full'])/2e-5
        require(abs(actual - expected[index]) <= 1e-8, 'Independent manual derivative finite difference differs')
    # A diagnostic difference cannot identify the feature-level mechanism.
    base = np.asarray([1e10, 2., -3.], dtype=np.float32)
    output = np.asarray([1., -1., 4.], dtype=np.float32)
    difference = output-base
    require(bit_equal(difference, np.asarray([-1e10,-3.,7.],dtype=np.float32)) and
            not bit_equal(base+difference, output), 'Diagnostic subtraction/composition distinction disappeared')
    receipt = dict(shape=[2,32,128,128], dtype='torch.float32', sha256='0'*64, l2=5., absolute_max=3.)
    check_tensor_receipt(receipt,[2,32,128,128])
    rejected=0
    for key, value in [('shape',[2,1,128,128]),('dtype','torch.float64'),('sha256','x'*64),('l2',float('nan')),('absolute_max',6.)]:
        bad=dict(receipt);bad[key]=value
        try:check_tensor_receipt(bad,[2,32,128,128])
        except ValueError:rejected+=1
        else:raise ValueError('Corrupt fusion receipt accepted: '+key)
    return dict(status='PASS', manual_derivative_finite_difference_coordinates=4,
                diagnostic_difference_not_composition_cases=1, rejected_fusion_receipt_cases=rejected,
                synthetic_only=True, scientific_array_reads=0)


def isolation(receipt):
    require(receipt['unchanged'] is True and receipt['before'] == receipt['after'], 'Measurement state/RNG isolation failed')
    require(set(receipt['before']) == {'adapter', 'optimizer', 'gradients', 'modes', 'torch_rng', 'numpy_rng', 'python_rng'},
            'Measurement isolation schema differs')


def check_plan(plan, pin, review, require_go):
    require(plan['kind'] == 'phase20-depth-fusion-v1' and plan['ready_for_execution'] is True and
            plan['arms'] == list(ARMS) and plan['training'] == TRAINING, 'Frozen study/training contract differs')
    require(plan['protocol']['sha256'] == PROTOCOL_SHA256, 'Root protocol pin differs')
    review.verify(plan['protocol'])
    require(plan['runtime'] == dict(python='3.12.14', numpy='2.5.3', torch='2.12.0+cpu'), 'Runtime contract differs')
    require(platform.python_version() == plan['runtime']['python'] and np.__version__ == plan['runtime']['numpy'],
            'Reviewer runtime differs')
    config = review.read_json(plan['configuration'])
    require(all(config[key] == plan[key] for key in ('architecture', 'training', 'limits', 'gates')),
            'Root configuration linkage differs')
    for item in [*plan['code'].values(), *plan['source_code'].values()]:
        review.verify(item)
    freeze = review.read_json(plan['code_freeze'])
    require(freeze['no_further_edits'] is True and freeze['code'] == plan['code'] and freeze['protocol'] == plan['protocol'],
            'Producer code freeze differs')
    require([case['id'] for case in plan['cases']] == CASES, 'Six-context development scope differs')
    for key, value in dict(training_improvement_min=1e-4, candidate_vs_control_full_loss_improvement_min=1e-4,
                           paris2_auc_improvement_min=.005, paris2_ap_improvement_min=.005,
                           maximum_per_case_rank_loss_vs_baseline=.02, fixed_threshold=.7558947205543518,
                           automatic_model_promotion=False, automatic_GPU_launch=False).items():
        require(plan['gates'][key] == value, 'Frozen screen configuration differs: ' + key)
    if require_go:
        go = review.read_json(record(plan['source_go_path']))
        require(go['status'] == 'GO' and go['plan'] == pin and go['protocol'] == plan['protocol'] and
                go['code_freeze'] == plan['code_freeze'], 'Root execution GO differs')


def check_losses(review, actual, expected, label):
    require(set(actual) == set(expected) == {'base', 'full', 'kl', 'hard_bce', 'probability_drift_manual'},
            'Manual objective schema differs')
    for key in expected:
        tolerance = 1e-12 if key in ('kl', 'probability_drift_manual') else 2e-7
        review.close(actual[key], expected[key], label + '/' + key, tolerance)
    review.close(actual['full'], actual['base'] + actual['kl'], label + '/component_arithmetic', 1e-12)


def check_reduction(review, reported, expected, label):
    require(len(reported['per_batch']) == len(expected) == 32, 'Fixed risk must contain all32 batches')
    for index, (actual, truth) in enumerate(zip(reported['per_batch'], expected)):
        check_losses(review, actual, truth, label + '/batch' + str(index + 1))
    require(reported['equal_batch_means'] == equal_batch_means(reported['per_batch']), 'Equal-batch reduction differs')
    independent = equal_batch_means(expected)
    check_losses(review, reported['equal_batch_means'], independent, label + '/mean')
    return independent


def check_tensor_receipt(receipt, shape):
    require(set(receipt) == {'shape','dtype','sha256','l2','absolute_max'} and
            receipt['shape'] == shape and receipt['dtype'] == 'torch.float32' and
            len(receipt['sha256']) == 64 and all(c in '0123456789abcdef' for c in receipt['sha256']) and
            math.isfinite(receipt['l2']) and math.isfinite(receipt['absolute_max']) and
            0 <= receipt['absolute_max'] <= receipt['l2'], 'Tensor execution receipt schema differs')


def check_measurement_features(receipt, steps):
    require([item['step'] for item in receipt['fusion_receipts']] == steps, 'Measurement feature receipt membership differs')
    for item in receipt['fusion_receipts']:
        check_tensor_receipt(item['feature_residual'], [2,32,128,128])
        check_tensor_receipt(item['adapted_features'], [2,32,128,128])


def check_backbone(receipt, counts, plan):
    require(receipt['state_sha256'] == plan['backbone_state_sha256'] and
            receipt['before'] == receipt['after'] and receipt['unchanged'] is True and
            receipt['checkpoint_loads'] == 1 and receipt['actual_calls'] == counts and
            receipt['updates'] == 0 and receipt['parameter_gradients_absent'] is True,
            'Original backbone state/count receipts differ')
    before = receipt['before']
    require(before['state_sha256'] == plan['backbone_state_sha256'] and
            before['trainable_parameters'] == before['gradient_parameters'] == [] and
            before['modes'] and all(value is False for value in before['modes'].values()),
            'Original backbone freeze/mode receipt differs')


def check_initialization(review, result, arrays, teacher, batches, plan):
    require(bit_equal(arrays['initial-original-logits'], teacher) and
            bit_equal(arrays['initial-zero-adapter-logits'], teacher),
            'All 32 original/zero-adapter outputs differ bitwise from cached teacher')
    initialization = review.read_json(result['initialization'])
    require(initialization['status'] == 'PASS' and initialization['all32_exact_original_teacher'] is True and
            initialization['all32_exact_zero_original'] is True and len(initialization['batches']) == 32,
            'All-input initialization receipt differs')
    for index, (item, batch) in enumerate(zip(initialization['batches'], batches)):
        require(item['step'] == index+1 and item['input_sha256'] == batch['input_sha256'] and
                item['original_state_sha256'] == plan['backbone_state_sha256'] and
                all(item[key] is True for key in ('original_modes_all_eval','ordinary_detached_tensor',
                    'original_teacher_bitwise_equal','zero_adapter_original_bitwise_equal','zero_residual',
                    'zero_adapted_features_bitwise_equal')), 'Per-input initialization lineage differs')
        check_tensor_receipt(item['fused'], [2,32,128,128])
        require(item['teacher'] == item['original_logits'] == item['zero_adapter_logits'],
                'Initialization output tensor receipts differ')
        receipt = item['teacher']
        check_tensor_receipt(receipt, [2,1,128,128])
        value = teacher[index]
        require(receipt['sha256'] == hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
                'Initialization teacher tensor hash differs')
        review.close(receipt['l2'], float(np.sqrt(np.square(value.astype(np.float64)).sum())),
                     result['arm']+'/initialization/l2/'+str(index+1), 1e-12)
        require(receipt['absolute_max'] == float(np.max(np.abs(value))), 'Teacher absolute maximum differs')
    return initialization


def training_checks(plan, plan_pin, summary, summary_pin, review):
    from review_optimizer import verify_trace
    require(summary['status'] == 'PASS' and summary['phase'] == 'training' and summary['plan'] == plan_pin and
            summary['retries'] == 0 and summary['all_required_arms_retained'] is True and
            set(summary['results']) == {'training'} and set(summary['results']['training']) == set(ARMS),
            'Completed training controller schema/lineage differs')
    require(summary['actual_calls'] == dict(adapter_forwards=448, backwards=64, optimizer_steps=64, original_full_model=64, original_depth_fusion=64, original_2d_network=512, decoder_input_backwards=64), 'Training controller call accounting differs')
    require(summary['threshold_fits'] == 0 and summary['wall_seconds'] <= plan['limits']['training_controller_wall_seconds'], 'Training controller resource/scope differs')
    complete = review.read_json(summary['training_complete'])
    require(complete == dict(status='PASS', plan=plan_pin, training=summary['results']['training'],
                             matched_initial_adapter_and_rng=True, development_predictions=0), 'Training completion gate differs')
    batches, teacher = training_batches(plan, review)
    baseline_expected = [manual_record(teacher[i], teacher[i], batch['mask'], batch['target'])
                         for i, batch in enumerate(batches)]
    require(all(row['kl'] == row['probability_drift_manual'] == 0. for row in baseline_expected),
            'Cached initializer KL/drift is not zero')
    output, raw_results = {}, {}
    for arm in ARMS:
        result_pin = summary['results']['training'][arm]
        result = review.read_json(result_pin)
        raw_results[arm] = result
        require(result['status'] == 'PASS' and result['phase'] == 'training' and result['arm'] == arm and
                result['plan'] == plan_pin and result['teacher_caches'] == plan['teacher_caches'], 'Training worker lineage differs')
        require(result['actual_calls'] == dict(adapter_forwards=224, backwards=32, optimizer_steps=32, original_full_model=32, original_depth_fusion=32, original_2d_network=256, decoder_input_backwards=32),
                'Training worker call accounting differs')
        check_backbone(result['backbone'], result['actual_calls'], plan)
        require(result['paris_training_reads'] == 0 and result['pseudo_data_use'] ==
                'PHerc0139 historical identity verification only; no pseudo loss or teacher term', 'Training leakage receipt differs')
        require(result['initialization_proof'] == dict(all32_actual_original_teacher_bitwise_equal=True, all32_zero_adapter_original_bitwise_equal=True, all32_feature_residual_zero=True, additional_adapter_initialization_forwards=32), 'Zero-head initialization proof differs')
        require(result['checkpoint_reload']['weights_only'] is True and result['checkpoint_reload']['strict'] is True and
                result['checkpoint_reload']['finite'] is True and result['checkpoint_reload']['parameter_count'] == 5296 and
                result['checkpoint_reload']['state_sha256'] == result['final_adapter_state_sha256'] and
                result['initial_adapter_state_sha256'] != result['final_adapter_state_sha256'], 'Final adapter reload receipt differs')
        require(result['peak_rss_bytes'] <= plan['limits']['rss_bytes'] and result['wall_seconds'] <= plan['limits']['training_wall_seconds'],
                'Training worker resource acceptance differs')
        logs = review.read_jsonl(result['steps'])
        require([row['step'] for row in logs] == list(range(1, 33)), 'Training step records incomplete')
        names = {'logit-gradients', 'initial-original-logits', 'initial-zero-adapter-logits'} | {stage + '-' + kind for stage in ('preupdate', 'postupdate', 'fixed-risk')
                                     for kind in ('logits', 'logit-differences')}
        require(set(result['arrays']) == names, 'Retained training array schema differs')
        arrays = {name: review.array(pin, shape=(4,32,2,1,128,128) if name.startswith('fixed-risk') else
                                    (32,2,1,128,128), dtype=np.float32, finite=True)
                  for name, pin in result['arrays'].items()}
        for stage in ('preupdate', 'postupdate', 'fixed-risk'):
            reference = teacher[None] if stage == 'fixed-risk' else teacher
            require(bit_equal(arrays[stage+'-logit-differences'], arrays[stage+'-logits'] - reference),
                    'Diagnostic student-minus-teacher difference differs: ' + arm + '/' + stage)
        require(bit_equal(arrays['preupdate-logits'][0], teacher[0]) and
                (arrays['preupdate-logit-differences'][0] == 0).all(), 'Executed first-step initialization identity differs')
        initialization = check_initialization(review, result, arrays, teacher, batches, plan)
        baseline = check_reduction(review, result['baseline'], baseline_expected, arm + '/baseline')
        derivative_error = 0.
        for i, (batch, row) in enumerate(zip(batches, logs)):
            for key in ('step', 'patches', 'exposures', 'input_target_sha256', 'input_sha256', 'manual_sha256'):
                require(row[key] == batch[key], 'Independent batch identity differs: ' + key)
            require(row['feature_sha256'] == batch['feature_sha256'][arm], 'Independent feature fingerprint differs')
            for stage, field in [('preupdate', 'pre'), ('postupdate', 'post')]:
                truth = manual_record(arrays[stage+'-logits'][i], teacher[i], batch['mask'], batch['target'])
                check_losses(review, row[field], truth, arm + '/' + stage + '/' + str(i+1))
            mask = batch['mask']
            student_probability = np.exp(-np.logaddexp(0., -arrays['preupdate-logits'][i][mask].astype(np.float64)))
            teacher_probability = np.exp(-np.logaddexp(0., -teacher[i][mask].astype(np.float64)))
            expected_gradient = (2 * student_probability - teacher_probability - (.25 + .5 * batch['target'][mask])) / int(mask.sum())
            actual_gradient = arrays['logit-gradients'][i]
            require((actual_gradient[~mask] == 0).all() and row['unknown_output_gradient_max'] == 0.,
                    'Nonmanual outputs contributed gradient')
            error = float(np.max(np.abs(actual_gradient[mask].astype(np.float64) - expected_gradient)))
            require(error <= 2e-10, 'Supported manual BCE+KL logit derivative differs')
            derivative_error = max(derivative_error, error)
            require(row['frozen_cache_sha256'] == initialization['batches'][i]['fused']['sha256'], 'Fused-cache step identity differs')
            check_tensor_receipt(row['feature_residual'], [2,32,128,128])
            check_tensor_receipt(row['adapted_features'], [2,32,128,128])
            check_tensor_receipt(row['decoder_input_gradient'], [2,32,128,128])
            require(row['decoder_input_gradient']['l2'] > 0, 'Decoder input gradient is zero')
            if i == 0:
                require(row['feature_residual']['l2'] == row['feature_residual']['absolute_max'] == 0, 'Initial feature residual not zero')
            isolation(row['measurement_isolation'])
            check_measurement_features(row['measurement_isolation'], [i+1])
            require(row['measurement_isolation']['no_grad'] is True, 'Immediate measurement was not no-grad')
        require(set(result['fixed_evaluations']) == {str(step) for step in UPDATES}, 'Fixed checkpoint schedule differs')
        fixed = {}
        for position, update in enumerate(UPDATES):
            receipt = result['fixed_evaluations'][str(update)]
            require(receipt['current_batch_post_bitwise_equal'] is True and
                    bit_equal(arrays['fixed-risk-logits'][position,update-1], arrays['postupdate-logits'][update-1]) and
                    bit_equal(arrays['fixed-risk-logit-differences'][position,update-1], arrays['postupdate-logit-differences'][update-1]),
                    'Immediate/current fixed-risk outputs differ')
            isolation(receipt['isolation'])
            check_measurement_features(receipt['isolation'], list(range(1,33)))
            require(receipt['isolation']['no_grad'] is True, 'Fixed measurement was not no-grad')
            expected = [manual_record(arrays['fixed-risk-logits'][position,i], teacher[i], batch['mask'], batch['target'])
                        for i, batch in enumerate(batches)]
            values = check_reduction(review, receipt, expected, arm + '/fixed' + str(update))
            fixed[str(update)] = dict(equal_batch_means=values,
                                     deltas_vs_initializer={key: values[key] - baseline[key] for key in values},
                                     producer_equal_batch_means=receipt['equal_batch_means'])
        sentinel = review.read_json(result['first_step_sentinel'])
        require(sentinel['initial_adapter_state_sha256'] == result['initial_adapter_state_sha256'] and
                sentinel['initial_rng_sha256'] == result['initial_rng_sha256'] and
                sentinel['baseline_logits_bitwise_equal'] is True and sentinel['initial_residual_all_zero'] is True and
                sentinel['initial_stem_gradient_l2'] == 0. and sentinel['initial_head_gradient_l2'] > 0 and
                logs[1]['stem_gradient_l2'] > 0, 'First-step gradient path sentinel differs')
        optimizer = verify_trace(review, result, logs)
        output[arm] = dict(training_result=result_pin, baseline=baseline, producer_baseline=result['baseline']['equal_batch_means'],
                           all_fixed_checkpoints=fixed, optimizer_verification=optimizer,
                           logit_gradient_max_absolute_error=derivative_error,
                           final_checkpoint_metadata_only=result['checkpoint'], checkpoint_decoded=False, fused_cache_receipts=initialization,
                           original_backbone_execution_receipt=result['backbone'])
    require(output[ARMS[0]]['fused_cache_receipts'] == output[ARMS[1]]['fused_cache_receipts'], 'Matched original fused cache receipts differ')
    first, second = [raw_results[arm] for arm in ARMS]
    require(first['initial_adapter_state_sha256'] == second['initial_adapter_state_sha256'] and
            first['initial_rng_sha256'] == second['initial_rng_sha256'], 'Initial state/RNG differs across matched arms')
    require(output[ARMS[0]]['optimizer_verification']['initial_parameters_sha256'] ==
            output[ARMS[1]]['optimizer_verification']['initial_parameters_sha256'], 'Independent initial parameter vectors differ')
    return dict(status='PASS', training_results=summary_pin, arms=output,
                all32_batch_and_feature_fingerprints_independently_reconstructed=True,
                all_loss_and_logit_gradient_arithmetic_independently_reconstructed=True,
                convolution_autograd_replayed=False, reviewer_backbone_loads=0, reviewer_backbone_forwards=0,
                producer_backbone_checkpoint_loads=2, producer_actual_module_calls=summary['actual_calls'],
                pseudo_arrays_used_for_historical_identity_only=True)


def margin_component(value, minimum):
    return dict(value=float(value), required_minimum=float(minimum), margin=float(value-minimum), passed=bool(value >= minimum))


def screen(training, cases, use_producer):
    arms = training['arms']
    def risk(arm, field):
        record = arms[arm]['all_fixed_checkpoints']['32']
        return record['producer_equal_batch_means' if use_producer else 'equal_batch_means'][field]
    baseline = arms['explicit_depth']['producer_baseline' if use_producer else 'baseline']
    components = dict(candidate_base_improvement=margin_component(baseline['base']-risk('explicit_depth','base'), 1e-4),
                      candidate_full_improvement=margin_component(baseline['full']-risk('explicit_depth','full'), 1e-4),
                      candidate_full_advantage_over_raw=margin_component(risk('raw_control','full')-risk('explicit_depth','full'), 1e-4))
    macro = {}
    for model in ('unchanged_seed42', *ARMS):
        fragments = {}
        for fragment in ('PHercParis2Fr47', 'PHercParis2Fr143'):
            selected = [row for row in cases if row['fragment'] == fragment]
            require(len(selected) == 2, 'Paris2 fragment/context membership differs')
            fragments[fragment] = {metric: math.fsum(row['models'][model]['rank'][metric] for row in selected)/2
                                   for metric in ('auc', 'ap')}
        macro[model] = dict(fragments=fragments, equal_fragment_mean={metric: math.fsum(row[metric] for row in fragments.values())/2
                                                                    for metric in ('auc', 'ap')})
    for metric in ('auc', 'ap'):
        components['paris2_macro_' + metric + '_advantage'] = margin_component(
            macro['explicit_depth']['equal_fragment_mean'][metric] - macro['raw_control']['equal_fragment_mean'][metric], .005)
    for row in cases:
        for metric in ('auc', 'ap'):
            value = row['models']['explicit_depth']['rank'][metric] - row['models']['unchanged_seed42']['rank'][metric]
            components[row['case'] + '_' + metric + '_retention'] = margin_component(value, -.02)
    return dict(passed=all(row['passed'] for row in components.values()), components=components, paris2_macro=macro,
                paris1_separate_contexts=[row['case'] for row in cases if row['parent'] == 'Paris1'],
                scope='Computational development screen only; no confirmation, promotion, or GPU authorization')


def development_checks(plan, plan_pin, summary, summary_pin, training, training_pin, review):
    require(summary['status'] == 'PASS' and summary['phase'] == 'development' and summary['plan'] == plan_pin and
            summary['training_verification'] == training_pin and summary['training_results'] == training['training_results'] and
            summary['all_six_contexts_both_final_arms_retained'] is True and summary['retries'] == 0,
            'Development controller lineage differs')
    require(summary['actual_calls'] == dict(adapter_forwards=300, backwards=0, optimizer_steps=0, original_full_model=300, original_depth_fusion=300, original_2d_network=600, decoder_input_backwards=0), 'Development calls differ')
    require(summary['wall_seconds'] <= plan['limits']['development_controller_wall_seconds'], 'Development controller wall limit exceeded')
    results = {arm: review.read_json(summary['results']['development'][arm]) for arm in ARMS}
    for arm, result in results.items():
        require(result['status'] == 'PASS' and result['arm'] == arm and result['plan'] == plan_pin and
                result['training_result'] == training['arms'][arm]['training_result'] and set(result['runs']) == set(CASES),
                'Development worker scope/lineage differs')
        require(result['actual_calls'] == dict(adapter_forwards=150, backwards=0, optimizer_steps=0, original_full_model=150, original_depth_fusion=150, original_2d_network=300, decoder_input_backwards=0), 'Development worker calls differ')
        check_backbone(result['backbone'], result['actual_calls'], plan)
        require(result['threshold_fits'] == 0 and summary['threshold_fits'] == 0, 'Undeclared cutoff fitting')
        require(result['checkpoint_reload']['state_sha256'] == training['arms'][arm]['optimizer_verification']['final_semantic_state_sha256'] and
                result['checkpoint_reload']['parameter_count'] == 5296 and
                result['checkpoint_reload']['weights_only'] is result['checkpoint_reload']['strict'] is result['checkpoint_reload']['finite'] is True,
                'Development final-adapter reload identity differs')
        isolation(result['isolation'])
        require(result['wall_seconds'] <= plan['limits']['development_wall_seconds'] and
                result['peak_rss_bytes'] <= plan['limits']['rss_bytes'], 'Development resource acceptance differs')
    source_plan = review.read_json(plan['inference_base_plan'])
    require(source_plan['initializer'] == plan['backbone_checkpoint_lineage'] and
            source_plan['models']['frozen_seed42']['state_sha256'] == plan['backbone_state_sha256'],
            'Development historical initializer lineage differs')
    for case in plan['cases']:
        original = next(row for row in source_plan['cases'] if row['id'] == case['id'])
        require(all(case[key] == value for key, value in original.items()), 'Development source case differs')
        baseline_result = review.read_json(case['baseline_result'])
        baseline_metadata = review.read_json(case['baseline_metadata'])
        require(baseline_result['status'] == 'PASS' and baseline_result['role'] == 'frozen_seed42' and
                baseline_result['state_sha256'] == plan['backbone_state_sha256'] and
                baseline_result['plan'] == plan['inference_base_plan'] and baseline_result['runs'][case['id']] == baseline_metadata,
                'Development cached tile lineage differs')
    for case in plan['cases']:
        fused_hashes = {}
        for arm in ARMS:
            item = results[arm]['runs'][case['id']]
            require(len(item['fusion_receipts']) == 25, 'Development fusion receipt membership differs')
            fused_hashes[arm] = []
            for receipt in item['fusion_receipts']:
                require(receipt['original_teacher_bitwise_equal'] is True and receipt['original_state_sha256'] == plan['backbone_state_sha256'],
                        'Development original-model lineage differs')
                for field in ('fused', 'feature_residual', 'adapted_features'):
                    check_tensor_receipt(receipt[field], [1,32,128,128])
                fused_hashes[arm].append(receipt['fused'])
        require(fused_hashes[ARMS[0]] == fused_hashes[ARMS[1]], 'Development original fused receipts differ across arms')
    cases = [development_case(case, {arm: results[arm]['runs'][case['id']] for arm in ARMS}, review)
             for case in plan['cases']]
    declared_screen, independent_screen = screen(training, cases, True), screen(training, cases, False)
    ambiguity = [name for name in declared_screen['components'] if declared_screen['components'][name]['passed'] !=
                 independent_screen['components'][name]['passed']]
    return dict(status='PASS', development_results=summary_pin, training_verification=training_pin,
                cases=cases, declared_scalar_screen=declared_screen, independently_reconstructed_screen=independent_screen,
                numerical_margin_ambiguity_components=ambiguity,
                rank_metric_status='Independently calculated directly from retained published score maps; producer-summary comparison is separate',
                development_only=True, fresh_confirmation_available=False, threshold_fits=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--stage', choices=['training', 'development'])
    parser.add_argument('--authorized-retained-read', action='store_true')
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--plan-sha256')
    parser.add_argument('--summary', type=Path)
    parser.add_argument('--summary-sha256')
    parser.add_argument('--training-verification', type=Path)
    parser.add_argument('--training-verification-sha256')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    signal.alarm(600)
    resource.setrlimit(resource.RLIMIT_CPU, (1200, 1200))
    resource.setrlimit(resource.RLIMIT_AS, (12*1024**3, 12*1024**3))
    fixture_receipt = fixture_checks()
    source_pins = {path.name: record(path) for path in sorted(Path(__file__).parent.glob('*.py'))}
    numerical_contract = record(Path(__file__).parent/'NUMERICAL_CONTRACT.md')
    review = Review()
    try:
        if args.self_test:
            result = dict(status='PASS', fixtures=fixture_receipt, scientific_array_reads=0)
        else:
            require(args.plan is not None and args.plan_sha256 is not None, 'Frozen plan pin required')
            plan_pin = record(args.plan)
            require(plan_pin['sha256'] == args.plan_sha256, 'Root plan SHA256 differs')
            plan = review.read_json(plan_pin)
            check_plan(plan, plan_pin, review, require_go=not args.preflight)
            if args.preflight:
                result = dict(status='READY', plan=plan_pin, fixtures=fixture_receipt,
                              scientific_array_reads=0, scope='Code, schema and mathematical fixture readiness; no training result yet')
            else:
                require(args.authorized_retained_read and args.stage is not None and args.summary is not None and
                        args.summary_sha256 is not None, 'Root retained-output authorization and summary pin required')
                summary_pin = record(args.summary)
                require(summary_pin['sha256'] == args.summary_sha256, 'Root summary SHA256 differs')
                summary = review.read_json(summary_pin)
                if args.stage == 'training':
                    result = training_checks(plan, plan_pin, summary, summary_pin, review)
                else:
                    require(args.training_verification is not None and args.training_verification_sha256 is not None,
                            'Completed independent training verification required')
                    training_pin = record(args.training_verification)
                    require(training_pin['sha256'] == args.training_verification_sha256, 'Training verification pin differs')
                    training = review.read_json(training_pin)
                    require(training['status'] == 'PASS', 'Training verification did not pass')
                    result = development_checks(plan, plan_pin, summary, summary_pin, training, training_pin, review)
                result['plan'] = plan_pin
        result.update(code=source_pins, numerical_contract=numerical_contract,
                      integrity=review.finish(), finished_utc=datetime.now(timezone.utc).isoformat(),
                      peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024)
        target = args.output_dir/'verification.json'
        target.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n')
        print(json.dumps(dict(status=result['status'], receipt=record(target)), sort_keys=True))
    except Exception as error:
        failure = dict(status='STOP', exception_type=type(error).__name__, message=str(error), code=source_pins,
                       numerical_contract=numerical_contract, input_pins=list(review.pins.values()), array_reads=review.array_reads)
        (args.output_dir/'failure.json').write_text(json.dumps(failure, indent=2, sort_keys=True)+'\n')
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    main()
