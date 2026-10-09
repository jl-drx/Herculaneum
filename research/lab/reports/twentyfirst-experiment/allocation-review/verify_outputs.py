"""Independent phase21-allocation saved-output verification without Torch or model execution."""
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
from review_allocation import FIELDS, allocation_record, allocation_derivative, diagnostic_difference, fixture_checks as allocation_fixtures
from review_data import development_case, fixture_checks as data_fixtures, training_batches

PROTOCOL_SHA256 = '498bf89ca0944ca26f596bdc8041951a82365667ccddf4eb88a8a1f0b33998b1'
UPDATES = [1, 8, 16, 32]
CASES = ['frag3-context1', 'frag3-context2', 'frag1-context1', 'frag1-context2', 'frag2-context1', 'frag2-context2']
TRAINING = dict(steps=32, batch_size=2, seed=20261008, learning_rate=.001, optimizer='AdamW',
                betas=[.9, .999], eps=1e-8, weight_decay=0, gradient_clip_norm=1.,
                loss_smoothing=.5, kl_coefficients={'manual_double':[2.,0.],'manual_retired':[1.,1.]}, coefficient_order=['manual','retired_validation'], foreach=False, fused=False, fixed_updates=UPDATES)


def fixture_checks():
    from review_optimizer import fixture_tests
    for minimum in (0.,1e-4,.005,-.02):
        require(margin_component(minimum, minimum)['passed'] and
                not margin_component(np.nextafter(minimum, -math.inf), minimum)['passed'],
                'Inclusive decision boundary fixture differs')
    fusion_fixture = fusion_fixture_checks()
    return dict(allocation=allocation_fixtures(),fusion=fusion_fixture, mathematics=math_fixtures(), features=feature_fixtures(), io=io_fixtures(),
                development=data_fixtures(), optimizer=fixture_tests(), inclusive_gate_cases=8)


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
    require(plan['kind'] == 'phase21-preservation-allocation-v1' and plan['ready_for_execution'] is True and
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
    expected_gates = dict(mechanism=dict(own_full_improvement_min=1e-4,candidate_manual_BCE_improvement_min=1e-4,
        candidate_retired_KL_reduction_vs_control_min=1e-4,candidate_manual_BCE_excess_vs_control_max=.001),
        practical=dict(candidate_paris2_auc_gain_vs_seed42_min=.005,candidate_paris2_ap_gain_vs_seed42_min=.005,
                       candidate_maximum_per_case_rank_loss_vs_seed42=.02),
        fixed_threshold=.7558947205543518,automatic_model_promotion=False,automatic_GPU_launch=False)
    for key,value in expected_gates.items():require(plan['gates'][key]==value,'Frozen allocation screen differs: '+key)
    require(plan['architecture']['fixed_feature_mode']=='explicit_depth' and plan['architecture']['trainable_parameters']==5296,
            'Matched explicit-depth architecture differs')
    support_pins = dict(preservation_support='762cb1e7ae7649d042358f80fea52712ce6471ebd5d4c4aaafc8f7a547694a8f',
        support_source_seal='0caeb2f1d60ac644951db8728ae5a0f422f6af2bce43c6be34eb06e1145be96c',
        independent_support_summary='5dc755a577fbfaa0594392c4880e9adbdc214601179471e6eedfde0703954fae',
        independent_support_seal='fb945b48e9693c5969df110fddfbd74112cb88316eb944a4c0a72a61749d18c4',
        prior_fusion_plan='58bb5c64ad64db19995c87c84d1e6a1ce6585d380afeaa7ef347ac82b138d78a',
        historical_phase20_metrics='60bb52efcff810830543941e5081f61e9570cb598a772b97c4c2633a1cce0be6')
    for key,digest in support_pins.items():
        require(plan[key]['sha256']==digest,'Frozen preservation support anchor differs: '+key)
        review.verify(plan[key])
    prior=review.read_json(plan['prior_fusion_plan'])
    for key in ('training_base_plan','inference_base_plan','schedule','backbone_state_sha256','backbone_checkpoint_lineage',
                'teacher_caches','historical_step_logs','historical_training_results','checkpoint'):
        require(plan[key]==prior[key],'Historical phase20 lineage differs: '+key)
    if require_go:
        go = review.read_json(record(plan['source_go_path']))
        require(go['status'] == 'GO' and go['plan'] == pin and go['protocol'] == plan['protocol'] and
                go['code_freeze'] == plan['code_freeze'], 'Root execution GO differs')


def check_losses(review, actual, expected, label, arm):
    require(set(actual) == set(expected) == FIELDS,
            'Manual objective schema differs')
    for key in expected:
        tolerance = 1e-12 if key in ('kl_manual','kl_retired','probability_drift_manual','probability_drift_retired') else 2e-7
        review.close(actual[key], expected[key], label + '/' + key, tolerance)
    review.close(actual['objective_A'],actual['base']+2*actual['kl_manual'],label+'/objective_A_arithmetic',1e-12)
    review.close(actual['objective_B'],actual['base']+actual['kl_manual']+actual['kl_retired'],label+'/objective_B_arithmetic',1e-12)
    require(actual['full']==actual['objective_A' if arm=='manual_double' else 'objective_B'],'Own/common objective field differs')


def check_reduction(review, reported, expected, label, arm):
    require(len(reported['per_batch']) == len(expected) == 32, 'Fixed risk must contain all32 batches')
    for index, (actual, truth) in enumerate(zip(reported['per_batch'], expected)):
        check_losses(review, actual, truth, label + '/batch' + str(index + 1), arm)
    require(reported['equal_batch_means'] == equal_batch_means(reported['per_batch']), 'Equal-batch reduction differs')
    independent = equal_batch_means(expected)
    check_losses(review, reported['equal_batch_means'], independent, label + '/mean', arm)
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


def first_step_checks(review,summary,results):
    reported=review.read_json(summary['first_step_comparison'])
    require(reported['status']=='DESCRIPTIVE' and reported['training_results']==summary['results']['training'] and
            reported['matched_all32_input_target_feature_role_identities'] is True and
            reported['no_bitwise_first_update_requirement'] is True and reported['no_outcome_dependent_tolerance'] is True,
            'First-step diagnostic scope differs')
    values={}
    for arm in ARMS:
        result=results[arm];values[arm]={}
        pin=result['optimizer_traces']
        with np.load(review.verify(pin),allow_pickle=False) as archive:
            for name in ('parameters_after','gradients_before_clip','exp_avg_after','exp_avg_sq_after'):
                values[arm][name]=archive[name][0].copy();review.array_reads+=1
        for name in ('preupdate-logits','postupdate-logits','logit-gradients'):
            values[arm][name]=review.array(result['arrays'][name],shape=(32,2,1,128,128),dtype=np.float32,finite=True)[0].copy()
    diagnostics={}
    for name in values[ARMS[0]]:
        left,right=values[ARMS[0]][name],values[ARMS[1]][name]
        result=diagnostic_difference(left,right)
        delta=right.astype(np.float64)-left.astype(np.float64)
        result['l2_difference']=float(np.sqrt(np.square(delta).sum()))
        if name!='logit-gradients':
            item=reported['differences'][name]
            require(item['max_absolute_difference']==result['maximum_absolute_difference'] and
                    item['bitwise_equal']==result['bitwise_equal'],'Published first-step difference differs')
            review.close(item['l2_difference'],result['l2_difference'],'first_step/'+name+'/l2',1e-12)
        diagnostics[name]=result
    require(set(reported['differences'])==set(diagnostics)-{'logit-gradients'},'First-step diagnostic membership differs')
    return dict(receipt=summary['first_step_comparison'],differences=diagnostics,acceptance_threshold=None,
                scope='Diagnostic differences only; each arm is independently checked under the frozen optimizer contract')


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
    baseline_expected = [allocation_record(teacher[i],teacher[i],batch['mask'],batch['target'],batch['retired_mask'],ARMS[0])
                         for i, batch in enumerate(batches)]
    require(all(row['kl_manual'] == row['kl_retired'] == row['probability_drift_manual'] == row['probability_drift_retired'] == 0. for row in baseline_expected),
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
                'PHerc0139 historical identity verification only; no pseudo loss or pseudo teacher term', 'Training leakage receipt differs')
        require(result['preservation_support']==plan['preservation_support'] and result['fixed_feature_mode']=='explicit_depth' and
                result['objective_role']==dict(manual_coefficient=TRAINING['kl_coefficients'][arm][0],
                   retired_validation_coefficient=TRAINING['kl_coefficients'][arm][1],reduction='Each role separately averaged over both batch patches'),
                'Declared feature/support/objective role differs')
        require(result['retired_labels_use']=='Whole historical arrays may be read for identity; V labels do not participate in targets, objectives, teacher construction or selection',
                'Retired label-use receipt differs')
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
        baseline = check_reduction(review, result['baseline'], baseline_expected, arm + '/baseline', arm)
        derivative_error = 0.
        for i, (batch, row) in enumerate(zip(batches, logs)):
            for key in ('step', 'patches', 'exposures', 'input_target_sha256', 'input_sha256', 'manual_sha256', 'retired_sha256','participation_sha256','source_index_sha256','retired_pixels'):
                require(row[key] == batch[key], 'Independent batch identity differs: ' + key)
            require(row['feature_sha256'] == batch['feature_sha256'][arm], 'Independent feature fingerprint differs')
            for stage, field in [('preupdate', 'pre'), ('postupdate', 'post')]:
                truth = allocation_record(arrays[stage+'-logits'][i],teacher[i],batch['mask'],batch['target'],batch['retired_mask'],arm)
                check_losses(review, row[field], truth, arm + '/' + stage + '/' + str(i+1), arm)
            mask = batch['mask'] if arm=='manual_double' else batch['mask']|batch['retired_mask']
            expected_gradient = allocation_derivative(arrays['preupdate-logits'][i],teacher[i],batch['mask'],batch['target'],batch['retired_mask'],arm)
            actual_gradient = arrays['logit-gradients'][i]
            require((actual_gradient[~mask] == 0).all() and row['unknown_output_gradient_max'] == 0.,
                    'Unsupported output contributed gradient')
            error = float(np.max(np.abs(actual_gradient[mask].astype(np.float64) - expected_gradient[mask])))
            require(error <= 2e-10, 'Supported allocation output derivative differs')
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
            expected = [allocation_record(arrays['fixed-risk-logits'][position,i],teacher[i],batch['mask'],batch['target'],batch['retired_mask'],arm)
                        for i, batch in enumerate(batches)]
            values = check_reduction(review, receipt, expected, arm + '/fixed' + str(update), arm)
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
    differences=first_step_checks(review,summary,raw_results)
    return dict(status='PASS', training_results=summary_pin, arms=output,first_step_interarm_diagnostics=differences,
                all32_batch_and_feature_fingerprints_independently_reconstructed=True,
                all_loss_and_logit_gradient_arithmetic_independently_reconstructed=True,
                convolution_autograd_replayed=False, reviewer_backbone_loads=0, reviewer_backbone_forwards=0,
                producer_backbone_checkpoint_loads=2, producer_actual_module_calls=summary['actual_calls'],
                pseudo_arrays_used_for_historical_identity_only=True,retired_labels_excluded_from_objective=True,
                preservation_support=plan['preservation_support'],
                declared_mechanism_screen=mechanism_screen({'arms':output},True),
                independently_reconstructed_mechanism_screen=mechanism_screen({'arms':output},False))


def margin_component(value, minimum):
    return dict(value=float(value), required_minimum=float(minimum), margin=float(value-minimum), passed=bool(value >= minimum))


def mechanism_screen(training,use_producer):
    arms=training['arms']
    def final(arm):return arms[arm]['all_fixed_checkpoints']['32']['producer_equal_batch_means' if use_producer else 'equal_batch_means']
    def initial(arm):return arms[arm]['producer_baseline' if use_producer else 'baseline']
    components={arm+'_own_full_improvement':margin_component(initial(arm)['full']-final(arm)['full'],1e-4) for arm in ARMS}
    control,candidate=final('manual_double'),final('manual_retired')
    components['candidate_manual_BCE_improvement']=margin_component(initial('manual_retired')['base']-candidate['base'],1e-4)
    components['candidate_retired_KL_reduction_vs_control']=margin_component(control['kl_retired']-candidate['kl_retired'],1e-4)
    components['candidate_manual_BCE_excess_margin']=margin_component(control['base']+.001-candidate['base'],0.)
    require(len(components)==5,'Mechanism component count differs')
    return dict(passed=all(item['passed'] for item in components.values()),components=components)


def screen(training,cases,use_producer):
    macro={}
    for model in ('unchanged_seed42',*ARMS):
        fragments={}
        for fragment in ('PHercParis2Fr47','PHercParis2Fr143'):
            selected=[row for row in cases if row['fragment']==fragment]
            require(len(selected)==2,'Paris2 fragment/context membership differs')
            fragments[fragment]={metric:math.fsum(row['models'][model]['rank'][metric] for row in selected)/2 for metric in ('auc','ap')}
        macro[model]=dict(fragments=fragments,equal_fragment_mean={metric:math.fsum(row[metric] for row in fragments.values())/2 for metric in ('auc','ap')})
    practical={}
    for metric in ('auc','ap'):
        practical['Paris2_candidate_'+metric+'_gain_vs_seed42']=margin_component(
            macro['manual_retired']['equal_fragment_mean'][metric]-macro['unchanged_seed42']['equal_fragment_mean'][metric],.005)
    for row in cases:
        for metric in ('auc','ap'):
            practical[row['case']+'_'+metric+'_retention']=margin_component(
                row['models']['manual_retired']['rank'][metric]-row['models']['unchanged_seed42']['rank'][metric],-.02)
    require(len(practical)==14,'Practical component count differs')
    return dict(mechanism=mechanism_screen(training,use_producer),
                practical=dict(passed=all(item['passed'] for item in practical.values()),components=practical),
                paris2_macro=macro,paris1_separate_contexts=[row['case'] for row in cases if row['parent']=='Paris1'],
                scope='Empirical mechanism and reused development screens only; no confirmation, promotion or GPU authorization')


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
    ambiguity = [group+'/'+name for group in ('mechanism','practical') for name in declared_screen[group]['components']
                 if declared_screen[group]['components'][name]['passed'] != independent_screen[group]['components'][name]['passed']]
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
    source_pins = {str(path.relative_to(Path(__file__).parent)): record(path) for path in sorted(Path(__file__).parent.rglob('*.py'))}
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
