"""Independent retained-logit risk reconstruction; no model or driver imports."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys

for option in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[option] = '1'

import numpy as np
from numerical_kernels import (fixture_checks as math_fixtures, loss_record, mean, require)
from reconstruct_batches import build_batches, fixture_checks as batch_fixtures
from review_io import Review, record

ARMS = ['A_preserve', 'B_preserve']
UPDATES = [1, 8, 16, 32]
FP32_FIELDS = {'loss_U', 'loss_P', 'loss_R', 'base', 'full', 'hard_base'}
FORBIDDEN_SUFFIXES = {'.pth', '.pt', '.ckpt', '.safetensors', '.bin'}


def bit_equal(a, b):
    require(a.dtype == b.dtype == np.float32 and a.shape == b.shape, 'FP32 bit comparison contract changed')
    return np.array_equal(a.view(np.uint32), b.view(np.uint32))


def rows(review, pin):
    return [json.loads(line) for line in review.verify(pin).read_text().splitlines()]


def equal_batch_means(losses):
    require(len(losses) == 32, 'Exactly 32 batch losses required')
    return {key: math.fsum(row[key] for row in losses) / 32 for key in losses[0]}


def isolation_guard(value):
    require(value['before'] == value['after'] and value['rng_unchanged'] is True and
            value['no_grad'] is True and value['model_mode_unchanged'] is True,
            'Worker measurement-isolation receipt failed')
    require(set(value['before']) == {'model', 'optimizer', 'gradients', 'modes'},
            'Isolation receipt fields changed')
    for key in ('model', 'optimizer', 'gradients'):
        require(isinstance(value['before'][key], str) and len(value['before'][key]) == 64,
                'Invalid worker isolation hash')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--authorized-retained-read', action='store_true')
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--plan-sha256')
    parser.add_argument('--gate', type=Path)
    parser.add_argument('--gate-sha256')
    parser.add_argument('--summary', type=Path)
    parser.add_argument('--summary-sha256')
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    require('torch' not in sys.modules, 'Torch must not be imported')
    fixtures = {'mathematics': math_fixtures(), 'batch_reconstruction': batch_fixtures()}
    source_pins = {p.name: record(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if args.self_test:
        result = dict(status='PASS', fixtures=fixtures, code=source_pins,
                      actual_retained_array_reads=0, Torch_imports=0, network_calls=0)
        target = args.output_dir / 'synthetic-fixtures.json'
        target.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
        print(json.dumps(dict(status='PASS', receipt=record(target))))
        return
    require(args.authorized_retained_read, 'Root authorization for new retained arrays is required')
    require(all(x is not None for x in (args.plan, args.gate, args.summary, args.run_dir)),
            'Completed frozen plan/gate/summary/run paths required')
    review = Review()
    entry_pins = {name: record(getattr(args, name)) for name in ('plan', 'gate', 'summary')}
    for name, pin in entry_pins.items():
        require(pin['sha256'] == getattr(args, name + '_sha256'), 'Root pin differs: ' + name)
    plan, gate, summary = [review.read(entry_pins[name]) for name in ('plan', 'gate', 'summary')]
    require(plan['ready_for_execution'] is True and plan['kind'] == 'conservative-fixed-risk-continuation-v1',
            'Frozen ready plan required')
    require(plan['new_arms'] == ARMS and plan['measurement']['updates'] == UPDATES,
            'Arm or checkpoint scope changed')
    require(plan['logits_contract'] == dict(pre_post_shape=[32,2,1,128,128],
            fixed_risk_shape=[4,32,2,1,128,128], dtype='float32', fixed_risk_update_order=UPDATES),
            'Raw logit contract changed')
    require(gate['status'] == summary['status'] == 'PASS' and
            gate['plan'] == summary['plan'] == entry_pins['plan'] and summary['gate'] == entry_pins['gate'],
            'Completed analysis lineage changed')
    skipped_checkpoint_pins = []
    for item in plan['verified_inputs']:
        if Path(item['path']).suffix.lower() in FORBIDDEN_SUFFIXES:
            skipped_checkpoint_pins.append(item)
        else:
            review.verify(item)
    base_plan = review.read(plan['base_plan'])
    require(base_plan['ready_for_execution'] is True, 'Historical matched plan not ready')
    require(plan['training'] == dict(base_plan['training'], learning_rate=1e-6),
            'More than the declared learning rate changed')
    for key in ('adaptation', 'initializer', 'teacher', 'schedule', 'exposure', 'helper',
                'preservation', 'historical_initial_state_sha256'):
        require(plan[key] == base_plan[key], 'Matched source contract changed: ' + key)
    require(plan['gate']['minimum_improvement'] == 1e-4 and plan['gate']['base_and_full_required'] is True and
            plan['gate']['fixed_exposures'] == 32 and plan['gate']['delta_arithmetic'] ==
            'final equal-batch mean minus initializer equal-batch mean; inclusive <= -1e-4; no gate tolerance' and
            gate['gate_contract'] == plan['gate'], 'Gate contract changed')
    require(plan['measurement']['reduction'] ==
            'math.fsum of32 logged per-batch scalars /32; BCE FP32; full FP32 base + FP64 KL; deltas final minus initializer',
            'Risk reduction/dtype contract changed')
    schedule = review.read(plan['schedule'])
    batches = build_batches(plan, schedule, review.array)
    batch_hashes = [b['input_target_sha256'] for b in batches]
    histories = {arm: rows(review, plan['historical'][arm]['steps']) for arm in ARMS}
    old_results = {arm: review.read(plan['historical'][arm]['result']) for arm in ARMS}
    teachers = {arm: review.array(plan['historical'][arm]['teacher'],
                    shape=(32,2,1,128,128), dtype=np.float32, finite=True) for arm in ARMS}
    require(bit_equal(teachers[ARMS[0]], teachers[ARMS[1]]), 'Historical teacher cache bits differ')
    for arm in ARMS:
        old = old_results[arm]
        require(old['status'] == 'PASS' and old['arm'] == arm and old['plan'] == plan['base_plan'] and
                old['initial_state_sha256'] == plan['historical_initial_state_sha256'], 'Historical source lineage changed')
        require(old['training']['preupdate_logits']['teacher'] == plan['historical'][arm]['teacher'] and
                old['training']['steps_log'] == plan['historical'][arm]['steps'], 'Teacher cache/log linkage changed')
        require(len(histories[arm]) == 32, 'Historical batch receipts incomplete')
        for batch, historical in zip(batches, histories[arm]):
            for key in ('step', 'patches', 'exposures', 'input_target_sha256'):
                require(historical[key] == batch[key], 'Independently rebuilt historical batch differs: ' + key)
    arm_checks = {}
    total_calls = dict(student_forwards=0, teacher_forwards=0, backwards=0, optimizer_steps=0)
    def check_losses(actual, expected, label):
        require(set(actual) == set(expected), 'Loss schema changed: ' + label)
        for key, value in expected.items():
            review.close(actual[key], value, label + '/' + key, 2e-6 if key in FP32_FIELDS else 1e-12)
        require(actual['loss_D_M'] >= -1e-6 and actual['loss_D_P'] >= -1e-6, 'Negative KL receipt')
        # Independently check the logged arithmetic as well as raw-logit reconstruction.
        c = .25 if label.startswith('B_preserve') else 0.
        review.close(actual['base'], actual['loss_U'] + c * actual['loss_P'], label + '/FP32_base_arithmetic', 2e-6)
        review.close(actual['full'], actual['base'] + actual['loss_D_M'] + c * actual['loss_D_P'],
                     label + '/full_arithmetic', 1e-12)
    def check_reduction(actual, per_batch, expected, label):
        exact = equal_batch_means(per_batch)
        require(actual == exact, 'Worker equal-batch reduction differs: ' + label)
        independent = equal_batch_means(expected)
        check_losses(actual, independent, label)
        return independent
    for arm in ARMS:
        coefficient = 0. if arm == 'A_preserve' else .25
        result_pin = gate['arms'][arm]['training_result']
        require(Path(result_pin['path']) == args.run_dir.resolve() / arm / 'result.json', 'Worker result location changed')
        result = review.read(result_pin)
        require(result['status'] == 'PASS' and result['arm'] == arm and result['plan'] == entry_pins['plan'] and
                result['initial_state_sha256'] == plan['historical_initial_state_sha256'] and
                result['final_state_sha256'] != result['initial_state_sha256'], 'Completed worker lineage changed')
        require(gate['arms'][arm]['checkpoint'] == result['checkpoint'] and
                gate['arms'][arm]['final_state_sha256'] == result['final_state_sha256'], 'Gate/final-state metadata changed')
        skipped_checkpoint_pins.append(result['checkpoint'])
        require(result['teacher_cache'] == plan['historical'][arm]['teacher'] and
                result['batch_fingerprints'] == batch_hashes, 'Teacher/batch fingerprint contract changed')
        arrays = {name: review.array(pin, shape=(4,32,2,1,128,128) if name == 'fixed_risk' else
                  (32,2,1,128,128), dtype=np.float32, finite=True) for name, pin in result['raw_logits'].items()}
        require(set(arrays) == {'preupdate', 'postupdate', 'fixed_risk'}, 'Three raw logit files required')
        logs = rows(review, result['steps_log'])
        require([row['step'] for row in logs] == list(range(1,33)), 'All 32 update receipts required')
        teacher = teachers[arm]
        require(bit_equal(arrays['preupdate'][0], teacher[0]), 'First preupdate teacher identity differs')
        probe = plan['first_step_probe'][arm]
        expected_post = review.array(probe['post_logits'], shape=(2,1,128,128), dtype=np.float32, finite=True)
        require(bit_equal(arrays['postupdate'][0], expected_post), 'Twelfth first postupdate logits differ')
        baseline_report = result['baseline_fixed_evaluation']
        require(len(baseline_report['per_batch']) == 32 and baseline_report['scope'] ==
                'cache-derived initializer risk; no new model forward', 'Baseline scope changed')
        baseline_expected = []
        steps_expected = []
        for index, (batch, log) in enumerate(zip(batches, logs)):
            for key in ('step', 'patches', 'exposures', 'input_target_sha256'):
                require(log[key] == batch[key], 'Current batch identity/support changed: ' + key)
            baseline = loss_record(teacher[index], teacher[index], batch['arrays'], coefficient)
            require(baseline['loss_D_M'] == baseline['loss_D_P'] == baseline['probability_drift_manual'] ==
                    baseline['probability_drift_pseudo'] == 0., 'Initializer KL/drift must be exactly zero')
            check_losses(baseline_report['per_batch'][index], baseline, arm + '/baseline-batch')
            baseline_expected.append(baseline)
            pre = loss_record(arrays['preupdate'][index], teacher[index], batch['arrays'], coefficient)
            post = loss_record(arrays['postupdate'][index], teacher[index], batch['arrays'], coefficient)
            check_losses(log['pre'], pre, arm + '/pre')
            check_losses(log['post'], post, arm + '/post')
            changes = {key: post[key] - pre[key] for key in pre}
            reported_changes = {key: log['post'][key] - log['pre'][key] for key in pre}
            for key in changes:
                review.close(reported_changes[key], changes[key], arm + '/immediate/' + key,
                             2e-6 if key in FP32_FIELDS else 1e-12)
            require(math.isfinite(log['preclip_global_gradient_norm']) and log['preclip_global_gradient_norm'] > 0 and
                    math.isfinite(log['parameter_update_l2']) and log['parameter_update_l2'] > 0 and
                    log['unknown_output_gradient_max'] == 0., 'Gradient/update receipt failed')
            isolation_guard(log['immediate_measurement_isolation'])
            active = batch['arrays']['manual_mask'] | (batch['arrays']['pseudo_mask'] if coefficient else False)
            support_counts = {name: {label: sum(e[name][label] for e in batch['exposures'])
                                     for label in ('positive','negative')} for name in ('manual','pseudo')}
            steps_expected.append(dict(step=index+1,pre=pre,post=post,immediate_change=changes,
                support_counts=support_counts,unknown_active_output_count=int((~active).sum())))
        baseline_means = check_reduction(baseline_report['equal_batch_means'], baseline_report['per_batch'],
                                          baseline_expected, arm + '/baseline-fixed')
        sentinel = review.read(result['first_step_sentinel'])
        require(sentinel['initial_state_sha256'] == plan['historical_initial_state_sha256'] == probe['initial_state_sha256'] and
                sentinel['post_state_sha256'] == probe['post_state_sha256'] and
                sentinel['input_target_sha256'] == batch_hashes[0] and
                sentinel['historical_first_logits_bitwise_equal'] is True and sentinel['twelfth_post_logits_bitwise_equal'] is True,
                'First-step sentinel lineage failed')
        require(sentinel['pre'] == logs[0]['pre'] and sentinel['post'] == logs[0]['post'] and
                sentinel['gradient_norm'] == logs[0]['preclip_global_gradient_norm'], 'First-step sentinel values changed')
        review.close(sentinel['gradient_norm'], histories[arm][0]['preclip_global_gradient_norm'], arm + '/first_gradient', 1e-6)
        for new, old in [('base','original_objective'),('loss_U','loss_U'),('loss_P','loss_P')]:
            review.close(logs[0]['pre'][new], histories[arm][0][old], arm + '/first_' + new, 1e-7)
        fixed_checks = {}
        require(set(result['fixed_evaluation']) == {'1','8','16','32'}, 'Fixed evaluation scope changed')
        for position, update in enumerate(UPDATES):
            reported = result['fixed_evaluation'][str(update)]
            require(len(reported['per_batch']) == 32 and reported['current_batch_post_bitwise_equal'] is True,
                    'Fixed batch accounting failed')
            require(bit_equal(arrays['fixed_risk'][position,update-1], arrays['postupdate'][update-1]),
                    'Current fixed-risk logit bits differ from immediate postupdate')
            expected_losses = []
            for index, batch in enumerate(batches):
                expected = loss_record(arrays['fixed_risk'][position,index], teacher[index], batch['arrays'], coefficient)
                check_losses(reported['per_batch'][index], expected, arm + '/fixed-batch')
                expected_losses.append(expected)
            independent_means = check_reduction(reported['equal_batch_means'], reported['per_batch'],
                                                 expected_losses, arm + '/fixed-mean')
            isolation_guard(reported['isolation'])
            require(reported['state_sha256'] == reported['isolation']['after']['model'], 'Fixed-state receipt failed')
            summary_row = summary['arms'][arm]['all_fixed_checkpoints'][str(update)]
            require(summary_row['equal_batch_means'] == reported['equal_batch_means'] and
                    summary_row['current_batch_post_bitwise_equal'] is True, 'Summary fixed-risk linkage failed')
            delta = {k: reported['equal_batch_means'][k] - baseline_report['equal_batch_means'][k] for k in independent_means}
            require(summary_row['deltas_vs_cache_initializer'] == delta, 'Published delta arithmetic failed')
            independent_delta = {k: independent_means[k] - baseline_means[k] for k in independent_means}
            fixed_checks[str(update)] = dict(equal_batch_means=independent_means,
                deltas_vs_cache_initializer=independent_delta,per_batch=expected_losses,
                current_batch_post_bitwise_equal=True,worker_state_isolation_receipts_passed=True)
        final = result['fixed_evaluation']['32']['equal_batch_means']
        initial = baseline_report['equal_batch_means']
        raw_deltas = {key: final[key] - initial[key] for key in ('base','full')}
        exact_gate = raw_deltas['base'] <= -1e-4 and raw_deltas['full'] <= -1e-4
        independent_deltas = fixed_checks['32']['deltas_vs_cache_initializer']
        independent_gate = independent_deltas['base'] <= -1e-4 and independent_deltas['full'] <= -1e-4
        require(exact_gate == result['eligible'] == gate['arms'][arm]['eligible'], 'Exact reported gate arithmetic failed')
        require(exact_gate == independent_gate, 'Independent risk reconstruction changes gate decision; no tolerance allowed')
        expected_gate_values = dict(exposures=32,base_initial=initial['base'],base_final=final['base'],
            full_initial=initial['full'],full_final=final['full'],base_delta=raw_deltas['base'],full_delta=raw_deltas['full'])
        require(gate['arms'][arm]['fixed_evaluation'] == expected_gate_values, 'Gate mean-risk linkage failed')
        report = summary['arms'][arm]
        require(report['initializer'] == initial and report['training_result'] == result_pin,
                'Published initializer/result linkage failed')
        for key in ('base','full'):
            count = sum(row['post'][key] > row['pre'][key] for row in logs)
            require(report['immediate_' + key + '_increases'] == count, 'Published immediate-increase count differs')
        for key, value in report['online_preupdate_mean'].items():
            review.close(value, math.fsum(row['pre'][key] for row in logs)/32, arm + '/online_pre_mean/' + key)
        require(result['actual_calls'] == dict(student_forwards=192,teacher_forwards=0,backwards=32,optimizer_steps=32),
                'Worker call accounting failed')
        for key, value in result['actual_calls'].items(): total_calls[key] += value
        require(result['mode_guard'] == dict(no_running_buffers=True,no_active_stochastic_layers=True,
            measurement_mode='unchanged training mode'), 'Model mode-guard receipt failed')
        require(result['persisted_validation']['weights_only'] is True and
                result['persisted_validation']['exact_tensor_values'] is True and
                result['persisted_validation']['strict'] is True, 'Worker checkpoint validation receipt failed')
        require(result['threads'] == 4 and result['torch_version'] == '2.12.0+cpu' and
                result['peak_rss_bytes'] <= plan['limits']['rss_acceptance_bytes'] and
                result['wall_seconds'] <= plan['limits']['worker_wall_seconds'], 'Worker runtime receipt failed')
        require(result['fixed_evaluation']['32']['state_sha256'] == result['final_state_sha256'],
                'Final fixed-risk state/final checkpoint metadata differs')
        arm_checks[arm] = dict(status='PASS',training_result=result_pin,steps=steps_expected,
            baseline_equal_batch_means=baseline_means,fixed_evaluation=fixed_checks,
            gate=dict(exact_reported_deltas=raw_deltas,independent_FP64_deltas={k:independent_deltas[k] for k in ('base','full')},
                      eligible=exact_gate,arithmetic='final equal-batch mean minus initializer equal-batch mean <= -1e-4, inclusive, no gate tolerance'),
            first_pre_teacher_bitwise_equal=True,first_post_twelfth_bitwise_equal=True,
            cached_teacher_lineage_verified=True,all_input_target_fingerprints_reconstructed=True,
            reported_FP32_immediate_increases={k:sum(row['post'][k] > row['pre'][k] for row in logs) for k in ('base','full')},
            independent_FP64_immediate_increases={k:sum(row['immediate_change'][k] > 0 for row in steps_expected) for k in ('base','full')})
    require(gate['eligible_arms'] == [a for a in ARMS if arm_checks[a]['gate']['eligible']], 'Eligible-arm list differs')
    require(total_calls == dict(student_forwards=384,teacher_forwards=0,backwards=64,optimizer_steps=64),
            'Combined accounting failed')
    execution_pin = record(args.run_dir / 'execution.json')
    execution = review.read(execution_pin)
    require(execution['plan'] == entry_pins['plan'] and execution['retries'] == 0 and
            [w['arm'] for w in execution['workers']] == ARMS and all(w['exit_code'] == 0 for w in execution['workers']),
            'Controller accounting or retry receipt failed')
    for pin in source_pins.values(): review.verify(pin)
    review.finish()
    result = dict(status='PASS',completed_at_utc=datetime.now(timezone.utc).isoformat(),
        inputs=entry_pins,code=source_pins,synthetic_fixtures=fixtures,arm_checks=arm_checks,
        steps_verified=64,fixed_schedule_batch_evaluations_verified=256,baseline_cache_batches_verified=64,
        immediate_pre_post_loss_records_verified=128,scalar_comparisons=review.comparisons,
        maximum_absolute_errors=review.maximum_absolute_errors,total_worker_call_receipts=total_calls,
        source_pins_unchanged=True,verified_inputs_before_and_after=list(review.pins.values()),
        checkpoint_pins_excluded_from_reads=skipped_checkpoint_pins,
        independent_scope='Retained logits, supports, losses, risks and gates. Parameter/optimizer/RNG isolation, gradients and checkpoint validation remain worker-reported receipts, not independently executed by this reviewer.',
        actual_operations=dict(retained_NPY_JSON_reads=True,Torch_imports=0,model_calls=0,
                              checkpoint_reads=0,network_calls=0,new_predictions_or_training=0))
    target = args.output_dir / 'independent-verification.json'
    target.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps(dict(status='PASS',result=record(target),scalar_comparisons=review.comparisons)))


if __name__ == '__main__':
    main()
