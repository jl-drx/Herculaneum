"""Read only source/text receipts; never construct or execute a model."""
from pathlib import Path
import hashlib
import json
import math

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
PLAN_SHA = '7130f486072df98df6a723d2afcb85089b11e7ff91b66afdfb05299228fda331'
WORKER_SHA = '698a7f804e6d7653fc1f9ced9ac866909b4fb1e8ad5315aba18a8b85766d7cc2'
PRIOR_SEAL_SHA = 'd4b9f384457250882b038ef00114f5dce035732d9625ea25cc2bc03e079069dc'


def pin(path):
    data = path.read_bytes()
    return dict(path=str(path), size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def main():
    checks = []

    def check(name, condition):
        assert condition, name
        checks.append(dict(name=name, passed=True))

    prior_path = HERE / 'retention-manifest.json'
    check('original_review_seal_unchanged', pin(prior_path)['sha256'] == PRIOR_SEAL_SHA)
    prior = json.loads(prior_path.read_text())
    for record in prior['records']:
        check('sealed_review_file_unchanged:' + Path(record['path']).name, pin(Path(record['path'])) == record)
    plan_path = PHASE / 'fusion-prototype/smoke-plan.json'
    check('execution_plan_exact', pin(plan_path)['sha256'] == PLAN_SHA)
    plan = json.loads(plan_path.read_text())
    for record in plan['records']:
        check('source_record_unchanged:' + record['path'], pin(Path(record['path'])) == record)
    run = PHASE / 'actual-backbone-smoke'
    start_path, controller_path, worker_path, log_path = [run / name for name in
        ('controller-start.json', 'controller-result.json', 'worker-result.json', 'worker.log')]
    start = json.loads(start_path.read_text())
    controller = json.loads(controller_path.read_text())
    worker = json.loads(worker_path.read_text())
    check('worker_external_identity', pin(worker_path)['sha256'] == WORKER_SHA)
    check('controller_binds_worker_receipt', controller['worker_result_sha256'] == WORKER_SHA)
    check('both_receipts_pass_without_stop_or_retry', controller['status'] == worker['status'] == 'PASS'
          and controller['exit_code'] == 0 and controller['stop_reason'] is None
          and controller['retries'] == start['retries'] == 0)
    check('start_worker_bind_reviewed_plan', start['execution_plan_sha256'] ==
          worker['execution_plan_sha256'] == PLAN_SHA)
    check('exact_pinned_worker_command', start['command'] == [
        '/workspace/.venvs/herculaneum/bin/python',
        str(PHASE / 'fusion-prototype/actual_backbone_smoke.py'), '--worker',
        '--output', str(run), '--expected-plan-sha256', PLAN_SHA])
    check('unchanged_strict_checkpoint_identity', worker['strict_checkpoint'] == plan['checkpoint'])
    check('exact_observed_call_schedule', worker['actual_calls'] == plan['expected_observed_calls'])
    check('one_pretrained_load', worker['pretrained_checkpoint_loads'] == 1)
    check('matched_initial_adapter_states', len(set(worker['initial_arm_state_sha256'].values())) == 1)
    check('original_backbone_state_unchanged', worker['backbone_state_before_sha256'] == worker['backbone_state_after_sha256'])
    check('no_named_buffers_in_loaded_model', worker['backbone_named_buffer_names'] == [])
    check('fixed_synthetic_input_identity_contract', worker['input']['generator_seed'] == 20261009
          and worker['input']['shape'] == [1, 1, 17, 128, 128]
          and worker['input']['kind'] == 'synthetic iid CPU FP32 Gaussian')
    check('synthetic_objective_contract', worker['objective']['kind'] == 'synthetic MSE to fixed spatial linspace(-1,1)')
    for arm in ('raw_control', 'explicit_depth'):
        observed = worker['observations'][arm]
        check(arm + '_exact_feature_and_logit_identity', observed['exact_initial_feature_identity']
              and observed['exact_initial_logit_identity'])
        check(arm + '_frozen_eval_unchanged_backbone', observed['all_backbone_modules_eval']
              and observed['backbone_gradients_absent'] and observed['unchanged_backbone_state'])
        check(arm + '_matched_capacity', observed['parameter_count'] == 5296)
        check(arm + '_first_projection_gradient', math.isfinite(observed['projection_gradient_norm'])
              and observed['projection_gradient_norm'] > 0)
        check(arm + '_first_stem_zero', observed['first_stem_gradient_norm'] == 0)
        check(arm + '_second_stem_gradient', math.isfinite(observed['second_stem_gradient_norm'])
              and observed['second_stem_gradient_norm'] > 0)
        check(arm + '_finite_synthetic_losses', all(math.isfinite(observed[key]) for key in
              ('fixed_synthetic_loss_before', 'fixed_synthetic_loss_after_one_step')))
    check('equal_initial_synthetic_losses', worker['observations']['raw_control']['fixed_synthetic_loss_before'] ==
          worker['observations']['explicit_depth']['fixed_synthetic_loss_before'])
    check('no_scientific_data_network_gpu_or_saved_weights', all(worker[key] == 0 for key in
        ('scientific_array_reads', 'annotation_reads', 'network_calls', 'gpu_calls', 'saved_model_checkpoints')))
    check('runtime_cpu_threads', worker['resources']['threads'] == 4 and worker['resources']['interop_threads'] == 1)
    check('wall_and_rss_limits', 0 < worker['resources']['wall_seconds'] <= controller['wall_seconds'] < 300
          and max(worker['resources']['peak_rss_bytes'], controller['peak_sampled_rss_bytes']) < 4 * 1024**3)
    check('retained_output_cap', sum(path.stat().st_size for path in run.rglob('*') if path.is_file()) < 64 * 1024**2)
    final_log_record = json.loads(log_path.read_text().splitlines()[-1])
    check('log_matches_structured_worker_result', final_log_record == {
        'status': worker['status'], 'actual_calls': worker['actual_calls'], 'resources': worker['resources']})
    result = dict(kind='phase19-independent-runtime-receipt-review-v1', status='PASS',
        check_count=len(checks), checks=checks,
        reviewed_receipts=[pin(path) for path in (start_path, controller_path, worker_path, log_path)],
        reviewed_plan=pin(plan_path), prior_review_seal=pin(prior_path),
        actual_smoke_calls=worker['actual_calls'], worker_resources=worker['resources'],
        controller_wall_seconds=controller['wall_seconds'],
        execution=dict(additional_pretrained_loads=0, additional_model_forwards=0,
                       scientific_array_reads=0, network_calls=0, gpu_calls=0),
        limitations=[
            'Receipt/source consistency review; actual pretrained execution was performed once by root, not rerun here.',
            'No independent numerical recomputation of real-model gradients or tensor equality from raw outputs.',
            'The actual model has no named buffers; the separate independent toy used BatchNorm buffers to test preservation.',
            'Synthetic loss changes are engineering observations and must not rank the arms for ink detection.',
            'Nested full-model/component forward counts overlap; they are not additive independent whole-model evaluations.'
        ])
    with (HERE / 'runtime-receipt-review-v1.json').open('x') as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(dict(status='PASS', check_count=len(checks), execution=result['execution'])))


if __name__ == '__main__':
    main()
