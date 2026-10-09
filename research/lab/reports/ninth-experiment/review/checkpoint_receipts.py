"""Additive tensor-only adapter; call only after both ninth workers complete.

Import is pure. The immutable eighth helper supplies bounded loading, exact
factory schema/alias checks, and independent semantic hash byte contracts.
Historical finals are hash-checked and supported by their retained PASS proof;
only seed42 and the two new final states are deserialized, one at a time.
"""
from collections.abc import Mapping
import gc
import hashlib
import importlib.util
import json
from pathlib import Path

LAB = Path('/workspace/herculaneum-lab')
HERE = LAB / 'reports/ninth-experiment/review'
IMPLEMENTATION = HERE.parent / 'implementation'
EIGHTH = LAB / 'reports/eighth-experiment'
OLD_HELPER = EIGHTH / 'metric-review/checkpoint_receipts.py'
OLD_HELPER_SHA = '3556305cd2662b79ba87b1cf0a6f61161731e073dc5661ce0f8a14dfab2d705c'
OLD_HELPER_BYTES = 19165
OLD_REVIEW = EIGHTH / 'metric-review/result/independent-verification.json'
OLD_REVIEW_SHA = '9ec0ca5f0f0db7352afdaa2d3b31254ef7ed7a14794b81a20e3ad7789fe0207b'
OLD_REVIEW_BYTES = 196720
OLD_COMMON_SHA = '8d933dae02248d7b5fbbd26370c95fca53191bcff4e1dc51e984c40d8095b8cf'
HISTORICAL = ('frozen_seed42', 'A', 'B')
NEW = ('A_smooth', 'B_smooth')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _old_helper():
    _require(OLD_HELPER.stat().st_size == OLD_HELPER_BYTES
             and hashlib.sha256(OLD_HELPER.read_bytes()).hexdigest() == OLD_HELPER_SHA,
             'Immutable eighth checkpoint helper changed')
    spec = importlib.util.spec_from_file_location('ninth_immutable_eighth_checkpoint_receipts', OLD_HELPER)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


def _json(item, helper, receipts):
    _require(0 < item['size_bytes'] <= 8 * 1024**2, 'Bounded retained JSON required')
    return json.loads(helper._pin(item, receipts).read_text())


def verify_checkpoints(plan, workers):
    """Return JSON proof; workers contains the two new arms or all five roles.

    The caller owns worker result-file receipts. Historical result identities
    are bound by the new plan and immutable eighth independent review.
    No old final state is re-deserialized, and no model is run or strict-loaded.
    """
    _require(plan['kind'] == 'smoothing-continuation-v1' and plan['ready_for_execution'] is True
             and plan['new_arms'] == list(NEW), 'Ready fixed smoothing plan required')
    _require(isinstance(workers, Mapping) and set(workers) in [set(NEW), set(HISTORICAL + NEW)],
             'Supply two new workers or the merged five-role mapping')
    _require(all(workers[arm]['status'] == 'PASS' for arm in NEW), 'Both new workers must be complete')
    helper = _old_helper()
    receipts = {}
    helper._pin(dict(path=str(OLD_HELPER), size_bytes=OLD_HELPER_BYTES, sha256=OLD_HELPER_SHA), receipts)
    helper._pin(helper._record(Path(__file__)), receipts)
    plan_record = workers[NEW[0]]['plan']
    _require(_json(plan_record, helper, receipts) == plan
             and all(workers[arm]['plan'] == plan_record for arm in NEW), 'New worker plan differs')
    base = _json(plan['base_plan'], helper, receipts)
    _require(base['kind'] == 'matched-continuation-v1' and base['ready_for_execution'] is True,
             'Ready immutable eighth base required')
    for key in ['adaptation', 'development', 'initializer', 'teacher', 'schedule', 'exposure',
                'helper', 'metrics', 'cutoff', 'calibration']:
        _require(plan[key] == base[key], 'Matched base contract differs: ' + key)
    _require(plan['training'] == dict(base['training'], loss_smoothing=.5)
             and plan['limits'] == dict(base['limits'], output_bytes=400 * 1024**2)
             and plan['diagnostic'] == base['fresh'], 'Fixed smoothing or retained contexts differ')
    for name, item in plan['code'].items():
        _require(helper._pin(item, receipts) == IMPLEMENTATION / name, 'Unexpected ninth source path')
    _require('control_common.py' in plan['code'] and 'smoothing.py' in plan['code'],
             'Smoothing source contracts absent')
    common = base['code']['experiment_common.py']
    _require(helper._pin(common, receipts) == EIGHTH / 'implementation/experiment_common.py'
             and common['sha256'] == OLD_COMMON_SHA, 'Immutable semantic hash source differs')
    old_review = _json(dict(path=str(OLD_REVIEW), size_bytes=OLD_REVIEW_BYTES,
                            sha256=OLD_REVIEW_SHA), helper, receipts)
    old_tensors = old_review['checkpoint_review']
    _require(old_review['status'] == 'PASS' and old_tensors['status'] == 'PASS'
             and old_review['plan_sha256'] == plan['base_plan']['sha256']
             and old_tensors['all_four_initializer_fingerprints_equal'] is True,
             'Retained eighth independent PASS contract differs')
    historical_records = {item['path']: item for item in
                          old_review['verified_input_output_records_before_and_after']}
    _require(set(plan['historical_results']) == set(HISTORICAL), 'Historical roles differ')
    historical = {}
    reused = {}
    for arm in HISTORICAL:
        item = plan['historical_results'][arm]
        result = _json(item, helper, receipts)
        actual = receipts[str(Path(item['path']).resolve())]
        _require(historical_records.get(actual['path']) == actual,
                 'Historical result absent from immutable independent proof')
        _require(result['status'] == 'PASS' and result['arm'] == arm
                 and result['plan'] == plan['base_plan'] and result['initializer'] == plan['initializer']
                 and result['initial_state_sha256'] == plan['historical_initial_state_sha256'],
                 'Historical result contract differs')
        if arm in workers:
            _require(workers[arm] == result, 'Caller changed a historical worker dictionary')
        historical[arm] = result
        proof = dict(result_record=actual, raw_result_unchanged_since_eighth_review=True)
        if arm == 'frozen_seed42':
            _require(result['training'] is None, 'Historical frozen role was adapted')
            proof['checkpoint_role'] = 'Unchanged seed42 initializer, inspected below'
        else:
            training = result['training']
            old_final = old_tensors['adapted_checkpoints'][arm]
            checkpoint = training['checkpoint']
            _require(checkpoint == old_final['checkpoint'] and 0 < checkpoint['size_bytes'] <= helper.MAX_FILE_BYTES,
                     'Historical final checkpoint receipt differs')
            path = helper._pin(checkpoint, receipts)
            actual_checkpoint = receipts[str(path)]
            _require(historical_records.get(str(path)) == actual_checkpoint
                     and training['final_state_sha256'] == old_final['semantic_state_sha256']
                     and training['final_parameter_sha256'] == old_final['semantic_parameter_sha256']
                     and old_final['actual_parameters_changed'] is True,
                     'Historical final absent from retained tensor proof')
            proof.update(checkpoint_record=actual_checkpoint, raw_checkpoint_unchanged_since_eighth_review=True,
                         retained_semantic_state_sha256=old_final['semantic_state_sha256'],
                         retained_semantic_parameter_sha256=old_final['semantic_parameter_sha256'],
                         re_deserialized=False)
        reused[arm] = proof
    audited_json = _json(dict(path=str(helper.AUDIT), size_bytes=helper.AUDIT_BYTES,
                              sha256=helper.AUDIT_SHA), helper, receipts)
    _require(audited_json['status'] == 'passed' and audited_json['tensor_schema_identical_across_four_arms'],
             'Retained model audit failed')
    audited = audited_json['models']['seed42']
    initializer = plan['initializer']
    _require(initializer['label'] == 'seed42' and initializer['sha256'] == audited['sha256']
             and initializer['size_bytes'] == audited['size_bytes'], 'Audited seed42 initializer differs')
    for arm in NEW:
        result = workers[arm]
        _require(result['arm'] == arm and result['initializer'] == initializer
                 and result['base_plan'] == plan['base_plan'], 'New worker role/initializer differs')
    import torch
    torch.set_num_threads(4)
    _require(torch.get_num_threads() == 4, 'Four Torch threads required')
    schema, groups, parameters, buffers, parameter_elements = helper._factory_schema(plan, audited, receipts, torch)
    schema_hash = hashlib.sha256(json.dumps(schema, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    names_hash = hashlib.sha256(json.dumps(sorted(parameters), separators=(',', ':')).encode()).hexdigest()
    _require(schema_hash == old_tensors['factory_schema_sha256']
             and names_hash == old_tensors['canonical_parameter_names_sha256'],
             'Canonical factory schema/parameter identities differ from retained proof')
    payload, initializer_archive = helper._bounded_load(initializer, receipts, torch)
    state = helper._initializer_state(payload, audited)
    initial_storage = helper._check_state(state, schema, groups, audited, torch)
    initial_hash = helper._semantic_hash(state)
    initial_parameter_hash = helper._semantic_hash({name: state[name] for name in parameters})
    initial_tensor_hashes = {name: helper._tensor_hash(state[name]) for name in parameters}
    _require(initial_hash == old_tensors['initial_state_sha256'] == plan['historical_initial_state_sha256']
             and initial_parameter_hash == old_tensors['initial_parameter_sha256']
             and all(result['initial_state_sha256'] == initial_hash
                     for result in [*historical.values(), *(workers[arm] for arm in NEW)]),
             'All five roles must share the actual seed42 initializer')
    del state, payload
    gc.collect()
    final_results = {}
    for arm in NEW:
        training = workers[arm]['training']
        _require(isinstance(training, dict) and training['steps'] == 32
                 and training['smoothing_epsilon'] == .5 and training['unsmoothed_targets'] is False
                 and training['participating_targets'] == '0.25 + 0.5 * hard_target'
                 and training['schedule'] == plan['schedule'] and training['exposure'] == plan['exposure'],
                 'Completed fixed smoothing receipt required')
        _require(training['initial_state_sha256'] == initial_hash
                 and training['initial_parameter_sha256'] == initial_parameter_hash,
                 'New training initializer differs from actual tensors')
        payload, archive = helper._bounded_load(training['checkpoint'], receipts, torch)
        state = payload
        storage = helper._check_state(state, schema, groups, audited, torch)
        final_hash = helper._semantic_hash(state)
        final_parameter_hash = helper._semantic_hash({name: state[name] for name in parameters})
        _require(final_hash == training['final_state_sha256']
                 and final_parameter_hash == training['final_parameter_sha256']
                 and final_parameter_hash != initial_parameter_hash, 'Final tensor hashes/change differ')
        claimed = training['persisted_validation']
        _require(claimed['weights_only'] is True and claimed['exact_tensor_values'] is True
                 and claimed['missing_keys'] == [] and claimed['unexpected_keys'] == []
                 and claimed['semantic_state_sha256'] == final_hash, 'Persisted validation receipt differs')
        for key in ['tensor_count', 'unique_storage_count', 'alias_group_count',
                    'named_tensor_bytes', 'unique_storage_bytes']:
            _require(claimed['storage_contract'][key] == storage[key], 'Persisted storage count differs')
        changed = [name for name in parameters if helper._tensor_hash(state[name]) != initial_tensor_hashes[name]]
        _require(changed, 'No independently measured changed parameter tensor')
        final_results[arm] = dict(checkpoint=dict(training['checkpoint']), archive=archive,
                                 storage_contract=storage, semantic_state_sha256=final_hash,
                                 semantic_parameter_sha256=final_parameter_hash,
                                 initial_state_sha256=initial_hash, initial_parameter_sha256=initial_parameter_hash,
                                 actual_parameters_changed=True, changed_parameter_tensor_count=len(changed),
                                 changed_parameter_names=changed,
                                 elements_in_changed_parameter_tensors=sum(schema[name]['numel'] for name in changed),
                                 changed_element_count_not_measured=True)
        del state, payload
        gc.collect()
    for item in receipts.values():
        _require(helper._record(item['path']) == item, 'Retained file changed during checkpoint review')
    return dict(status='PASS', immutable_eighth_helper_sha256=OLD_HELPER_SHA,
                retained_eighth_independent_review_sha256=OLD_REVIEW_SHA,
                historical_reuse=reused, historical_final_checkpoints_deserialized=0,
                deserialization="torch.load(map_location='cpu', weights_only=True)",
                torch_threads=4, checkpoints_loaded=3, peak_concurrent_checkpoint_payloads=1,
                factory_constructions=1, factory_forward_calls=0, strict_model_load_calls=0,
                factory_schema_sha256=schema_hash, canonical_parameter_names_sha256=names_hash,
                unique_parameter_tensor_count=len(parameters), unique_buffer_tensor_count=len(buffers),
                parameter_element_count=parameter_elements, initial_state_sha256=initial_hash,
                initial_parameter_sha256=initial_parameter_hash, all_five_initializer_fingerprints_equal=True,
                initializer_archive=initializer_archive, initializer_storage_contract=initial_storage,
                adapted_checkpoints=final_results,
                verified_source_and_checkpoint_records_before_and_after=list(receipts.values()),
                benchmark_metric_code_imported=False, worker_code_imported=False,
                forward_calls=0, backward_calls=0, inference_calls=0, training_calls=0, network_requests=0,
                limitation='Tensor identity, literal historical checkpoint/result reuse and parameter changes; no causal mechanism, new holdout accuracy, calibrated confidence or ink authenticity established.')
