"""Add phase21 completion/register/startup records after all bounded work is sealed."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAB = HERE.parent.parent


def record(path):
    path = Path(path).resolve(strict=True)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(path), size_bytes=path.stat().st_size, sha256=digest)


def read(rel, expected=None):
    path = HERE / rel
    if expected:
        assert record(path)['sha256'] == expected, rel
    return json.loads(path.read_text())


def save(path, data):
    with Path(path).open('x') as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


metrics = read('allocation-analysis/metrics.json', '75ec95d6f03185d78088fd787512b2e5aabf9e23d845db195a9eef4a0986558c')
training = read('allocation-implementation/run/training-results.json', 'e11ac5058a89019e938f8965811a26dc8e9ac1ccfca9dff1a0f2e5136c998eb3')
development = read('allocation-implementation/run/development-results.json', '92f95de0ba6164d9cbaf1d3491ccb0b3ace574aeb68d4012562b14b95df205ad')
review = read('allocation-review/verification.json', 'dcd7029592f5b163dd2ee3896fa58cf672132c9fabd77c872f21bed3fe3ed730')
assert metrics['status'] == training['status'] == development['status'] == review['status'] == 'PASS'
assert training['retries'] == development['retries'] == review['checker_retries'] == 0
assert metrics['mechanism_screen_all'] and not metrics['practical_screen_all']
assert len(metrics['mechanism_screen']) == 5 and all(g['passed'] for g in metrics['mechanism_screen'].values())
assert len(metrics['practical_screen']) == 14 and sum(g['passed'] for g in metrics['practical_screen'].values()) == 9
assert not review['numerical_margin_ambiguity_components']
assert development['all_six_contexts_both_final_arms_retained']

geometry = read('pherc0139-geometry/public-geometry-implementation-v2/run/result.json')
assert geometry['status'] == 'PASS' and geometry['request_count'] == 6
assert geometry['retained_range_body_bytes'] == 2739744
native = read('native-checkpoint-contract/query-run/result.json', 'c274c8e530d58a0c7a0743adfdc3b046b1cfef6dbfa0b319ca9b1a27f4434472')
assert native['status'] == 'HEADER_SCHEMA_COMPATIBLE_VALUES_AND_LOAD_UNVERIFIED'
assert native['tensor_value_bytes_requested'] == native['model_calls'] == native['retries'] == 0
frag_a = read('frag6-stage-a/run-v1/acquisition/request-ledger.json')
frag_b = read('frag6-stage-b/run-v1/acquisition/request-ledger.json')
text_phase = read('pherc0139-geometry/text-run/result.json')
structural = read('pherc0139-geometry/structural/run/result.json')
assert text_phase['status'] == structural['status'] == 'PASS'
network = dict(
    Frag6_A=dict(attempts=len(frag_a['requests']), observed_body_bytes=frag_a['retained_response_body_bytes']),
    Frag6_B=dict(attempts=len(frag_b['requests']), observed_body_bytes=frag_b['retained_response_body_bytes']),
    PHerc0139_text=dict(attempts=len(text_phase['requests']), observed_body_bytes=text_phase['actual_retained_body_bytes']),
    PHerc0139_structure=dict(attempts=structural['requests'], observed_body_bytes=structural['actual_retained_body_bytes']),
    PHerc0139_coordinates=dict(attempts=geometry['request_count'], observed_body_bytes=geometry['retained_range_body_bytes']),
    native_checkpoint_header=dict(attempts=native['https_attempt_count'], observed_body_bytes=native['observed_response_body_bytes']))
network_totals = dict(attempts=sum(v['attempts'] for v in network.values()),
                      observed_response_body_bytes=sum(v['observed_body_bytes'] for v in network.values()),
                      automatic_retries=0, external_uploads=0, scientific_worker_requests=0,
                      accounting='Response bodies only, including hash-only native redirect body; not full wire traffic')
calls = {k: training['actual_calls'][k]+development['actual_calls'][k] for k in training['actual_calls']}
checkpoints = {}
for arm in ['manual_double', 'manual_retired']:
    data = read('allocation-implementation/run/training/'+arm+'/result.json')
    assert data['initialization_proof']['all32_actual_original_teacher_bitwise_equal']
    assert data['initialization_proof']['all32_zero_adapter_original_bitwise_equal']
    assert record(data['checkpoint']['path']) == data['checkpoint']
    checkpoints[arm] = data['checkpoint']

paths = {
    'allocation_protocol':'PRESERVATION_ALLOCATION_PROTOCOL.md',
    'allocation_plan':'allocation-implementation/final-plan/experiment-plan.json',
    'allocation_metrics':'allocation-analysis/metrics.json',
    'allocation_verification':'allocation-review/verification.json',
    'allocation_review_seal':'allocation-review/retention-manifest.json',
    'allocation_report':'ALLOCATION_RESULTS.md',
    'root_report_review':'ROOT_REPORT_REVIEW.json',
    'root_visual_review':'ROOT_VISUAL_REVIEW.json',
    'support_contract':'preservation-support/audit-v2/support-contract.json',
    'support_review':'support-review/RESULTS.md',
    'Frag6_A_contract':'frag6-stage-a/run-v1/acquisition/STRUCTURAL_CONTRACT.json',
    'Frag6_B':'frag6-stage-b/run-v1/acquisition/SPARSE_GEOMETRY.json',
    'Frag6_B_review':'frag6-b-runtime-review/verification.json',
    'PHerc0139_geometry':'pherc0139-geometry/public-geometry-implementation-v2/run/result.json',
    'PHerc0139_review':'pherc0139-runtime-review/verification.json',
    'native_header':'native-checkpoint-contract/query-run/result.json',
    'native_header_review':'native-header-review/verification.json',
    'source_correction':'research-context/EVIDENCE.md',
    'report':'RESULTS.md',
    'next_study':'NEXT_STUDY_DRAFT.md'}
evidence = {k:record(HERE/v) for k,v in paths.items()}
assert read(paths['PHerc0139_review'])['status'] == 'PASS'
assert read(paths['native_header_review'])['status'] == 'PASS'
assert read(paths['Frag6_B_review'])['status'] == 'PASS'
contract = dict(schema_version=1, created_at_utc=datetime.now(timezone.utc).isoformat(),
    status='completed_verified_training_mechanism_passed_development_screen_failed',
    purpose='Matched output-preservation allocation study and separately bounded geometry/source audits',
    conclusion='All five training mechanism criteria pass; development passes 9/14 and fails five. Retain seed42.',
    decision='No promotion or GPU scaling; prioritize independent evaluation and source-compatible inputs before further tuning.',
    evidence=evidence, mechanism_screen_passed=True, practical_screen_passed=False,
    mechanism_screen=metrics['mechanism_screen'], practical_screen=metrics['practical_screen'],
    Paris2_equal_fragment=metrics['Paris2_equal_fragment'], Paris1_equal_context=metrics['Paris1_equal_context'],
    training=metrics['training'], actual_calls=calls, calls_overlap=True,
    scientific_optimizer_updates=64, retained_backbone_loads=4, saved_backbone_checkpoints=0,
    final_adapter_checkpoints=checkpoints, scientific_worker_retries=0,
    threshold=metrics['threshold'], threshold_fits=0,
    controller_wall_seconds=dict(training=training['wall_seconds'],development=development['wall_seconds']),
    network=network, network_totals=network_totals,
    independent_confirmation_ready=False, adopted_baseline='unchanged strict seed42', promoted_model=None,
    native_checkpoint_values_verified=False, native_checkpoint_strict_load_verified=False,
    historical_private_geometry_validity_authenticated=False, Frag6_stage_C_authorized=False,
    paid_GPU_spend_GBP=0, total_first_GPU_pilot_cap_GBP=10,
    R2_worker_secret_delivery_verified=False, new_artifacts_uploaded_to_R2=False,
    publication=False, transcription=None, translation=None,
    limitations=['All six Paris contexts are reused development, not independent confirmation.',
      'Paris2 and its aliases remain excluded from gradients, teachers and threshold fitting.',
      'One initialization and 32 updates per arm do not establish population or converged-training behavior.',
      'Preservation targets do not authenticate ink or the supplied-reference labels.',
      'Geometry bounds are not native observed coverage, material support or authenticated historical private masks.',
      'Native header schema does not verify full-file SHA, finite tensor values, alias-value equality or strict load.',
      'Local retention and draft saving do not establish external backup or fresh-task restoration.'])
save(HERE/'COMPLETION_CONTRACT.json',contract)

prior = LAB/'reports/research-register-v16.json'
assert record(prior)['sha256'] == '8deafeec845c377ae29a7dbf6a0208af66e0f93ac4de80c85be2ee0eb4d88263'
register = json.loads(prior.read_text())
assert len(register['experiments']) == 20
register['previous_register'] = record(prior)
register['created_at_utc'] = datetime.now(timezone.utc).isoformat()
register['scope'] = 'Additive register after phase21; first twenty studies unchanged'
register['experiments'].append(dict(id='twentyfirst-experiment',purpose=contract['purpose'],status=contract['status'],
    conclusion=contract['conclusion'],decision=contract['decision'],
    completion_contract=record(HERE/'COMPLETION_CONTRACT.json'),report=evidence['report'],
    numerical_verification=evidence['allocation_verification'],
    mechanism_screen_passed=True,practical_screen_passed=False,scientific_optimizer_updates=64,
    independent_confirmation_ready=False, network_requests=network_totals['attempts'],
    network_response_body_bytes=network_totals['observed_response_body_bytes'],GPU_spend_GBP=0,
    GPU_pilot_total_cap_GBP=10,promoted_model=None,new_artifacts_uploaded_to_r2=False,transcription=None,translation=None))
register['next_study']={**evidence['next_study'],'status':'Post-result recommendation; no new scientific or paid execution protocol'}
register['readiness_check']['CPU_research']='Phase21 complete and verified; retained numerical and bounded geometry/source workflows support further CPU research'
register['readiness_check']['GPU_launch']='Not ready: practical study criteria failed; useful workload, pinned CUDA, pricing, independent shutdown and scoped worker delivery remain required'
save(LAB/'reports/research-register-v17.json',register)
print(json.dumps(dict(status='PASS',contract=record(HERE/'COMPLETION_CONTRACT.json'),register=record(LAB/'reports/research-register-v17.json'),network_totals=network_totals)))
