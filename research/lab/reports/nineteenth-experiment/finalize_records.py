"""Record phase19 source-audit and synthetic-integration outcomes, additively."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAB = HERE.parent.parent


def record(path):
    p = Path(path).resolve(strict=True)
    with p.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(p), size_bytes=p.stat().st_size, sha256=digest)


def read(rel, expected=None):
    p = HERE / rel
    if expected:
        assert record(p)['sha256'] == expected, str(p)
    return json.loads(p.read_text())


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


worker = read('actual-backbone-smoke/worker-result.json',
    '698a7f804e6d7653fc1f9ced9ac866909b4fb1e8ad5315aba18a8b85766d7cc2')
controller = read('actual-backbone-smoke/controller-result.json')
runtime_review = read('fusion-review/runtime-receipt-review-v1.json',
    '2f3c99b15b1403674e9940021abec6453f974b001782be5afe7ca12a30976655')
technical = read('fusion-review/TECHNICAL_GO.json',
    'd0202e843a2997ab4f91a770d4f96e54d7ad018745df9d1c9fa52340d6cfcfed')
network = read('NETWORK_SUMMARY.json')
assert worker['status'] == controller['status'] == runtime_review['status'] == 'PASS'
assert technical['decision'] == 'GO' and not technical['blocking_issues']
assert worker['actual_calls'] == technical['expected_actual_smoke_calls']
assert controller['retries'] == 0 and controller['stop_reason'] is None
assert worker['backbone_state_before_sha256'] == worker['backbone_state_after_sha256']
assert worker['scientific_array_reads'] == worker['annotation_reads'] == worker['network_calls'] == 0
assert worker['gpu_calls'] == worker['saved_model_checkpoints'] == 0
assert worker['pretrained_checkpoint_loads'] == 1
for observation in worker['observations'].values():
    assert observation['parameter_count'] == 5296
    assert all(observation[k] for k in ['exact_initial_feature_identity',
        'exact_initial_logit_identity', 'unchanged_backbone_state',
        'backbone_gradients_absent', 'all_backbone_modules_eval'])
    assert observation['projection_gradient_norm'] > 0
    assert observation['first_stem_gradient_norm'] == 0
    assert observation['second_stem_gradient_norm'] > 0
assert network['requests'] == 50 and network['response_body_bytes'] == 3631835

evidence = {k: record(HERE / p) for k, p in {
    'scope': 'SCOPE.md', 'data_roles': 'DATA_ROLES.json',
    'scroll_audit': 'scroll-audit/RESULTS.md',
    'scroll_contract': 'scroll-audit/eligibility-matrix.json',
    'optical_audit': 'optical-audit/OPTICAL_CANDIDATE_AUDIT.md',
    'optical_contract': 'optical-audit/CANDIDATE_CONTRACT.json',
    'upstream_exposure': 'exposure-audit/AUDIT.md',
    'network_accounting': 'NETWORK_SUMMARY.json',
    'prototype': 'fusion-prototype/fusion_adapter.py',
    'prototype_source_seal': 'fusion-prototype/source-manifest.json',
    'smoke_plan': 'fusion-prototype/smoke-plan.json',
    'author_synthetic_checks': 'fusion-prototype/synthetic-results-v1.json',
    'author_targeted_checks': 'fusion-prototype/synthetic-results-v2.json',
    'independent_synthetic_checks': 'fusion-review/independent-toy-results-v1.json',
    'technical_GO': 'fusion-review/TECHNICAL_GO.json',
    'root_GO': 'ROOT_EXECUTION_GO.json',
    'worker': 'actual-backbone-smoke/worker-result.json',
    'controller': 'actual-backbone-smoke/controller-result.json',
    'runtime_receipt_review': 'fusion-review/runtime-receipt-review-v1.json',
    'final_technical_review_seal': 'fusion-review/retention-manifest-v2.json',
    'report': 'RESULTS.md', 'next_study': 'NEXT_STUDY_DESIGN.md'}.items()}
contract = dict(schema_version=1, status='completed_audit_and_synthetic_integration_verified',
    created_at_utc=datetime.now(timezone.utc).isoformat(),
    purpose='Resolve fresh-parent eligibility and validate matched feature-fusion adapter mechanics',
    conclusion='Three candidates remain conditional source-contract reserves; 5,296-parameter matched fusion adapters pass actual seed42 synthetic integration.',
    decision='Proceed next with a separately implemented bounded developmental comparison; prioritize Frag6 source-contract completion; no independent confirmation or paid scaling claim.',
    evidence=evidence, source_audit_requests=network,
    prototype_parameters_per_arm=5296, actual_backbone_synthetic_calls=worker['actual_calls'],
    actual_backbone_resources=dict(worker=worker['resources'],
        controller_wall_seconds=controller['wall_seconds'],
        peak_sampled_rss_bytes=controller['peak_sampled_rss_bytes']),
    actual_backbone_observations=worker['observations'],
    scientific_payload_reads=0, new_candidate_scores=0, scientific_training_updates=0,
    saved_model_checkpoints=0, retained_checkpoint_loads=1, retries=0,
    independent_confirmation_ready=False,
    seed42_exposure_summary='No configured pretrained backbone; candidate parents absent from 29 declared roots; original annotation/teacher ancestry unresolved.',
    adopted_baseline='unchanged strict seed42', promoted_model=None,
    paid_GPU_spend_GBP=0, total_first_GPU_pilot_cap_GBP=10,
    new_artifacts_uploaded_to_R2=False, publication=False, transcription=None, translation=None,
    limitations=[
        'A synthetic MSE mechanics check does not measure ink-detection efficacy or full scientific-training runtime.',
        'Unknown ancestry proves neither candidate contamination nor upstream independence.',
        'Paper R50 and production canonical R152 training histories must remain separate.',
        'Candidate public-package and rendered-surface gaps prevent freezing a defensible new scientific acquisition plan.',
        'Locally unscored candidates may support qualified supplied-reference agreement after geometry/support closure.',
        'All six Paris contexts are reused development; Paris2 remains excluded from fitting across aliases.',
        'The fusion branch changes capacity versus phase18; only within-study raw/depth arms are capacity matched.',
        'Local retention and draft saving are not external backup or fresh-task restoration.'])
save(HERE / 'COMPLETION_CONTRACT.json', contract)

prior = LAB / 'reports/research-register-v14.json'
assert record(prior)['sha256'] == '9790e6d8101b7de5103487b91739c2044c27cf2f77a2809c4cf060a89c8fb369'
register = json.loads(prior.read_text())
assert len(register['experiments']) == 18
register['previous_register'] = record(prior)
register['created_at_utc'] = datetime.now(timezone.utc).isoformat()
register['scope'] = 'Additive register after nineteenth source-audit and fusion mechanics phase; first 18 studies unchanged'
register['experiments'].append(dict(id='nineteenth-experiment', purpose=contract['purpose'],
    status=contract['status'], conclusion=contract['conclusion'], decision=contract['decision'],
    completion_contract=record(HERE / 'COMPLETION_CONTRACT.json'), report=evidence['report'],
    runtime_receipt_review=evidence['runtime_receipt_review'], data_roles=evidence['data_roles'],
    synthetic_backbone_calls=worker['actual_calls'], scientific_training_updates=0,
    scientific_payload_reads=0, new_candidate_scores=0, independent_confirmation_ready=False,
    network_requests=50, network_response_body_bytes=3631835,
    GPU_spend_GBP=0, GPU_pilot_total_cap_GBP=10, promoted_model=None,
    new_artifacts_uploaded_to_r2=False, transcription=None, translation=None))
register['next_study'] = {**evidence['next_study'],
    'status': 'Concrete matched fusion developmental design; scientific runner and execution-plan review remain to be completed; fresh confirmation unavailable'}
register['readiness_check']['CPU_research'] = 'Actual frozen seed42 fusion integration verified on synthetic input; scientific developmental runner is next'
register['readiness_check']['GPU_launch'] = 'Not ready: mechanics are verified, not efficacy or scaling value; useful frozen workload, CUDA, pricing, shutdown and scoped worker delivery remain prerequisites'
save(LAB / 'reports/research-register-v15.json', register)

startup = f'''# Startup after the nineteenth source-audit and fusion phase

Use /workspace/Herculaneum at commit 49fabf3e890d5b880f391b27d581f1e7a1ebdbcc; preserve tracked source and existing untracked .codex/. Lab /workspace/herculaneum-lab; use /workspace/.venvs/herculaneum/bin/python (3.12.14, NumPy 2.5.3, Torch 2.12.0+cpu). No new installation, service or worktree is needed.

Phase 19 is COMPLETE. Read reports/nineteenth-experiment/RESULTS.md, DATA_ROLES.json and COMPLETION_CONTRACT.json (SHA256 {record(HERE / 'COMPLETION_CONTRACT.json')['sha256']}). Research register v15 SHA256 {record(LAB / 'reports/research-register-v15.json')['sha256']}. All first 18 evidence remains immutable. Before new work run this phase's verify_retained_state.py with the literal external anchor in these startup instructions. Never regenerate a manifest to conceal a mismatch. Missing files require restoration; startup text is not a backup.

Three source audits made 50 public metadata/text requests / 3,631,835 retained response-body bytes. No scientific payload was acquired or evaluated. PHerc0009B and PHerc0500P2 public label prefixes lack advertised mapping-validity and linked provenance; dirty transfer, annotation-depth and participation contracts remain unresolved. Renderer mesh centers 54 and 58.5 are established, not ink locations. PHerc0500P2 recorded energy is 111 keV despite literal110-keV source filenames. Frag6 has four native scans but unresolved surface rendering/label lineage; prioritize its source-contract completion. Frag5 shares exposed PHerc1667. Keep all candidate parents/aliases reserved from fitting until their roles are explicitly revised.

Exposure correction: seed42 has no configured pretrained backbone and excludes these three parents from its 29 declared roots. Its original annotation/teacher ancestry remains unresolved. Paper R50 iteration-0 explicitly reports 9b and500p2a training. Production canonical R152 is a different model with unresolved roots; no proven ancestry edge to seed42 is established. Unknown ancestry is neither clean independence nor confirmed contamination. No independent confirmation set is ready; qualified prospective supplied-reference agreement remains possible after geometry/support closure.

The matched 5,296-parameter adapters inject into32-channel depth_fusion output before the frozen2D U-Net. Author45 + targeted4 synthetic checks and independent42 checks passed. Root ran the actual-seed42 synthetic smoke once: controller5.10s, about705MB peak RSS, exact zero-init output equality, projection then stem gradients through frozen network, unchanged original state. One checkpoint load, full-model1/fusion2/network5/adapter4 forwards,4 backwards,2 synthetic SGD steps; full-model counts overlap component calls. No trained state saved. Independent78 receipt consistency checks passed. This is engineering readiness, not ink efficacy; do not rerun or treat synthetic MSE as a scientific objective.

NEXT_STUDY_DESIGN.md specifies a separate matched fusion developmental study using the retained manual PHerc0139 schedule/objective and unchanged six-context development reporting. Implement and review its scientific runner/immutable execution plan before training. All six Paris contexts remain reused DEVELOPMENT; keep Paris2 and aliases out of gradients, teachers and cutoff fitting. Keep phase18 negative result and seed42 reference unchanged. No model promotion or authentic text.

GPU spend GBP0; first paid-pilot TOTAL cap GBP10 persists including retries/overheads. No paid launch before a useful frozen workload, pinned CUDA environment, current pricing/currency margin, independent shutdown and verified bucket-scoped R2 worker delivery. Secret import/injection remains unverified; temporary credentials expire conservatively2026-10-15 09:58:59UTC. Never request raw keys in chat. No uploads, publication, outreach or new terms acceptance. EduceLab/source attribution and retained restrictions apply.

Saving startup instructions persists a draft, not publication, snapshot, restart, external backup or proven fresh-task restoration. Preserve unrelated settings and prior startup history verbatim.
'''
for old, new in [('literal110','literal 110'),('and500','and 500'),('into32','into 32'),
    ('frozen2D','frozen 2D'),('Author45','Author 45'),('targeted4','targeted 4'),
    ('independent42','independent 42'),('controller5.10','controller 5.10'),('about705MB','about 705 MB'),
    ('full-model1/fusion2/network5/adapter4','full-model 1/fusion 2/network 5/adapter 4'),
    ('forwards,4 backwards,2','forwards, 4 backwards, 2'),('Independent78','Independent 78'),
    ('phase18','phase 18'),('GBP0','GBP 0'),('GBP10','GBP 10'),
    ('conservatively2026','conservatively 2026'),('09:58:59UTC','09:58:59 UTC')]:
    startup = startup.replace(old, new)
with (HERE / 'STARTUP_AFTER_NINETEENTH.md').open('x') as stream:
    stream.write(startup)
print(json.dumps(dict(status='PASS', contract=record(HERE / 'COMPLETION_CONTRACT.json'),
    register=record(LAB / 'reports/research-register-v15.json'),
    startup=record(HERE / 'STARTUP_AFTER_NINETEENTH.md'))))
