"""Bind the completed phase20 result without altering prior evidence."""
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
    if expected is not None:
        assert record(path)['sha256'] == expected, rel
    return json.loads(path.read_text())


def save(path, data):
    with Path(path).open('x') as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


metrics = read('analysis/metrics.json', '60bb52efcff810830543941e5081f61e9570cb598a772b97c4c2633a1cce0be6')
train = read('implementation/run/training-results.json', 'f5fe986942710d05b24980e4a16946bfd7efec2a50cf12b579f1c062679b2626')
dev = read('implementation/run/development-results.json', 'a04a7b305fd444f5adf551ccaba2ae1f47fe7b2c13e2e5e6e219afc124944e5f')
review = read('independent-review/verification.json', '7bc5049ceb9a04f728b334594261cda77c01dd34cebfb20310cdf9c8779b7ac0')
heads = read('frag6-plan/head-request-ledger.json')
assert metrics['status'] == train['status'] == dev['status'] == review['status'] == 'PASS'
assert train['retries'] == dev['retries'] == review['checker_retries'] == 0
assert not metrics['feature_screen_all'] and not review['feature_screen_passed']
assert len(metrics['feature_screen']) == 17
assert sum(g['passed'] for g in metrics['feature_screen'].values()) == 12
assert review['screen_components_passed'] == 12 and review['screen_components_failed'] == 5
assert not review['producer_independent_gate_ambiguities']
assert dev['all_six_contexts_both_final_arms_retained']
assert len(heads['requests']) == 4 and heads['response_body_reads'] == heads['retained_body_bytes'] == 0
assert heads['automatic_retries'] == heads['redirects_followed'] == 0
assert read('implementation/synthetic-fixtures-v2.json')['check_count'] == 30
assert read('technical-review/synthetic-controls-v2.json')['check_count'] == 44
calls = {k: train['actual_calls'][k] + dev['actual_calls'][k] for k in train['actual_calls']}
checkpoints = {}
resources = {}
for arm in ['raw_control', 'explicit_depth']:
    data = read('implementation/run/training/' + arm + '/result.json')
    assert data['initialization_proof']['all32_actual_original_teacher_bitwise_equal']
    assert data['initialization_proof']['all32_zero_adapter_original_bitwise_equal']
    assert data['checkpoint'] == record(data['checkpoint']['path'])
    checkpoints[arm] = data['checkpoint']
    resources[arm] = {k: data[k] for k in ['wall_seconds', 'peak_rss_bytes']}

presentation = read('figures/presentation-receipt.json')
save(HERE / 'ROOT_VISUAL_REVIEW.json', dict(status='PASS', reviewer='root',
    method='Root visually inspected the rendered PNG after generation',
    findings='Legible three panels; all six regions retained on common difference axes; training iteration 0 and 1 ticks close but readable; no selective rescaling or outcome omission.',
    presentation=record(HERE / 'figures/presentation-receipt.json'),
    outputs=presentation['outputs'], new_model_calls=0, rerenders=0))

evidence = {k: record(HERE / p) for k, p in {
    'protocol': 'RESEARCH_DESIGN.md', 'configuration': 'EXPERIMENT_CONFIG.json',
    'code_freeze': 'implementation/code-freeze.json',
    'plan': 'implementation/final-plan/experiment-plan.json',
    'root_GO': 'EXECUTION_GO.json', 'root_prelaunch_stop': 'ROOT_PRELAUNCH_STOP.json',
    'training': 'implementation/run/training-results.json',
    'development': 'implementation/run/development-results.json',
    'metrics': 'analysis/metrics.json', 'numerical_verification': 'independent-review/verification.json',
    'numerical_review_seal': 'independent-review/retention-manifest.json',
    'technical_GO': 'technical-review/TECHNICAL_GO.json',
    'technical_seal': 'technical-review/retention-manifest.json',
    'frag6_plan': 'frag6-plan/GEOMETRY_ACQUISITION_PLAN.json',
    'frag6_seal': 'frag6-plan/retention-manifest.json',
    'head_ledger': 'frag6-plan/head-request-ledger.json',
    'figure': 'figures/depth-fusion-study.png', 'figure_review': 'ROOT_VISUAL_REVIEW.json',
    'report': 'RESULTS.md', 'next_study': 'NEXT_STUDY_DRAFT.md'}.items()}

contract = dict(schema_version=1, created_at_utc=datetime.now(timezone.utc).isoformat(),
    status='completed_execution_verified_scientific_screen_failed',
    purpose='Matched raw versus explicit-depth feature fusion on the frozen seed42 backbone',
    conclusion='Explicit depth beats matched raw control but fails five baseline-retention criteria; retain unchanged seed42.',
    decision='No promotion or GPU scaling. Next: bounded Frag6 structural check and training-only preservation-support audit before a separately frozen objective study.',
    evidence=evidence, scientific_screen_passed=False, scientific_screen=metrics['feature_screen'],
    training=metrics['training'], Paris2_equal_fragment=metrics['Paris2_equal_fragment'],
    Paris1_equal_context=metrics['Paris1_equal_context'],
    numerical_verification_status=review['status'], actual_calls=calls,
    call_count_note='Original full-model calls include fusion/network component calls; counts overlap.',
    retained_backbone_loads=4, scientific_optimizer_updates=64, final_adapter_checkpoints=checkpoints,
    saved_backbone_checkpoints=0, scientific_worker_retries=0, root_prelaunch_stops_before_worker_start=1,
    controller_wall_seconds=dict(training=train['wall_seconds'], development=dev['wall_seconds']),
    training_worker_resources=resources, threshold=metrics['threshold'], threshold_fits=0,
    network=dict(public_HEAD_requests=4, response_body_reads=0, response_body_bytes=0,
                 scientific_worker_requests=0, external_uploads=0),
    new_candidate_scores=0, independent_confirmation_ready=False,
    adopted_baseline='unchanged strict seed42', promoted_model=None,
    paid_GPU_spend_GBP=0, total_first_GPU_pilot_cap_GBP=10,
    R2_worker_secret_delivery_verified=False, new_artifacts_uploaded_to_R2=False,
    publication=False, transcription=None, translation=None,
    limitations=[
        'All six Paris contexts are reused development, not independent confirmation.',
        'Paris2 and all aliases remain excluded from gradients, teachers and cutoff fitting.',
        'The coordinate/difference feature bundle changes jointly; no separate causal attribution.',
        'Original annotation and teacher ancestry remain unresolved.',
        'Numerical verification does not replay convolution/autograd or independently decode checkpoints.',
        'Lower training risk and better relative-to-control scores do not preserve baseline performance.',
        'Frag6 Stage A is a plan awaiting implementation, synthetic fixtures and source/plan review; Stage C remains blocked.',
        'Local retention and draft saving are not external backup or verified fresh-task restoration.'])
save(HERE / 'COMPLETION_CONTRACT.json', contract)

prior = LAB / 'reports/research-register-v15.json'
assert record(prior)['sha256'] == 'f5e991a0926ecd84ccdf37c9f357da159a268c671525850d66bbe5c34eef4389'
register = json.loads(prior.read_text())
assert len(register['experiments']) == 19
register['previous_register'] = record(prior)
register['created_at_utc'] = datetime.now(timezone.utc).isoformat()
register['scope'] = 'Additive register after phase20 matched feature-fusion experiment; first 19 studies unchanged'
register['experiments'].append(dict(id='twentieth-experiment', purpose=contract['purpose'],
    status=contract['status'], conclusion=contract['conclusion'], decision=contract['decision'],
    completion_contract=record(HERE / 'COMPLETION_CONTRACT.json'), report=evidence['report'],
    numerical_verification=evidence['numerical_verification'], scientific_screen_passed=False,
    scientific_optimizer_updates=64, new_candidate_scores=0, independent_confirmation_ready=False,
    network_requests=4, network_response_body_bytes=0, GPU_spend_GBP=0, GPU_pilot_total_cap_GBP=10,
    promoted_model=None, new_artifacts_uploaded_to_r2=False, transcription=None, translation=None))
register['next_study'] = {**evidence['next_study'], 'status': 'Post-result draft: implement/freeze Frag6 Stage A and audit train-only preservation support; no scientific follow-on execution readiness claim'}
register['readiness_check']['CPU_research'] = 'Phase20 complete and independently verified, scientific screen failed; bounded structural work and train-only support audit recommended'
register['readiness_check']['GPU_launch'] = 'Not ready: adaptation loses baseline performance; useful frozen workload, CUDA, current pricing, shutdown and scoped worker delivery still required'
save(LAB / 'reports/research-register-v16.json', register)

startup = f'''# Startup after phase 20: matched depth-feature fusion

Use /workspace/Herculaneum at commit 49fabf3e890d5b880f391b27d581f1e7a1ebdbcc; preserve tracked source and existing untracked .codex/. Lab /workspace/herculaneum-lab. Use /workspace/.venvs/herculaneum/bin/python: Python 3.12.14, NumPy 2.5.3, Torch 2.12.0+cpu. No installation, service or new worktree is needed.

Phase 20 is COMPLETE. Read reports/twentieth-experiment/RESULTS.md and COMPLETION_CONTRACT.json (SHA256 {record(HERE / 'COMPLETION_CONTRACT.json')['sha256']}); additive register v16 SHA256 {record(LAB / 'reports/research-register-v16.json')['sha256']}. Before new work verify retained state using this phase's verifier and the literal external manifest anchor saved below. Preserve all first-twentieth evidence. Never regenerate a manifest to conceal a mismatch; missing files need restoration. Startup text is not a backup.

Both matched 5,296-parameter branches trained exactly 32 CPU updates from fresh identical initialization, using the unchanged PHerc0139 manual schedule. Original seed42 weights/buffers stayed frozen; decoder input gradients were active. Initial original/teacher/zero-adapter logits were bitwise equal on all 32 batches per arm. Both final arms completed all six development regions with original-cache equality checks. No scientific worker retry, new cutoff fit or checkpoint selection occurred. Four workers loaded the original checkpoint once each. Total actual calls: full-model 364, fusion 364, 2D network 1112, adapter 748, backwards/decoder-input backwards/updates 64 each. Full-model counts overlap component calls. Controller times: training 55.72s, development 52.21s; training peak about 1.33GB. Only two final adapter files (23,705 bytes each) were saved.

Independent training, development and report arithmetic all PASS with frozen tolerances, zero checker retries and no decision ambiguities. Scientific screen FAIL: 12/17 pass, five baseline-retention failures. Paris2 equal-fragment AUROC/AP: seed42 0.657642/0.384286, raw 0.628518/0.304992, depth 0.647074/0.313037. Depth beats raw by 0.018557/0.008045 but loses 0.010568/0.071248 against seed42. Large losses: frag1-context2 AUROC/AP, frag2-context2 AP, frag3-context1 AUROC/AP. Final full training risk raw 0.5940976292/depth 0.5938324379; depth has higher KL/probability drift. Retain unchanged seed42. No promotion, text or GPU scaling. Do not claim depth universally useless or a proven cause of transfer loss.

All six Paris regions remain reused DEVELOPMENT; Paris2 and every alias stay excluded from fitting, teachers and cutoff selection. No independent confirmation set is ready; seed42 original annotation/teacher ancestry remains unresolved. Numerical checker did not replay convolution/autograd or decode model checkpoints. A root metadata assertion stop happened before worker creation and was corrected without changing frozen scientific code; retain its receipt.

Frag6 planning made four public HEAD requests, zero body bytes and no scientific acquisition. Its sealed plan proposes Stage A (13 requests / 202,301 reserved body bytes) for PPM/TIFF structural metadata; implement, fixture-test and freeze the reader/plan before running. Later Stage C remains blocked by missing declared source/render contract. Keep Frag6/PHerc51/PHerc0051 and reserved PHerc0009B/PHerc0500P2 across aliases/scans out of fitting. Do not infer offset/normal/ink depth from file count or plausible coordinates.

NEXT_STUDY_DRAFT.md is a post-result recommendation, not an execution-ready scientific protocol. Next: small Frag6 structural check and train-only PHerc0139 support audit. Conditional follow-on hypothesis: broader valid-training-output teacher preservation versus manual-only KL, with supervision still manual-only. Source-valid support, reductions, weight, margins and limits must be fixed before any run. Do not sweep hyperparameters against reused Paris regions, warm-start rejected adapters or run another GPU study by default. Root owns scientific judgment; narrow technical/numerical delegation remains useful.

Paid GPU spend GBP 0; first paid pilot TOTAL cap GBP 10 includes failures/retries and overheads. No paid launch before a useful frozen workload, pinned CUDA, current pricing/currency margin, independent shutdown and verified bucket-scoped R2 worker delivery. Import/injection remains unverified; temporary credentials conservatively expire 2026-10-15 09:58:59 UTC. Never request raw credentials in chat. No uploads, external publication, outreach, new terms or prize submission; preserve attribution and retained source-use restrictions.

Saving startup instructions persists a draft, not publication, snapshot, restart, external backup or verified fresh-task restoration. Preserve unrelated settings and append prior startup history verbatim. The current phase supersedes older pending-study suggestions without altering their records.
'''
with (HERE / 'STARTUP_AFTER_TWENTIETH.md').open('x') as stream:
    stream.write(startup)
print(json.dumps(dict(status='PASS', contract=record(HERE / 'COMPLETION_CONTRACT.json'),
    register=record(LAB / 'reports/research-register-v16.json'),
    startup=record(HERE / 'STARTUP_AFTER_TWENTIETH.md'))))
