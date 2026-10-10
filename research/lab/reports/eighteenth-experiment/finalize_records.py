"""Seal completed phase18 results and append an additive research-register entry."""
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


def read(path, expected=None):
    p = HERE / path
    if expected:
        assert record(p)['sha256'] == expected, str(p)
    return json.loads(p.read_text())


def save(path, data):
    with Path(path).open('x') as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


review = read('independent-review/verification.json',
              '622e81c5abd9a061c6bbdfc64e737ccad7efbf2f5d2b5749a2f867a77b36fc96')
seal = read('independent-review/retention-manifest.json',
            'a1491a8b74ae859aa259135b62a4c62c69ae3804227b6831e2d0b1d750f6ed10')
assert review['status'] == 'PASS' and not review['feature_screen_passed']
assert not review['producer_independent_gate_ambiguities']
metrics = read('analysis/metrics.json',
               'd2995ff521a87255ad0cb2d991ebc15f0134f4cd7b84e1278361c3d13dffd662')
training = read('implementation/run/training-results.json',
                'c2cb84984f3cd7fc2ba0656da901f8d433e22e3119acaafa6c75c5e74ea9f4a5')
development = read('implementation/run/development-results.json')
assert metrics['status'] == training['status'] == development['status'] == 'PASS'
assert not metrics['feature_screen_all'] and not metrics['automatic_promotion']
failed = {k: v for k, v in metrics['feature_screen'].items() if not v['passed']}
assert len(metrics['feature_screen']) == 17 and len(failed) == 3
assert training['retries'] == development['retries'] == 0
assert development['all_six_contexts_both_final_arms_retained']
assert training['actual_calls'] == dict(adapter_forwards=384, backbone_forwards=0,
    backwards=64, optimizer_steps=64, threshold_fits=0)
assert development['actual_calls'] == dict(adapter_forwards=300, backbone_forwards=0,
    backwards=0, optimizer_steps=0, threshold_fits=0)
workers = [read(f'implementation/run/{phase}/{arm}/result.json')
           for phase in ['training', 'development']
           for arm in ['raw_control', 'explicit_depth']]
assert all(w['status'] == 'PASS' for w in workers)
assert all(w['backbone']['loads'] == w['backbone']['forwards'] ==
           w['backbone']['backwards'] == w['backbone']['updates'] == 0
           for w in workers[:2])
assert all(w['checkpoint_reload']['strict'] and w['checkpoint_reload']['parameter_count'] == 801
           for w in workers[:2])
assert workers[0]['initial_adapter_state_sha256'] == workers[1]['initial_adapter_state_sha256']
assert workers[0]['initial_rng_sha256'] == workers[1]['initial_rng_sha256']

save(HERE / 'ROOT_VISUAL_REVIEW.json', dict(
    recorded_at_utc=datetime.now(timezone.utc).isoformat(), status='completed', reviewer='root',
    image=record(HERE / 'figures/depth-residual-pilot.png'),
    scope='Prior root visual inspection of the final training-trajectory and six-context comparison figure',
    observations=[
        'Legible axes, legends and complete scheduled trajectories for both arms.',
        'All six regional candidate-minus-control AUROC/AP comparisons are present.',
        'The plot is descriptive development evidence; it adds no fitted choice or threshold.'],
    new_numerical_or_model_execution=False))

evidence = {k: record(HERE / p) for k, p in {
    'design': 'RESEARCH_DESIGN.md', 'configuration': 'EXPERIMENT_CONFIG.json',
    'execution_GO': 'EXECUTION_GO.json', 'runtime': 'runtime-preflight.json',
    'code_freeze': 'implementation/code-freeze.json',
    'plan': 'implementation/final-plan/experiment-plan.json',
    'architecture_audit': 'architecture-audit/code-review/TECHNICAL_GO.json',
    'data_audit': 'data-audit/DATA_AUDIT.md',
    'training': 'implementation/run/training-results.json',
    'development': 'implementation/run/development-results.json',
    'independent_review': 'independent-review/verification.json',
    'independent_seal': 'independent-review/retention-manifest.json',
    'independent_training': 'independent-review/training-v2/verification.json',
    'independent_development': 'independent-review/development-v1/verification.json',
    'independent_report_arithmetic': 'independent-review/publication-review/comparison-v1/comparison.json',
    'metrics': 'analysis/metrics.json', 'presentation': 'figures/presentation-receipt.json',
    'root_visual_review': 'ROOT_VISUAL_REVIEW.json',
    'report': 'RESULTS.md', 'next_study': 'NEXT_STUDY_DRAFT.md'}.items()}
contract = dict(
    schema_version=1, status='completed_independently_verified',
    created_at_utc=datetime.now(timezone.utc).isoformat(),
    purpose='Matched explicit-depth versus raw-neighbor output-residual experiment on frozen cached seed42 logits',
    conclusion='Explicit depth better preserves the initializer than the raw control but fails three predeclared superiority margins; Paris2 ranking remains slightly below unchanged seed42.',
    decision='Retain both final adapters as research evidence; keep seed42 as reference; do not scale this recipe to GPU on these results.',
    evidence=evidence,
    feature_screen=dict(passed=False, component_count=17, passed_count=14,
                        failed_count=3, failed_components=failed, independent_boundary_ambiguities=[]),
    training_metrics=metrics['training'], Paris2_equal_fragment=metrics['Paris2_equal_fragment'],
    Paris1_equal_context=metrics['Paris1_equal_context'], threshold=metrics['threshold'],
    numerical_execution=dict(training_calls=training['actual_calls'],
        development_calls=development['actual_calls'],
        training_controller_wall_seconds=training['wall_seconds'],
        development_controller_wall_seconds=development['wall_seconds'],
        training_peak_rss_bytes=max(w['peak_rss_bytes'] for w in workers[:2]),
        training_retries=0, development_retries=0, independent_training_checker_schema_stops=1,
        new_backbone_loads=0, new_backbone_forwards=0, new_backbone_backwards=0,
        new_backbone_updates=0, network_calls=0, GPU_seconds=0),
    final_adapter_checkpoints={w['arm']: w['checkpoint'] for w in workers[:2]},
    adopted_baseline='unchanged strict seed42', promoted_model=None,
    evaluation_scope='All six reused contexts are DEVELOPMENT only; no fresh confirmatory set was ready.',
    paid_GPU_spend_GBP=0, total_first_GPU_pilot_cap_GBP=10,
    new_artifacts_uploaded_to_R2=False, publication=False, transcription=None, translation=None,
    limitations=[
        'The joint coordinate/axial-difference feature bundle does not isolate either feature individually.',
        'Nominal stack indices do not authenticate anatomical depth, physical normal sign or ink location.',
        'Paris2 is one parent with two fragments; all six inspected contexts are reused development evidence.',
        'Upstream backbone and teacher training exposure remains incompletely known.',
        'Manual PHerc0139 targets supplied the objective; historical pseudo arrays supplied identity checks only.',
        'No conclusion about all depth-aware architectures follows from this 801-parameter, 32-update output-head pilot.',
        'Independent arithmetic checks do not independently replay convolutional autograd or prove execution history.',
        'Local retention and a saved draft do not establish backup or fresh-task restoration.'])
save(HERE / 'COMPLETION_CONTRACT.json', contract)

prior = LAB / 'reports/research-register-v13.json'
assert record(prior)['sha256'] == 'cfade40a9f6428f6ece6df1ed770e3da9ec84f392caa7dfd783f72935acecfba'
register = json.loads(prior.read_text())
assert len(register['experiments']) == 17
register['previous_register'] = record(prior)
register['created_at_utc'] = datetime.now(timezone.utc).isoformat()
register['scope'] = 'Additive register after eighteenth matched depth-residual pilot; prior 17 studies unchanged'
register['experiments'].append(dict(
    id='eighteenth-experiment', purpose=contract['purpose'], status=contract['status'],
    conclusion=contract['conclusion'], decision=contract['decision'],
    completion_contract=record(HERE / 'COMPLETION_CONTRACT.json'), report=evidence['report'],
    independent_verification=evidence['independent_review'], metrics=evidence['metrics'],
    feature_screen_passed=False, failed_screen_components=list(failed),
    adapter_forwards=684, backwards=64, optimizer_steps=64, new_backbone_calls=0,
    GPU_spend_GBP=0, GPU_pilot_total_cap_GBP=10, promoted_model=None,
    new_artifacts_uploaded_to_r2=False, evaluation_informed=True,
    transcription=None, translation=None))
register['next_study'] = {**evidence['next_study'],
    'status': 'Draft fresh-parent/exposure contract followed by matched depth-aware feature fusion; not executed or GPU-ready'}
register['readiness_check']['CPU_research'] = 'Matched depth-residual pilot complete and independently verified; feature screen failed; seed42 remains reference'
register['readiness_check']['GPU_launch'] = 'Not ready: no scaling case established; fresh evaluation, useful frozen workload, pinned CUDA, current pricing, shutdown and scoped worker delivery remain prerequisites'
save(LAB / 'reports/research-register-v14.json', register)

startup = f'''# Startup after the eighteenth depth-residual experiment

Use /workspace/Herculaneum at commit 49fabf3e890d5b880f391b27d581f1e7a1ebdbcc; preserve tracked files and pre-existing untracked .codex/. Lab: /workspace/herculaneum-lab. Use /workspace/.venvs/herculaneum/bin/python (3.12.14, NumPy 2.5.3, Torch 2.12.0+cpu). No installation, service or new worktree is needed.

Phase 18 is COMPLETE. Read reports/eighteenth-experiment/RESULTS.md and COMPLETION_CONTRACT.json (SHA256 {record(HERE / 'COMPLETION_CONTRACT.json')['sha256']}). Research register v14 SHA256 {record(LAB / 'reports/research-register-v14.json')['sha256']}. Two matched 801-parameter residual adapters each received the same 32 manual PHerc0139 updates; both final states were evaluated on all six contexts. Independent training, optimizer, blending and metric checks passed. Three of 17 feature-screen components failed. Paris2 AUROC/AP: seed42 0.657642/0.384286; raw control 0.655292/0.382845; explicit depth 0.657107/0.384051. Candidate full-risk advantage is mostly smaller preservation KL, not better manual BCE fitting. Keep seed42; no promotion or justification to scale this output-only recipe.

Before new work run reports/eighteenth-experiment/verify_retained_state.py with --expected-manifest-sha256 and the literal anchor in these saved startup instructions. Never regenerate a manifest to hide a mismatch. Preserve all prior evidence, both final adapters, all numerical outputs, the checker v1 schema STOP and its separate v2 clarification. Missing files need restoration; instructions are not a backup. This run used zero new backbone executions, zero network calls and no threshold fitting. The figure was visually inspected; no model rerun is needed.

All six Paris1/Paris2 contexts remain reused DEVELOPMENT. Keep Paris2 parents/fragments/aliases out of gradient, teacher and cutoff fitting. Phase 17 optical-registration and exact anatomical-depth limitations persist. There is no ready fresh confirmatory set. NEXT_STUDY_DRAFT.md recommends resolving a fresh-parent/exposure contract, then a separately frozen matched depth-aware feature-fusion experiment. PHerc0009B/PHerc0500P2 are metadata-only candidates; Frag6 provenance remains incomplete. New to this project does not mean absent from upstream pretraining. Do not retune this completed pilot on its results.

GPU spend GBP 0; first paid pilot TOTAL cap GBP 10, including failures, retries and attributable overheads. Before any paid launch: useful frozen experiment, pinned CUDA environment, current pricing/currency margin, independent shutdown and verified bucket-scoped R2 worker delivery. Import/injection remains unverified; temporary credentials expire conservatively 2026-10-15 09:58:59 UTC. Never request keys in chat. No upload, publication, new terms acceptance or outreach. Existing EduceLab attribution and use restrictions apply.

For plots use writable /tmp MPLCONFIGDIR and XDG_CACHE_HOME. A draft save is not publication, a snapshot, restart, external backup or verified fresh-task restoration. Preserve unrelated environment settings and prior startup history.
'''
with (HERE / 'STARTUP_AFTER_EIGHTEENTH.md').open('x') as stream:
    stream.write(startup)
print(json.dumps(dict(status='PASS', contract=record(HERE / 'COMPLETION_CONTRACT.json'),
    register=record(LAB / 'reports/research-register-v14.json'),
    startup=record(HERE / 'STARTUP_AFTER_EIGHTEENTH.md'))))
