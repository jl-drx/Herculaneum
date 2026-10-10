"""Write retention metadata after the root's numerical and visual review."""
from copy import deepcopy
from study_common import *

run=json.loads((HERE/'run-v1/COMPLETE.json').read_text())
tv=json.loads((HERE/'TRAINING_VERIFICATION.json').read_text())
ev=json.loads((HERE/'EVALUATION_VERIFICATION.json').read_text())
review=json.loads((HERE/'ROOT_REPORT_REVIEW.json').read_text())
require(run['status']==tv['status']==ev['status']=='PASS','Completed numerical evidence')
require(review['reviewer']=='root' and review['figures_visually_reviewed'] and review['report_reviewed'],'Completed root review')
require(not (HERE/'COMPLETION_CONTRACT.json').exists(),'Write-once metadata')
metrics=json.loads((HERE/'METRICS.json').read_text())
refs=dict(schema_version=1,created_at_utc=now(),training_parent='PHerc0139',training_regions=2,
    evaluation_parents=['Paris1','Paris2','PHerc0814'],evaluation_regions=8,
    all_evaluation_regions_reused_development=True,independent_ink_accuracy_established=False,
    supervision='Supplied manual participation, acquired support, retired validation excluded; ancestry unresolved',
    teacher='Original seed42 regularization on manual support only; ancestry unresolved',
    Frag6_used=False,physical_optical_CT_registration='unresolved; no new alignment claim',
    physical_ink_depth='unresolved; no recentering',maintainer_request='last user-confirmed NOT SENT; no reply supplied',
    other_reserves_untouched=['PHerc0009B','PHerc0500P2','PHerc0841'])
save(HERE/'REFERENCE_STATUS.json',refs)
evidence=['PROTOCOL.md','PLAN.json','GO.json','TEST_RECEIPT.json','REVIEW_FREEZE.json','REVIEW_FREEZE_V2.json',
    'TRAINING_VERIFICATION.json','EVALUATION_VERIFICATION.json','METRICS.json','FIGURES.json',
    'run-v1/COMPLETE.json','ROOT_REPORT_REVIEW.json','REFERENCE_STATUS.json','RESULTS.md',
    'NEXT_STUDY_DRAFT.md','STARTUP_AFTER_THIRTYSEVENTH.md']
contract=dict(schema_version=1,created_at_utc=now(),status=review['status'],decision=review['decision'],
    evidence={f:pin(HERE/f) for f in evidence},scientific_run_attempts=1,scientific_retries=0,
    completed_training_arms=2,optimizer_updates=1024,scientific_model_forwards=2146,
    training_batch_two_forwards=1346,development_single_tile_forwards=800,teacher_forwards=0,
    combined_scientific_wall_seconds=run['elapsed_seconds'],wall_cap_seconds=1800,worker_RSS_cap_bytes=4*1024**3,
    fixtures_passed=7,fixtures_failed=0,fixtures_errors=0,fixtures_skipped=0,
    training_scalar_values_verified=tv['scalar_values_checked'],rank_values_verified=ev['rank_values_checked'],
    blended_score_pixels_verified=ev['blended_score_pixels'],
    all_evaluation_regions_reused_development=True,independent_ink_accuracy_established=False,
    historical_control32_exact=True,baseline_promoted=False,full_17_plane_contract_retained=True,
    baseline_screen_passing_endpoints=[r for r,s in metrics['screens'].items() if s['passes']],
    network_requests=0,GPU_spend_GBP=0,GPU_pilot_total_cap_GBP=10,
    new_artifacts_uploaded_to_r2=False,submission=False,publication=False,outreach=False,
    transcription=None,translation=None,retention_scope='Local bytes; no external backup or restoration proof')
save(HERE/'COMPLETION_CONTRACT.json',contract)
prior=HERE.parent/'research-register-v32.json'
require(pin(prior)['sha256']=='58cb262a11aa1d173304de3cf351ac1290e33b6a1ec8447a01e4aa80160d3771','Prior register identity')
old=json.loads(prior.read_text());new=deepcopy(old)
new.update(created_at_utc=now(),scope='Additive phase37 register; all 36 prior experiment entries unchanged',
    previous_register=pin(prior),next_study=pin(HERE/'NEXT_STUDY_DRAFT.md'),
    readiness_check=dict(script=str(HERE/'verify_retained_state.py'),external_manifest_anchor='Saved additive phase37 start_skill prefix'))
new['experiments'].append(dict(id='thirtyseventh-experiment',purpose='Matched full-network CPU optimization budget and manual Dice objective comparison',
    status=review['status'],decision=review['decision'],conclusion=review['conclusion'],
    report=pin(HERE/'RESULTS.md'),completion_contract=pin(HERE/'COMPLETION_CONTRACT.json'),
    numerical_verification=[pin(HERE/'TRAINING_VERIFICATION.json'),pin(HERE/'EVALUATION_VERIFICATION.json')],
    training_parent_count=1,evaluation_parent_count=3,evaluation_regions=8,
    all_evaluation_regions_reused_development=True,independent_ink_accuracy_established=False,
    scientific_optimizer_updates=1024,scientific_model_forwards=2146,
    network_requests=0,GPU_spend_GBP=0,GPU_pilot_total_cap_GBP=10,new_artifacts_uploaded_to_r2=False,
    transcription=None,translation=None))
require(new['experiments'][:-1]==old['experiments'] and len(new['experiments'])==37,'All prior entries preserved')
save(HERE.parent/'research-register-v33.json',new)
print(json.dumps(dict(completion=pin(HERE/'COMPLETION_CONTRACT.json'),register=pin(HERE.parent/'research-register-v33.json'),prior_entries_preserved=36)))
