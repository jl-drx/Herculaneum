"""Seal phase34 contracts and append a register entry without changing old evidence."""
import subprocess
from audit_common import *
require(not (HERE/'COMPLETION_CONTRACT.json').exists(),'New completion contract')
run=json.loads((HERE/'run-v1/COMPLETE.json').read_text());v=json.loads((HERE/'VERIFICATION.json').read_text());check_freeze()
require(subprocess.check_output(['git','-C','/workspace/Herculaneum','rev-parse','HEAD'],text=True).strip()=='49fabf3e890d5b880f391b27d581f1e7a1ebdbcc','Checkout HEAD')
require(subprocess.check_output(['git','-C','/workspace/Herculaneum','status','--short'],text=True)=='?? .codex/\n','Checkout status')
reference=json.loads((REPORTS/'thirtythird-experiment/REFERENCE_STATUS.json').read_text())
reference['phase34']='Fixed cached CT scalar-feature audit; reference/geometry and independent accuracy remain unqualified'
save(HERE/'REFERENCE_STATUS.json',reference)
review=dict(status='PASS_WITH_INCONCLUSIVE_REPRESENTATION_INFERENCE',reviewer='root',independent_agent=False,
    numerical=pin(HERE/'VERIFICATION.json'),figures=[pin(p) for p in sorted((HERE/'figures').glob('*.png'))],
    all_contexts_and_depths_inspected=True,all_signed_variants_retained=True,original_primary_unchanged=True,
    decision='Retain baseline; stop feature/depth sweeps and prepare a concrete geometry/reference review packet',
    limits=['Scalar-feature failure cannot establish absent ink or exclude a learned representation',
        'Fixed contrast also weak in regions where the CT backbone ranks labels well',
        'Historical layers do not authenticate physical ink depth or optical registration',
        'Interior sparse positives and dependent pixels/depths limit inference',
        'One previously scored parent with unresolved exposure/reference history'])
save(HERE/'ROOT_REPORT_REVIEW.json',review)
evidence=['PROTOCOL.md','PLAN.json','GO.json','TEST_RECEIPT.json','SETUP_RECEIPT.json','run-v1/COMPLETE.json','VERIFICATION.json',
    'all-metrics.csv','all-profile-summaries.csv','all-class-distributions.csv','RESULTS.md','NEXT_STUDY_DRAFT.md',
    'REFERENCE_STATUS.json','ROOT_REPORT_REVIEW.json','STARTUP_AFTER_THIRTYFOURTH.md']
contract=dict(schema_version=1,status='completed_verified_cached_CT_feature_audit_inconclusive_for_representation',created_at_utc=now(),
    evidence={Path(p).stem:pin(HERE/p) for p in evidence},parent_count=1,posthoc_development=True,
    fixtures_passed=7,fixtures_failed=0,fixtures_errors=0,fixtures_skipped=0,metric_records=2688,undefined_metric_records=0,
    verified_counts=v['counts'],maximum_rank_difference=v['maximum_rank_difference'],maximum_distribution_difference=v['maximum_distribution_difference'],
    CPU_seconds=run['elapsed_seconds'],CPU_peak_RSS_bytes=run['peak_RSS_bytes'],verification_seconds=v['elapsed_seconds'],verification_peak_RSS_bytes=v['peak_RSS_bytes'],
    original_primary_unchanged=True,scientific_run_attempts=1,scientific_model_forwards=0,scientific_optimizer_updates=0,network_requests=0,
    new_artifacts_uploaded_to_r2=False,independent_agent_review=False,independent_ink_accuracy_established=False,
    maintainer_status='NOT SENT; no new reply supplied',GPU_spend_GBP=0,GPU_pilot_total_cap_GBP=10,decision=review['decision'],
    publication=False,submission=False,transcription=None,translation=None)
save(HERE/'COMPLETION_CONTRACT.json',contract)
prior=REPORTS/'research-register-v29.json';reg=json.loads(prior.read_text());old=reg['experiments'][:];require(len(old)==33,'Prior register count')
reg['experiments'].append(dict(id='thirtyfourth-experiment',status=contract['status'],purpose='Fixed all-depth cached intensity/local-contrast audit',
    conclusion='No original-support local-contrast AUROC >0.6 at any depth/sign/scale/context; learned representation inference inconclusive',
    decision=review['decision'],completion_contract=pin(HERE/'COMPLETION_CONTRACT.json'),report=pin(HERE/'RESULTS.md'),numerical_verification=pin(HERE/'VERIFICATION.json'),
    parent_count=1,posthoc=True,metric_records=2688,scientific_model_forwards=0,scientific_optimizer_updates=0,network_requests=0,
    candidate_now_model_scored=True,independent_ink_accuracy_established=False,new_artifacts_uploaded_to_r2=False,
    GPU_spend_GBP=0,GPU_pilot_total_cap_GBP=10,transcription=None,translation=None))
reg['created_at_utc']=now();reg['previous_register']=pin(prior);reg['next_study']=pin(HERE/'NEXT_STUDY_DRAFT.md')
reg['readiness_check']=dict(script=str(HERE/'verify_retained_state.py'),external_manifest_anchor='Saved additive phase34 start_skill prefix')
require(reg['experiments'][:-1]==old,'Prior entries changed');require(not (REPORTS/'research-register-v30.json').exists(),'New register')
save(REPORTS/'research-register-v30.json',reg)
(HERE/'verify_retained_state.py').write_bytes((REPORTS/'thirtythird-experiment/verify_retained_state.py').read_bytes())
s=(REPORTS/'thirtythird-experiment/build_retention_manifest.py').read_text()
s=s.replace('prior thirtysecond identity','prior thirtythird identity').replace('reports/thirtysecond-experiment/final-verification.json','reports/thirtythird-experiment/final-verification.json')
s=s.replace('93359692e5ff98a40526a58869fd33df2168b62a37038a4a20b43f1b3bffcaa3','7e0a3d35f77f9802e1339b9621b434a9a9f84b6057d6b5fb19919e20ad25b3bb')
s=s.replace('research-register-v29.json','research-register-v30.json').replace('phase33 output','phase34 output')
(HERE/'build_retention_manifest.py').write_text(s)
print(json.dumps(dict(status='PASS',contract=pin(HERE/'COMPLETION_CONTRACT.json'),register=pin(REPORTS/'research-register-v30.json'),prior_entries_preserved=len(old))))
