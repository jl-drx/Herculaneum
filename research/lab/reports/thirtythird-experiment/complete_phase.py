"""Write phase33 contracts and append one register entry, preserving prior entries."""
from diag_common import *
import subprocess
j=json.loads((HERE/'run-v1/COMPLETE.json').read_text());v=json.loads((HERE/'VERIFICATION.json').read_text())
require(not (HERE/'COMPLETION_CONTRACT.json').exists(),'Completion is write once')
require(subprocess.check_output(['git','-C','/workspace/Herculaneum','rev-parse','HEAD'],text=True).strip()=='49fabf3e890d5b880f391b27d581f1e7a1ebdbcc','Checkout HEAD')
require(subprocess.check_output(['git','-C','/workspace/Herculaneum','status','--short'],text=True)=='?? .codex/\n','Checkout status')
status=dict(parent='PHerc51',fragment='Frag6',contexts=4,model_scored_development=True,independent_ink_accuracy_established=False,
    optical_annotation_shared_provenance_possible=True,teacher_model_history='unresolved',maintainer_request='user-confirmed not sent',
    optical_label_consistency='strong in this panel; not independent truth',physical_registration='not established by raster shifts',
    other_reserves_untouched=['PHerc0009B','PHerc0500P2','PHerc0841'])
save(HERE/'REFERENCE_STATUS.json',status)
review=dict(status='PASS_WITH_SCIENTIFIC_LIMITATIONS',reviewer='root',independent_agent=False,
    numerical=pin(HERE/'VERIFICATION.json'),figures=[pin(p) for p in sorted((HERE/'figures-v2').glob('*.png'))],
    all_four_contexts_inspected=True,all_nine_shifts_retained=True,original_scores_preserved=True,
    layout_revision='Separate figures-v2 generated with fixed subplot margins; first renders and script retained',
    caveats=['Common support eliminates most changeable labels; no alignment proof','Moving support confounds displacement and coverage',
        'Interior AP depends on changed prevalence; only 37 and 167 ink pixels in weak interiors',
        'Optical-label agreement can be circular','Simple model failure does not establish absent CT signal',
        'Same four contexts, one parent, post hoc development; no independent benchmark claim'],
    decision='Retain baseline; one cached CT observability audit before a representation hypothesis and disjoint development replication')
save(HERE/'ROOT_REPORT_REVIEW.json',review)
evidence=['PROTOCOL.md','PLAN.json','GO.json','TEST_RECEIPT.json','run-v1/COMPLETE.json','VERIFICATION.json','all-metrics.csv',
          'RESULTS.md','NEXT_STUDY_DRAFT.md','REFERENCE_STATUS.json','ROOT_REPORT_REVIEW.json','STARTUP_AFTER_THIRTYTHIRD.md']
contract=dict(schema_version=1,status='completed_verified_cached_reference_sensitivity',created_at_utc=now(),
    evidence={Path(p).stem:pin(HERE/p) for p in evidence},parent_count=1,posthoc=True,
    fixtures_passed=6,fixtures_failed=0,fixtures_errors=0,fixtures_skipped=0,metric_records=276,undefined_metric_records=0,
    independent_rank_scalar_values=552,maximum_rank_difference=v['maximum_metric_difference'],
    original_primary_unchanged=True,common_known_pixels=15470,common_potentially_changeable_pixels=20,
    scientific_model_forwards=0,scientific_optimizer_updates=0,network_requests=0,new_artifacts_uploaded_to_r2=False,
    independent_agent_review=False,independent_ink_accuracy_established=False,maintainer_status='NOT SENT',
    CPU_seconds=j['elapsed_seconds'],CPU_peak_RSS_bytes=j['peak_RSS_bytes'],verification_seconds=v['elapsed_seconds'],
    verification_peak_RSS_bytes=v['peak_RSS_bytes'],GPU_spend_GBP=0,GPU_pilot_total_cap_GBP=10,
    decision=review['decision'],publication=False,submission=False,transcription=None,translation=None)
save(HERE/'COMPLETION_CONTRACT.json',contract)
prior=REPORTS/'research-register-v28.json';reg=json.loads(prior.read_text());old=reg['experiments'][:]
require(len(old)==32,'Prior register count')
reg['experiments'].append(dict(id='thirtythird-experiment',status=contract['status'],purpose='Cached label-shift, boundary-band and optical consistency diagnostic',
    conclusion='Small shifts and boundary exclusion do not resolve weak regions; common support mostly removes changing labels',decision=review['decision'],
    completion_contract=pin(HERE/'COMPLETION_CONTRACT.json'),report=pin(HERE/'RESULTS.md'),numerical_verification=pin(HERE/'VERIFICATION.json'),
    parent_count=1,posthoc=True,metric_records=276,scientific_model_forwards=0,scientific_optimizer_updates=0,network_requests=0,
    candidate_now_model_scored=True,independent_ink_accuracy_established=False,new_artifacts_uploaded_to_r2=False,
    GPU_spend_GBP=0,GPU_pilot_total_cap_GBP=10,transcription=None,translation=None))
reg['created_at_utc']=now();reg['previous_register']=pin(prior);reg['next_study']=pin(HERE/'NEXT_STUDY_DRAFT.md')
reg['readiness_check']=dict(script=str(HERE/'verify_retained_state.py'),external_manifest_anchor='Saved additive phase33 start_skill prefix')
require(reg['experiments'][:-1]==old,'Prior entries changed')
require(not (REPORTS/'research-register-v29.json').exists(),'Register is new')
save(REPORTS/'research-register-v29.json',reg)
# Exact byte-preserving adapter of the read-only verifier; new builder keeps prior identities.
(HERE/'verify_retained_state.py').write_bytes((REPORTS/'thirtysecond-experiment/verify_retained_state.py').read_bytes())
s=(REPORTS/'thirtysecond-experiment/build_retention_manifest.py').read_text()
s=s.replace('prior thirtyfirst identity','prior thirtysecond identity').replace("reports/thirtyfirst-experiment/final-verification.json","reports/thirtysecond-experiment/final-verification.json")
s=s.replace('85a25bc893ae1bda4f2e4ecece78010cf68fdf6b5680315e05476ee3ccb92459','93359692e5ff98a40526a58869fd33df2168b62a37038a4a20b43f1b3bffcaa3')
s=s.replace('research-register-v28.json','research-register-v29.json').replace('1024*1024**2','64*1024**2').replace('phase32 output','phase33 output')
(HERE/'build_retention_manifest.py').write_text(s)
print(json.dumps(dict(status='PASS',contract=pin(HERE/'COMPLETION_CONTRACT.json'),register=pin(REPORTS/'research-register-v29.json'),prior_entries_preserved=len(old))))
