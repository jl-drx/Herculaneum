"""Seal the scientific comparison, leaving Git publication as a separate receipt."""
from copy import deepcopy
from common import *

prior=LAB/'reports/thirtyeighth-experiment/final-verification.json'
require(pin(prior)['sha256']=='85079e27d46a441059cb225178781a3d1b6cc85ffa5d41151cc9b0de9312a837','Phase38 anchor')
old_manifest=json.loads(prior.read_text());records={}
for r in old_manifest['records']:
    require(pin(r['path'])==r,'Prior evidence identity: '+r['path']);records[r['path']]=r
metrics=json.loads((HERE/'run-v1/METRICS.json').read_text());v=json.loads((HERE/'run-v1/VERIFICATION.json').read_text())
require(v['status']=='PASS' and v['rank_values_checked']==64,'Verified science')
review=dict(status='COMPLETED_NOT_PROMOTED',created_at_utc=now(),reviewer='root',figures_visually_reviewed=True,report_reviewed=True,decision='Retain seed42; propose one disjoint source-region replication; GPU deferred',conclusion='This fixed midpoint does not match the coverage aggregate/worst-region tradeoff; changed original-source exposure remains a confound',advancement_passed=metrics['advancement_screen']['passes'],descriptive_relation_passed=metrics['descriptive_relation']['passes'])
save(HERE/'ROOT_REPORT_REVIEW.json',review)
reference=json.loads((LAB/'reports/thirtyeighth-experiment/REFERENCE_STATUS.json').read_text());reference.update(created_at_utc=now(),new_training=False,new_acquisition=False,prediction_comparison_only=True)
save(HERE/'REFERENCE_STATUS.json',reference)
evidence=['PROTOCOL.md','PLAN.json','GO.json','TEST_RECEIPT.json','run-v1/METRICS.json','run-v1/COMPLETE.json','run-v1/VERIFICATION.json','RESULTS.md','FIGURES.json','ROOT_REPORT_REVIEW.json','REFERENCE_STATUS.json','NEXT_STUDY_DRAFT.md','STARTUP_AFTER_THIRTYNINTH.md']
contract=dict(schema_version=1,status=review['status'],decision=review['decision'],evidence={f:pin(HERE/f) for f in evidence},scientific_attempts=1,scientific_retries=0,optimizer_updates=0,new_model_forwards=0,new_source_acquisitions=0,rank_values_verified=64,midpoint_pixels_verified=1179648,fixtures_passed=4,all_evaluation_regions_reused_development=True,baseline_promoted=False,full_17_plane_contract_retained=True,CPU_monetary_cap=None,GPU_spend_GBP=0,GPU_pilot_total_cap_GBP=10,publication_scope='Local science sealed here; user-authorized GitHub export recorded separately',transcription=None,translation=None)
save(HERE/'COMPLETION_CONTRACT.json',contract)
old_register=LAB/'reports/research-register-v34.json'
require(pin(old_register)['sha256']=='f51d9af522fe677388f64ac1e29188cd4e518df79bef4aaaa54f409a26207229','Prior register')
old=json.loads(old_register.read_text());new=deepcopy(old)
new.update(created_at_utc=now(),scope='Additive phase39 register; all 38 prior entries unchanged',previous_register=pin(old_register),next_study=pin(HERE/'NEXT_STUDY_DRAFT.md'),readiness_check=dict(script=str(HERE/'verify_retained_state.py'),external_manifest_anchor='Saved phase39 startup prefix'))
new['experiments'].append(dict(id='thirtyninth-experiment',purpose='One fixed midpoint prediction comparison against coverage512',status=review['status'],decision=review['decision'],conclusion=review['conclusion'],report=pin(HERE/'RESULTS.md'),completion_contract=pin(HERE/'COMPLETION_CONTRACT.json'),numerical_verification=pin(HERE/'run-v1/VERIFICATION.json'),all_evaluation_regions_reused_development=True,independent_ink_accuracy_established=False,scientific_optimizer_updates=0,scientific_model_forwards=0,GPU_spend_GBP=0,GPU_pilot_total_cap_GBP=10,transcription=None,translation=None))
require(new['experiments'][:-1]==old['experiments'] and len(new['experiments'])==39,'Prior entries preserved')
register=LAB/'reports/research-register-v35.json';save(register,new)
destination=HERE/'final-verification.json'
for path in [prior,register,*[p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts]]:
    item=pin(path);require(item['path'] not in records or records[item['path']]==item,'No historical mutation');records[item['path']]=item
ordered=[records[k] for k in sorted(records)];phase_bytes=sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())
manifest=dict(schema_version=1,status='PASS',created_at_utc=now(),prior_manifest=pin(prior),prior_records_preserved=len(old_manifest['records']),records=ordered,record_count=len(ordered),retained_bytes=sum(r['size_bytes'] for r in ordered),phase_bytes_before_final_manifest=phase_bytes,checkout_baseline='49fabf3e890d5b880f391b27d581f1e7a1ebdbcc',checkout_policy='Authorized additive Git research export; sealed research bytes remain immutable',scope='Local byte-integrity retention; GitHub export and replay evidence recorded separately',gpu_spend_GBP=0,total_first_GPU_pilot_cap_GBP=10)
save(destination,manifest);require(phase_bytes+destination.stat().st_size<64*1024**2,'Phase output bound')
print(json.dumps(dict(manifest=pin(destination),register=pin(register),records=len(ordered),retained_bytes=manifest['retained_bytes'])))
