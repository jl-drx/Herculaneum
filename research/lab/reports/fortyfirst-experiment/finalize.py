"""Seal the completed control while preserving every earlier evidence record."""
from copy import deepcopy
from common import *
prior=REPORTS/'fortieth-experiment/final-verification.json'
require(pin(prior)['sha256']=='1412022014d6fedb7bfd4e0540c5ee0558c4f0c1f87e7afe4a667d6824a52915','Prior manifest anchor')
old_manifest=json.loads(prior.read_text());records={}
for item in old_manifest['records']:verify(item);records[item['path']]=item
review=json.loads((HERE/'ROOT_REPORT_REVIEW.json').read_text());run=json.loads((HERE/'run-v1/COMPLETE.json').read_text());tv=json.loads((HERE/'TRAINING_VERIFICATION.json').read_text());ev=json.loads((HERE/'EVALUATION_VERIFICATION.json').read_text());m=json.loads((HERE/'METRICS.json').read_text())
require(review['reviewer']=='root' and review['report_reviewed'] and review['figures_visually_reviewed'],'Root review complete')
require(run['status']==tv['status']==ev['status']=='PASS','Verified science')
reference=json.loads((REPORTS/'fortieth-experiment/REFERENCE_STATUS.json').read_text())
reference.update(created_at_utc=now(),new_training=True,new_acquisition=False,new_scientific_payload_reads=0,reused_phase40_teacher=True,new_teacher_forwards=0,intervention='Remove new crop BCE gradient only; retain label-selected support and joint KL',public_metadata_research=True,paired_learning_performed=False,independent_ink_accuracy=False)
save(HERE/'REFERENCE_STATUS.json',reference)
evidence=['PROTOCOL.md','PLAN.json','GO.json','TEST_RECEIPT.json','PREFLIGHT_NOTES.md','TRAINING_VERIFICATION.json','EVALUATION_VERIFICATION.json','METRICS.json','run-v1/COMPLETE.json','ROOT_REPORT_REVIEW.json','REFERENCE_STATUS.json','RESULTS.md','FIGURES.json','PAIRED_ACQUISITION_REVIEW.md','NEXT_STUDY_DRAFT.md','STARTUP_AFTER_FORTYFIRST.md','research/PAIR_FEASIBILITY.json']
contract=dict(schema_version=1,status=review['status'],decision=review['decision'],evidence={f:pin(HERE/f) for f in evidence},scientific_attempts=1,scientific_retries=0,optimizer_updates=512,new_model_forwards=905,teacher_forwards=0,reused_teacher_batches=32,student_batch_two_forwards=705,development_single_tile_forwards=200,scientific_payload_acquisition_requests=0,training_loss_rows_verified=tv['loss_rows_checked'],training_scalar_values_verified=tv['scalar_values_checked'],rank_values_verified=ev['rank_values_checked'],blended_score_pixels_verified=ev['blended_score_pixels'],fixtures_passed=8,advancement_passed=m['screen']['passes'],supervision_contribution=m['supervision_contribution'],all_evaluation_regions_reused_development=True,independent_ink_accuracy=False,baseline_promoted=False,full_17_plane_contract_retained=True,CPU_monetary_cap=None,GPU_spend_GBP=0,GPU_pilot_total_cap_GBP=10,new_artifacts_uploaded_to_R2=False,next_paired_learning_run_performed=False,publication_scope='Local science sealed; authorized GitHub source/results publication recorded separately',transcription=None,translation=None)
save(HERE/'COMPLETION_CONTRACT.json',contract)
old_register=REPORTS/'research-register-v36.json';require(pin(old_register)['sha256']=='6976984e650c5bf10735a5df114c80da13f7f40ddf9e69e4ea4bd4c2f6c81d6d','Prior register anchor')
old=json.loads(old_register.read_text());new=deepcopy(old)
new.update(created_at_utc=now(),scope='Additive phase41 register; all40 prior entries unchanged',previous_register=pin(old_register),next_study=pin(HERE/'NEXT_STUDY_DRAFT.md'),readiness_check=dict(script=str(HERE/'verify_retained_state.py'),external_manifest_anchor='Saved phase41 startup prefix'))
new['experiments'].append(dict(id='fortyfirst-experiment',purpose='Isolate phase40 new-crop supervised-gradient contribution with fixed original BCE weighting and joint teacher KL',status=review['status'],decision=review['decision'],conclusion=review['conclusion'],report=pin(HERE/'RESULTS.md'),completion_contract=pin(HERE/'COMPLETION_CONTRACT.json'),numerical_verification=[pin(HERE/'TRAINING_VERIFICATION.json'),pin(HERE/'EVALUATION_VERIFICATION.json')],all_evaluation_regions_reused_development=True,independent_ink_accuracy_established=False,scientific_optimizer_updates=512,scientific_model_forwards=905,new_scientific_payload_acquisition_requests=0,GPU_spend_GBP=0,GPU_pilot_total_cap_GBP=10,transcription=None,translation=None))
require(new['experiments'][:-1]==old['experiments'] and len(new['experiments'])==41,'All prior register entries preserved')
register=REPORTS/'research-register-v37.json';save(register,new)
for path in [prior,register,*[p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts]]:
 item=pin(path);require(item['path'] not in records or records[item['path']]==item,'Historical mutation');records[item['path']]=item
ordered=[records[k] for k in sorted(records)];phase_bytes=sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())
manifest=dict(schema_version=1,status='PASS',created_at_utc=now(),prior_manifest=pin(prior),prior_records_preserved=len(old_manifest['records']),records=ordered,record_count=len(ordered),retained_bytes=sum(r['size_bytes'] for r in ordered),phase_bytes_before_final_manifest=phase_bytes,scope='Local byte integrity; GitHub source/results publication tracked separately; full archive not remotely backed up',gpu_spend_GBP=0,total_first_GPU_pilot_cap_GBP=10)
destination=HERE/'final-verification.json';save(destination,manifest);require(phase_bytes+destination.stat().st_size<2*1024**3,'Phase bound')
print(json.dumps(dict(manifest=pin(destination),register=pin(register),records=len(ordered),retained_bytes=manifest['retained_bytes'])))
