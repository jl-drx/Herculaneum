"""Seal the bounded reference search without changing prior evidence."""
from copy import deepcopy
from datetime import datetime,timezone
import json
from pathlib import Path
from analyze_references import P,pin,check
R=P.parent
def now():return datetime.now(timezone.utc).isoformat()
def put(path,value):
 with Path(path).open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
prior=R/'fortysecond-experiment/final-verification.json'
check(pin(prior)['sha256']=='9d51f515f138bb1f07a0cb89d2912424cb87546d86ae92f021326779b3a55fb1','Prior manifest anchor')
old=json.loads(prior.read_text());records={}
for row in old['records']:check(pin(row['path'])==row,'Prior evidence changed');records[row['path']]=row
v=json.loads((P/'SOURCE_VERIFICATION.json').read_text());f=json.loads((P/'FINDINGS.json').read_text());review=json.loads((P/'ROOT_REPORT_REVIEW.json').read_text())
check(v['status']=='PASS' and review['report_reviewed'] and not f['CT_reference_admitted'],'Verified reference outcome')
for item in json.loads((P/'GO.json').read_text())['pins']:check(pin(item['path'])==item,'GO pin changed')
for row in json.loads((P/'SOURCE_EXPORT_MAP.json').read_text()):
 check(Path(row['source']).read_bytes()==Path(row['copy']).read_bytes(),'Exact source metadata export copy')
evidence=['PROTOCOL.md','GO.json','REQUESTS.json','SOURCE_VERIFICATION.json','FINDINGS.json','LISTING_SUMMARY.json','EXTENT_ARITHMETIC.json','AUDIT_CORRECTION.md','RESULTS.md','NEXT_STUDY_DRAFT.md','REFERENCE_REQUEST_DRAFT.md','ROOT_REPORT_REVIEW.json','SOURCE_EXPORT_MAP.json','STARTUP_AFTER_FORTYTHIRD.md']
contract=dict(schema_version=1,status=review['status'],decision=review['decision'],evidence={name:pin(P/name) for name in evidence},reference_targets=['PHercParis4','PHerc0500P2'],new_metadata_GET_requests=24,new_metadata_bytes=429661,HTTP_retries=0,failed_HTTP_requests=0,new_scientific_media_payload_requests=0,new_CT_chunks=0,new_label_arrays=0,new_model_calls=0,new_optimizer_updates=0,CT_reference_admitted=False,profilometry_reference_admitted=False,profilometry_release_metadata_accessible=True,profilometry_image_bodies_inspected=False,profilometry_sampling_calibration_resolved=False,external_messages_sent=0,request_status='DRAFT_UNSENT',CPU_monetary_cap=None,GPU_spend_GBP=0,GPU_pilot_total_cap_GBP=10,new_R2_objects=0,previous_baseline_promoted=False,full_historical_depth_contract_retained=True,transcription=None,translation=None,scope='Reference-source search and supporting dataset discovery; no physical validation or learning')
put(P/'COMPLETION_CONTRACT.json',contract)
oldreg=R/'research-register-v38.json';check(pin(oldreg)['sha256']=='5d238b68304f275e225f281d334fbc8bd39c28fbdc46a04749f27828686b4aca','Prior register anchor');oreg=json.loads(oldreg.read_text());reg=deepcopy(oreg)
reg.update(created_at_utc=now(),scope='Additive phase43 register; all42 prior entries unchanged',previous_register=pin(oldreg),next_study=pin(P/'NEXT_STUDY_DRAFT.md'),readiness_check=dict(script=str(P/'verify_retained_state.py'),external_manifest_anchor='Saved phase43 startup prefix'))
reg['experiments'].append(dict(id='fortythird-experiment',purpose='Bounded coordinate/reference search and public profilometry-release assessment',status=review['status'],decision=review['decision'],conclusion=review['conclusion'],report=pin(P/'RESULTS.md'),completion_contract=pin(P/'COMPLETION_CONTRACT.json'),numerical_verification=[pin(P/'SOURCE_VERIFICATION.json')],independent_ink_accuracy_established=False,scientific_optimizer_updates=0,scientific_model_forwards=0,new_scientific_media_payload_acquisition_requests=0,new_metadata_requests=24,new_metadata_bytes=429661,GPU_spend_GBP=0,GPU_pilot_total_cap_GBP=10,transcription=None,translation=None))
check(reg['experiments'][:-1]==oreg['experiments'] and len(reg['experiments'])==43,'Prior register entries preserved');destreg=R/'research-register-v39.json';put(destreg,reg)
for path in [prior,destreg,*[p for p in P.rglob('*') if p.is_file() and '__pycache__' not in p.parts]]:
 item=pin(path);check(item['path'] not in records or records[item['path']]==item,'Historical record collision');records[item['path']]=item
ordered=[records[k] for k in sorted(records)];manifest=dict(schema_version=1,status='PASS',created_at_utc=now(),prior_manifest=pin(prior),prior_records_preserved=len(old['records']),records=ordered,record_count=len(ordered),retained_bytes=sum(r['size_bytes'] for r in ordered),scope='Local retained byte integrity; selected GitHub publication tracked separately; full archive not remotely backed up',gpu_spend_GBP=0,total_first_GPU_pilot_cap_GBP=10)
dest=P/'final-verification.json';put(dest,manifest)
print(json.dumps(dict(manifest=pin(dest),register=pin(destreg),records=len(ordered),retained_bytes=manifest['retained_bytes'])))
