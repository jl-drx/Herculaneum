from pathlib import Path
import hashlib,json,datetime
P=Path(__file__).resolve().parent;R=P.parent

def pin(path):
 path=Path(path).resolve(strict=True)
 with path.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
 return dict(path=str(path),sha256=digest,size_bytes=path.stat().st_size)
def write(path,d):
 with path.open('x') as f:json.dump(d,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
seals={'input-audit/RETENTION_MANIFEST.json':'a5efffb3dd741ba1fd9d20218f6ba95147cb679e8f63fdc9af7a29f817588b8c','implementation/RETENTION_MANIFEST.json':'0d59cd5fb90d1c2f5c1cf488dabf4040f72798e63c3fb18c772e5607ef3ad726','independent-review/RETENTION_MANIFEST.json':'66f81d2c341063af67cac9b54c5f2d234fefff589014a03faa7fe139219bb883','visuals/RETENTION_MANIFEST.json':'07c392e5ce2273a6d67d0192bc79cc6cc719954740bd5621115c74cc7a5f827f'}
verified=[]
for rel,sha in seals.items():
 f=P/rel;assert pin(f)['sha256']==sha;d=json.loads(f.read_text())
 records=d.get('files',d.get('records'));assert isinstance(records,list)
 for item in records:
  path=Path(item['path']);path=path if path.is_absolute() else f.parent/path
  actual=pin(path);assert actual['sha256']==item['sha256'];assert actual['size_bytes']==item.get('size_bytes',item.get('byte_length'))
 verified.append(dict(manifest=pin(f),count=len(records)))
metrics=json.loads((P/'implementation/analysis-v1/metrics.json').read_text());review=json.loads((P/'independent-review/verification-v1.json').read_text());assert metrics['status']==review['status']=='PASS'
controller=json.loads((P/'implementation/run-v1/controller-result.json').read_text());assert controller['status']=='PASS'
counts=dict(AP_improved=sum(x['native_minus_seed42']['average_precision']>0 for x in metrics['per_context'].values()),AUC_improved=sum(x['native_minus_seed42']['roc_auc']>0 for x in metrics['per_context'].values()))
assert counts==dict(AP_improved=4,AUC_improved=3)
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
write(P/'ROOT_REPORT_REVIEW.json',dict(status='PASS',created_at_utc=now,seals=verified,report=pin(P/'RESULTS.md'),visual_review=pin(P/'ROOT_VISUAL_REVIEW.json'),next_study=pin(P/'NEXT_STUDY_DRAFT.md'),context_direction_counts=counts,interpretation='Mixed descriptive development result, no performance gate/promotion/independence claim; all6 contexts reported and all panels viewed.'))
evidence={name:pin(P/rel) for name,rel in {'report':'RESULTS.md','root_review':'ROOT_REPORT_REVIEW.json','input_contract':'input-audit/INPUT_CONTRACT.json','input_verification':'input-audit/verification.json','plan':'implementation/PLAN.json','protocol':'PROTOCOL.md','native_result':'implementation/run-v1/native/result.json','metrics':'implementation/analysis-v1/metrics.json','independent_verification':'independent-review/verification-v1.json','visual_review':'ROOT_VISUAL_REVIEW.json','plot_receipt':'visuals/render-v1/PLOT_RECEIPT.json','recovery':'INFRASTRUCTURE_RECOVERY.json','next_study':'NEXT_STUDY_DRAFT.md'}.items()}
contract=dict(status='completed_execution_and_numerics_verified_mixed_development_transfer',created_at_utc=now,
 adopted_baseline='unchanged strict seed42',promoted_model=None,
 conclusion='Native improves Paris2 but regresses Paris1; not a consistent replacement under matched inputs.',
 decision='Retain seed42 baseline and native contrasting reference; next a fixed paired depth-direction diagnostic, no tuning/GPU scaling.',
 scientific_model_forwards=150,seed42_model_forwards=0,native_model_forwards=150,model_constructions=1,strict_state_loads=1,
 scientific_optimizer_updates=0,threshold_fits=0,new_candidate_scores=0,reused_development_contexts=6,physical_parents=2,
 primary_pixels=520794,primary_positive=86535,primary_negative=434259,parent_metrics=metrics['per_parent'],context_direction_counts=counts,
 scientific_performance_gate=None,independent_confirmation_ready=False,source_input_state_unchanged=True,
 external_inference_seconds=controller['elapsed_seconds'],network_requests=0,network_response_body_bytes=0,automatic_retries=0,
 GPU_spend_GBP=0,GPU_pilot_total_cap_GBP=10,R2_worker_secret_delivery_verified=False,new_artifacts_uploaded_to_r2=False,
 Frag6_stage_C_authorized=False,publication=None,submission=None,transcription=None,translation=None,evidence=evidence)
write(P/'COMPLETION_CONTRACT.json',contract)
previous=R/'research-register-v18.json';assert pin(previous)['sha256']=='1f01526ed82b375a3c4dccb4817bbc3889ea8156bab4b0d3b40e375fa69c35d8'
r=json.loads(previous.read_text());old=json.loads(json.dumps(r['experiments']));assert len(old)==22
r['experiments'].append(dict(id='twentythird-experiment',purpose='Fixed native versus seed42 reference comparison on exact reused development inputs',completion_contract=pin(P/'COMPLETION_CONTRACT.json'),report=pin(P/'RESULTS.md'),numerical_verification=pin(P/'independent-review/verification-v1.json'),**{k:contract[k] for k in ('status','conclusion','decision','promoted_model','scientific_model_forwards','scientific_optimizer_updates','new_candidate_scores','independent_confirmation_ready','GPU_spend_GBP','GPU_pilot_total_cap_GBP','network_requests','network_response_body_bytes','new_artifacts_uploaded_to_r2','transcription','translation')},parent_metrics=metrics['per_parent']))
assert r['experiments'][:-1]==old
r.update(created_at_utc=now,scope='Additive register after phase23; first22 experiments unchanged',previous_register=pin(previous),next_study={**pin(P/'NEXT_STUDY_DRAFT.md'),'status':'User requested continued testing; separate phase24 execution protocol must be frozen before reversal outputs'})
r['readiness_check'].update(CPU_research='Phase23 independently verified; exact inputs and both checkpoint forward outputs ready for a separately frozen direction diagnostic',GPU_launch='No launch: mixed development transfer; useful workload,pinnedCUDA,pricing,independentshutdown and actualscopedworker delivery still required')
write(R/'research-register-v19.json',r)
print(json.dumps(dict(status='PASS',completion=pin(P/'COMPLETION_CONTRACT.json'),register=pin(R/'research-register-v19.json'),seals_verified=len(seals))))
