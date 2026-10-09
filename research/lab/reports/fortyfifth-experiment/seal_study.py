"""Append the completed study without rewriting any preceding scientific evidence."""
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
P=Path(__file__).resolve().parent
REPORTS=P.parent
def pin(p):
    with p.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(path=str(p),size_bytes=p.stat().st_size,sha256=sha)
def save(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def now():return datetime.now(timezone.utc).isoformat()
assert json.loads((P/'RESULT_VERIFICATION.json').read_text())['status']=='PASS'
assert json.loads((P/'geometry/VERIFICATION.json').read_text())['status']=='PASS'
assert json.loads((P/'PHERC343_MESH_REPLAY.json').read_text())['status']=='PASS'
prior_path=REPORTS/'fortyfourth-experiment/final-verification.json'
assert pin(prior_path)['sha256']=='d8d2d85a4320345419a3d444d33f0a5f77c3038551f10158b957080e92bb852e'
prior=json.loads(prior_path.read_text())
register=json.loads((REPORTS/'research-register-v40.json').read_text())
assert len(register['experiments'])==44
register['created_at_utc']=now()
register['previous_register']=pin(REPORTS/'research-register-v40.json')
register['scope']='Research history through45; selected Git export and full cloud evidence are distinct.'
register['next_study']=pin(P/'NEXT_STUDY_DRAFT.md')
register['research_direction']='Reproduce PHerc343 positive control, then PHerc0800 target transfer; paired PHerc0139 supporting benchmark. Profilometry and Paris4 request shelved.'
register['gpu_admission']=dict(criterion='Concrete experiment/data/runtime/cost/shutdown readiness; no universal CPU AP prerequisite.',gpu_total_cap_GBP=10,paid_job_launched=False)
register['readiness_check']=dict(script=str(P/'verify_retained_state.py'),manifest=str(P/'final-verification.json'),requires_external_manifest_sha256=True)
register['experiments'].append(dict(id='fortyfifth-experiment',purpose='Approved native8192 continuation, released-checkpoint variability, PHerc0800 mesh–M7 measurements and target-return source audit',status='completed; profilometry closed and PHerc343 positive-control reproduction recommended',result_path=str(P/'RESULTS.md'),result_sha256=pin(P/'RESULTS.md')['sha256'],held_out_accuracy_established=False,inference_scope='One optical-native training fold and four released checkpoints on reused eight-region panel; no new target ink inference',new_artifacts_uploaded_to_r2=False,letter=None,transcription=None,translation=None,gpu_spend_GBP=0))
regpath=REPORTS/'research-register-v41.json';save(regpath,register)
records={r['path']:r for r in prior['records']}
for original in list(records.values()):assert pin(Path(original['path']))==original,original['path']
new=[p for p in P.rglob('*') if p.is_file() and '__pycache__' not in p.parts and 'plot-cache' not in p.parts and p.name!='final-verification.json']+[regpath,prior_path]
for p in new:
    entry=pin(p)
    if str(p) in records:assert records[str(p)]==entry
    records[str(p)]=entry
ordered=[records[k] for k in sorted(records)]
out=dict(created_at_utc=now(),prior_manifest=pin(prior_path),prior_records_preserved=len(prior['records']),record_count=len(ordered),retained_bytes=sum(r['size_bytes'] for r in ordered),records=ordered,gpu_spend_GBP=0)
save(P/'final-verification.json',out)
print(json.dumps(dict(status='PASS',manifest=pin(P/'final-verification.json'),record_count=out['record_count'],retained_bytes=out['retained_bytes'],register=pin(regpath),prior_records_preserved=len(prior['records']))))
