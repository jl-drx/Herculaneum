"""Append verified phase50 evidence without rewriting preceding studies."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

P=Path(__file__).resolve().parent
R=P.parent

def pin(p):
    with p.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(path=str(p),size_bytes=p.stat().st_size,sha256=digest)

def save(p,v):
    with p.open('x') as f: json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')

assert not (P/'final-verification.json').exists()
for name in ['ARCHIVE_VERIFICATION.json','AUDIT_VERIFICATION_V3.json','ADMISSION.json','CONTEXT_VERIFICATION.json','RENDER_VERIFICATION.json','INFERENCE_RESULTS.json','VERIFICATION.json']:
    assert json.loads((P/name).read_text())['status']=='PASS',name
priorpath=R/'fortyninth-experiment/final-verification.json'
assert pin(priorpath)['sha256']=='0e864a6e95214235ee1d13c28cd6b415d8f0a51f497503fb1bab6972e0eef618'
prior=json.loads(priorpath.read_text())
records={r['path']:r for r in prior['records']}
for n,r in enumerate(prior['records']):
    assert pin(Path(r['path']))==r,r['path']
    if n%5000==0: print('Verified prior',n,flush=True)
register=json.loads((R/'research-register-v45.json').read_text())
assert len(register['experiments'])==49
register.update(created_at_utc=datetime.now(timezone.utc).isoformat(),previous_register=pin(R/'research-register-v45.json'),
                scope='History through50; independent audit replay, grid-aware128-square admission, observed rendering and16 fixed-centre target maps complete.',
                next_study=pin(P/'NEXT_STUDY_DRAFT.md'),
                readiness_check=dict(script=str(P/'verify_retained_state.py'),manifest=str(P/'final-verification.json'),requires_external_manifest_sha256=True))
register['experiments'].append(dict(id='fiftieth-experiment',purpose='PHerc0800 independent geometry replay, complete model-context admission and fixed-centre four-model comparison',
    status='COMPLETED_VERIFIED_GRID_AWARE_CANDIDATE_AND_FIXED_CENTRE_COMPARISON',result_path=str(P/'RESULTS.md'),result_sha256=pin(P/'RESULTS.md')['sha256'],
    held_out_accuracy_established=False,inference_scope='288 unique CPU forwards;16 maps on a fixed128-square core, both phases and depth orders; no held-out accuracy or reading claim',
    new_artifacts_uploaded_to_r2=False,letter=None,transcription=None,translation=None,gpu_spend_GBP=0))
omitted=[]
for f in sorted(P.rglob('*')):
    if not f.is_file() or f.is_symlink():continue
    rel=f.relative_to(P)
    if any(x in rel.parts for x in ['__pycache__']):
        omitted.append(dict(path=str(rel),reason='Generated Python bytecode cache'));continue
    entry=pin(f)
    assert str(f) not in records or records[str(f)]==entry
    records[str(f)]=entry
regpath=R/'research-register-v46.json'
save(regpath,register)
for f in [regpath,priorpath]:records[str(f)]=pin(f)
ordered=[records[k] for k in sorted(records)]
result=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),prior_manifest=pin(priorpath),
    prior_records_preserved=len(prior['records']),record_count=len(ordered),retained_bytes=sum(r['size_bytes'] for r in ordered),
    records=ordered,excluded=omitted,gpu_spend_GBP=0)
save(P/'final-verification.json',result)
print(json.dumps(dict(status='PASS',manifest=pin(P/'final-verification.json'),records=len(ordered),
    retained_bytes=result['retained_bytes'],prior_records_preserved=len(prior['records']),register=pin(regpath))))
