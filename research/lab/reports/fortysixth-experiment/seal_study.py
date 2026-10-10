"""Append verified phase46 evidence without rewriting preceding studies."""
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
assert json.loads((P/'completion-check-20261010/VERIFICATION.json').read_text())['status']=='PASS'
for name in ['REPLAY.json','WHOLE_CROP_KDTREE.json','SYNTHETIC_REPLAY.json','completed-review/COMPLETION_REVIEW.json']:
    assert json.loads((P/'completion-review'/name).read_text())['status']=='PASS'
assert json.loads((P/'completion-review/ACTUAL_FINAL_MAP_CHECK.json').read_text())['status']=='FINAL_MAP_INTEGRITY_PASS'
state=json.loads((P/'PIPELINE_STATUS.json').read_text())
assert state['status']=='COMPUTE_COMPLETE_AWAITING_REVIEW' and all(x['exit_code']==0 for x in state['stages'])
priorpath=R/'fortyfifth-experiment/final-verification.json'
assert pin(priorpath)['sha256']=='96062d1bca457c134a814d2016862155e400dab02db91bb953fa92554f680851'
prior=json.loads(priorpath.read_text())
records={r['path']:r for r in prior['records']}
for n,r in enumerate(prior['records']):
    assert pin(Path(r['path']))==r,r['path']
    if n%5000==0: print('Verified prior',n,flush=True)
register=json.loads((R/'research-register-v41.json').read_text())
assert len(register['experiments'])==45
register.update(created_at_utc=datetime.now(timezone.utc).isoformat(),previous_register=pin(R/'research-register-v41.json'),
                scope='History through46; published-image reproduction qualifies PHerc343 positive control, not ink accuracy.',
                next_study=pin(P/'NEXT_STUDY_DRAFT.md'),
                readiness_check=dict(script=str(P/'verify_retained_state.py'),manifest=str(P/'final-verification.json'),requires_external_manifest_sha256=True))
register['experiments'].append(dict(id='fortysixth-experiment',purpose='Fixed PHerc343 four-model d+1 positive-control reproduction and independent completed-output review',
    status='COMPLETED_VERIFIED_PUBLISHED_IMAGE_REPRODUCTION',result_path=str(P/'RESULTS.md'),result_sha256=pin(P/'RESULTS.md')['sha256'],
    held_out_accuracy_established=False,inference_scope='Eight full PHerc343 maps; source-linked exposed published area; no new PHerc0800 ink inference',
    new_artifacts_uploaded_to_r2=False,letter=None,transcription=None,translation=None,gpu_spend_GBP=0))
omitted=[]
for f in sorted(P.rglob('*')):
    if not f.is_file() or f.is_symlink():continue
    rel=f.relative_to(P)
    if any(x in rel.parts for x in ['__pycache__','plot-cache','ct-cache','squashfs-root','renderer-cache']) or f.suffix.lower() in ['.appimage','.zip']:
        omitted.append(dict(path=str(rel),reason='Reacquirable raw CT cache, installed/downloaded software or generated cache'));continue
    entry=pin(f)
    assert str(f) not in records or records[str(f)]==entry
    records[str(f)]=entry
regpath=R/'research-register-v42.json'
save(regpath,register)
for f in [regpath,priorpath]:records[str(f)]=pin(f)
ordered=[records[k] for k in sorted(records)]
result=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),prior_manifest=pin(priorpath),
    prior_records_preserved=len(prior['records']),record_count=len(ordered),retained_bytes=sum(r['size_bytes'] for r in ordered),
    records=ordered,excluded_phase46=omitted,gpu_spend_GBP=0)
save(P/'final-verification.json',result)
print(json.dumps(dict(status='PASS',manifest=pin(P/'final-verification.json'),records=len(ordered),
    retained_bytes=result['retained_bytes'],prior_records_preserved=len(prior['records']),register=pin(regpath))))
