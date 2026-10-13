"""Package finished PHerc343 results only; no inference or credential import."""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
import tarfile


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


ap=argparse.ArgumentParser()
ap.add_argument('--lab',type=Path,required=True)
ap.add_argument('--bundle',type=Path,required=True)
ap.add_argument('--manifest',type=Path,required=True)
a=ap.parse_args()
p=a.lab/'reports/fortysixth-experiment'
assert json.loads((p/'FIXED_DEPTH_INFERENCE.json').read_text())['status']=='PASS'
assert json.loads((p/'PIPELINE_STATUS.json').read_text())['status']=='COMPUTE_COMPLETE_AWAITING_REVIEW'
assert json.loads((p/'completion-check-20261010/VERIFICATION.json').read_text())['status']=='PASS'
files=set()
for directory in ['maps_plain','maps_padded','maps','image']:
    files.update(f for f in (p/'work'/directory).rglob('*') if f.is_file())
for name in ['FIXED_DEPTH_INFERENCE.json','IMAGE_COMPARISON.json','PIPELINE_STATUS.json',
             'FIXED_DISPLAY_STATS.json','PROTOCOL.md','INFERENCE_QUALIFICATION.json',
             'REAL_INPUT_VERIFICATION.json','RENDER_VERIFICATION.json',
             'run_fixed_depth.py','run_fixed_depth_resume.py','compose_and_compare.py',
             'execute_pipeline.py','execute_resume.py','prepare_and_qualify.py',
             'work/valid.npy','work/hecate-checkpoints/BINDING.json','work/hecate-checkpoints/LATEST.json',
             'work/hecate-checkpoints/WATCHER_STATUS.json']:
    files.add(p/name)
files.update((p/'completion-check-20261010').glob('*.py'))
files.update((p/'completion-check-20261010').glob('*.json'))
files.update((p/'work/hecate-checkpoints').glob('rows-*/MANIFEST.json'))
pipeline=Path(json.loads((p/'PIPELINE_STATUS.json').read_text())['run_directory'])
files.update(f for f in pipeline.iterdir() if f.is_file())
old=a.lab/'reports/fortyfifth-experiment'
stem='concat_w047-w048_R5B2_z9500-11000'
files.update([old/'source-evidence'/(stem+'.png'),
              old/'source-evidence/pherc343/outputs'/(stem+'.json'),
              old/'ph343-mesh-replay'/(stem+'_node_source.npz')])
patterns=[rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
          rb'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|(?:AKIA|ASIA)[A-Z0-9]{16})\b',
          rb'[?&](?:X-Amz-Signature|X-Amz-Security-Token|access_token)=',
          rb'https?://[^\s/:]+:[^\s/@]+@']
records=[]
for f in sorted(files):
    assert f.is_file() and not f.is_symlink()
    if f.suffix in {'.json','.py','.md','.log','.csv'}:
        assert not any(re.search(rx,f.read_bytes(),re.I) for rx in patterns), 'Review credential-like content in '+str(f)
    records.append({'path':f.relative_to(a.lab).as_posix(),'bytes':f.stat().st_size,'sha256':sha(f)})
with a.bundle.open('xb') as raw, gzip.GzipFile(fileobj=raw,mode='wb',compresslevel=1,mtime=0) as gz, tarfile.open(fileobj=gz,mode='w|') as tf:
    for r in records:
        f=a.lab/r['path']
        info=tf.gettarinfo(str(f),arcname=r['path'])
        info.uid=info.gid=0;info.uname=info.gname='';info.mode=0o644;info.mtime=0
        with f.open('rb') as stream:tf.addfile(info,stream)
        assert sha(f)==r['sha256'] and f.stat().st_size==r['bytes']
with tarfile.open(a.bundle,'r:gz') as tf:
    members=tf.getmembers()
    assert [m.name for m in members]==[r['path'] for r in records]
    for m,r in zip(members,records):
        assert m.isfile() and m.size==r['bytes']
        assert hashlib.file_digest(tf.extractfile(m),'sha256').hexdigest()==r['sha256']
report={'schema':1,'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'Completed eight raw maps, four ownership merges, fixed composition and components, published reference, valid mask, receipts, frozen recipes, checkpoint metadata. No model weights, CT volumes or credential values.',
        'bundle':{'file':a.bundle.name,'bytes':a.bundle.stat().st_size,'sha256':sha(a.bundle)},
        'members':records,'expanded_bytes':sum(r['bytes'] for r in records),
        'archive_readback':'PASS: every archived member byte count and SHA256 checked against its source',
        'prior_snapshot_unchanged':True,
        'checkpoint_note':'All 60 prefix arrays matched the finished map before packaging. Metadata/verification ledger retained; redundant prefix arrays are not included. This bundle is completed output, not a restart checkpoint.',
        'scientific_scope':'Published-image reproduction; no raw-author-map equality, ink accuracy or letter authentication.'}
with a.manifest.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps({'status':'PASS','members':len(records),'bundle':report['bundle'],'expanded_bytes':report['expanded_bytes']},indent=2))
