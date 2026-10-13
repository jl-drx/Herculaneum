"""Read-only verification of archived audit, render metrics,16 maps and cache receipts."""
from pathlib import Path
import hashlib,json,itertools
import numpy as np
P=Path(__file__).resolve().parent;W=P/'work';A=P/'independent-audit'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def translated(p):
 p=Path(p);return P/p.relative_to('/workspace/herculaneum-lab/reports/fiftieth-experiment')
def check_pin(r):
 p=translated(r['path']);assert p.stat().st_size==r['size_bytes'] and sha(p)==r['sha256'],str(p)
archive=json.loads((P/'ARCHIVE_VERIFICATION.json').read_text());assert sha(P/'independent-audit.zip')==archive['archive_sha256']
files=0
for line in (A/'SHA256SUMS').read_text().splitlines():
 h,n=line.split(maxsplit=1);assert sha(A/n.lstrip('*'))==h;files+=1
receipts=json.loads((P/'SOURCE_RECEIPTS.json').read_text())['receipts'];keys=json.loads((P/'RENDER_GO.json').read_text())['source_keys']
for key in keys:
 rec=next(r for r in receipts if r['key']==key);assert sha(W/'source-cache'/key)==rec['sha256'];assert (W/'source-cache'/key).stat().st_size==128**3
mask=np.load(W/'render-observed-mask.npy');a=np.load(W/'candidate-render66.npy');b=np.load(W/'candidate-independent66.npy');d=np.abs(a[:,mask].astype(np.int16)-b[:,mask].astype(np.int16))
st=dict(samples=int(d.size),exact=int((d==0).sum()),exact_fraction=float((d==0).mean()),mean_abs=float(d.mean()),max_abs=int(d.max()),within_one_fraction=float((d<=1).mean()))
assert st==json.loads((P/'RENDER_VERIFICATION.json').read_text())['candidate_comparison'];assert st['max_abs']<=1 and st['exact_fraction']>=.95 and st['mean_abs']<=.1
r=json.loads((P/'INFERENCE_RESULTS.json').read_text());go=json.loads((P/'INFERENCE_GO.json').read_text());go_sha=sha(P/'INFERENCE_GO.json');models=[x['name'] for x in go['models']]
assert {(x['model'],x['phase'],x['direction']) for x in r['maps']}==set(itertools.product(models,['plain','padded'],['forward','reverse']))
count=0;new=0
for row in r['maps']:
 assert row['status']=='PASS' and row['go_sha256']==go_sha
 for p in row['outputs'].values():check_pin(p)
 aligned=np.load(translated(row['outputs']['aligned']['path']));display=np.load(translated(row['outputs']['display']['path']));assert aligned.shape==(128,128) and np.isfinite(aligned).all();assert np.array_equal(display,np.rint(aligned*255).astype(np.uint8))
 folder=translated(row['outputs']['aligned']['path']).parent;logs=sorted(folder.glob('tile-*.json'));assert len(logs)==row['counts']['model_calls']
 for f in logs:
  rec=json.loads(f.read_text());assert rec['binding']==dict(go_sha256=go_sha,job=[row['model'],row['phase'],row['direction']]);check_pin(rec['logits']);assert np.isfinite(np.load(translated(rec['logits']['path']))).all();count+=1
 assert row['read_only_replay']['status']=='PASS' and row['read_only_replay']['new_forwards']==0
 new+=row['counts']['new_forwards']+row['initial_fault_injection_forwards']
assert count==r['total_unique_model_forwards']==new==288
comparison=json.loads((P/'MAP_COMPARISONS.json').read_text());assert len(comparison['maps'])==16 and len(comparison['comparisons'])==40
out=dict(status='PASS',archive_payload_files_verified=files,raw_source_chunks_verified=len(keys),render_metric_samples_recomputed=int(d.size),completed_maps_verified=16,committed_forward_receipts_verified=count,new_model_forwards_in_check=0,original_framework_cached_replays_passed=16,gpu_spend_GBP=0)
print(json.dumps(out))
