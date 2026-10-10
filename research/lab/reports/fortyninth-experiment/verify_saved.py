"""Read-only recomputation of qualification metrics; no rendering or inference."""
from pathlib import Path
import hashlib,json
import numpy as np
P=Path(__file__).resolve().parent;W=P/'work'
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def stats(a,b):
    d=np.abs(a.astype(np.int16)-b.astype(np.int16))
    return dict(shape=list(a.shape),n=a.size,exact=int((d==0).sum()),exact_fraction=float((d==0).mean()),mean_abs=float(d.mean()),max_abs=int(d.max()),within_one_fraction=float((d<=1).mean()))
plan=json.loads((P/'GO.json').read_text());r=json.loads((P/'RENDER_VERIFICATION.json').read_text());q=json.loads((P/'INFERENCE_QUALIFICATION.json').read_text())
assert sha(P/'PROTOCOL.md')==plan['protocol']['sha256']
for a,b,k in [('render31','published31-crop','actual31_vs_published'),('render66','independent66','actual66_vs_independent'),('independent31','published31-crop','independent31_vs_published')]:
    assert stats(np.load(W/f'{a}.npy'),np.load(W/f'{b}.npy'))==r[k],k
assert r['actual31_vs_published']['max_abs']<=1 and r['actual31_vs_published']['exact_fraction']>=.99
assert r['actual66_vs_independent']['max_abs']<=1 and r['actual66_vs_independent']['exact_fraction']>=.95 and r['actual66_vs_independent']['mean_abs']<=.1
receipts=json.loads((P/'SOURCE_RECEIPTS.json').read_text())['receipts'];bykey={r['key']:r for r in receipts}
for key in plan['source_chunk_keys']:
    f=W/'source-cache'/key;assert f.stat().st_size==128**3 and sha(f)==bykey[key]['sha256']
assert len(q['models'])==4 and q['total_tile_forwards']==20 and q['full_target_maps']==0
for row in q['models']:
    n=row['model'];a=np.load(W/f'qualify-{n}-single.npy');b=np.load(W/f'qualify-{n}-batch4.npy');raw=np.load(W/f'qualify-{n}-raw.npy')
    assert a.shape[0]==1 and b.shape[0]==4 and a.shape[1:]==b.shape[1:]
    assert np.isfinite(a).all() and np.isfinite(b).all()
    assert float(np.abs(b-a).max())==row['max_batch_difference']<1e-4
    assert hashlib.sha256(raw.tobytes()).hexdigest()==row['raw_sha256']
    assert list(raw.shape)==row['input_shape']
support=json.loads((P/'OUTPUT_SUPPORT_PLAN.json').read_text());m=np.load(W/'planned-output-support.npz');common=m['common']
assert np.array_equal(common,np.logical_and.reduce([m[n] for n in m.files if n!='common']))
assert common.sum()==support['common_support_pixels'] and float(common.mean())==support['common_support_fraction']
forecast=json.loads((P/'TRANSFER_PLAN.json').read_text())
assert forecast['source_chunk_count']==len(set(tuple(k) for k in forecast['source_chunk_indices_zyx']))==384
assert forecast['source_payload_upper_bound_bytes']==forecast['source_chunk_count']*128**3
assert abs(sum(r['roi_estimated_seconds'] for r in forecast['compute'])-forecast['roi_model_seconds'])<1e-9
print(json.dumps(dict(status='PASS',render_comparisons_recomputed=3,source_chunks_verified=7,model_output_sets_verified=4,full_target_maps=0,common_support_pixels=int(common.sum()),gpu_spend_GBP=0)))
