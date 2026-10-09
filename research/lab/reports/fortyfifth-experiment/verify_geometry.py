"""Reconstruct sparse normal samples and independently score contiguous runs."""
import json,hashlib,itertools,statistics
from pathlib import Path
import numpy as np
import numcodecs
P=Path(__file__).resolve().parent
g=P/'geometry';plan=json.loads((g/'PLAN.json').read_text());result=json.loads((g/'RESULTS.json').read_text())
for pin in plan['pins']:
    path=Path(pin['path'])
    assert path.stat().st_size==pin['size_bytes']
    with path.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==pin['sha256']
coords=np.load(g/'sample-indices-zyx.npy');profile=np.load(g/'profiles.npy');observed=np.load(g/'observed.npy')
assert len(result['nodes'])==243 and coords.shape==(104,31,3)
for row in plan['records']:
    i=row['profile_index']
    if i is None:continue
    normal=np.asarray(row['normal_xyz']);assert abs(float(np.linalg.norm(normal))-1)<1e-6
    # Preserve authored float32 sampling arithmetic, checked independently per offset.
    base=np.asarray(row['xyz'],np.float32);normal=normal.astype(np.float32)
    points=np.array([np.rint((base+np.float32(k)*normal)[::-1]).astype(np.int64) for k in range(-15,16)])
    assert np.array_equal(points,coords[i])
rebuilt=np.zeros_like(profile);seen=np.zeros_like(observed);chunk_shape=np.array(plan['array']['chunks']);codec=numcodecs.get_codec(plan['array']['compressor'])
requests=json.loads((g/'REQUESTS.json').read_text())
for row in requests:
    raw=Path(row['body']['path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==row['body']['sha256']
    if row['status']!=200:continue
    a=np.frombuffer(codec.decode(raw),np.uint8).reshape(tuple(chunk_shape))
    for i in range(len(coords)):
        for j,point in enumerate(coords[i]):
            if np.array_equal(point//chunk_shape,row['chunk']):
                rebuilt[i,j]=a[tuple(point%chunk_shape)];seen[i,j]=True
assert np.array_equal(rebuilt,profile) and np.array_equal(seen,observed) and observed.all()
checks=0
for row in result['nodes']:
    i=row['profile_index']
    if i is None:assert row['measurement']['status']=='invalid_mesh_normal';continue
    runs=[]
    for on,group in itertools.groupby(enumerate(profile[i]),lambda item:item[1]>0):
        indices=[j for j,_ in group]
        if on:runs.append((indices[0],indices[-1],(indices[0]+indices[-1])/2-15))
    distances=[abs(v[2]) for v in runs]
    if not runs:status,offset='no_predicted_run',None
    elif distances.count(min(distances))>1:status,offset='equal_nearest_runs',None
    else:
        lo,hi,offset=runs[distances.index(min(distances))]
        if lo==0 or hi==30:status,offset='censored_nearest_run',None
        else:status='measured'
    assert row['measurement']==dict(status=status,runs=len(runs),offset=offset)
    checks+=1
for summary in result['summaries']:
    vals=[abs(r['measurement']['offset']) for r in result['nodes'] if r['segment']==summary['segment'] and r['measurement']['status']=='measured']
    assert statistics.median(vals)==summary['median_absolute_offset']
    assert sum(v>2 for v in vals)==summary['measured_over_two']
receipt=dict(status='PASS',selected_nodes=243,valid_normal_profiles=104,independent_run_classifications=checks,redecoded_chunks=len(requests),all_sample_indices_and_values_match=True,unknown_support_filled=False,accuracy_claim=False)
(g/'VERIFICATION.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
