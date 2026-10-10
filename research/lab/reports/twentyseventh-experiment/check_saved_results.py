"""Root's separate scalar accounting check; no live data, models or agents."""
import hashlib
import itertools
import json
import math
from pathlib import Path
import struct

from fetch_metadata import pin,save

HERE=Path(__file__).resolve().parent
out=HERE/'run-v1'
plan=json.loads((out/'ACTUAL_LAYOUT_DRY_PLAN.json').read_text())
records=list(struct.iter_unpack('<6d',Path(plan['geometry']['path']).read_bytes()))
point_keys=[];valid_count=0;corners=0
for depth in [-3.24,0,3.24]:
    for record in records:
        keys=set()
        if any(record):
            xyz=[record[i]+depth*(record[3+i]/3.24) for i in range(3)]
            if all(math.isfinite(x) and 0<=x<=s-1 for x,s in zip(xyz,[6300,2260,8855])):
                valid_count+=1
                lower=[math.floor(x) for x in xyz];f=[x-l for x,l in zip(xyz,lower)]
                for delta in itertools.product((0,1),repeat=3):
                    weight=math.prod(t if d else 1-t for t,d in zip(f,delta))
                    if weight>0:
                        corners+=1
                        keys.add(tuple((lower[i]+delta[i])//128 for i in (2,1,0)))
        point_keys.append(keys)
assert len(records)==729 and sum(bool(any(r)) for r in records)==558
assert valid_count==plan['geometric_valid_count']==1674
assert corners==plan['positive_weight_corners']==13392
union=set().union(*point_keys)
assert sorted(union)==[tuple(k) for k in plan['unique_keys_zyx']]
assert len(union)*128**3*2==plan['unique_physical_bytes']==293601280
start=0;current=set();ranges=[]
for i,keys in enumerate(point_keys):
    if len(current|keys)>16:
        ranges.append((start,i));start=i;current=set(keys)
    else:current|=keys
ranges.append((start,len(point_keys)))
assert ranges==[(b['start_inclusive'],b['stop_exclusive']) for b in plan['batches']]
for batch in plan['batches']:
    keys=set().union(*point_keys[batch['start_inclusive']:batch['stop_exclusive']])
    assert sorted(keys)==[tuple(k['key_zyx']) for k in batch['chunks']]
    assert len(keys)*4194304==batch['physical_storage_bytes']<=67108864
assert len(ranges)==13
assert sum(b['unique_chunks'] for b in plan['batches'])==200
result=json.loads((out/'NATIVE_SIZE_SYNTHETIC_RESULT.json').read_text())
errors=[]
for p,v,valid in zip(result['sample_points_xyz'],result['sample_values'],result['valid']):
    if valid:errors.append(abs(v-(p[0]*7+p[1]*5+p[2]*3)))
    else:assert v is None
assert max(errors)<1e-10 and result['all_voxels_exact'] and result['full_and_edge_chunks_checked']==8
receipt=json.loads((out/'RUN_RECEIPT.json').read_text())
for item in receipt['outputs']:assert pin(item['path'])==item
freeze=json.loads((HERE/'SOURCE_FREEZE.json').read_text())
for item in freeze['files']:assert pin(item['path'])==item
ledger=json.loads((HERE/'metadata-run-v1/request-ledger.json').read_text())
assert len(ledger['requests'])==4 and ledger['returned_body_bytes']==3443 and ledger['reserved_body_bytes']==102404
assert [r['success'] for r in ledger['requests']]==[True,True,True,False]
assert ledger['requests'][-1]['status']==404 and ledger['retries']==0
save(HERE/'SAVED_RESULT_VERIFICATION.json',dict(status='PASS',reviewer='root separate scalar second pass; not independent agent',
    geometry_scalar_corner_and_chunk_union_match=True,deterministic_ranges_match=True,all2187slots_preserved=True,
    synthetic_scalar_errors_max=max(errors),frozen_sources_and_outputs_match=True,
    acquisition_stop_preserved=True,requests=4,returned_bytes=3443,unique_chunks=70,batches=13,total_chunk_uses=200,
    candidate_voxels_read=0,network_requests_this_check=0))
print('PASS: scalar geometry, batch coverage, synthetic values, source hashes and partial acquisition accounting')
