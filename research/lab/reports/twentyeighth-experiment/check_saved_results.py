"""Root scalar verification of persisted geometry, contributions and synthetic values."""
import itertools
import json
import math
from pathlib import Path
import struct
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'twentyseventh-experiment'))
from fetch_metadata import pin,save

out=HERE/'run-v1';plan=json.loads((out/'CHUNK_PLAN.json').read_text());comparison=json.loads((out/'COMPARISON.json').read_text())
data=np.load(out/'COMPARISON_ARRAYS.npz',allow_pickle=False)
records=list(struct.iter_unpack('<6d',Path(plan['geometry']['path']).read_bytes()))
keys=set();per_point=[];expected_counts=[];truth=[];valid=[]
for depth in [-3.24,0,3.24]:
    for record in records:
        p=[record[i]+depth*(record[3+i]/3.24) for i in range(3)] if any(record) else [math.nan]*3
        ok=all(math.isfinite(v) and 0<=v<=s-1 for v,s in zip(p,[6300,2260,8855]))
        valid.append(ok);expected_counts.append(0);contributions=[]
        if ok:
            lo=[math.floor(v) for v in p];f=[v-l for v,l in zip(p,lo)]
            for d in itertools.product((0,1),repeat=3):
                w=math.prod(a if b else 1-a for a,b in zip(f,d))
                if w>0:
                    xyz=[l+b for l,b in zip(lo,d)]
                    keys.add(tuple(xyz[i]//128 for i in (2,1,0)))
                    contributions.append(w*(xyz[2]+2*xyz[1]+3*xyz[0]));expected_counts[-1]+=1
        truth.append(math.fsum(contributions) if ok else math.nan);per_point.append(p)
np.testing.assert_array_equal(data['points_xyz'],np.array(per_point))
np.testing.assert_array_equal(data['valid'],valid);np.testing.assert_array_equal(data['valid'],data['baseline_valid'])
np.testing.assert_array_equal(data['expected'],expected_counts);np.testing.assert_array_equal(data['processed'],expected_counts)
np.testing.assert_allclose(data['values'],truth,atol=1e-10,rtol=0,equal_nan=True)
np.testing.assert_array_equal(data['values'],data['baseline_values'])
assert [list(k) for k in sorted(keys)]==plan['keys_zyx']==comparison['scheduler_stats']['loader_calls']
assert plan['batch_chunks']==[16,16,16,16,6] and sum(plan['batch_physical_bytes'])==293601280
assert sum(expected_counts)==13392 and sum(valid)==1674 and len(valid)==2187
manifest=json.loads((out/'SYNTHETIC_MANIFEST.json').read_text());root=out/'synthetic-store'
for name,item in {**manifest['metadata'],**manifest['objects']}.items():
    got=pin(root/name);got.pop('path');assert got==item
assert len(manifest['objects'])==70
encoded=sum(v['size_bytes'] for v in manifest['objects'].values())
assert encoded==comparison['optimized_stats']['source_bytes']==3315894
assert comparison['baseline_stats']['decoded']==200 and comparison['optimized_stats']['decoded']==70
assert (200-70)/200==comparison['source_read_reduction_fraction']==0.65
for item in json.loads((out/'RUN_RECEIPT.json').read_text())['outputs']:assert pin(item['path'])==item
for item in json.loads((HERE/'SOURCE_FREEZE.json').read_text())['files']:assert pin(item['path'])==item
err=np.abs(data['values'][valid]-np.array(truth)[valid])
save(HERE/'SAVED_RESULT_VERIFICATION.json',dict(status='PASS',reviewer='root separate scalar check, not an independent agent',
    all_original_slots_preserved=True,contribution_counts_exact=True,chunk_union_exact=True,values_identical_to_baseline=True,
    scalar_maximum_error=float(err.max()),source_hashes_verified=True,unique_encoded_bytes=encoded,
    candidate_pixels_read=0,network_requests=0))
print('PASS: scalar geometry/support, all saved sample values, native-layout synthetic identities and65% decode reduction')
