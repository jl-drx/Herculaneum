import hashlib
import json
import struct
from pathlib import Path
import numpy as np
from independent_ppm import candidate,select


def packed(rows,endian): return b''.join(struct.pack(endian+'6d',*row) for row in rows)

checks=[]
xy=np.array([[5,7],[6,7],[7,7]],dtype=np.float64)
rows=[[5,7,32,0,0,-1],[6,7,32,0,0,-1],[7,7,32,0,0,-1]]
for endian in ['<','>']:
    data=packed(rows,endian)
    results={e:candidate(data,e,'surface_volume',xy)for e in ['<','>']}
    assert select(results)==endian and results[endian]['flat_normal_sign']==-1
    results={e:candidate(data,e,'result',xy)for e in ['<','>']}
    assert select(results)==endian
checks.append('both_true_serializations_and_wrong_endian_subnormal_rejection')
rows[0]=[0]*6
r=candidate(packed(rows,'<'),'<','result',xy)
assert r['pass_invariant'] and r['zero_sentinels']==1
assert not candidate(packed([[0]*6]*3,'<'),'<','result',xy)['pass_invariant']
assert not candidate(packed(rows,'<'),'<','surface_volume',xy)['pass_invariant']
checks.append('exact_zero_sentinel_nonvacuity_and_flat_sentinel_rejection')
for bad in [[5,7,32,0,0,1e-310],[5,7,32,0,0,1.001],[float('nan'),7,32,0,0,1]]:
    rows[0]=bad
    assert not candidate(packed(rows,'<'),'<','result',xy)['pass_invariant']
checks.append('tiny_nonzero_nonunit_and_nonfinite_rejection')
rows=[[5,7,32,0,0,-1],[6,7,32,0,0,1],[7,7,32,0,0,-1]]
assert not candidate(packed(rows,'<'),'<','surface_volume',xy)['pass_invariant']
rows[1]=[6.00000001,7,32,0,0,-1]
assert not candidate(packed(rows,'<'),'<','surface_volume',xy)['pass_invariant']
checks.append('mixed_flat_sign_and_exact_coordinate_mismatch_rejection')
assert select({'<':{'pass_invariant':True},'>':{'pass_invariant':True}})is None
assert select({'<':{'pass_invariant':False},'>':{'pass_invariant':False}})is None
checks.append('both_or_neither_candidate_unresolved')
p=Path(__file__).resolve();out=p.with_name('ppm-fixture-receipt.json')
r=dict(status='PASS',checks=checks,check_count=len(checks),scientific_ppm_decodes=0,
       sources={q.name:hashlib.sha256(q.read_bytes()).hexdigest()for q in [p,p.with_name('independent_ppm.py')]})
with out.open('x')as f:json.dump(r,f,indent=2);f.write('\n')
print(json.dumps(r,indent=2))
