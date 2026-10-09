"""Prospective depth-permutation fixtures; no scientific inputs or model calls."""
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

REFERENCE=Path('/workspace/herculaneum-lab/reports/twentysecond-experiment/parity-review/verify_parity.py')
REFERENCE_SHA='98a7356225792d26705d176ea53237531d7b4da960f56ff5d39aa5827cc768ee'

def sha(a):return hashlib.sha256(a.tobytes(order='C')).hexdigest()
def run():
    assert hashlib.sha256(REFERENCE.read_bytes()).hexdigest()==REFERENCE_SHA
    spec=importlib.util.spec_from_file_location('phase24_reversal_fixture_reference',REFERENCE)
    ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
    rows=[]
    z,y,x=np.indices((17,128,128),dtype=np.int32)
    pattern=(1+(17*z+3*y+5*x+(y*x)%29+(z*y)%23)%251).astype(np.float32)
    rng=np.random.default_rng(3);fallback=np.zeros((17,128,128),np.float32)
    selected=rng.choice(fallback.size,int(fallback.size*.35),replace=False)
    fallback.ravel()[selected]=rng.uniform(.01,10000,len(selected)).astype(np.float32)
    for name,raw in [('structured',pattern),('constant',np.full_like(pattern,17)),('fallback',fallback)]:
        source_hash=sha(raw);original=ref.normalize(raw);original_hash=sha(original)
        reverse=np.ascontiguousarray(original[::-1]);back=np.ascontiguousarray(reverse[::-1])
        assert sha(raw)==source_hash and sha(original)==original_hash
        assert reverse.dtype==np.float32 and reverse.shape==(17,128,128)
        assert back.tobytes()==original.tobytes()
        for plane in range(17):assert reverse[plane].tobytes()==original[16-plane].tobytes()
        reordered_normalization=ref.normalize(raw[::-1])
        row=dict(case=name,source_unchanged=True,original_unchanged=True,exact17plane_permutation=True,
            double_reversal_identity=True,original_sha256=original_hash,reversed_sha256=sha(reverse),
            normalize_reversed_raw_bitwise_equal=reverse.tobytes()==reordered_normalization.tobytes(),
            normalize_reversed_raw_max_absolute=float(np.max(np.abs(reverse-reordered_normalization))),
            std_forward=float(np.std(raw)),std_reversed=float(np.std(raw[::-1])))
        rows.append(row)
    assert rows[0]['original_sha256']!=rows[0]['reversed_sha256']
    assert rows[1]['original_sha256']==rows[1]['reversed_sha256']
    assert not rows[2]['normalize_reversed_raw_bitwise_equal']
    # Preserve signed-zero bits when testing the permutation itself.
    signed=np.zeros((17,2,2),np.float32);signed[0,0,0]=-0.
    assert np.ascontiguousarray(signed[::-1])[16,0,0].tobytes()==signed[0,0,0].tobytes()
    return dict(status='PASS',cases=rows,recipe='Normalize original order, verify original bytes, reverse normalized depth16..0, verify exact reversed bytes.',
        signed_zero_preserved=True,actual_array_reads=0,checkpoint_reads=0,model_calls=0)
if __name__=='__main__':
    import sys
    result=run();out=Path(sys.argv[1])
    with out.open('x') as f:json.dump(result,f,sort_keys=True,indent=2);f.write('\n')
    print(json.dumps(dict(status=result['status'],path=str(out),sha256=hashlib.sha256(out.read_bytes()).hexdigest(),size_bytes=out.stat().st_size)))
