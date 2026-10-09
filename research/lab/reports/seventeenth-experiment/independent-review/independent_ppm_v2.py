"""Fixed two-candidate PPM interpretation with no source/label/score search."""
import math
import struct
import numpy as np


def decode_records(data,endian):
    assert endian in ('<','>') and len(data)%48==0
    return np.asarray(list(struct.iter_unpack(endian+'6d',data)),dtype=np.float64)


def candidate(data,endian,variant,xy):
    rows=decode_records(data,endian)
    assert xy.shape==(len(rows),2)
    zeros=np.all(rows==0,axis=1)
    finite=np.isfinite(rows).all(axis=1)
    norms=np.array([math.hypot(*normal.tolist()) if np.isfinite(normal).all() else math.nan for normal in rows[:,3:]],dtype=np.float64)
    if variant=='result':
        units=finite & (np.abs(norms-1)<=1e-5)
        bad=(~zeros)&(~units)
        ok=bool(np.any(~zeros) and not np.any(bad))
        return dict(pass_invariant=ok,total_records=len(rows),zero_sentinels=int(zeros.sum()),
                    nonzero_records=int((~zeros).sum()),nonfinite_records=int((~finite).sum()),
                    nonunit_nonzero_records=int(((~zeros)&finite&~units).sum()),
                    flat_normal_sign=None)
    assert variant=='surface_volume'
    expected=np.column_stack((xy,np.full(len(rows),32.0),np.zeros((len(rows),2))))
    sign=rows[np.flatnonzero(~zeros)[0],5] if np.any(~zeros) else math.nan
    fixed=finite & np.all(rows[:,:5]==expected,axis=1)
    fixed &= (rows[:,5]==sign) & (abs(sign)==1)
    return dict(pass_invariant=bool(np.any(~zeros) and (fixed|zeros).all()),total_records=len(rows),
                zero_sentinels=int(zeros.sum()),nonzero_records=int((~zeros).sum()),
                nonfinite_records=int((~finite).sum()),mismatch_records=int((~fixed&~zeros).sum()),
                flat_normal_sign=float(sign) if np.any(~zeros) and (fixed|zeros).all() else None)


def select(candidates):
    passing=[name for name,row in candidates.items() if row['pass_invariant']]
    return passing[0] if len(passing)==1 else None
