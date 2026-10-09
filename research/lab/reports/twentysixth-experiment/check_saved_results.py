"""Root second-pass check: scalar chunk enumeration and saved output identities."""
import hashlib
import itertools
import json
import math
from pathlib import Path
import struct

ROOT=Path(__file__).resolve().parent


def pin(path):
    path=Path(path)
    return dict(path=str(path.resolve()),size_bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    run=ROOT/'run-v1'
    receipt=json.loads((run/'RUN_RECEIPT.json').read_text())
    assert receipt['status']=='PASS' and receipt['tests']==37
    for record in receipt['outputs']:assert pin(record['path'])==record
    freeze=json.loads((ROOT/'SOURCE_FREEZE.json').read_text())
    for record in freeze['files']:assert pin(record['path'])==record
    raw=freeze['sparse_geometry_records'];assert pin(raw['path'])==raw
    records=list(struct.iter_unpack('<6d',Path(raw['path']).read_bytes()))
    unique=set();mapped=invalid=corners=0
    for record in records:
        if all(x==0 for x in record):invalid+=1;continue
        mapped+=1;x,y,z,nx,ny,nz=record
        denominator=math.sqrt((nx*3.24)**2+(ny*3.24)**2+(nz*3.24)**2)
        for distance in (-3.24,0,3.24):
            xyz=(x+distance*nx/denominator,y+distance*ny/denominator,z+distance*nz/denominator)
            assert all(math.isfinite(a) and 0<=a<=b for a,b in zip(xyz,(6299,2259,8854)))
            floors=tuple(math.floor(a) for a in xyz)
            for delta in itertools.product((0,1),repeat=3):
                weight=math.prod((a-b) if d else (1-(a-b)) for a,b,d in zip(xyz,floors,delta))
                if weight<=0:continue
                corner=tuple(b+d for b,d in zip(floors,delta));corners+=1
                unique.add(tuple(c//64 for c in corner[::-1]))
    byte_count=0
    expected=[]
    for key in sorted(unique):
        shape=[min(64,s-k*64) for s,k in zip((8855,2260,6300),key)]
        size=math.prod(shape)*2;byte_count+=size
        expected.append(dict(key_zyx=list(key),shape_zyx=shape,uncompressed_bytes=size))
    plan=json.loads((run/'SPARSE_GEOMETRY_DRY_PLAN.json').read_text())
    assert (mapped,invalid,corners,len(unique),byte_count)==(558,171,13392,78,40894464)
    assert plan['chunks']==expected
    assert plan['positive_weight_corners']==corners and plan['uncompressed_source_bytes']==byte_count
    assert plan['geometric_valid_count']==mapped*3 and plan['point_count']==len(records)*3
    assert plan['requests_made']==plan['source_voxels_loaded']==0 and not plan['stage_C_authorized']
    diagnostics=json.loads((run/'SYNTHETIC_DIAGNOSTICS.json').read_text())
    assert diagnostics['sampling']['correct']['maximum_absolute_error']<=1e-10
    reg=diagnostics['registration'];assert not set(reg['fit']['ids'])&set(reg['check']['ids'])
    assert reg['fit']['maximum_pixels']<1e-12 and abs(reg['check']['maximum_pixels']-5)<1e-12
    result=dict(status='PASS',review='Root scalar re-enumeration and saved-output audit; not independent-agent review',
        mapped_records=mapped,unmapped_records=invalid,positive_weight_corners=corners,
        unique_chunks=len(unique),hypothetical_uncompressed_bytes=byte_count,
        source_hashes_unchanged=True,network_calls=0,model_calls=0,candidate_source_voxels_loaded=0,
        inputs=[pin(run/f) for f in ['RUN_RECEIPT.json','SPARSE_GEOMETRY_DRY_PLAN.json','SYNTHETIC_DIAGNOSTICS.json']])
    out=ROOT/'SAVED_RESULT_VERIFICATION.json';assert not out.exists()
    out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
