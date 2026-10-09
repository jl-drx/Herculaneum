"""Independent scalar expected sums and artificial endian-aware row/coverage fixtures."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

from pooling_kernels import pool_triplet,assemble_group,row_range


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=False);checks=[]
    def passed(name):checks.append(name)
    def reject(name,action):
        try:action()
        except (ValueError,KeyError):passed(name);return
        raise AssertionError('Expected rejection: '+name)
    values=np.fromfunction(lambda z,y,x:1000*z+10*y+x,(3,6,9),dtype=int).astype(np.uint16)
    expected=np.empty((2,3),np.float32)
    for y in range(2):
        for x in range(3):
            total=sum(int(values[z,3*y+dy,3*x+dx]) for z in range(3) for dy in range(3) for dx in range(3))
            expected[y,x]=struct.unpack('=f',struct.pack('=f',total/6939))[0]
    for dtype in ['<u2','>u2']:
        result=pool_triplet(values.astype(dtype));assert np.array_equal(result.view(np.uint32),expected.view(np.uint32));passed('scalar-axes-'+dtype)
    maximum=pool_triplet(np.full((3,3,3),65535,np.uint16));assert maximum[0,0]==255;passed('no-uint16-overflow')
    fractional=np.zeros((3,3,3),np.uint16);fractional[0,0,0]=1
    assert 0<pool_triplet(fractional)[0,0]<1 and pool_triplet(fractional)[0,0]==np.float32(1/6939);passed('fraction-preserved')
    reject('out-of-range-source',lambda:pool_triplet(np.full((3,3,3),65536,np.uint32)))
    reject('signed-source',lambda:pool_triplet(values.astype(np.int16)))
    reject('wrong-depth',lambda:pool_triplet(values[:2]))
    reject('nonaligned-XY',lambda:pool_triplet(values[:,:,:8]))
    contract=dict(width=20,height=20,strip_offset=100,strip_bytes=800,object_total_bytes=950)
    words=np.array([[0,1,256,32768,65535,17],[65535,32768,256,1,0,23]],np.uint16)
    for dtype in ['<u2','>u2']:
        c=dict(contract,dtype=dtype);body=bytearray(b'prefix');rows=[]
        # Physical response order is reversed; native row routing must preserve XY.
        for i in [1,0]:
            payload=words[i].astype(dtype).tobytes();offset=len(body);body.extend(payload);body.extend(b'framing')
            rows.append(dict(native_y=4+i,range_inclusive=row_range(c,3,4+i,6),payload_offset=offset,payload_bytes=len(payload),payload_sha256=hashlib.sha256(payload).hexdigest()))
        result=assemble_group(bytes(body),rows,c,3,4,2,6);assert np.array_equal(result,words.astype(np.uint32));passed('row-order-and-endian-'+dtype)
        reject('missing-row-'+dtype,lambda:assemble_group(bytes(body),rows[:1],c,3,4,2,6))
        reject('duplicate-row-'+dtype,lambda:assemble_group(bytes(body),[rows[0],rows[0]],c,3,4,2,6))
        bad=[dict(row) for row in rows];bad[0]['payload_sha256']='0'*64
        reject('bad-row-SHA-'+dtype,lambda:assemble_group(bytes(body),bad,c,3,4,2,6))
        bad=[dict(row) for row in rows];bad[0]['payload_offset']=len(body)
        reject('truncated-row-'+dtype,lambda:assemble_group(bytes(body),bad,c,3,4,2,6))
        bad=[dict(row) for row in rows];bad[0]['range_inclusive']=[0,11]
        reject('wrong-source-range-'+dtype,lambda:assemble_group(bytes(body),bad,c,3,4,2,6))
    reject('row-outside-canvas',lambda:row_range(contract,18,0,6))
    reject('strip-bound',lambda:row_range(dict(contract,strip_bytes=10),0,4,6))
    def record(path):
        data=path.read_bytes();return dict(path=str(path),size_bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    receipt=dict(status='PASS',checks=checks,source_files=[record(p) for p in sorted(Path(__file__).parent.glob('*.py'))],
        actual_operations=dict(synthetic_numpy_arrays=True,real_CT_body_reads=0,real_CT_array_reads=0,model_calls=0,network_calls=0,checkpoint_reads=0))
    out=args.output_dir/'fixture-receipt.json';out.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(status='PASS',checks=len(checks),receipt=record(out))))


if __name__=='__main__':main()
