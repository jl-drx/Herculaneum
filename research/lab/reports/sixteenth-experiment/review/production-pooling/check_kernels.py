"""Independent scalar/mixed-endian synthetic verification; no scientific inputs."""
from pathlib import Path
import hashlib
import json
import socket
import struct
import sys
import numpy as np
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
IMPL=HERE.parent.parent/'production-pooling'
sys.path.insert(0,str(IMPL))
def blocked(*a,**k):raise AssertionError('Network blocked in independent synthetic review')
socket.socket.connect=socket.socket.connect_ex=socket.create_connection=blocked
from pooling_kernels import assemble_group,pool_triplet,row_range,decode_row
CASES=[]
def record(p):
 p=Path(p).resolve();return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def reject(name,f):
 try:f()
 except (ValueError,KeyError):CASES.append(name);return
 raise AssertionError(name)
def main():
 out=HERE/'synthetic-fixture-result.json';assert not out.exists()
 for endian_pattern in [('<','>','<'),('>','<','>')]:
  planes=[];original=[]
  for z,e in enumerate(endian_pattern):
   values=[[(z*17381+y*1703+x*109+258)%65536 for x in range(9)]for y in range(6)]
   original.append(values)
   contract=dict(dtype=e+'u2',width=17,height=30,strip_offset=173,strip_bytes=1020,object_total_bytes=1300)
   body=bytearray(b'header');rows=[]
   for y in [5,1,3,0,4,2]:
    payload=struct.pack(e+'9H',*values[y]);offset=len(body);body+=payload+b'framing'
    rows.append(dict(native_y=7+y,range_inclusive=[173+2*((7+y)*17+4),173+2*((7+y)*17+13)-1],payload_offset=offset,payload_bytes=18,payload_sha256=hashlib.sha256(payload).hexdigest()))
   planes.append(assemble_group(bytes(body),rows,contract,4,7,6,9))
   assert planes[-1].dtype==np.uint32 and planes[-1].tolist()==values
   CASES.append('mixed-endian-row-routing-'+str(endian_pattern)+'-'+str(z))
  actual=pool_triplet(np.stack(planes));expected=np.empty((2,3),np.float32)
  for oy in range(2):
   for ox in range(3):
    total=sum(original[z][3*oy+dy][3*ox+dx]for z in range(3)for dy in range(3)for dx in range(3))
    expected[oy,ox]=struct.unpack('=f',struct.pack('=f',total/6939))[0]
  assert np.array_equal(actual.view(np.uint32),expected.view(np.uint32))
  CASES.append('mixed-endian-scalar-27-value-bitwise-'+str(endian_pattern))
 for value in (0,1,258,32768,65535):
  x=np.full((3,3,3),value,np.uint32);actual=pool_triplet(x)
  expected=struct.unpack('=f',struct.pack('=f',(27*value)/6939))[0]
  assert actual.item()==expected
  CASES.append('constant-'+str(value))
 impulse=np.zeros((3,6,9),np.uint32);impulse[2,5,8]=1
 expected=np.zeros((2,3),np.float32);expected[1,2]=np.float32(1/6939)
 assert np.array_equal(pool_triplet(impulse).view(np.uint32),expected.view(np.uint32))
 CASES.append('corner-impulse-axis-and-fraction')
 reject('reject-overflow-sample',lambda:pool_triplet(np.full((3,3,3),65536,np.uint32)))
 reject('reject-float-input',lambda:pool_triplet(np.zeros((3,3,3),np.float32)))
 reject('reject-depth-gap',lambda:pool_triplet(np.zeros((2,3,3),np.uint16)))
 reject('reject-nonmultiple-axis',lambda:pool_triplet(np.zeros((3,4,3),np.uint16)))
 reject('reject-unknown-endian',lambda:decode_row(b'\x01\x02',dict(payload_offset=0,payload_bytes=2,range_inclusive=[0,1],payload_sha256=hashlib.sha256(b'\x01\x02').hexdigest()),[0,1],'u2',1))
 assert 'torch' not in sys.modules
 result=dict(status='PASS',count=len(CASES),cases=CASES,source=record(IMPL/'pooling_kernels.py'),reviewer=record(__file__),scientific_arrays_read=0,real_network_requests=0,model_calls=0)
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(record(out)))
if __name__=='__main__':main()
