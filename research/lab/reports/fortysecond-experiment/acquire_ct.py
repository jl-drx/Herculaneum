"""Execute an exact frozen chunk manifest; never fall back to implicit zeros."""
import gzip,time,urllib.request
import numpy as np
from common import *
def assemble(chunks,start_zyx):
 out=np.empty((256,256,256),np.uint8);base=np.asarray(start_zyx)//128
 for dz in range(2):
  for dy in range(2):
   for dx in range(2):
    key=tuple((base+[dz,dy,dx]).tolist());a=chunks[key]
    need(a.shape==(128,128,128) and a.dtype==np.uint8,'Exact uint8 native chunks')
    out[dz*128:(dz+1)*128,dy*128:(dy+1)*128,dx*128:(dx+1)*128]=a
 return out
def main():
 go=json.loads((HERE/'CT_GO.json').read_text())
 for item in go['pins']:verify(item)
 p=json.loads((HERE/'CT_PLAN.json').read_text());out=HERE/'ct';out.mkdir(exist_ok=False);(out/'chunks').mkdir();records=[];delivered=0;started=time.monotonic()
 deadline=datetime.fromisoformat(p['admission_deadline_utc']).timestamp()
 try:
  for item in p['chunks']:
   need(time.time()<deadline,'Admission wall deadline');need(delivered+2097152+65536<=p['hard_CT_cap_bytes'],'Delivered-byte cap')
   record=dict(id=item['id'],url=item['url'],volume=item['volume'],index_zyx=item['index_zyx'])
   try:
    with urllib.request.urlopen(urllib.request.Request(item['url'],headers={'Accept-Encoding':'identity'}),timeout=30) as response:
     need(response.status==200,'Explicit chunk response');record.update(status=response.status,content_encoding=response.headers.get('Content-Encoding'),etag=response.headers.get('ETag'))
     body=response.read(2097152+65537);need(len(body)<=2097152+65536,'Chunk transport cap')
    delivered+=len(body);path=out/'chunks'/(item['id']+'.bin');path.write_bytes(body);record.update(raw=pin(path),delivered_bytes=len(body))
    decoded=gzip.decompress(body) if record['content_encoding']=='gzip' else body;need(len(decoded)==2097152,'Complete native chunk decoded bytes')
    record.update(decoded_sha256=hashlib.sha256(decoded).hexdigest(),decoded_bytes=len(decoded));records.append(record)
    (out/'REQUESTS.json').write_text(json.dumps(dict(records=records,delivered_bytes=delivered),indent=2)+'\n')
   except Exception as e:
    if record not in records:record.update(error=type(e).__name__+': '+str(e));records.append(record)
    raise
  arrays=[]
  for b in p['blocks']:
   pieces={}
   for item in records:
    if item['volume']!=b['volume'] or item['id'] not in b['chunk_ids']:continue
    body=verify(item['raw']).read_bytes();body=gzip.decompress(body) if item['content_encoding']=='gzip' else body
    need(hashlib.sha256(body).hexdigest()==item['decoded_sha256'],'Decoded identity');pieces[tuple(item['index_zyx'])]=np.frombuffer(body,np.uint8).reshape(128,128,128)
   need(len(pieces)==8,'Explicit eight chunks per block');a=assemble(pieces,b['start_zyx']);path=out/(b['id']+'.npy');np.save(path,a,allow_pickle=False)
   arrays.append(dict(**b,array=pin(path),nonzero_voxels=int(np.count_nonzero(a)),minimum=int(a.min()),maximum=int(a.max())))
  save(out/'COMPLETE.json',dict(status='PASS_EXACT_FROZEN_CHUNKS',created_at_utc=now(),GO=pin(HERE/'CT_GO.json'),requests=len(records),delivered_bytes=delivered,decoded_bytes=sum(x['decoded_bytes'] for x in records),arrays=arrays,records=records,elapsed_seconds=time.monotonic()-started,network_retries=0,new_model_calls=0))
  print(json.dumps(dict(status='PASS',requests=len(records),bytes=delivered,blocks=len(arrays),elapsed_seconds=time.monotonic()-started)))
 except Exception as e:
  save(out/'FAILURE.json',dict(status='STOP',created_at_utc=now(),error=type(e).__name__+': '+str(e),records=records,delivered_bytes=delivered,elapsed_seconds=time.monotonic()-started));raise
if __name__=='__main__':main()
