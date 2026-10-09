"""Offline full source-byte reconstruction and two formulations of exact pooling."""
import resource,time
import numpy as np
from common import *
sys.path.insert(0,str(REPORTS/'sixteenth-experiment/production-pooling-v2'))
from pooling_kernels import pool_triplet

def main():
    started=time.monotonic();check_freeze('PIXEL_GO.json')
    plan=json.loads((HERE/'PLAN.json').read_text())
    receipt=json.loads((HERE/'pixels-v1/COMPLETE.json').read_text())
    ledger=json.loads(verify(receipt['ledger']).read_text())
    objects=json.loads((HERE/'metadata-v1/OBJECTS.json').read_text())['objects']
    source={o['url']:o for o in objects}
    actual={k:np.load(verify(v),mmap_mode='r',allow_pickle=False) for k,v in receipt['raw_arrays'].items()}
    reconstructed={k:np.empty((63,384,384),dtype=np.uint16) for k in actual}
    seen=np.zeros((63,4,384),bool);body_count=0
    for entry in ledger['requests']:
        require(entry['success'],'Incomplete HTTP request')
        if len(entry['ranges'])==1:continue
        path=verify(entry['file']);data=path.read_bytes();obj=source[entry['url']]
        parts=parse_multipart(data,entry['response_headers']['Content-Type'],entry['ranges'],obj['identity']['total_bytes'])
        require(parts==entry['parts'],'Multipart interpretation changed')
        body_count+=1
        for part in parts:
            a,b=part['range_inclusive'];relative=a-obj['layout']['strip_offset']
            require(relative>=0 and relative%2==0 and b-a+1==768,'Physical byte layout')
            y,x=divmod(relative//2,6205)
            matches=[(i,c) for i,c in enumerate(plan['contexts']) if c['source_bbox_yxhw'][0]<=y<c['source_bbox_yxhw'][0]+384 and x==c['source_bbox_yxhw'][1]]
            require(len(matches)==1,'Byte address not uniquely in frozen crop')
            ci,c=matches[0];row=y-c['source_bbox_yxhw'][0];z=obj['layer']-1
            require(not seen[z,ci,row],'Duplicate physical crop row');seen[z,ci,row]=True
            payload=data[part['payload_offset']:part['payload_offset']+768]
            require(hashlib.sha256(payload).hexdigest()==part['payload_sha256'],'Payload digest differs')
            reconstructed[c['id']][z,row]=np.frombuffer(payload,dtype=obj['layout']['dtype'])
    require(body_count==378 and seen.all(),'Incomplete physical support')
    out=HERE/'pooled-v1';out.mkdir(exist_ok=False);records={}
    for c in plan['contexts']:
        name=c['id'];raw=reconstructed[name]
        require(np.array_equal(raw,actual[name]),'Reconstructed crop differs')
        pooled=np.stack([pool_triplet(raw[z:z+3]) for z in range(0,63,3)])
        sums=np.zeros((21,128,128),np.uint32)
        for dz in range(3):
            for dy in range(3):
                for dx in range(3):sums+=raw[dz::3,dy::3,dx::3].astype(np.uint32)
        independently_pooled=(sums.astype(np.float64)/6939.0).astype(np.float32)
        require(np.array_equal(pooled,independently_pooled),'Independent pooling differs')
        require(np.isfinite(pooled).all() and np.all((pooled>=0)&(pooled<=255)),'Pooled input range')
        np.save(out/(name+'.npy'),pooled,allow_pickle=False)
        records[name]=dict(array=pin(out/(name+'.npy')),raw_source=receipt['raw_arrays'][name],
            shape=list(pooled.shape),dtype=str(pooled.dtype),min=float(pooled.min()),max=float(pooled.max()),
            zero_values=int(np.count_nonzero(pooled==0)),verified_source_samples=int(raw.size),verified_pooled_values=int(pooled.size))
    result=dict(status='PASS',objects=pin(HERE/'metadata-v1/OBJECTS.json'),source_completion=pin(HERE/'pixels-v1/COMPLETE.json'),
        multipart_bodies_verified=body_count,physical_rows_verified=int(seen.sum()),
        source_uint16_values_verified=37158912,pooled_FP32_values_verified=1376256,contexts=records,
        elapsed_seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        scientific_model_forwards=0,created_at_utc=now())
    require(result['peak_rss_bytes']<=2*1024**3,'Verification RSS cap')
    save(out/'COMPLETE.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='contexts'},indent=2))

if __name__=='__main__':main()
