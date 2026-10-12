"""Pin TIFF structure using bounded ranges; no prediction pixels requested."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import struct
import urllib.request


def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);a=p.parse_args()
    out=a.work/'teacher-metadata';out.mkdir(exist_ok=False)
    cohort=json.loads((a.work/'COHORT_CANDIDATES.json').read_text())
    ledger=[];headers=[];total=0
    def save():
        (out/'LEDGER.json').write_text(json.dumps(dict(requests=ledger,body_bytes=total),indent=2)+'\n')
    for row in cohort['native_v2_audit']:
        if row['role']!='additional_regional_coverage_candidate':continue
        ident=row['sample'].lower()+'-'+row['segment'].split('-')[0]
        url='https://vesuvius-challenge-open-data.s3.amazonaws.com/'+row['teacher']['path']
        with urllib.request.urlopen(urllib.request.Request(url,method='HEAD'),timeout=40) as r:
            assert r.status==200;length=int(r.headers['Content-Length']);etag=r.headers['ETag']
        ranges=[];used=0
        def get(offset,size,label):
            nonlocal total,used
            assert 0<=offset<offset+size<=length and used+size<=256*1024
            used+=size
            req=urllib.request.Request(url,headers={'Range':f'bytes={offset}-{offset+size-1}',
                                                  'If-Match':etag,'Accept-Encoding':'identity'})
            with urllib.request.urlopen(req,timeout=40) as r:
                assert r.status==206 and r.headers['Content-Range']==f'bytes {offset}-{offset+size-1}/{length}'
                assert r.headers['ETag']==etag
                data=r.read(size+1);assert len(data)==size
            name=f'{ident}-{label}.bin';(out/name).write_bytes(data);total+=size;ranges.append([offset,offset+size])
            ledger.append(dict(id=ident,url=url,offset=offset,bytes=size,etag=etag,path=name,
                sha256=hashlib.sha256(data).hexdigest()));save();return data
        raw=get(0,8,'header');assert raw[:2] in [b'II',b'MM']
        endian='<' if raw[:2]==b'II' else '>';kind,offset=struct.unpack(endian+'HI',raw[2:]);assert kind==42
        count=struct.unpack(endian+'H',get(offset,2,'count'))[0];assert 1<=count<=128
        data=get(offset+2,count*12+4,'ifd');tags={}
        sizes={1:1,2:1,3:2,4:4,5:8,6:1,7:1,8:2,9:4,10:8,11:4,12:8}
        formats={1:'B',3:'H',4:'I',6:'b',8:'h',9:'i',11:'f',12:'d'}
        for i in range(count):
            entry=data[i*12:(i+1)*12];tag,typ,n,ptr=struct.unpack(endian+'HHII',entry)
            size=n*sizes[typ];assert tag not in tags and 0<size<=256*1024
            value=entry[8:8+size] if size<=4 else get(ptr,size,f'tag{tag}')
            tags[tag]=list(struct.unpack(endian+str(n)+formats[typ],value)) if typ in formats else value.hex()
        assert struct.unpack(endian+'I',data[-4:])[0]==0
        assert tags[258]==[8] and tags[277]==[1] and tags[262]==[1]
        assert tags.get(274,[1])==[1] and tags.get(317,[1])==[1] and tags.get(284,[1])==[1]
        assert tags[259] in [[8],[32946]] and tags[322]==tags[323]==[1024]
        offsets=tags[324];counts=tags[325]
        width,height=tags[256][0],tags[257][0]
        assert len(offsets)==len(counts)==((width+1023)//1024)*((height+1023)//1024)
        assert all(0<o<o+n<=length for o,n in zip(offsets,counts))
        assert all(not (lo<o+n and o<hi) for lo,hi in ranges for o,n in zip(offsets,counts)), 'Header range overlaps image payload'
        result=dict(id=ident,segment=row['segment'],url=url,etag=etag,object_bytes=length,
            shape_yx=[height,width],tags=tags,header_ranges=ranges,header_bytes=used,
            no_image_payload_requested=True)
        (out/f'{ident}-HEADER.json').write_text(json.dumps(result,indent=2)+'\n');headers.append(result)
        print(json.dumps(dict(id=ident,shape=[height,width],object_bytes=length,header_bytes=used)),flush=True)
    (out/'COMPLETE.json').write_text(json.dumps(dict(status='PASS',maps=len(headers),
        header_bytes=total,full_map_bytes=sum(x['object_bytes'] for x in headers),
        image_payload_bytes=0,implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2)+'\n')


if __name__=='__main__':main()
