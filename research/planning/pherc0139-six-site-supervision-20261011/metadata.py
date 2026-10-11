"""Read exact TIFF structural ranges only; no image payload requests."""
from common import *
import datetime,struct,urllib.request
CAP=256*1024

def main():
    ledger=[];total=0;out=R/'metadata';out.mkdir(exist_ok=False)
    admitted={s['winding'] for s in read(R/'ALTERNATIVES.json')['sites'] if s['admitted_for_bounded_target_sensitivity']}
    for p in read(R/'PROVENANCE.json')['sites']:
        w=p['winding']
        if w not in admitted:continue
        url=p['url']
        with urllib.request.urlopen(urllib.request.Request(url,method='HEAD'),timeout=30) as response:
            assert response.status==200;length=int(response.headers['Content-Length']);etag=response.headers['ETag']
            ledger.append(dict(winding=w,method='HEAD',status=200,url=url,bytes=length,etag=etag))
        ranges=[]
        def get(off,n,label):
            nonlocal total
            assert 0<=off<off+n<=length and total+n<=CAP
            req=urllib.request.Request(url,headers={'Range':f'bytes={off}-{off+n-1}','If-Match':etag})
            with urllib.request.urlopen(req,timeout=30) as response:
                assert response.status==206 and response.headers.get('ETag')==etag
                assert response.headers.get('Content-Range')==f'bytes {off}-{off+n-1}/{length}'
                b=response.read(n+1);assert len(b)==n
            file=out/f'{w}-{label}.bin';file.write_bytes(b);total+=n;ranges.append((off,off+n))
            ledger.append(dict(winding=w,method='GET',status=206,url=url,range_half_open=[off,off+n],etag=etag,**receipt(file)))
            save(out/'LEDGER.partial.json',dict(complete=False,requests=ledger,header_body_bytes=total))
            return b
        h=get(0,8,'header');assert h[:2] in [b'II',b'MM'];endian='<' if h[:2]==b'II' else '>'
        typ,off=struct.unpack(endian+'HI',h[2:]);assert typ==42
        n=struct.unpack(endian+'H',get(off,2,'ifd-count'))[0];assert n<=64
        raw=get(off+2,n*12+4,'ifd');tags={}
        sizes={1:1,2:1,3:2,4:4,5:8,6:1,7:1,8:2,9:4,10:8,11:4,12:8};fmt={1:'B',3:'H',4:'I',6:'b',8:'h',9:'i',11:'f',12:'d'}
        for i in range(n):
            e=raw[i*12:(i+1)*12];tag,t,count,ptr=struct.unpack(endian+'HHII',e);size=count*sizes[t]
            assert tag not in tags and 0<size<=CAP
            value=e[8:8+size] if size<=4 else get(ptr,size,f'tag-{tag}')
            tags[tag]=list(struct.unpack(endian+str(count)+fmt[t],value)) if t in fmt else value.hex()
        expected={256:p['canvas_xy'][0],257:p['canvas_xy'][1],258:8,259:32946,262:1,277:1,322:1024,323:1024}
        assert all(tags[k]==[v] for k,v in expected.items())
        assert tags.get(317,[1])==[1] and tags.get(274,[1])==[1] and tags.get(284,[1])==[1]
        assert struct.unpack(endian+'I',raw[-4:])[0]==0
        assert len(tags[324])==len(tags[325])==((p['canvas_xy'][0]+1023)//1024)*((p['canvas_xy'][1]+1023)//1024)
        assert all(0<o<o+c<=length for o,c in zip(tags[324],tags[325]))
        assert all(not(a<o+c and o<b) for a,b in ranges for o,c in zip(tags[324],tags[325])), 'Metadata range overlaps image payload'
        save(out/f'{w}-HEADER.json',dict(url=url,length=length,etag=etag,tags=tags,canvas_xy=p['canvas_xy'],metadata_ranges=ranges,no_image_payload_requested=True))
        print(w,'TIFF canvas',p['canvas_xy'],'metadata only',flush=True)
    save(out/'LEDGER.json',dict(complete=True,requests=ledger,header_body_bytes=total,cap_bytes=CAP,utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))

if __name__=='__main__':main()
