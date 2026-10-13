"""Explicit finite TIFF image plan, conditional fetch, and offline reconstruction."""
from common import *
import argparse,datetime,urllib.request,zlib

def decode(body):
    z=zlib.decompressobj();b=z.decompress(body,1024*1024+1)
    assert len(b)==1024*1024 and z.eof and not z.unused_data and not z.unconsumed_tail
    return np.frombuffer(b,np.uint8).reshape(1024,1024)

def prepare():
    sites=[s for s in read(R/'ALTERNATIVES.json')['sites'] if s['admitted_for_bounded_target_sensitivity']];rows=[]
    for w in sorted({s['winding'] for s in sites}):
        h=read(R/'metadata'/f'{w}-HEADER.json');tags=h['tags'];nx=(h['canvas_xy'][0]+1023)//1024;indices=set()
        for s in [s for s in sites if s['winding']==w]:
            x0,y0,x1,y1=s['source_bbox_fullfine_xyxy']
            assert 0<=x0<x1<=h['canvas_xy'][0] and 0<=y0<y1<=h['canvas_xy'][1]
            indices|={yy*nx+xx for yy in range(y0//1024,(y1-1)//1024+1) for xx in range(x0//1024,(x1-1)//1024+1)}
        for i in sorted(indices):
            rows.append(dict(winding=w,index=i,xy=[i%nx,i//nx],url=h['url'],etag=h['etag'],length=h['length'],
                offset=tags['324'][i],bytes=tags['325'][i],file=f'teacher/{w}-tile-{i:04d}.deflate'))
    total=sum(x['bytes'] for x in rows);assert total<=32*1024**2 and len(rows)<=256
    assert not (R/'REQUEST_PLAN.json').exists()
    paths=['PROTOCOL.md','INPUTS.json','ALTERNATIVES.json','PROVENANCE.json','common.py','prepare.py','metadata.py','acquire.py','analyse.py','synthetic_checks.py','SYNTHETIC_CHECKS.json']
    paths += [str(p.relative_to(R)) for p in sorted((R/'metadata').glob('*')) if p.is_file() and '.partial.' not in p.name]
    save(R/'REQUEST_PLAN.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),target_pixels_read=False,
        requests=rows,body_bytes=total,cap_bytes=32*1024**2,code_and_inputs=[receipt(R/p) for p in paths]))
    print('Frozen',len(rows),'image ranges,',total,'bytes')

def reconstruct(plan,receipts,out,compare=False):
    records={r['file']:r for r in receipts};data={}
    for t in plan['requests']:
        p=R/t['file'];r=records[t['file']];assert p.stat().st_size==t['bytes'] and sha(p)==r['compressed_sha256']
        a=decode(p.read_bytes());assert hashlib.sha256(a.tobytes()).hexdigest()==r['decoded_sha256'];data[(t['winding'],t['index'])]=a
    reports=[]
    for site in read(R/'ALTERNATIVES.json')['sites']:
        if not site['admitted_for_bounded_target_sensitivity']:continue
        sid=site['core_id'];w=site['winding'];x0,y0,x1,y1=site['source_bbox_fullfine_xyxy'];crop=np.zeros((y1-y0,x1-x0),np.uint8);valid=np.zeros(crop.shape,bool)
        for t in [t for t in plan['requests'] if t['winding']==w]:
            ox,oy=[v*1024 for v in t['xy']];xa,xb=max(x0,ox),min(x1,ox+1024);ya,yb=max(y0,oy),min(y1,oy+1024)
            if xa>=xb or ya>=yb:continue
            crop[ya-y0:yb-y0,xa-x0:xb-x0]=data[(w,t['index'])][ya-oy:yb-oy,xa-ox:xb-ox];valid[ya-y0:yb-y0,xa-x0:xb-x0]=True
        assert valid.all()
        p=out/(sid+'.npy')
        if compare:assert np.array_equal(crop,np.load(p,allow_pickle=False))
        else:np.save(p,crop)
        reports.append(dict(core_id=sid,origin_xy=[x0,y0],shape=list(crop.shape),sha256=sha(p),decoded_sha256=hashlib.sha256(crop.tobytes()).hexdigest()))
    return reports

def execute(replay=False):
    plan=read(R/'REQUEST_PLAN.json');verify(plan['code_and_inputs']);verify(read(R/'INPUTS.json')['files'])
    out=R/'teacher'
    if replay:
        ledger=read(out/'LEDGER.json');assert ledger['complete'] and ledger['request_plan_sha256']==sha(R/'REQUEST_PLAN.json')
        reports=reconstruct(plan,ledger['requests'],out,compare=True)
        assert reports==read(out/'CROPS.json');print('Offline compressed tile/crop replay PASS');return
    out.mkdir(exist_ok=False);rows=[]
    for t in plan['requests']:
        off,n=t['offset'],t['bytes'];req=urllib.request.Request(t['url'],headers={'Range':f'bytes={off}-{off+n-1}','If-Match':t['etag']})
        with urllib.request.urlopen(req,timeout=30) as response:
            assert response.status==206 and response.headers.get('Content-Range')==f'bytes {off}-{off+n-1}/{t["length"]}'
            assert response.headers.get('ETag')==t['etag'];body=response.read(n+1);assert len(body)==n
            headers={k:response.headers.get(k) for k in ['Content-Range','Content-Length','ETag','Last-Modified']}
        a=decode(body);p=R/t['file'];p.write_bytes(body)
        rows.append(dict(**t,status=206,response_headers=headers,compressed_sha256=sha(p),decoded_sha256=hashlib.sha256(a.tobytes()).hexdigest(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        save(out/'LEDGER.partial.json',dict(complete=False,requests=rows,request_plan_sha256=sha(R/'REQUEST_PLAN.json')))
        print(t['winding'],t['index'],n,'bytes',flush=True)
    reports=reconstruct(plan,rows,out)
    save(out/'CROPS.json',reports);save(out/'LEDGER.json',dict(complete=True,requests=rows,request_plan_sha256=sha(R/'REQUEST_PLAN.json'),body_bytes=sum(r['bytes'] for r in rows)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','fetch','replay']);a=p.parse_args()
    prepare() if a.mode=='prepare' else execute(a.mode=='replay')
