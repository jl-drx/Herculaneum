"""Bounded acquisition and offline reconstruction of predeclared TIFF tiles."""
import argparse,datetime,hashlib,json,urllib.request,zlib
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def prepare():
    header=json.loads((R/'provenance/TEACHER_TIFF_HEADER.json').read_text());foot=json.loads((R/'SOURCE_FOOTPRINT.json').read_text());tags={x['tag']:x['value'] for x in header['tags']}
    for t,v in {256:31040,257:24880,258:8,259:32946,262:1,277:1,322:1024,323:1024}.items():assert tags[t]==[v],(t,tags[t])
    assert 317 not in tags and 274 not in tags and header['next_ifd_offset']==0
    x0,y0,x1,y1=foot['source_bbox_fullfine_xy_half_open'];tiles=[]
    assert 0<=x0<x1<=31040 and 0<=y0<y1<=24880
    for yy in range(y0//1024,(y1-1)//1024+1):
        for xx in range(x0//1024,(x1-1)//1024+1):
            i=yy*31+xx;o=tags[324][i];n=tags[325][i]
            assert 0<o<o+n<=header['length']
            tiles.append(dict(index=i,xy=[xx,yy],offset=o,bytes=n,name=f'tile-{i:03d}.deflate'))
    paths=['PROTOCOL.md','CT_CANDIDATES.json','SOURCE_FOOTPRINT.json','analysis.py','synthetic_checks.py','SYNTHETIC_CHECKS.json','acquire_teacher.py','provenance/TEACHER_TIFF_HEADER.json','provenance/ADMISSION_SUMMARY.json','native_eligible.npy','NATIVE_ELIGIBILITY.json']
    plan=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),target_pixels_read=False,url=header['url'],etag=header['etag'],length=header['length'],tiles=tiles,body_bytes=sum(t['bytes'] for t in tiles),body_cap_bytes=8*1024**2,bbox=[x0,y0,x1,y1],code_and_inputs=[dict(path=p,sha256=sha(R/p)) for p in paths])
    assert plan['body_bytes']<=plan['body_cap_bytes']
    assert not (R/'REQUEST_PLAN.json').exists()
    write(R/'REQUEST_PLAN.json',plan);print(json.dumps({k:plan[k] for k in ['body_bytes','bbox','created_utc']}));print('tiles',len(tiles))
def execute(replay=False):
    plan=json.loads((R/'REQUEST_PLAN.json').read_text())
    for e in plan['code_and_inputs']:assert sha(R/e['path'])==e['sha256'],e['path']
    out=R/'teacher';out.mkdir(exist_ok=True);receipts=[]
    x0,y0,x1,y1=plan['bbox'];crop=np.zeros((y1-y0,x1-x0),np.uint8);coverage=np.zeros(crop.shape,bool)
    for t in plan['tiles']:
        p=out/t['name'];off=t['offset'];size=t['bytes']
        if replay:
            stored=json.loads((out/'REQUEST_LEDGER.json').read_text());entry=next(e for e in stored['requests'] if e['index']==t['index']);assert sha(p)==entry['compressed_sha256'];body=p.read_bytes();headers=entry['response_headers']
        else:
            assert not p.exists(),'Refusing accidental overwrite; use replay after complete successful acquisition'
            request=urllib.request.Request(plan['url'],headers={'Range':f'bytes={off}-{off+size-1}','If-Match':plan['etag']})
            with urllib.request.urlopen(request,timeout=45) as r:
                assert r.status==206,r.status
                assert r.headers.get('Content-Range')==f'bytes {off}-{off+size-1}/{plan["length"]}'
                assert r.headers.get('ETag')==plan['etag']
                body=r.read(size+1);headers={k:r.headers.get(k) for k in ['Content-Range','Content-Length','ETag','Last-Modified']}
            assert len(body)==size; p.write_bytes(body)
        d=zlib.decompressobj();raw=d.decompress(body,1024*1024+1)
        assert len(raw)==1024*1024 and d.eof and not d.unused_data and not d.unconsumed_tail
        a=np.frombuffer(raw,np.uint8).reshape(1024,1024);ox,oy=[v*1024 for v in t['xy']]
        xa,xb=max(x0,ox),min(x1,ox+1024);ya,yb=max(y0,oy),min(y1,oy+1024)
        crop[ya-y0:yb-y0,xa-x0:xb-x0]=a[ya-oy:yb-oy,xa-ox:xb-ox];coverage[ya-y0:yb-y0,xa-x0:xb-x0]=True
        receipts.append(dict(**t,compressed_sha256=sha(p),decoded_sha256=hashlib.sha256(raw).hexdigest(),response_headers=headers,retrieved_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        if not replay:write(out/'REQUEST_LEDGER.partial.json',dict(plan_sha256=sha(R/'REQUEST_PLAN.json'),complete=False,requests=receipts))
    assert coverage.all()
    if replay:
        original=np.load(out/'crop.npy',allow_pickle=False);assert np.array_equal(crop,original);print('Offline tile replay PASS');return
    np.save(out/'crop.npy',crop)
    write(out/'CROP.json',dict(origin_fullfine_xy=[x0,y0],crop_shape_yx=list(crop.shape),dtype='uint8',crop_sha256=sha(out/'crop.npy'),decoded_pixel_sha256=hashlib.sha256(crop.tobytes()).hexdigest(),meaning='current released teacher subset; not historical whole-object identity'))
    write(out/'REQUEST_LEDGER.json',dict(plan_sha256=sha(R/'REQUEST_PLAN.json'),complete=True,requests=receipts,body_bytes=sum(t['bytes'] for t in receipts)))
    print(json.dumps(dict(status='PASS',tiles=len(receipts),body_bytes=sum(t['bytes'] for t in receipts),shape=list(crop.shape))))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','fetch','replay']);a=p.parse_args();prepare() if a.mode=='prepare' else execute(a.mode=='replay')
