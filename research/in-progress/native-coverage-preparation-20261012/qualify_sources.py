"""Eight metadata-selected source crops, bounded HTTP reads and two decoders.

Exercises the proposed added stream. No model inference or cohort selection by
prediction values. It does not prove physical fine/native registration.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import urllib.request
import urllib.error
import zlib
import numpy as np
import cv2
import tifffile


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')


def prepare(root):
    path=root/'SOURCE_CROP_GO.json'
    if path.exists():raise FileExistsError(path)
    cohort=json.loads((root/'COHORT_CANDIDATES.json').read_text());cases=[];requests=[];pins={}
    for row in cohort['native_v2_audit']:
        if row['role']!='additional_regional_coverage_candidate':continue
        ident=row['sample'].lower()+'-'+row['segment'].split('-')[0]
        mp=root/'metadata'/f'{ident}-array.raw';hp=root/'teacher-metadata'/f'{ident}-HEADER.json'
        m=json.loads(mp.read_text());h=json.loads(hp.read_text());pins[str(mp.relative_to(root))]=sha(mp);pins[str(hp.relative_to(root))]=sha(hp)
        d,height,width=m['shape'];assert d==28 and m['chunks']==[28,128,128] and m['compressor'] is None and m['dtype']=='|u1'
        x=(width//2//128)*128;y=(height//2//128)*128
        assert x+128<=width and y+128<=height
        url='https://vesuvius-challenge-open-data.s3.amazonaws.com/'+row['native']['path'].rstrip('/')+f'/0/0/{y//128}/{x//128}'
        try:
            with urllib.request.urlopen(urllib.request.Request(url,method='HEAD'),timeout=40) as r:
                assert r.status==200 and int(r.headers['Content-Length'])==28*128*128
                etag=r.headers['ETag']
        except urllib.error.HTTPError as error:
            if error.code!=404:raise
            # Missing central storage is a recorded failed smoke-crop admission,
            # not permission to search for a more attractive location.
            cases.append(dict(id=ident,segment=row['segment'],native_shape=[d,height,width],
                native_crop_xy=[x,y,128,128],status='MISSING_CENTRAL_CHUNK',head_status=404,url=url,
                reason='No source body retrieved. No zero-filled substitute or replacement crop.'))
            continue
        cr=dict(id=ident+'-ct',url=url,etag=etag,bytes=28*128*128,path=f'source-crops/{ident}-ct.bin',kind='full')
        requests.append(cr)
        fh,fw=h['shape_yx'];x0=max(0,math.floor(x*fw/width)-2);x1=min(fw,math.ceil((x+128)*fw/width)+2)
        y0=max(0,math.floor(y*fh/height)-2);y1=min(fh,math.ceil((y+128)*fh/height)+2)
        tiles=[];columns=(fw+1023)//1024
        for ty in range(y0//1024,(y1-1)//1024+1):
            for tx in range(x0//1024,(x1-1)//1024+1):
                index=ty*columns+tx;offset=h['tags']['324'][index];size=h['tags']['325'][index]
                rq=dict(id=f'{ident}-tile-{index}',url=h['url'],etag=h['etag'],bytes=size,offset=offset,
                    object_bytes=h['object_bytes'],path=f'source-crops/{ident}-tile-{index}.bin',kind='range',tile_index=index,ty=ty,tx=tx)
                requests.append(rq);tiles.append(rq)
        cases.append(dict(id=ident,segment=row['segment'],status='SOURCE_PRESENT',native_shape=[d,height,width],native_crop_xy=[x,y,128,128],
            teacher_shape=[fh,fw],teacher_roi_xyxy=[x0,y0,x1,y1],header=str(hp.relative_to(root)),ct=cr,tiles=tiles))
    result=dict(status='FROZEN_BEFORE_CROP_PAYLOAD',selection='One globally aligned 128-square containing the metadata canvas centre per segment. All eight retained; do not replace blank/failed cases.',
        cases=cases,requests=requests,total_expected_body_bytes=sum(r['bytes'] for r in requests),
        pins=pins,cohort_sha256=sha(root/'COHORT_CANDIDATES.json'),implementation_sha256=sha(Path(__file__)),
        model_forwards=0,training_admission=False,physical_registration_claim=False)
    write(path,result);print(json.dumps(dict(status=result['status'],cases=8,requests=len(requests),bytes=result['total_expected_body_bytes'])),flush=True)


def acquire(root):
    plan=json.loads((root/'SOURCE_CROP_GO.json').read_text());assert sha(Path(__file__))==plan['implementation_sha256']
    for rel,digest in plan['pins'].items():assert sha(root/rel)==digest
    out=root/'source-crops';out.mkdir(exist_ok=True)
    ledger=[]
    for rq in plan['requests']:
        dest=root/rq['path'];receipt=dest.with_suffix('.json')
        if receipt.exists():
            r=json.loads(receipt.read_text());assert sha(dest)==r['sha256'] and dest.stat().st_size==rq['bytes'];ledger.append(r);continue
        headers={'If-Match':rq['etag'],'Accept-Encoding':'identity'}
        if rq['kind']=='range':headers['Range']=f'bytes={rq["offset"]}-{rq["offset"]+rq["bytes"]-1}'
        with urllib.request.urlopen(urllib.request.Request(rq['url'],headers=headers),timeout=40) as response:
            assert response.status==(206 if rq['kind']=='range' else 200) and response.headers['ETag']==rq['etag']
            if rq['kind']=='range':assert response.headers['Content-Range']==f'bytes {rq["offset"]}-{rq["offset"]+rq["bytes"]-1}/{rq["object_bytes"]}'
            data=response.read(rq['bytes']+1);assert len(data)==rq['bytes']
        dest.write_bytes(data);r=dict(rq,sha256=sha(dest));write(receipt,r);ledger.append(r)
    write(root/'SOURCE_CROP_ACQUISITION.json',dict(status='PASS',requests=ledger,body_bytes=sum(r['bytes'] for r in ledger),
        scope='Current retrieved objects pinned by ETag/length/SHA; no publisher historical byte identity claim'))
    print(json.dumps(dict(status='ACQUIRED',requests=len(ledger),bytes=sum(r['bytes'] for r in ledger))),flush=True)


def area_weights(start,n,dst_size,src_size,lo,hi):
    out=np.zeros((n,hi-lo),np.float64);scale=src_size/dst_size
    for i in range(n):
        left=(start+i)*scale;right=(start+i+1)*scale
        for j in range(math.floor(left),math.ceil(right)):
            if 0<=j<src_size:
                assert lo<=j<hi
                out[i,j-lo]=max(0,min(right,j+1)-max(left,j))/scale
    assert np.allclose(out.sum(1),1,atol=1e-10)
    return out


def assemble(root):
    cv2.setNumThreads(1)
    plan=json.loads((root/'SOURCE_CROP_GO.json').read_text());assert sha(Path(__file__))==plan['implementation_sha256']
    receipts=json.loads((root/'SOURCE_CROP_ACQUISITION.json').read_text())['requests']
    for r in receipts:assert sha(root/r['path'])==r['sha256'] and (root/r['path']).stat().st_size==r['bytes']
    out=root/'source-fixtures';out.mkdir(exist_ok=False);results=[]
    header_ledger=json.loads((root/'teacher-metadata/LEDGER.json').read_text())['requests']
    # First qualify sparse full-canvas resizing against a known complete image.
    rng=np.random.default_rng(120814)
    for sh,sw,dh,dw in [(359,517,91,131),(509,409,129,103),(257,385,64,96)]:
        src=rng.integers(0,256,(sh,sw),dtype=np.uint8);x,y=dw//3,dh//3;n=16
        lo=max(0,math.floor(x*sw/dw)-2);hi=min(sw,math.ceil((x+n)*sw/dw)+2)
        top=max(0,math.floor(y*sh/dh)-2);bottom=min(sh,math.ceil((y+n)*sh/dh)+2)
        sparse=np.zeros_like(src);sparse[top:bottom,lo:hi]=src[top:bottom,lo:hi]
        assert np.array_equal(cv2.resize(src,(dw,dh),interpolation=cv2.INTER_AREA)[y:y+n,x:x+n],
                              cv2.resize(sparse,(dw,dh),interpolation=cv2.INTER_AREA)[y:y+n,x:x+n])
    for case in plan['cases']:
        if case['status']!='SOURCE_PRESENT':
            results.append(case);continue
        ident=case['id'];header=json.loads((root/case['header']).read_text());fh,fw=case['teacher_shape'];_,h,w=case['native_shape'];x,y,_,_=case['native_crop_xy']
        # Large virtual zero canvas preserves OpenCV's global area-sampling phase.
        # Only frozen tiles are populated; zeros outside the dependency footprint
        # have no influence on the crop. These are not reconstructed full maps.
        canvas=np.zeros((fh,fw),np.uint8)
        temporary=out/(ident+'-sparse.tif')
        with temporary.open('wb') as f:
            f.truncate(header['object_bytes'])
            for hr in header_ledger:
                if hr['id']!=ident:continue
                data=(root/'teacher-metadata'/hr['path']).read_bytes();assert hashlib.sha256(data).hexdigest()==hr['sha256']
                f.seek(hr['offset']);f.write(data)
        with tifffile.TiffFile(temporary) as tf:
            page=tf.pages[0];assert page.shape==(fh,fw)
            for tile in case['tiles']:
                payload=(root/tile['path']).read_bytes();raw=zlib.decompress(payload);assert len(raw)==1024**2
                explicit=np.frombuffer(raw,np.uint8).reshape(1024,1024)
                official=page.decode(payload,tile['tile_index'])[0].reshape(1024,1024)
                assert np.array_equal(explicit,official),'Independent TIFF decode mismatch'
                ty,tx=tile['ty']*1024,tile['tx']*1024;hh=min(1024,fh-ty);ww=min(1024,fw-tx)
                canvas[ty:ty+hh,tx:tx+ww]=explicit[:hh,:ww]
        temporary.unlink()
        resized=cv2.resize(canvas,(w,h),interpolation=cv2.INTER_AREA)
        teacher=resized[y:y+128,x:x+128].copy();del resized
        x0,y0,x1,y1=case['teacher_roi_xyxy'];wx=area_weights(x,128,w,fw,x0,x1);wy=area_weights(y,128,h,fh,y0,y1)
        independent=wy@canvas[y0:y1,x0:x1].astype(np.float64)@wx.T
        diff=np.abs(independent-teacher.astype(np.float64));assert diff.max()<=.5002,float(diff.max())
        del canvas
        raw=np.fromfile(root/case['ct']['path'],np.uint8).reshape(28,128,128)
        valid=raw[14]>0;target=teacher>=128;mask=valid&((teacher>=128)|(teacher<=64))
        path=out/(ident+'.npz');np.savez_compressed(path,raw=raw,teacher=teacher,target=np.where(mask,target,False),mask=mask,valid=valid)
        row=dict(id=ident,native_crop_xy=case['native_crop_xy'],valid_pixels=int(valid.sum()),supervised_pixels=int(mask.sum()),
            teacher_positive_supervised=int((mask&target).sum()),teacher_negative_supervised=int((mask&~target).sum()),
            ignored_pixels=int((~mask).sum()),teacher_min=int(teacher.min()),teacher_max=int(teacher.max()),
            independent_area_max_abs_error=float(diff.max()),tiff_decoders_equal=True,
            file=str(path.relative_to(root)),bytes=path.stat().st_size,sha256=sha(path),
            interpretation='Published weak-label recipe mechanically reproduced on a fixed smoke crop; not physical registration or final support admission')
        results.append(row);print(json.dumps(row),flush=True)
    write(root/'SOURCE_QUALIFICATION.json',dict(status='ALL_FIXED_CASES_ACCOUNTED',cases=results,source_crop_go_sha256=sha(root/'SOURCE_CROP_GO.json'),
        sparse_resize_known_image_checks=3,official_and_explicit_tiff_decoders_equal=True,
        all_eight_cases_retained=True,body_bytes=sum(r['bytes'] for r in receipts),new_inference=0,
        final_training_population=False,physical_fine_native_registration=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','acquire','assemble']);p.add_argument('--work',type=Path,required=True);a=p.parse_args()
    globals()[a.mode](a.work)
