"""Bounded, replayable mask-first intake. Never imports or executes a model."""
from __future__ import annotations
import argparse, csv, hashlib, json, os, shutil, sys, time
from datetime import datetime, timezone
from pathlib import Path
import cv2, numpy as np, requests, zarr
import indexed_shard
from sampling import context, spread

HERE=Path(__file__).resolve().parent
BASE='https://vesuvius-challenge-open-data.s3.amazonaws.com/'
CODE=['PROTOCOL.md','sampling.py','intake.py','indexed_shard.py','hecate_prep_original.py']


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def ash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def load(path):return json.loads(Path(path).read_text())
def stamp():return datetime.now(timezone.utc).isoformat()
def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.partial')
    with tmp.open('w') as f:
        json.dump(obj,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
def pin(p):return {'bytes':p.stat().st_size,'sha256':sha(p)}
def hist(a):
    h=np.bincount(a.ravel(),minlength=256)
    return {str(i):int(v) for i,v in enumerate(h) if v}
def csvfile(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def freeze(args):
    root=args.work
    if (root/'FREEZE.json').exists():raise FileExistsError('Already frozen')
    root.mkdir(parents=True,exist_ok=True)
    (root/'implementation').mkdir(exist_ok=True);(root/'metadata').mkdir(exist_ok=True)
    for name in CODE:shutil.copyfile(HERE/name,root/'implementation'/name)
    for name in ['CATALOG_ADMISSION.json','METADATA_RESULTS.json']:
        shutil.copyfile(args.packet/name,root/name)
    meta=load(root/'METADATA_RESULTS.json');cases=[]
    for r in meta['source_pins']:
        source=args.metadata/r['file'];assert pin(source)=={k:r[k] for k in ['bytes','sha256']}
        shutil.copyfile(source,root/'metadata'/source.name)
        rp=source.with_suffix('.receipt.json');shutil.copyfile(rp,root/'metadata'/rp.name)
    for r in meta['rows']:
        ident=r['sample'].lower()+'-'+r['segment'];case=dict(r,id=ident,arrays=[])
        for role in ['supervision','validation','inklabels']:
            mp=root/'metadata'/f'{ident}-{role}-array-v3.raw'
            receipt=load(mp.with_suffix('.receipt.json'))
            m=load(mp);indexed_shard.validate_metadata(m)
            url=receipt['url'].removesuffix('zarr.json')+'c/0/0'
            with requests.head(url,headers={'Accept-Encoding':'identity'},timeout=(10,40),allow_redirects=False) as response:
                assert response.status_code==200,f'HEAD {response.status_code}: {ident}/{role}'
                size=int(response.headers['Content-Length']);etag=response.headers['ETag']
                assert 0<size<=4*1024**2
                item=dict(id=ident+'-'+role,case=ident,role=role,url=url,expected_bytes=size,etag=etag,
                          metadata=mp.name,relative_path=f'arrays/{ident}/{role}/c/0/0',
                          last_modified=response.headers.get('Last-Modified'),head_utc=stamp())
            case['arrays'].append(item)
        cases.append(case)
    plan={'cases':cases,'body_cap_bytes':sum(i['expected_bytes'] for c in cases for i in c['arrays'])+1024**2,
          'metadata_intake_sha256':sha(root/'METADATA_RESULTS.json'),'ink_opened_before_selection':False}
    write(root/'SOURCE_PLAN.json',plan)
    pins={str(p.relative_to(root)):pin(p) for p in sorted(root.rglob('*')) if p.is_file()}
    write(root/'FREEZE.json',{'utc':stamp(),'pins':pins,'annotation_payload_bytes_before_freeze':0,'new_ct_bytes_before_freeze':0})
    print(json.dumps({'status':'FROZEN','arrays':12,'expected_annotation_bytes':sum(i['expected_bytes'] for c in cases for i in c['arrays'])}),flush=True)


def verify_freeze(root):
    f=load(root/'FREEZE.json')
    for name,expected in f['pins'].items():assert pin(root/name)==expected,name
    for name in CODE:assert pin(HERE/name)==f['pins']['implementation/'+name],name
    return load(root/'SOURCE_PLAN.json')


def fetch(root,item,stage,cap):
    dest=root/item['relative_path'];receipt=root/'receipts'/(item['id']+'.json')
    if receipt.exists():
        r=load(receipt);assert r['url']==item['url'] and r['bytes']==item['expected_bytes']
        assert pin(dest)=={k:r[k] for k in ['bytes','sha256']}
        if 'etag' in item:assert r['etag']==item['etag']
        return r
    bp=root/(stage+'-BODY_BUDGET.json')
    b=load(bp) if bp.exists() else dict(cap_bytes=cap,reserved_bytes=0,observed_bytes=0,attempts=0,completed=0,
          accounting='Whole expected body reserved before each attempt; failed/interrupted reservation retained. Observed bytes may omit a process-interrupted attempt.')
    assert b['cap_bytes']==cap and b['reserved_bytes']+item['expected_bytes']<=cap
    b['reserved_bytes']+=item['expected_bytes'];b['attempts']+=1;write(bp,b)
    dest.parent.mkdir(parents=True,exist_ok=True);partial=dest.with_suffix('.partial');seen=0
    headers={'Accept-Encoding':'identity'}
    if 'etag' in item:headers['If-Match']=item['etag']
    try:
        with requests.get(item['url'],headers=headers,stream=True,timeout=(10,45),allow_redirects=False) as response:
            assert response.status_code==200,f"HTTP {response.status_code}: {item['id']}"
            assert response.headers.get('Content-Encoding','identity')=='identity'
            assert int(response.headers['Content-Length'])==item['expected_bytes']
            assert response.headers.get('ETag')
            if 'etag' in item:assert response.headers['ETag']==item['etag']
            with partial.open('wb') as stream:
                for part in response.iter_content(65536):
                    seen+=len(part)
                    if seen>item['expected_bytes']:raise ValueError('Unexpected object length')
                    stream.write(part)
                stream.flush();os.fsync(stream.fileno())
            assert seen==item['expected_bytes']
            os.replace(partial,dest)
            r=dict(id=item['id'],url=item['url'],**pin(dest),etag=response.headers['ETag'],
                   last_modified=response.headers.get('Last-Modified'),utc=stamp(),stage=stage,
                   relative_path=item['relative_path'],identity_scope='Current retrieved object, not a historical scan checksum')
            write(receipt,r);b['completed']+=1
    except Exception as e:
        write(root/'errors'/f'{stage}-{b["attempts"]:04}.json',dict(id=item['id'],error_type=type(e).__name__,observed_bytes=seen,utc=stamp()))
        raise
    finally:
        b['observed_bytes']+=seen;write(bp,b)
    return r


def decode(root,item):
    mp=root/'metadata'/item['metadata'];meta=load(mp)
    p=root/item['relative_path'];r=load(root/'receipts'/(item['id']+'.json'))
    assert pin(p)=={k:r[k] for k in ['bytes','sha256']}
    store=p.parents[2];(store/'zarr.json').write_bytes(mp.read_bytes())
    h,w=meta['shape'];blob=p.read_bytes()
    arr,present,receipt=indexed_shard.decode_shard_roi(blob,meta,[0,0,w,h],maximum_encoded_bytes=item['expected_bytes'],
        maximum_expanded_bytes=3*1024**3,expected_sha256=r['sha256'],expected_byte_length=r['bytes'])
    del present
    official=np.asarray(zarr.open_array(str(store),mode='r')[:])
    assert official.dtype==arr.dtype==np.uint8 and np.array_equal(official,arr),'Installed Zarr differs'
    summary={'shape':list(arr.shape),'histogram':hist(arr),'decoded_sha256':ash(arr),
             'official_zarr_sha256':ash(official),'official_zarr_equals_explicit':True,
             'zarr_version':zarr.__version__,'index_crc32c':receipt['index_crc32c'],
             'index_sha256':receipt['index_sha256'],'present_chunks':receipt['present_inner_chunks']}
    write(root/'decode-receipts'/(item['id']+'.json'),summary)
    return arr,summary


def resize_mask(a,shape):
    return cv2.resize((a>0).astype(np.uint8)*255,(shape[1],shape[0]),interpolation=cv2.INTER_AREA)>=128


def select(args):
    root=args.work;p=verify_freeze(root)
    if (root/'SELECTION.json').exists():raise FileExistsError('Selection already sealed; use replay destination')
    assert not any((root/i['relative_path']).exists() for c in p['cases'] for i in c['arrays'] if i['role']=='inklabels')
    assert not (root/'ct').exists()
    cases=[];requests_plan=[]
    for c in p['cases']:
        arrs={}
        for i in c['arrays']:
            if i['role']=='inklabels':continue
            fetch(root,i,'annotations',p['body_cap_bytes']);arrs[i['role']]=decode(root,i)[0]
        s,v=arrs['supervision'],arrs['validation'];h,w=c['native']['shape_zyx'][1:]
        sn,vn=resize_mask(s,(h,w)),resize_mask(v,(h,w));separate=sn|vn
        union=resize_mask(np.maximum(s,v),(h,w))
        support=separate if c['sample']=='PHerc0841' else union
        stats=dict(fine_overlap=int(np.count_nonzero((s>0)&(v>0))),native_overlap=int((sn&vn).sum()),
                   native_supervision=int(sn.sum()),native_validation=int(vn.sum()),
                   union_order_disagreement=int(np.count_nonzero(separate!=union)),support_pixels=int(support.sum()),
                   support_policy='resize_then_union' if c['sample']=='PHerc0841' else 'union_then_resize',
                   support_sha256=ash(support))
        del s,v,arrs
        folder=root/'support';folder.mkdir(exist_ok=True)
        np.savez_compressed(folder/(c['id']+'.npz'),supervision=sn,validation=vn,separate_union=separate,
                            fine_union_resized=union,participation=support)
        rows=[];eligible=[]
        for y in range(0,h-127,128):
            for x in range(0,w-127,128):
                n=int(support[y:y+128,x:x+128].sum());ctx=context((h,w),c['native']['spacing_zyx_um'][0],x,y)
                row=dict(id=f'{c["id"]}-x{x:05}-y{y:05}',x=x,y=y,supported_pixels=n,
                         complete_context=ctx is not None,eligible=n>=4096 and ctx is not None)
                rows.append(row)
                if row['eligible']:eligible.append(row)
        csvfile(root/'support'/(c['id']+'-all-cores.csv'),rows)
        if len(eligible)<6:
            write(root/'BLOCKED.json',dict(case=c['id'],eligible=len(eligible),required=6,selection_changed=False))
            raise ValueError('Insufficient eligible cores')
        chosen=spread(eligible);cores=[];needed=set()
        for rank,r in enumerate(chosen):
            ctx=context((h,w),c['native']['spacing_zyx_um'][0],r['x'],r['y']);cores.append(dict(r,selection_order=rank,**ctx))
            needed.update(map(tuple,ctx['native_chunks_zyx']))
        ctmeta=load(root/'metadata'/f'{c["id"]}-{c["native"]["volume"]}-array.raw')
        assert ctmeta['dtype']=='|u1' and ctmeta['compressor'] is None and ctmeta['filters'] is None
        assert ctmeta['order']=='C' and ctmeta['dimension_separator']=='/' and ctmeta['chunks']==[c['native']['shape_zyx'][0],128,128]
        for z,cy,cx in sorted(needed):
            ident=f'{c["id"]}-z{z}-y{cy}-x{cx}';key=f'{z}/{cy}/{cx}'
            requests_plan.append(dict(id=ident,case=c['id'],chunk_zyx=[z,cy,cx],url=BASE+c['native']['source_path']+'0/'+key,
                         expected_bytes=int(np.prod(ctmeta['chunks'])),relative_path=f'ct/{c["id"]}/0/{key}'))
        cases.append(dict(id=c['id'],sample=c['sample'],segment=c['segment'],native=c['native'],mask_summary=stats,
                          eligible_cores=len(eligible),cores=cores,source_chunk_count=len(needed)))
        print(json.dumps({'case':c['id'],'eligible':len(eligible),'chosen':[(r['x'],r['y']) for r in chosen],**stats}),flush=True)
    summary=dict(status='FROZEN_BEFORE_INK_CT_OR_INFERENCE',utc=stamp(),freeze_sha256=sha(root/'FREEZE.json'),
                 cv2_version=cv2.__version__,cases=cases,selected_cores=sum(len(c['cores']) for c in cases),
                 ink_payload_bytes_opened=0,CT_bytes_opened=0,model_forwards=0)
    write(root/'SELECTION.json',summary)
    expected=sum(r['expected_bytes'] for r in requests_plan)
    write(root/'CT_PLAN.json',dict(selection_sha256=sha(root/'SELECTION.json'),requests=requests_plan,
                                 expected_bytes=expected,body_cap_bytes=expected+8*1024**2))


def acquire_ct(args):
    root=args.work;verify_freeze(root);p=load(root/'CT_PLAN.json')
    assert p['selection_sha256']==sha(root/'SELECTION.json')
    started=time.monotonic()
    for n,r in enumerate(p['requests']):
        fetch(root,r,'ct',p['body_cap_bytes'])
        if (n+1)%25==0:print(json.dumps({'CT_objects_complete':n+1,'total':len(p['requests'])}),flush=True)
    write(root/'CT_COMPLETE.json',dict(status='PASS',plan_sha256=sha(root/'CT_PLAN.json'),
          objects=len(p['requests']),bytes=p['expected_bytes'],seconds=time.monotonic()-started))


def source_canvas(root,c):
    """Sparse canvas + independent known mask. Unknown zeros are never observed CT."""
    d,h,w=c['native']['shape_zyx'];a=np.zeros((d,h,w),np.uint8);known=np.zeros((h,w),bool)
    plan=load(root/'CT_PLAN.json')
    for r in plan['requests']:
        if r['case']!=c['id']:continue
        path=root/r['relative_path'];rec=load(root/'receipts'/(r['id']+'.json'))
        assert pin(path)=={k:rec[k] for k in ['bytes','sha256']}
        z,cy,cx=r['chunk_zyx'];assert z==0
        tile=np.frombuffer(path.read_bytes(),dtype=np.uint8).reshape(d,128,128)
        yy,xx=cy*128,cx*128;ny,nx=min(128,h-yy),min(128,w-xx)
        a[:,yy:yy+ny,xx:xx+nx]=tile[:,:ny,:nx];known[yy:yy+ny,xx:xx+nx]=True
    return a,known


def inputs(args):
    root=args.work;p=verify_freeze(root);selection=load(root/'SELECTION.json')
    assert load(root/'CT_PLAN.json')['selection_sha256']==sha(root/'SELECTION.json')
    assert load(root/'CT_COMPLETE.json')['status']=='PASS'
    out=root/'cores';out.mkdir(exist_ok=True);results=[]
    for c in selection['cases']:
        origin=next(r for r in p['cases'] if r['id']==c['id'])
        item=next(i for i in origin['arrays'] if i['role']=='inklabels')
        fetch(root,item,'annotations',p['body_cap_bytes']);fine,_=decode(root,item)
        native=resize_mask(fine,c['native']['shape_zyx'][1:]);del fine
        a,known=source_canvas(root,c);d,h,w=a.shape
        support=np.load(root/'support'/(c['id']+'.npz'))['participation']
        cores=[]
        for r in c['cores']:
            x,y,ww,hh=r['core_xywh'];sl=np.s_[y:y+hh,x:x+ww]
            assert known[sl].all()
            for px,py,pw,ph in r['hybrid_tiles_xywh']:assert known[py:py+ph,px:px+pw].all()
            x0,y0,x1,y1=r['hecate_source_bbox_xyxy'];assert known[y0:y1,x0:x1].all()
            valid=a[d//2][sl]>0;mask=support[sl]&valid;ink=native[sl]
            np.savez_compressed(out/(r['id']+'.npz'),ink=ink,participation=support[sl],native_centre_nonzero=valid,
                                score_mask=mask,origin_xy=np.array([x,y],np.int32))
            cores.append(dict(id=r['id'],annotation_pixels=int(support[sl].sum()),score_pixels=int(mask.sum()),
                         positive_pixels=int((mask&ink).sum()),negative_pixels=int((mask&~ink).sum()),
                         zero_ct_annotation_pixels=int((support[sl]&~valid).sum()),score_mask_sha256=ash(mask),ink_sha256=ash(ink)))
        results.append(dict(id=c['id'],cores=cores));del a,known,support,native
        print(json.dumps({'qualified_inputs':c['id'],'score_pixels':sum(r['score_pixels'] for r in cores)}),flush=True)
    write(root/'INPUTS_COMPLETE.json',dict(status='PASS',selection_sha256=sha(root/'SELECTION.json'),cases=results,
          score_pixels=sum(r['score_pixels'] for c in results for r in c['cores']),model_forwards=0,
          claim='Array and context qualification only; no model or physical registration validation'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['freeze','select','ct','inputs'])
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--packet',type=Path);parser.add_argument('--metadata',type=Path)
    a=parser.parse_args();{'freeze':freeze,'select':select,'ct':acquire_ct,'inputs':inputs}[a.command](a)


if __name__=='__main__':main()
