"""Frozen native annotation coverage intake; no CT, model or training operations."""
from __future__ import annotations
import argparse, csv, hashlib, json, math, pathlib, shutil, struct, sys, tempfile
from datetime import datetime, timezone
import google_crc32c, numcodecs, numpy as np, requests, zarr

ROOT=pathlib.Path(__file__).resolve().parent
CAP=4*1024*1024
SENTINEL=(1<<64)-1

def sha(b):return hashlib.sha256(b).hexdigest()
def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def load(path):return json.loads(path.read_text())
def stamp():return datetime.now(timezone.utc).isoformat()
def metadata(item):
    p=ROOT/'provenance'/pathlib.Path(item['metadata_file']).name
    assert sha(p.read_bytes())==item['metadata_sha256']
    return load(p)
def layout(meta):
    assert meta['zarr_format']==3 and meta['node_type']=='array'
    assert meta['data_type']=='uint8' and meta['fill_value']==0
    assert meta['chunk_grid']['name']=='regular'
    assert meta['chunk_key_encoding']=={'name':'default','configuration':{'separator':'/'}}
    assert meta.get('storage_transformers',[])==[]
    shape=tuple(meta['shape']);outer=tuple(meta['chunk_grid']['configuration']['chunk_shape'])
    assert len(shape)==len(outer)==2 and all(0<s<=o for s,o in zip(shape,outer))
    assert len(meta['codecs'])==1 and meta['codecs'][0]['name']=='sharding_indexed'
    conf=meta['codecs'][0]['configuration'];inner=tuple(conf['chunk_shape'])
    assert inner==(512,512) and all(o%i==0 for o,i in zip(outer,inner))
    assert conf['index_location']=='end'
    assert conf['index_codecs']==[{'name':'bytes','configuration':{'endian':'little'}},{'name':'crc32c'}]
    codecs=conf['codecs'];assert len(codecs)==2 and codecs[0]['name']=='bytes' and codecs[0].get('configuration',{}).get('endian','little')=='little'
    assert codecs[1]['name']=='blosc'
    cfg=codecs[1]['configuration']
    assert cfg=={'typesize':1,'cname':'zstd','clevel':3,'shuffle':'bitshuffle','blocksize':0}
    return shape,outer,inner

def parse_index(body,meta):
    shape,outer,inner=layout(meta);grid=tuple(o//i for o,i in zip(outer,inner))
    n=math.prod(grid);index_size=n*16+4
    if len(body)<index_size:raise ValueError('short shard/index')
    ib=body[-index_size:];crc=struct.unpack('<I',ib[-4:])[0]
    if google_crc32c.value(ib[:-4])!=crc:raise ValueError('CRC32C mismatch')
    index=np.frombuffer(ib[:-4],dtype='<u8').reshape(*grid,2)
    payload_stop=len(body)-index_size;present=[];missing=0
    for y,x in np.ndindex(*grid):
        off,nbytes=map(int,index[y,x])
        if off==SENTINEL or nbytes==SENTINEL:
            if (off,nbytes)!=(SENTINEL,SENTINEL):raise ValueError('partial missing sentinel')
            missing+=1;continue
        if nbytes<=0 or off+nbytes>payload_stop:raise ValueError('index range out of bounds')
        present.append((off,off+nbytes,y,x))
    intervals=sorted(present)
    if any(a[1]>b[0] for a,b in zip(intervals,intervals[1:])):raise ValueError('overlapping inner ranges')
    return index,present,{'index_crc32c':f'{crc:08x}','index_bytes':index_size,'payload_bytes':payload_stop,'present_chunks':len(present),'missing_chunks':missing,'index_entries':n,'index_ranges_valid_nonoverlapping':True}

def decode_explicit(body,meta):
    shape,outer,inner=layout(meta);index,present,checks=parse_index(body,meta)
    arr=np.zeros(shape,dtype=np.uint8)
    codec=numcodecs.Blosc(cname='zstd',clevel=3,shuffle=numcodecs.Blosc.BITSHUFFLE,blocksize=0)
    for start,end,y,x in present:
        compressed=body[start:end]
        if len(compressed)<16:raise ValueError('short Blosc header')
        raw_bytes,block_bytes,compressed_bytes=struct.unpack_from('<III',compressed,4)
        if raw_bytes!=math.prod(inner) or compressed_bytes!=len(compressed):raise ValueError('Blosc header length mismatch')
        raw=codec.decode(compressed)
        if len(raw)!=math.prod(inner):raise ValueError('wrong decoded inner chunk size')
        chunk=np.frombuffer(raw,dtype='u1').reshape(inner)
        y0,x0=y*inner[0],x*inner[1];ny,nx=min(inner[0],shape[0]-y0),min(inner[1],shape[1]-x0)
        if ny>0 and nx>0:arr[y0:y0+ny,x0:x0+nx]=chunk[:ny,:nx]
    return arr,checks

def decode_crosscheck(store,meta):
    body=(store/'c'/'0'/'0').read_bytes();a,checks=decode_explicit(body,meta)
    b=np.asarray(zarr.open_array(str(store),mode='r')[:])
    assert a.shape==b.shape==tuple(meta['shape']) and a.dtype==b.dtype==np.uint8
    if not np.array_equal(a,b):raise ValueError('official Zarr disagrees with explicit decoder')
    checks.update({'shape':list(a.shape),'dtype':str(a.dtype),'decoded_sha256':sha(a.tobytes()),'official_zarr_decoded_sha256':sha(b.tobytes()),'official_zarr_equals_explicit':True,'zarr_version':zarr.__version__,'numcodecs_version':numcodecs.__version__})
    return a,checks

def histogram(a):
    h=np.bincount(a.ravel(),minlength=256)
    return {str(i):int(v) for i,v in enumerate(h) if v}

def fetch(item,ledger):
    store=ROOT/'arrays'/item['winding']/item['role'];out=store/'c'/'0'/'0'
    assert not out.exists(),'refuse overwrite; use replay for existing shards'
    out.parent.mkdir(parents=True,exist_ok=True)
    mp=ROOT/'provenance'/pathlib.Path(item['metadata_file']).name
    (store/'zarr.json').write_bytes(mp.read_bytes());meta=metadata(item);layout(meta)
    before=ledger['body_bytes'];chunks=[]
    with requests.get(item['url'],headers={'If-Match':item['etag'],'Accept-Encoding':'identity'},stream=True,timeout=(10,60),allow_redirects=False) as response:
        assert response.status_code==200,f'HTTP {response.status_code}'
        assert response.headers.get('ETag')==item['etag']
        assert int(response.headers['Content-Length'])==item['content_length']
        assert response.headers.get('Content-Encoding','identity')=='identity'
        for part in response.iter_content(8192):
            ledger['body_bytes']+=len(part)
            write(ROOT/'REQUEST_LEDGER.partial.json',ledger)
            if ledger['body_bytes']>CAP:raise ValueError('body cap exceeded')
            if ledger['body_bytes']-before>item['content_length']:raise ValueError('object body exceeded pinned length')
            chunks.append(part)
        body=b''.join(chunks)
        assert len(body)==item['content_length']
        out.write_bytes(body)
        row={'winding':item['winding'],'role':item['role'],'url':item['url'],'bytes':len(body),'sha256':sha(body),'etag':response.headers['ETag'],'last_modified':response.headers.get('Last-Modified'),'retrieved_utc':stamp(),'path':str(out.relative_to(ROOT))}
    ledger['requests'].append(row);write(ROOT/'REQUEST_LEDGER.partial.json',ledger)
    return store

def enumerate_coverage(mask,item):
    h,w=mask.shape;rows=[];native=load(ROOT/'provenance'/pathlib.Path(item['native_metadata_file']).name)
    assert native['shape']==[28,h,w] and native['chunks']==[28,128,128] and native['compressor'] is None and native['dtype']=='|u1'
    needed=set();total_supervised=int(np.count_nonzero(mask));eligible_supervised=0
    for y in range(0,h,128):
        for x in range(0,w,128):
            complete_core=(x+128<=w and y+128<=h)
            context_ok=(x>=64 and y>=64 and x+192<=w and y+192<=h)
            n=int(np.count_nonzero(mask[y:min(y+128,h),x:min(x+128,w)]))
            eligible=complete_core and context_ok and n>=1024
            row={'core_id':f"{item['winding']}-x{x:05d}-y{y:05d}",'x':x,'y':y,'width':128,'height':128,'supervised_pixels':n,'complete_core':complete_core,'complete_context':context_ok,'eligible':eligible,'reason':'eligible' if eligible else ('incomplete_core' if not complete_core else ('incomplete_context' if not context_ok else 'supervision_below_1024'))}
            rows.append(row)
            if eligible:
                eligible_supervised+=n
                for cy in range((y-64)//128,(y+191)//128+1):
                    for cx in range((x-64)//128,(x+191)//128+1):needed.add((0,cy,cx))
    elig=[r for r in rows if r['eligible']]
    return {'winding':item['winding'],'public_segment':item['public_segment'],'shape':list(mask.shape),'mask_histogram':histogram(mask),'supervised_pixels':total_supervised,'supervised_fraction':total_supervised/mask.size,'grid_positions_including_partial':len(rows),'complete_context_candidates':sum(r['complete_context'] for r in rows),'eligible_cores':len(elig),'eligible_output_area':len(elig)*128**2,'eligible_supervised_pixels':eligible_supervised,'supervised_pixels_outside_eligible_cores':total_supervised-eligible_supervised,'prospective_ct_chunks_zyx':[list(c) for c in sorted(needed)],'prospective_ct_unique_chunks':len(needed),'prospective_ct_bytes':len(needed)*28*128*128,'prospective_ct_chunk_bytes':28*128*128,'rows':rows}

def csv_rows(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['core_id']);w.writeheader();w.writerows(rows)

def frozen_inputs():
    return {str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in [ROOT/'PROTOCOL.md',ROOT/'SOURCE_PLAN.json',ROOT/'intake.py']+sorted((ROOT/'provenance').glob('*.json'))}

def verify_freeze():
    frozen=load(ROOT/'FREEZE.json')
    for path,digest in frozen['files'].items():assert sha((ROOT/path).read_bytes())==digest,f'frozen input changed: {path}'

def freeze():
    assert not (ROOT/'FREEZE.json').exists()
    assert not (ROOT/'arrays').exists()
    plan=load(ROOT/'SOURCE_PLAN.json');assert sum(r['content_length'] for r in plan['arrays'])<CAP
    assert len(plan['arrays'])==6
    for item in plan['arrays']:layout(metadata(item))
    write(ROOT/'FREEZE.json',{'created_utc':stamp(),'files':frozen_inputs(),'source_pixel_bytes_before_freeze':0,'maximum_body_bytes':CAP})

def masks():
    verify_freeze();assert load(ROOT/'SYNTHETIC_CHECKS.json')['all_passed']
    assert not (ROOT/'MASK_STAGE_COMPLETE.json').exists()
    plan=load(ROOT/'SOURCE_PLAN.json');ledger={'started_utc':stamp(),'freeze_sha256':sha((ROOT/'FREEZE.json').read_bytes()),'requests':[],'body_bytes':0,'complete':False}
    sites=[];decodes=[]
    for item in sorted((r for r in plan['arrays'] if r['role']=='supervision'),key=lambda r:r['winding']):
        store=fetch(item,ledger);mask,checks=decode_crosscheck(store,metadata(item));checks.update({'winding':item['winding'],'role':'supervision'});decodes.append(checks)
        cov=enumerate_coverage(mask,item);sites.append(cov)
        csv_rows(ROOT/f"{item['winding']}-all-grid.csv",cov['rows']);csv_rows(ROOT/f"{item['winding']}-eligible-cores.csv",[r for r in cov['rows'] if r['eligible']])
    write(ROOT/'MASK_COVERAGE.json',{'stage':'mask-only before inklabel acquisition','sites':sites})
    write(ROOT/'MASK_DECODE_CHECKS.json',decodes)
    paths=[ROOT/'MASK_COVERAGE.json',ROOT/'MASK_DECODE_CHECKS.json']+sorted(ROOT.glob('*-cores.csv'))+sorted(ROOT.glob('*-grid.csv'))
    receipt={'completed_utc':stamp(),'files':{str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in paths},'label_requests':0,'mask_requests':3,'network_body_bytes':ledger['body_bytes'],'protocol_sha256':sha((ROOT/'PROTOCOL.md').read_bytes())}
    write(ROOT/'MASK_STAGE_COMPLETE.json',receipt)
    print(json.dumps([{k:v for k,v in s.items() if k not in ['rows','prospective_ct_chunks_zyx']} for s in sites],indent=2))

def labels():
    verify_freeze();frozen=load(ROOT/'MASK_STAGE_COMPLETE.json')
    for p,h in frozen['files'].items():assert sha((ROOT/p).read_bytes())==h
    ledger=load(ROOT/'REQUEST_LEDGER.partial.json');assert len(ledger['requests'])==3 and all(r['role']=='supervision' for r in ledger['requests'])
    plan=load(ROOT/'SOURCE_PLAN.json');coverage={s['winding']:s for s in load(ROOT/'MASK_COVERAGE.json')['sites']};results=[];decodes=[]
    for item in sorted((r for r in plan['arrays'] if r['role']=='inklabels'),key=lambda r:r['winding']):
        store=fetch(item,ledger);labels,checks=decode_crosscheck(store,metadata(item));checks.update({'winding':item['winding'],'role':'inklabels'});decodes.append(checks)
        mi=next(r for r in plan['arrays'] if r['winding']==item['winding'] and r['role']=='supervision');mask,_=decode_crosscheck(ROOT/'arrays'/item['winding']/'supervision',metadata(mi));valid=mask!=0;positive=labels!=0
        cov=coverage[item['winding']];rows=[]
        for r in cov['rows']:
            if not r['eligible']:continue
            y,x=r['y'],r['x'];m=valid[y:y+128,x:x+128];p=positive[y:y+128,x:x+128]
            row=dict(r);row.update({'ink_supervised':int(np.count_nonzero(p&m)),'nonink_supervised':int(np.count_nonzero(~p&m)),'ink_outside_supervision':int(np.count_nonzero(p&~m))});rows.append(row)
        csv_rows(ROOT/f"{item['winding']}-eligible-core-class-counts.csv",rows)
        results.append({'winding':item['winding'],'label_histogram':histogram(labels),'mask_histogram':histogram(mask),'total_ink_pixels':int(np.count_nonzero(positive)),'supervised_pixels':int(np.count_nonzero(valid)),'ink_supervised':int(np.count_nonzero(positive&valid)),'nonink_supervised':int(np.count_nonzero(~positive&valid)),'ink_outside_supervision':int(np.count_nonzero(positive&~valid)),'eligible_cores':len(rows),'eligible_output_area':cov['eligible_output_area'],'eligible_supervised_pixels':sum(r['supervised_pixels'] for r in rows),'eligible_ink_supervised':sum(r['ink_supervised'] for r in rows),'eligible_nonink_supervised':sum(r['nonink_supervised'] for r in rows),'eligible_cores_with_both_classes':sum(r['ink_supervised']>0 and r['nonink_supervised']>0 for r in rows),'eligible_cores_without_ink':sum(r['ink_supervised']==0 for r in rows),'eligible_cores_without_nonink':sum(r['nonink_supervised']==0 for r in rows),'eligible_ink_outside_supervision':sum(r['ink_outside_supervision'] for r in rows),'prospective_ct_unique_chunks':cov['prospective_ct_unique_chunks'],'prospective_ct_bytes':cov['prospective_ct_bytes'],'rows':rows})
    write(ROOT/'LABEL_DECODE_CHECKS.json',decodes)
    write(ROOT/'RESULTS.json',{'created_utc':stamp(),'scope':'Current native annotation coverage/agreement endpoint readiness, not independent physical ink truth','mask_stage_sha256':sha((ROOT/'MASK_STAGE_COMPLETE.json').read_bytes()),'sites':results,'network_body_bytes':ledger['body_bytes'],'no_ct_model_training':True})
    ledger['complete']=True;ledger['completed_utc']=stamp();write(ROOT/'REQUEST_LEDGER.json',ledger)
    print(json.dumps([{k:v for k,v in s.items() if k!='rows'} for s in results],indent=2))

def synthetic():
    # Build tiny official Zarr arrays with the exact approved codec contract.
    checks=[]
    with tempfile.TemporaryDirectory(prefix='native-reference-synthetic-',dir='/workspace/scratch') as td:
        for name,sparse in [('asymmetric_edges',False),('missing_chunks',True)]:
            path=pathlib.Path(td)/name
            a=zarr.create_array(store=str(path),shape=(530,650),dtype='u1',chunks=(512,512),shards=(1024,1024),compressors=[zarr.codecs.BloscCodec(typesize=1,cname='zstd',clevel=3,shuffle='bitshuffle',blocksize=0)],fill_value=0)
            if sparse:
                expected=np.zeros((530,650),dtype='u1');expected[5:23,8:41]=np.arange(18*33,dtype='u2').reshape(18,33)%256
            else:
                yy,xx=np.indices((530,650));expected=((3*yy+7*xx)%256).astype('u1')
            a[:]=expected;meta=load(path/'zarr.json');body=(path/'c'/'0'/'0').read_bytes()
            decoded,info=decode_crosscheck(path,meta);assert np.array_equal(decoded,expected)
            checks.append({'test':name,'passed':True,'missing_chunks':info['missing_chunks'],'present_chunks':info['present_chunks']})
            if sparse:assert info['missing_chunks']>0
            if not sparse:
                bad=bytearray(body);bad[-1]^=1
                try:parse_index(bytes(bad),meta)
                except ValueError:checks.append({'test':'reject_bad_crc','passed':True})
                else:raise AssertionError('CRC accepted')
                idx_size=info['index_bytes'];raw=bytearray(body[-idx_size:-4]);struct.pack_into('<Q',raw,0,len(body))
                bad=body[:-idx_size]+bytes(raw)+struct.pack('<I',google_crc32c.value(bytes(raw)))
                try:parse_index(bad,meta)
                except ValueError:checks.append({'test':'reject_out_of_bounds_index','passed':True})
                else:raise AssertionError('bounds accepted')
                raw=bytearray(body[-idx_size:-4]);raw[16:32]=raw[:16]
                bad=body[:-idx_size]+bytes(raw)+struct.pack('<I',google_crc32c.value(bytes(raw)))
                try:parse_index(bad,meta)
                except ValueError:checks.append({'test':'reject_overlap_index','passed':True})
                else:raise AssertionError('overlap accepted')
                raw=bytearray(body[-idx_size:-4]);struct.pack_into('<Q',raw,0,SENTINEL)
                bad=body[:-idx_size]+bytes(raw)+struct.pack('<I',google_crc32c.value(bytes(raw)))
                try:parse_index(bad,meta)
                except ValueError:checks.append({'test':'reject_partial_missing_sentinel','passed':True})
                else:raise AssertionError('partial missing accepted')
    write(ROOT/'SYNTHETIC_CHECKS.json',{'created_utc':stamp(),'checks':checks,'all_passed':True,'no_real_pixels_read':True})
    print(json.dumps(checks,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['synthetic','freeze','masks','labels']);args=parser.parse_args()
    {'synthetic':synthetic,'freeze':freeze,'masks':masks,'labels':labels}[args.stage]()
