"""Bounded first-read native CT intake and offline support qualification."""
import argparse,concurrent.futures,hashlib,json,os,pathlib,sys,threading,uuid
from datetime import datetime,timezone
import numpy as np,requests
ROOT=pathlib.Path(__file__).resolve().parent
PACKET=ROOT.parent
sys.path.insert(0,str(PACKET));import intake as annotations
CAP=256*1024*1024;CHUNK_BYTES=28*128*128
lock=threading.Lock();halt=threading.Event()

def sha(b):return hashlib.sha256(b).hexdigest()
def load(p):return json.loads(p.read_text())
def stamp():return datetime.now(timezone.utc).isoformat()
def write(p,o):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp-'+uuid.uuid4().hex)
    tmp.write_text(json.dumps(o,indent=2,allow_nan=False)+'\n');os.replace(tmp,p)
def verify_freeze():
    f=load(ROOT/'FREEZE.json')
    for p,h in f['ct_files'].items():assert sha((ROOT/p).read_bytes())==h
    for p,h in f['packet_files'].items():assert sha((PACKET/p).read_bytes())==h

def freeze():
    assert not (ROOT/'FREEZE.json').exists() and not (ROOT/'chunks').exists()
    p=load(ROOT/'REQUEST_PLAN.json');assert p['expected_total_bytes']==245891072 and len(p['requests'])==536
    assert p['max_workers']==4 and p['body_cap_bytes']==CAP
    packet=dict(p['cohort_pins']);packet.update({r['path']:r['sha256'] for r in p['native_metadata']})
    for name,h in packet.items():assert sha((PACKET/name).read_bytes())==h
    for r in p['requests']:assert r['expected_bytes']==CHUNK_BYTES and r['chunk_zyx'][0]==0
    write(ROOT/'FREEZE.json',{'created_utc':stamp(),'ct_files':{f:sha((ROOT/f).read_bytes()) for f in ['PROTOCOL.md','REQUEST_PLAN.json','intake_ct.py']},'packet_files':packet,'source_CT_bytes_before_freeze':0})

def verified(r):
    p=ROOT/r['relative_path'];receipt=ROOT/'receipts'/(r['id']+'.json')
    if not receipt.exists():return False
    x=load(receipt)
    assert x['url']==r['url'] and x['bytes']==CHUNK_BYTES and x['chunk_zyx']==r['chunk_zyx']
    b=p.read_bytes();assert len(b)==CHUNK_BYTES and sha(b)==x['sha256']
    return True

def download_one(r):
    if halt.is_set():return 'cancelled'
    if verified(r):return 'reused'
    with lock:
        if halt.is_set():return 'cancelled'
        b=load(ROOT/'BODY_BUDGET.json')
        if b['reserved_body_bytes']+CHUNK_BYTES>CAP:halt.set();raise RuntimeError('cumulative reserved body cap would be exceeded')
        b['reserved_body_bytes']+=CHUNK_BYTES;b['attempts']+=1;attempt=b['attempts'];write(ROOT/'BODY_BUDGET.json',b)
    p=ROOT/r['relative_path'];p.parent.mkdir(parents=True,exist_ok=True);partial=p.with_name(p.name+'.partial')
    seen=0;headers={};error=None
    try:
        with requests.get(r['url'],headers={'Accept-Encoding':'identity'},stream=True,timeout=(15,60),allow_redirects=False) as response:
            headers={k:response.headers.get(k) for k in ['ETag','Last-Modified','Content-Length','Content-Encoding']}
            assert response.status_code==200,f'HTTP {response.status_code}'
            assert int(response.headers['Content-Length'])==CHUNK_BYTES,'unexpected Content-Length'
            assert response.headers.get('Content-Encoding','identity')=='identity'
            assert response.headers.get('ETag'),'missing first-read ETag'
            h=hashlib.sha256()
            with partial.open('wb') as f:
                for part in response.iter_content(65536):
                    seen+=len(part)
                    if seen>CHUNK_BYTES:raise RuntimeError('body exceeds pinned stored chunk length')
                    f.write(part);h.update(part)
            assert seen==CHUNK_BYTES,'short object'
            a=np.memmap(partial,dtype='u1',mode='r',shape=(28,128,128));assert a.shape==(28,128,128);del a
            receipt={'id':r['id'],'url':r['url'],'winding':r['winding'],'chunk_zyx':r['chunk_zyx'],'bytes':seen,'sha256':h.hexdigest(),'etag':response.headers['ETag'],'last_modified':response.headers.get('Last-Modified'),'retrieved_utc':stamp(),'attempt':attempt,'identity_scope':'current first-read object; not historical entire-volume hash'}
            os.replace(partial,p);write(ROOT/'receipts'/(r['id']+'.json'),receipt)
    except Exception as e:
        halt.set();error=f'{type(e).__name__}: {e}'
        write(ROOT/'errors'/f'attempt-{attempt:04d}.json',{'id':r['id'],'attempt':attempt,'url':r['url'],'observed_body_bytes':seen,'reserved_body_bytes':CHUNK_BYTES,'headers':headers,'error':error,'utc':stamp()})
        raise
    finally:
        with lock:
            b=load(ROOT/'BODY_BUDGET.json');b['recorded_observed_body_bytes']+=seen
            if error is None:b['completed_new_objects']+=1
            else:b['failed_attempts']+=1
            write(ROOT/'BODY_BUDGET.json',b)
            if b['completed_new_objects']%25==0 or error:print(json.dumps({k:b[k] for k in ['completed_new_objects','failed_attempts','reserved_body_bytes','recorded_observed_body_bytes']}),flush=True)
    return 'acquired'

def acquire():
    verify_freeze();assert load(ROOT/'SYNTHETIC_CHECKS.json')['all_passed']
    p=load(ROOT/'REQUEST_PLAN.json')
    if not (ROOT/'BODY_BUDGET.json').exists():write(ROOT/'BODY_BUDGET.json',{'started_utc':stamp(),'cap_bytes':CAP,'reserved_body_bytes':0,'recorded_observed_body_bytes':0,'attempts':0,'completed_new_objects':0,'failed_attempts':0,'accounting':'Every attempt reserves its entire expected body durably before GET; interrupted/failing attempts retain reservation. Observed counter may omit bytes delivered just before process interruption; reserved counter remains conservative.'})
    failed=[];counts={'acquired':0,'reused':0,'cancelled':0}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(download_one,r):r for r in p['requests']}
        for f in concurrent.futures.as_completed(futures):
            try:counts[f.result()]+=1
            except Exception as e:failed.append({'id':futures[f]['id'],'error':str(e)});halt.set()
    write(ROOT/'ACQUISITION_STATUS.json',{'utc':stamp(),'complete':not failed and counts['cancelled']==0,'this_invocation':counts,'errors':failed,'budget':load(ROOT/'BODY_BUDGET.json')})
    if failed or counts['cancelled']:raise RuntimeError(f'acquisition stopped; errors={failed[:3]}, cancelled={counts["cancelled"]}')
    for r in p['requests']:assert verified(r)
    print(json.dumps(load(ROOT/'ACQUISITION_STATUS.json'),indent=2))

def context(reader,x,y):
    out=np.empty((28,256,256),dtype='u1');x0,y0=x-64,y-64
    for cy in range(y0//128,(y0+255)//128+1):
        for cx in range(x0//128,(x0+255)//128+1):
            c=reader(cy,cx);assert c.shape==(28,128,128) and c.dtype==np.uint8
            left,top=max(x0,cx*128),max(y0,cy*128);right,bottom=min(x0+256,(cx+1)*128),min(y0+256,(cy+1)*128)
            out[:,top-y0:bottom-y0,left-x0:right-x0]=c[:,top-cy*128:bottom-cy*128,left-cx*128:right-cx*128]
    return out

def qualify(output=None):
    verify_freeze();p=load(ROOT/'REQUEST_PLAN.json')
    assert load(ROOT/'ACQUISITION_STATUS.json')['complete']
    for r in p['requests']:assert verified(r)
    cohort=load(PACKET/'MASK_COVERAGE.json');annplan=load(PACKET/'SOURCE_PLAN.json');sites=[]
    out=output or ROOT/'support';out.mkdir(parents=True,exist_ok=True)
    for site in cohort['sites']:
        w=site['winding'];cache={}
        for r in p['requests']:
            if r['winding']==w:cache[tuple(r['chunk_zyx'][1:])]=np.frombuffer((ROOT/r['relative_path']).read_bytes(),dtype='u1').reshape(28,128,128)
        mitem=next(r for r in annplan['arrays'] if r['winding']==w and r['role']=='supervision');iitem=next(r for r in annplan['arrays'] if r['winding']==w and r['role']=='inklabels')
        mask,_=annotations.decode_crosscheck(PACKET/'arrays'/w/'supervision',annotations.metadata(mitem));ink,_=annotations.decode_crosscheck(PACKET/'arrays'/w/'inklabels',annotations.metadata(iitem))
        rows=[]
        for r in site['rows']:
            if not r['eligible']:continue
            x,y=r['x'],r['y'];v=context(lambda cy,cx:cache[(cy,cx)],x,y).max(axis=0)>0;core=v[64:192,64:192]
            m=mask[y:y+128,x:x+128]>0;i=ink[y:y+128,x:x+128]>0
            record={'core_id':r['core_id'],'x':x,'y':y,'supervised_pixels':int(m.sum()),'native_valid_core_pixels':int(core.sum()),'native_valid_context_pixels':int(v.sum()),'supervised_native_valid_pixels':int((m&core).sum()),'supervised_zero_ct_pixels':int((m&~core).sum()),'ink_supervised_native_valid':int((m&i&core).sum()),'nonink_supervised_native_valid':int((m&~i&core).sum()),'ink_supervised_zero_ct':int((m&i&~core).sum()),'nonink_supervised_zero_ct':int((m&~i&~core).sum()),'core_valid_sha256':sha(core.astype('u1').tobytes()),'context_valid_sha256':sha(v.astype('u1').tobytes()),'geometry_status':'not determined by CT nonzero check; separate geometry audit'}
            q=out/'masks'/(r['core_id']+'.npz');q.parent.mkdir(exist_ok=True,parents=True);np.savez_compressed(q,core_valid=core,context_valid=v,core_origin_xy=np.array([x,y],dtype='<i4'),context_origin_xy=np.array([x-64,y-64],dtype='<i4'))
            rows.append(record)
        summary={'winding':w,'cores':len(rows),'native_valid_core_pixels':sum(r['native_valid_core_pixels'] for r in rows),'supervised_native_valid_pixels':sum(r['supervised_native_valid_pixels'] for r in rows),'supervised_zero_ct_pixels':sum(r['supervised_zero_ct_pixels'] for r in rows),'ink_supervised_zero_ct':sum(r['ink_supervised_zero_ct'] for r in rows),'nonink_supervised_zero_ct':sum(r['nonink_supervised_zero_ct'] for r in rows),'cores_with_any_zero_ct_context':sum(r['native_valid_context_pixels']<256**2 for r in rows),'cores_with_any_zero_ct_core':sum(r['native_valid_core_pixels']<128**2 for r in rows),'rows':rows};sites.append(summary)
        annotations.csv_rows(out/f'{w}-core-support.csv',rows)
    result={'scope':'array/nonzero native CT eligibility, all28planes; not geometry qualification or ink truth','original_cohort_unchanged':True,'sites':sites,'source_plan_sha256':sha((ROOT/'REQUEST_PLAN.json').read_bytes())}
    write(out/'RESULTS.json',result)
    print(json.dumps([{k:v for k,v in s.items() if k!='rows'} for s in sites],indent=2))
    return result

def synthetic():
    yy,xx=np.indices((512,512));whole=np.zeros((28,512,512),dtype='u1');whole[0]=((xx+3*yy)%255).astype('u1');whole[27,200,210]=255
    chunks={(y,x):whole[:,y*128:(y+1)*128,x*128:(x+1)*128] for y in range(4) for x in range(4)}
    got=context(lambda y,x:chunks[(y,x)],128,128);assert np.array_equal(got,whole[:,64:320,64:320]);assert got[27,136,146]==255
    whole[:]=0;whole[27,200,210]=255;v=context(lambda y,x:chunks[(y,x)],128,128).max(0)>0
    assert int(v.sum())==1 and v[136,146]
    write(ROOT/'SYNTHETIC_CHECKS.json',{'utc':stamp(),'all_passed':True,'checks':['asymmetric chunk assembly with nonzero64pixelorigin','all28planes validity includes finalplane27','allzero source remains invalid except knownfinalplanevoxel'],'no_source_CT_read':True})

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['synthetic','freeze','acquire','qualify']);a.add_argument('--output',type=pathlib.Path);args=a.parse_args()
    if args.stage=='qualify':qualify(args.output)
    else:{'synthetic':synthetic,'freeze':freeze,'acquire':acquire}[args.stage]()
