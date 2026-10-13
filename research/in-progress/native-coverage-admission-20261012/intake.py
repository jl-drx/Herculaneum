"""Frozen population requests and resumable bounded public-data acquisition."""
import argparse
import collections
import concurrent.futures
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil
import threading
import time
import urllib.error
import urllib.request


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.partial')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def prepare(a):
    a.work.mkdir(parents=True, exist_ok=False)
    base = a.base
    candidates = list(csv.DictReader((base/'support-reservation-v2/CANDIDATES.csv').open()))
    result = json.loads((base/'support-reservation-v2/RESULTS.json').read_text())
    assert len(candidates) == result['broad_candidates'] == 11887
    assert sha(base/'support-reservation-v2/CANDIDATES.csv') == result['csv_sha256']
    cohort = json.loads((base/'COHORT_CANDIDATES.json').read_text())
    requests = []; cases = []; pins = {}
    for source in cohort['native_v2_audit']:
        if source['role'] != 'additional_regional_coverage_candidate':
            continue
        segment = source['segment']; ident = source['sample'].lower()+'-'+segment.split('-')[0]
        rows = [r for r in candidates if r['segment'] == segment]
        metadata = base/'metadata'/f'{ident}-array.raw'
        hp = base/'teacher-metadata'/f'{ident}-HEADER.json'
        m = json.loads(metadata.read_text()); h = json.loads(hp.read_text())
        for f in [metadata, hp]: pins[str(f.relative_to(base))] = sha(f)
        depth, height, width = m['shape']; fh, fw = h['shape_yx']
        assert depth == 28 and m['chunks'] == [28,128,128] and m['compressor'] is None and m['dtype'] == '|u1'
        assert m['order'] == 'C' and m['dimension_separator'] == '/' and m['filters'] is None
        chunks = set(); tiles = set(); columns = (fw+1023)//1024
        for r in rows:
            x,y = int(r['x']),int(r['y'])
            for cy in range(y//128, (y+127)//128+1):
                for cx in range(x//128, (x+127)//128+1): chunks.add((cy,cx))
            lo=max(0,math.floor(x*fw/width)-2);hi=min(fw,math.ceil((x+128)*fw/width)+2)
            top=max(0,math.floor(y*fh/height)-2);bottom=min(fh,math.ceil((y+128)*fh/height)+2)
            for ty in range(top//1024,(bottom-1)//1024+1):
                for tx in range(lo//1024,(hi-1)//1024+1): tiles.add(ty*columns+tx)
        # Saved centres are controls, not replacements or extra cohort members.
        prior = next(c for c in json.loads((base/'SOURCE_CROP_GO.json').read_text())['cases'] if c['id']==ident)
        smoke_tiles = {t['tile_index'] for t in prior.get('tiles', [])}
        tiles |= smoke_tiles
        for cy,cx in sorted(chunks):
            key=f'{ident}-ct-y{cy:04}-x{cx:04}'
            url='https://vesuvius-challenge-open-data.s3.amazonaws.com/'+source['native']['path'].rstrip('/')+f'/0/0/{cy}/{cx}'
            requests.append(dict(id=key,source_id=ident,kind='ct',url=url,bytes=28*128*128,
                                 cy=cy,cx=cx,path=f'objects/{ident}/ct/{cy}/{cx}.bin'))
        for index in sorted(tiles):
            size=h['tags']['325'][index];offset=h['tags']['324'][index]
            requests.append(dict(id=f'{ident}-teacher-{index:05}',source_id=ident,kind='teacher',
                url=h['url'],bytes=size,offset=offset,object_bytes=h['object_bytes'],etag=h['etag'],
                tile_index=index,ty=index//columns,tx=index%columns,path=f'objects/{ident}/teacher/{index}.bin'))
        cases.append(dict(id=ident,segment=segment,native_shape=m['shape'],teacher_shape=h['shape_yx'],
                          candidates=len(rows),ct_chunks=len(chunks),teacher_tiles=len(tiles),
                          restricted=sum(int(r['in_restricted_A']) for r in rows)))
    for relative in ['COHORT_CANDIDATES.json','support-reservation-v2/CANDIDATES.csv','support-reservation-v2/RULE.json']:
        pins[relative]=sha(base/relative)
    retained=json.loads((base/'SOURCE_CROP_ACQUISITION.json').read_text())['requests']
    prior={(r['url'],r.get('offset',0),r['bytes']):r for r in retained}
    for r in requests:
        old=prior.get((r['url'],r.get('offset',0),r['bytes']))
        if old:
            r['reuse']=dict(path=old['path'],sha256=old['sha256'])
            r['etag']=old['etag']
    payload=sum(r['bytes'] for r in requests)
    assert payload <= int(6.3*1024**3)
    plan=dict(status='FROZEN_BEFORE_POPULATION_PAYLOAD',utc=stamp(),cases=cases,requests=requests,
              base_pins=pins,protocol_sha256=sha(Path(__file__).with_name('PROTOCOL.md')),
              implementation_sha256=sha(Path(__file__)),planned_body_bytes=payload,
              minimum_supervised_pixels=1024,maximum_concurrent_reads=16,
              maximum_body_bytes_with_retries=8*1024**3,model_forwards=0,paid_compute=False)
    write(a.work/'REQUEST_PLAN.json',plan)
    print(json.dumps(dict(cases=cases,requests=len(requests),planned_bytes=payload)),flush=True)


def acquire(a):
    plan=json.loads((a.work/'REQUEST_PLAN.json').read_text())
    assert sha(Path(__file__))==plan['implementation_sha256']
    assert sha(Path(__file__).with_name('PROTOCOL.md'))==plan['protocol_sha256']
    for rel,digest in plan['base_pins'].items(): assert sha(a.base/rel)==digest,rel
    (a.work/'receipts').mkdir(exist_ok=True)
    lock=threading.Lock(); state=dict(completed=0,missing=0,bytes=0,reused=0)
    existing_receipts=[json.loads(p.read_text()) for p in (a.work/'receipts').glob('*.json')]
    prior_bytes=sum(sum(v.get('body_bytes',0) for v in r['attempts']) for r in existing_receipts)
    failed=threading.Event()
    started=time.monotonic()
    def fetch(r):
        if failed.is_set():raise RuntimeError('Cancelled following acquisition failure')
        dest=a.work/r['path']; rp=a.work/'receipts'/(r['id']+'.json')
        if rp.exists():
            saved=json.loads(rp.read_text());assert saved['request']==r
            if saved['status']=='PRESENT':assert dest.stat().st_size==r['bytes'] and sha(dest)==saved['sha256']
            return saved,True
        dest.parent.mkdir(parents=True,exist_ok=True)
        if 'reuse' in r:
            source=a.base/r['reuse']['path']
            assert source.stat().st_size==r['bytes'] and sha(source)==r['reuse']['sha256']
            shutil.copyfile(source,dest)
            saved=dict(status='PRESENT',request=r,sha256=sha(dest),etag=r['etag'],bytes=r['bytes'],
                       attempts=[],reused_prior_source=True,utc=stamp())
            write(rp,saved);return saved,True
        headers={'Accept-Encoding':'identity'}
        if 'etag' in r: headers['If-Match']=r['etag']
        if r['kind']=='teacher':headers['Range']=f"bytes={r['offset']}-{r['offset']+r['bytes']-1}"
        attempts=[];payload=None;etag=None
        for attempt in range(1,4):
            got=0
            try:
                with urllib.request.urlopen(urllib.request.Request(r['url'],headers=headers),timeout=45) as response:
                    assert response.status==(206 if r['kind']=='teacher' else 200)
                    etag=response.headers['ETag'];assert etag
                    if 'etag' in r: assert etag==r['etag'],'Source changed'
                    if r['kind']=='teacher':assert response.headers['Content-Range']==f"bytes {r['offset']}-{r['offset']+r['bytes']-1}/{r['object_bytes']}"
                    assert int(response.headers['Content-Length'])==r['bytes']
                    payload=response.read(r['bytes']+1);got=len(payload)
                    with lock:
                        state['bytes']+=got
                        assert prior_bytes+state['bytes']<=plan['maximum_body_bytes_with_retries']
                    assert got==r['bytes']
                attempts.append(dict(attempt=attempt,status='SUCCESS',body_bytes=got));break
            except urllib.error.HTTPError as error:
                attempts.append(dict(attempt=attempt,http_status=error.code,body_bytes=got))
                if error.code==404:
                    saved=dict(status='MISSING',request=r,attempts=attempts,utc=stamp())
                    write(rp,saved);return saved,False
                if error.code not in [429,500,502,503,504] or attempt==3:raise
            except (TimeoutError,urllib.error.URLError) as error:
                attempts.append(dict(attempt=attempt,error=type(error).__name__,body_bytes=got))
                if attempt==3:raise
            time.sleep(2**(attempt-1))
        assert payload is not None
        tmp=dest.with_suffix('.partial');tmp.write_bytes(payload);tmp.replace(dest)
        saved=dict(status='PRESENT',request=r,sha256=hashlib.sha256(payload).hexdigest(),etag=etag,
                   bytes=len(payload),attempts=attempts,utc=stamp())
        write(rp,saved);return saved,False
    # Metadata-defined segments stay together to support interpretable progress.
    futures=[];rows=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        for r in plan['requests']:futures.append(pool.submit(fetch,r))
        try:
            for future in concurrent.futures.as_completed(futures):
                row,reused=future.result();rows.append(row);state['completed']+=1
                state['missing']+=row['status']=='MISSING';state['reused']+=reused
                if state['completed']%128==0 or state['completed']==len(futures):
                    progress=dict(state,requests=len(futures),seconds=time.monotonic()-started,utc=stamp())
                    write(a.work/'ACQUISITION_STATUS.json',progress);print(json.dumps(progress),flush=True)
        except Exception as error:
            failed.set()
            for future in futures:future.cancel()
            write(a.work/'ACQUISITION_FAILURE.json',dict(utc=stamp(),error_type=type(error).__name__,
                  reason=str(error),state=state))
            raise
    rows.sort(key=lambda r:r['request']['id'])
    result=dict(status='ALL_REQUESTS_ACCOUNTED',request_plan_sha256=sha(a.work/'REQUEST_PLAN.json'),
                objects=len(rows),present=sum(r['status']=='PRESENT' for r in rows),
                missing=sum(r['status']=='MISSING' for r in rows),new_body_bytes=state['bytes'],
                total_retained_payload_bytes=sum(r.get('bytes',0) for r in rows),
                retried_objects=sum(len(r['attempts'])>1 for r in rows),
                receipts_sha256={r['request']['id']:sha(a.work/'receipts'/(r['request']['id']+'.json')) for r in rows},
                seconds=time.monotonic()-started,utc=stamp(),paid_compute=False)
    write(a.work/'ACQUISITION_COMPLETE.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='receipts_sha256'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','acquire'])
    p.add_argument('--base',type=Path,required=True);p.add_argument('--work',type=Path,required=True)
    a=p.parse_args();(prepare if a.mode=='prepare' else acquire)(a)
