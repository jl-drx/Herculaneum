"""Account for unaffected frozen inputs while preserving unresolved transport errors.

No request list, scientific eligibility, source validator or existing receipt is
changed. New errors are unknown input availability, never zero-filled negatives.
"""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shutil
import threading
import time
import urllib.error
import urllib.request
from intake import sha, write, stamp


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True)
    p.add_argument('--work',type=Path,required=True);a=p.parse_args()
    plan=json.loads((a.work/'REQUEST_PLAN.json').read_text())
    freeze=a.work/'AVAILABILITY_PASS_PLAN.json'
    if freeze.exists():raise FileExistsError('Pass already declared')
    errors=a.work/'transport-errors';errors.mkdir()
    (a.work/'receipts').mkdir(exist_ok=True)
    known=set()
    diagnosis=a.work/'TRANSPORT_DIAGNOSIS.json'
    if diagnosis.exists():
        d=json.loads(diagnosis.read_text())
        if d.get('status')==503:
            known.add(d['id']);write(errors/(d['id']+'.json'),dict(status='TRANSPORT_UNRESOLVED',diagnosis=d,not_retried=True))
    write(freeze,dict(utc=stamp(),request_plan_sha256=sha(a.work/'REQUEST_PLAN.json'),
        implementation_sha256=sha(Path(__file__)),workers=4,attempts_per_untried_request=1,
        purpose='Collect unaffected frozen requests after proxy-tunnel failures; retain each inaccessible input as unresolved. This phase cannot certify a complete population if any input is unresolved.',
        skip_known_failed=sorted(known),stop_after_consecutive_transport_errors=64,
        source_coordinates_validators_and_eligibility_unchanged=True,paid_compute=False))
    state=dict(completed=0,new_present=0,reused=0,missing=0,transport_unresolved=0,new_body_bytes=0)
    lock=threading.Lock();stop=threading.Event();consecutive=0;started=time.monotonic()
    def fetch(r):
        rp=a.work/'receipts'/(r['id']+'.json');dest=a.work/r['path']
        if rp.exists():
            saved=json.loads(rp.read_text());assert saved['request']==r
            if saved['status']=='PRESENT':assert dest.stat().st_size==r['bytes'] and sha(dest)==saved['sha256']
            return 'reused',0
        if r['id'] in known:return 'transport_unresolved',0
        if 'reuse' in r:
            source=a.base/r['reuse']['path'];assert sha(source)==r['reuse']['sha256'] and source.stat().st_size==r['bytes']
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
            write(rp,dict(status='PRESENT',request=r,sha256=sha(dest),etag=r['etag'],bytes=r['bytes'],attempts=[],reused_prior_source=True,utc=stamp()))
            return 'reused',0
        if stop.is_set():return 'unattempted',0
        headers={'Accept-Encoding':'identity'}
        if 'etag' in r:headers['If-Match']=r['etag']
        if r['kind']=='teacher':headers['Range']=f"bytes={r['offset']}-{r['offset']+r['bytes']-1}"
        try:
            with urllib.request.urlopen(urllib.request.Request(r['url'],headers=headers),timeout=30) as response:
                assert response.status==(206 if r['kind']=='teacher' else 200)
                etag=response.headers['ETag'];assert etag
                if 'etag' in r:assert etag==r['etag'],'Source identity changed'
                if r['kind']=='teacher':assert response.headers['Content-Range']==f"bytes {r['offset']}-{r['offset']+r['bytes']-1}/{r['object_bytes']}"
                assert int(response.headers['Content-Length'])==r['bytes']
                data=response.read(r['bytes']+1);assert len(data)==r['bytes']
        except urllib.error.HTTPError as error:
            body=error.read(4096).decode(errors='replace')
            if error.code==404 and ('NoSuchKey' in body or 'NoSuchObject' in body):
                write(rp,dict(status='MISSING',request=r,attempts=[dict(attempt=1,http_status=404,body_bytes=0)],
                              provider_error=body,utc=stamp()))
                return 'missing',0
            if error.code==412:raise RuntimeError('Pinned source identity changed') from error
            write(errors/(r['id']+'.json'),dict(status='TRANSPORT_UNRESOLVED',request=r,http_status=error.code,
                body=body,utc=stamp(),zero_filled=False))
            return 'transport_unresolved',0
        except (TimeoutError,urllib.error.URLError) as error:
            write(errors/(r['id']+'.json'),dict(status='TRANSPORT_UNRESOLVED',request=r,
                error_type=type(error).__name__,utc=stamp(),zero_filled=False))
            return 'transport_unresolved',0
        dest.parent.mkdir(parents=True,exist_ok=True);temp=dest.with_suffix('.partial');temp.write_bytes(data);temp.replace(dest)
        write(rp,dict(status='PRESENT',request=r,sha256=hashlib.sha256(data).hexdigest(),etag=etag,
            bytes=len(data),attempts=[dict(attempt=1,status='SUCCESS',body_bytes=len(data))],availability_pass=True,utc=stamp()))
        return 'new_present',len(data)
    # Reuse local files first; this remains useful even during complete outage.
    requests=sorted(plan['requests'],key=lambda r:('reuse' not in r,r['id']))
    futures=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(fetch,r) for r in requests]
        try:
            for f in concurrent.futures.as_completed(futures):
                kind,size=f.result();state[kind]=state.get(kind,0)+1;state['completed']+=1;state['new_body_bytes']+=size
                assert state['new_body_bytes']<=plan['planned_body_bytes']
                if kind=='transport_unresolved':consecutive+=1
                elif kind=='new_present':consecutive=0
                if consecutive>=64:stop.set()
                if state['completed']%128==0 or state['completed']==len(futures):
                    snapshot=dict(state,requests=len(futures),seconds=time.monotonic()-started,utc=stamp())
                    write(a.work/'AVAILABILITY_STATUS.json',snapshot);print(json.dumps(snapshot),flush=True)
        except Exception:
            stop.set()
            for f in futures:f.cancel()
            raise
    receipts=[json.loads(p.read_text()) for p in (a.work/'receipts').glob('*.json')]
    missing=sum(r['status']=='MISSING' for r in receipts);present=sum(r['status']=='PRESENT' for r in receipts)
    unknown=len(plan['requests'])-len(receipts)
    out=dict(status='COMPLETE' if unknown==0 else 'PARTIAL_TRANSPORT_UNRESOLVED',utc=stamp(),present=present,
        confirmed_missing=missing,unresolved_or_unattempted=unknown,phase_counts=state,
        new_body_bytes=state['new_body_bytes'],retained_bytes=sum(r.get('bytes',0) for r in receipts),
        input_receipts_sha256={p.name:sha(p) for p in sorted((a.work/'receipts').glob('*.json'))},
        request_plan_sha256=sha(a.work/'REQUEST_PLAN.json'),no_unknown_input_is_zero_filled=True,
        scientific_population_fully_admitted=False,paid_compute=False)
    write(a.work/'AVAILABILITY_COMPLETE.json',out)
    print(json.dumps({k:v for k,v in out.items() if k!='input_receipts_sha256'}),flush=True)


if __name__=='__main__':main()
