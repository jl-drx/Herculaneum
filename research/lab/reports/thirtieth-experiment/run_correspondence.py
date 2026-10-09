"""Frozen bounded live CT acquisition followed by the fixed correspondence comparison."""
from datetime import datetime,timezone
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import resource
import signal
import ssl
import sys
import time
import urllib.error
import urllib.request
import numpy as np

HERE=Path(__file__).resolve().parent
REPORTS=HERE.parent
sys.path.insert(0,str(REPORTS/'twentyeighth-experiment'))
sys.path.insert(0,str(REPORTS/'twentyseventh-experiment'))
from chunk_accumulator import build_plan,execute
from local_zarr_reader import LocalReader
from compare_mapping import compare


def require(ok,why):
    if not ok:raise ValueError(why)


def pin(path):
    with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    return dict(path=str(path.resolve()),size_bytes=path.stat().st_size,sha256=digest)


def save(path,value):
    tmp=path.with_suffix(path.suffix+'.writing')
    with tmp.open('w') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    tmp.replace(path)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):return None


class Acquisition:
    def __init__(self,run,plan):
        self.run,self.plan=run,plan
        self.start=time.monotonic()
        self.ledger=dict(requests=[],reserved_body_bytes=0,returned_body_bytes=0,
                         retries=0,redirects=0,verified_TLS=True)
        self.opener=urllib.request.build_opener(NoRedirect(),urllib.request.HTTPSHandler(context=ssl.create_default_context()))

    def resources(self):
        limit=self.plan['limits']
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=limit['max_rss_bytes'],'Process RSS exceeded')
        require(sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())<=limit['max_phase_bytes'],'Phase disk exceeded')
        require(time.monotonic()-self.start<=limit['total_seconds'],'Whole acquisition deadline exceeded')

    def get(self,url,dest,*,cap,etag=None,span=None,total=None,expected_body=None):
        require(url.startswith(self.plan['native_base_url']) or any(url==t['object']['url'] for t in self.plan['layers']),'URL outside frozen sources')
        require(not dest.exists() and not dest.with_suffix(dest.suffix+'.partial').exists(),'Fresh object destination required')
        self.resources()
        limit=self.plan['limits']
        require(len(self.ledger['requests'])<limit['max_requests'],'Request count exceeded')
        require(self.ledger['reserved_body_bytes']+cap+1<=limit['max_reserved_bytes'],'Full reservation exceeds budget')
        headers={'Accept-Encoding':'identity','User-Agent':'Herculaneum-Fixed-CT-Correspondence'}
        if etag is not None:headers['If-Match']=etag
        if span is not None:headers['Range']=f'bytes={span[0]}-{span[1]}'
        entry=dict(url=url,request_headers=headers,reserved_body_bytes=cap+1,returned_body_bytes=0,status=None,success=False)
        self.ledger['requests'].append(entry);self.ledger['reserved_body_bytes']+=cap+1
        save(self.run/'REQUEST_LEDGER.json',self.ledger)
        response=None;started=time.monotonic();partial=dest.with_suffix(dest.suffix+'.partial')
        dest.parent.mkdir(parents=True,exist_ok=True)
        def alarm(*_):raise TimeoutError('Absolute request/remaining total deadline')
        old=signal.signal(signal.SIGALRM,alarm)
        signal.setitimer(signal.ITIMER_REAL,min(limit['request_seconds'],limit['total_seconds']-(time.monotonic()-self.start)))
        try:
            try:response=self.opener.open(urllib.request.Request(url,headers=headers),timeout=15)
            except urllib.error.HTTPError as e:response=e
            entry['status']=response.status
            require(sum(len(k)+len(v)+4 for k,v in response.headers.raw_items())<=16384,'Response header bound')
            keys=('Content-Length','Content-Range','Content-Type','Content-Encoding','Transfer-Encoding','ETag','Last-Modified')
            require(all(len(response.headers.get_all(k,[]))<=1 for k in keys),'Duplicate singleton response header')
            entry['response_headers']={k:response.headers.get(k) for k in keys}
            require(response.status==(206 if span is not None else 200),'Wrong HTTP status; no fallback body read')
            require(response.headers.get('Content-Encoding','identity') in ('','identity') and response.headers.get('Transfer-Encoding') is None,'Encoded/chunked body unsupported')
            observed_etag=response.headers.get('ETag','')
            require(re.fullmatch(r'"[\x21\x23-\x7e]+"',observed_etag) is not None,'Strong ETag required')
            if etag is not None:require(observed_etag==etag,'Pinned ETag changed')
            length=response.headers.get('Content-Length','')
            require(length.isdigit() and 0<int(length)<=cap,'Exact bounded Content-Length required')
            length=int(length)
            if span is not None:
                require(length==span[1]-span[0]+1 and response.headers.get('Content-Range')==f'bytes {span[0]}-{span[1]}/{total}','Exact requested byte range required')
            else:require(response.headers.get('Content-Range') is None,'Unexpected partial object')
            with partial.open('xb') as stream:
                while entry['returned_body_bytes']<length:
                    try:block=response.read(min(65536,length-entry['returned_body_bytes']))
                    except http.client.IncompleteRead as e:
                        block=e.partial;stream.write(block);entry['returned_body_bytes']+=len(block)
                        raise ValueError('Truncated body; partial retained') from None
                    require(bool(block),'Truncated response')
                    stream.write(block);entry['returned_body_bytes']+=len(block)
                extra=response.read(1)
                if extra:
                    stream.write(extra);entry['returned_body_bytes']+=len(extra)
                    raise ValueError('Excess response byte')
                stream.flush();os.fsync(stream.fileno())
            actual=pin(partial)
            if expected_body is not None:
                require(actual['sha256']==expected_body['sha256'] and actual['size_bytes']==expected_body['size_bytes'],'Pinned metadata body changed')
            partial.replace(dest);entry['file']=pin(dest);entry['success']=True
        except Exception as e:
            entry['error']=type(e).__name__+': '+str(e)
            if partial.exists():entry['partial_file']=pin(partial)
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
            if response is not None:response.close()
            entry['elapsed_seconds']=time.monotonic()-started
            self.ledger['returned_body_bytes']+=entry['returned_body_bytes']
            save(self.run/'REQUEST_LEDGER.json',self.ledger)
        self.resources()
        if len(self.ledger['requests'])%10==0:print(json.dumps(dict(requests_completed=len(self.ledger['requests']),returned_body_bytes=self.ledger['returned_body_bytes'])),flush=True)
        return entry


def main():
    freeze=json.loads((HERE/'SOURCE_FREEZE.json').read_text())
    for item in freeze['files']:require(pin(Path(item['path']))==item,'Frozen source/input changed')
    plan=json.loads((HERE/'SAMPLING_PLAN.json').read_text())
    run=HERE/'run-v1';run.mkdir(exist_ok=False)
    source=run/'source';source.mkdir()
    acq=Acquisition(run,plan);started=time.monotonic()
    receipt=dict(status='RUNNING',started_utc=datetime.now(timezone.utc).isoformat(),
                 source_freeze=pin(HERE/'SOURCE_FREEZE.json'),sampling_plan=pin(HERE/'SAMPLING_PLAN.json'),
                 scientific_model_forwards=0,ink_label_pixels_read=0,optical_pixels_read=0,GPU_spend_GBP=0)
    save(run/'RUN_RECEIPT.json',receipt)
    try:
        manifest=dict(metadata={},objects={})
        for item in plan['metadata']:
            dest=source/item['name']
            acq.get(item['url'],dest,cap=65536,etag=item['etag'],expected_body=item['body'])
            manifest['metadata'][item['name']]={k:pin(dest)[k] for k in ('size_bytes','sha256')}
        reader=LocalReader(source,manifest)
        points=np.load(HERE/'POINTS.npy',allow_pickle=False)
        sampling=build_plan(points,reader.spec,max_total_chunks=32,max_total_bytes=128*1024**2)
        require(sampling.summary()==plan['native_plan'],'Changed sampling support')
        for key in plan['native_plan']['keys_zyx']:
            name='0/'+'/'.join(map(str,key));dest=source/name
            acq.get(plan['native_base_url']+name,dest,cap=4194304+16)
            manifest['objects'][name]={k:pin(dest)[k] for k in ('size_bytes','sha256')}
        save(run/'SOURCE_MANIFEST.json',manifest)
        reader=LocalReader(source,manifest)
        sampled=execute(sampling,reader)
        require(sampled['valid'].all(),'Incomplete native sample support')
        np.save(run/'NATIVE_VALUES.npy',sampled['values'],allow_pickle=False)
        receipt['native_sampling']=sampled['stats'];receipt['native_reader']=reader.stats
        receipt['decoded_native_voxels']=len(manifest['objects'])*128**3
        blocks=sampled['values'].reshape(6,7,3,9)
        candidates=np.stack([blocks[i,:,::sign,:] for i in range(3) for sign in (1,-1)])
        controls=np.stack([blocks[i+3,:,::sign,:] for i in range(3) for sign in (1,-1)])
        # Positive sign follows [-32,0,+32]; negative sign reverses that layer axis.
        observed=[]
        for layer in plan['layers']:
            rows=[];obj=layer['object']
            for i,span in enumerate(layer['ranges']):
                dest=run/'tiff-ranges'/f"{layer['layer']:02d}-{i:02d}.bin"
                acq.get(obj['url'],dest,cap=6,etag=obj['etag'],span=span,total=obj['total_bytes'])
                rows.append(np.frombuffer(dest.read_bytes(),dtype='<u2'))
            observed.append(np.concatenate(rows).reshape(7,9))
        observed=np.stack(observed,axis=1)
        np.save(run/'PUBLISHED_VALUES.npy',observed,allow_pickle=False)
        decision=compare(candidates,controls,observed,plan)
        save(run/'CORRESPONDENCE.json',decision)
        receipt.update(status='ACQUISITION_AND_COMPARISON_COMPLETE',scientific_mapping_status=decision['status'],
                       selected_by_calibration=decision['selected_by_calibration'],published_CT_pixels_read=int(observed.size),
                       source_manifest=pin(run/'SOURCE_MANIFEST.json'),correspondence=pin(run/'CORRESPONDENCE.json'),
                       native_values=pin(run/'NATIVE_VALUES.npy'),published_values=pin(run/'PUBLISHED_VALUES.npy'))
    except Exception as e:
        receipt.update(status='STOP',error=type(e).__name__+': '+str(e))
        raise
    finally:
        receipt.update(elapsed_seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                       request_count=len(acq.ledger['requests']),reserved_body_bytes=acq.ledger['reserved_body_bytes'],
                       returned_body_bytes=acq.ledger['returned_body_bytes'],ledger=pin(run/'REQUEST_LEDGER.json'))
        save(run/'RUN_RECEIPT.json',receipt)
        print(json.dumps(receipt,indent=2,sort_keys=True),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
