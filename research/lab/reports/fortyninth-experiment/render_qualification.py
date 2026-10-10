"""Run the pinned renderer with a finite, audited source whitelist; no fill accepted."""
from pathlib import Path
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
import hashlib,json,os,re,subprocess,threading,time,requests
from prepare import pin,save
P=Path(__file__).resolve().parent;W=P/'work';plan=json.loads((P/'GO.json').read_text());cache=W/'source-cache';cache.mkdir(exist_ok=True)
assert pin(P/'PROTOCOL.md')==plan['protocol']
for r in plan['inputs']:assert pin(r['path'])==r,r['path']
assert not (P/'RENDER_EXECUTION.json').exists()
ledger=(W/'source-ledger.jsonl').open('x');lock=threading.Lock();fatal=[];metabytes=0;seen={};receipts=[]
def record(event):
    with lock:ledger.write(json.dumps(event,sort_keys=True)+'\n');ledger.flush()
def fetch(key,optional=False):
    global metabytes
    url=plan['source_url']+'/'+key
    cap=128**3 if key in plan['source_chunk_keys'] else 2*1024**2
    for attempt in range(1,5):
        start=time.monotonic()
        try:
            with requests.get(url,timeout=(15,45),stream=True) as r:
                status=r.status_code
                if status==404 and optional:
                    record(dict(key=key,url=url,status=status,optional=True));return None
                if status!=200:
                    record(dict(key=key,url=url,status=status,attempt=attempt))
                    if status in [500,502,503,504] and attempt<4:time.sleep(min(2**attempt,8));continue
                    r.raise_for_status()
                body=bytearray()
                for part in r.iter_content(65536):
                    body.extend(part)
                    if len(body)>cap:raise RuntimeError('Source byte cap exceeded: '+key)
                body=bytes(body)
                rec=dict(key=key,url=url,status=status,bytes=len(body),sha256=hashlib.sha256(body).hexdigest(),seconds=time.monotonic()-start)
                if key in plan['source_chunk_keys']:assert len(body)==128**3
                else:
                    json.loads(body);metabytes+=len(body);assert metabytes<=2*1024**2
                record(rec);receipts.append(rec);return body
        except requests.RequestException as e:
            record(dict(key=key,attempt=attempt,error=type(e).__name__))
            if getattr(e,'response',None) is not None and e.response.status_code not in [500,502,503,504]:raise
            if attempt==4:raise
            time.sleep(min(2**attempt,8))
    raise AssertionError(key)
# Materialize every whitelisted chunk before rendering. Retained byte identity is
# checked against the older sealed receipt in GO; no zero arrays are fabricated.
for key in plan['source_chunk_keys']:
    old=next((r for r in plan['retained_chunks'] if r['key']==key),None)
    if old:
        expected={k:old[k] for k in ['path','size_bytes','sha256']};assert pin(old['path'])==expected
        body=Path(old['path']).read_bytes();rec=dict(key=key,source='verified prior acquisition',**expected);record(rec);receipts.append(rec)
    else:body=fetch(key)
    dest=cache/key;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(body);seen[key]=body
for key in ['.zgroup','.zattrs','0/.zarray']:
    body=fetch(key);dest=cache/key;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(body);seen[key]=body
meta=json.loads(seen['0/.zarray']);assert meta['shape']==[24298,9867,9867] and meta['chunks']==[128]*3 and meta['dtype']=='|u1' and meta['compressor'] is None and not meta.get('filters')
save(P/'SOURCE_RECEIPTS.json',dict(status='PASS',all_chunks_present=True,fill_chunks=0,receipts=receipts,planned_chunks=len(plan['source_chunk_keys']),metadata_bytes=metabytes))
class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def do_HEAD(self):self.do_GET()
    def do_GET(self):
        key=self.path.lstrip('/')
        with lock:
            allowed=key in seen or key in ['.zmetadata','zarr.json','meta.json','metadata.json'] or bool(re.fullmatch(r'[0-5]/(?:zarr.json|\.zattrs|\.zarray)',key))
        if not allowed:
            fatal.append(dict(key=key,error='unplanned request'));record(dict(key=key,fatal=True));self.send_error(404);return
        if key not in seen:
            try:body=fetch(key,optional=True);seen[key]=body
            except Exception as e:
                fatal.append(dict(key=key,error=repr(e)));self.send_error(502);return
        body=seen[key];record(dict(renderer_request=key,bytes=None if body is None else len(body),status=404 if body is None else 200))
        if body is None:self.send_error(404);return
        self.send_response(200);self.send_header('Content-Length',str(len(body)));self.end_headers()
        if self.command!='HEAD':self.wfile.write(body)
server=ThreadingHTTPServer(('127.0.0.1',0),Handler);threading.Thread(target=server.serve_forever,daemon=True).start()
env=dict(os.environ,QT_QPA_PLATFORM='offscreen',VC3D_CONFIG_DIR=str(W/'vc3d-config'),OMP_NUM_THREADS='4')
rows=[]
try:
    for depth in plan['depths']:
        cmd=[str(P.parent/'fortysixth-experiment/work/squashfs-root/AppRun'),'vc_render_tifxyz','--volume',str(W/f'renderer-cache-{depth}'),'--remote-url',f'http://127.0.0.1:{server.server_port}','--group-idx','0','--scale','1','--segmentation',str(W/'target.tifxyz'),'--num-slices',str(depth),'--slice-step','1','--flip-normals','--crop-x','1216','--crop-y','1216','--crop-width','128','--crop-height','128','--cache-gb','1','--zarr-output',str(W/f'target-{depth}.zarr'),'--zarr-compressor','none','--pyramid','0','--timeout','3']
        t=time.monotonic()
        with (W/f'render-{depth}.log').open('xb') as log:r=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=210)
        rows.append(dict(depth=depth,command=cmd,exit_code=r.returncode,seconds=time.monotonic()-t))
        if r.returncode or fatal:break
finally:server.shutdown();server.server_close();ledger.close()
result=dict(status='PASS' if len(rows)==2 and all(r['exit_code']==0 for r in rows) and not fatal else 'FAIL',renders=rows,fatal_source_errors=fatal,transport='Loopback finite whitelist; public upstream HTTPS through inherited proxy and TLS verification',script=pin(__file__))
save(P/'RENDER_EXECUTION.json',result);print(json.dumps(result));assert result['status']=='PASS'
