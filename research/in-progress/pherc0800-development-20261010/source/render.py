"""Run the pinned renderer with a finite, audited source whitelist; no fill accepted."""
from pathlib import Path
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
import hashlib,json,os,re,subprocess,threading,time,requests
from common import pin,save,atomjson
P=Path(__file__).resolve().parent;W=P/'work';plan=json.loads((P/'RENDER_GO.json').read_text());cache=W/'source-cache';cache.mkdir(exist_ok=True)
assert pin(P/'PROTOCOL.md')==plan['protocol']
for r in plan['pins']:assert pin(r['path'])==r,r['path']
assert not (P/'RENDER_EXECUTION.json').exists()
ledger=(W/'source-ledger.jsonl').open('a');lock=threading.Lock();fatal=[];metabytes=0;seen={};receipts=[];active_keys=set();active_name='acquisition'
def record(event):
    with lock:ledger.write(json.dumps(event,sort_keys=True)+'\n');ledger.flush()
def fetch(key,optional=False):
    global metabytes
    url=plan['source_url']+'/'+key
    cap=128**3 if key in plan['source_keys'] else 2*1024**2
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
                if key in plan['source_keys']:assert len(body)==128**3
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
for key in plan['source_keys']:
    checkpoint=W/'acquisition-receipts'/key.replace('/','_')
    checkpoint.parent.mkdir(exist_ok=True)
    dest=cache/key
    if checkpoint.exists():
        rec=json.loads(checkpoint.read_text());assert pin(dest)==rec['file'];receipts.append(rec['source']);seen[key]=dest;continue
    old=next((r for r in plan['retained_chunks'] if r['key']==key),None)
    if old:
        expected={k:old[k] for k in ['path','size_bytes','sha256']};assert pin(old['path'])==expected
        body=Path(old['path']).read_bytes();rec=dict(key=key,source='verified prior acquisition',**expected);record(rec);receipts.append(rec)
    else:body=fetch(key)
    dest=cache/key;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(body);atomjson(checkpoint,dict(file=pin(dest),source=rec if old else receipts[-1]));seen[key]=dest
for key in ['.zgroup','.zattrs','0/.zarray']:
    body=fetch(key);dest=cache/key;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(body);seen[key]=body
meta=json.loads(seen['0/.zarray']);assert meta['shape']==[24298,9867,9867] and meta['chunks']==[128]*3 and meta['dtype']=='|u1' and meta['compressor'] is None and not meta.get('filters')
atomjson(P/'SOURCE_RECEIPTS.json',dict(status='PASS',all_chunks_present=True,fill_chunks=0,receipts=receipts,planned_chunks=len(plan['source_keys']),metadata_bytes=metabytes))
class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def do_HEAD(self):self.do_GET()
    def do_GET(self):
        key=self.path.lstrip('/')
        with lock:
            allowed=(key in seen and (not key.startswith('0/') or key=='0/.zarray' or key in active_keys)) or key in ['.zmetadata','zarr.json','meta.json','metadata.json'] or bool(re.fullmatch(r'[0-5]/(?:zarr.json|\.zattrs|\.zarray)',key))
        if not allowed:
            fatal.append(dict(key=key,error='unplanned request'));record(dict(key=key,fatal=True));self.send_error(404);return
        if key in plan['source_keys'] and key not in active_keys:
            fatal.append(dict(key=key,case=active_name,error='outside active render whitelist'));self.send_error(404);return
        if key not in seen:
            try:body=fetch(key,optional=True);seen[key]=body
            except Exception as e:
                fatal.append(dict(key=key,error=repr(e)));self.send_error(502);return
        body=seen[key];body=body.read_bytes() if isinstance(body,Path) else body;record(dict(renderer_request=key,case=active_name,bytes=None if body is None else len(body),status=404 if body is None else 200))
        if body is None:self.send_error(404);return
        self.send_response(200);self.send_header('Content-Length',str(len(body)));self.end_headers()
        if self.command!='HEAD':self.wfile.write(body)
server=ThreadingHTTPServer(('127.0.0.1',0),Handler);threading.Thread(target=server.serve_forever,daemon=True).start()
env=dict(os.environ,QT_QPA_PLATFORM='offscreen',VC3D_CONFIG_DIR=str(W/'vc3d-config'),OMP_NUM_THREADS='4')
rows=[]
try:
    for case in plan['cases']:
        done=W/(case['name']+'-COMPLETE.json')
        if done.exists():
            rec=json.loads(done.read_text());assert rec['go_sha256']==pin(P/'RENDER_GO.json')['sha256']
            for r in rec['output_pins']:assert pin(r['path'])==r
            rows.append(rec);continue
        active_keys=set(case['source_keys']);active_name=case['name'];x,y,w,h=case['crop_xywh']
        cmd=[str(P.parent/'fortysixth-experiment/work/squashfs-root/AppRun'),'vc_render_tifxyz','--volume',str(W/('renderer-cache-'+active_name)),'--remote-url',f'http://127.0.0.1:{server.server_port}','--group-idx','0','--scale','1','--segmentation',case['segmentation'],'--num-slices','66','--slice-step','1','--flip-normals','--crop-x',str(x),'--crop-y',str(y),'--crop-width',str(w),'--crop-height',str(h),'--cache-gb','1','--zarr-output',str(W/(active_name+'.zarr')),'--zarr-compressor','none','--pyramid','0','--voxel-size','8.64','--voxel-unit','micrometer','--timeout','3']
        t=time.monotonic()
        with (W/(active_name+'-render.log')).open('ab') as log:r=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=300)
        rec=dict(case=active_name,command=cmd,exit_code=r.returncode,seconds=time.monotonic()-t,go_sha256=pin(P/'RENDER_GO.json')['sha256'])
        rows.append(rec)
        if r.returncode or fatal:break
        rec['output_pins']=[pin(p) for p in sorted((W/(active_name+'.zarr')).rglob('*')) if p.is_file()]
        atomjson(done,rec)
        atomjson(W/'RENDER_PROGRESS.json',dict(completed_rectangles=len(rows),expected_rectangles=len(plan['cases']),last_completed=active_name))
        print('RENDERED',active_name,len(rows),flush=True)
finally:server.shutdown();server.server_close();ledger.close()
result=dict(status='PASS' if len(rows)==len(plan['cases']) and all(r['exit_code']==0 for r in rows) and not fatal else 'FAIL',renders=rows,fatal_source_errors=fatal,transport='Loopback finite whitelist; public upstream HTTPS through inherited proxy and TLS verification',script=pin(__file__))
save(P/'RENDER_EXECUTION.json',result);print(json.dumps({k:v for k,v in result.items() if k!='renders'}));assert result['status']=='PASS'
