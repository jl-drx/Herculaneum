"""Original renderer over a loopback cache; all source fetches retain HTTPS proxy/TLS.

Only transient transport/5xx errors are retried. No 403/404 bypass or missing-fill.
This changes transport, not the renderer, meshes, CT bytes or sampling.
"""
from pathlib import Path
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
import concurrent.futures,hashlib,json,os,re,subprocess,threading,time,requests
P=Path(__file__).resolve().parent;W=P/'work';OLD=P.parent/'fortyfifth-experiment';NAME='concat_w047-w048_R5B2_z9500-11000'
BASE='https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0343/volumes/20250521140437-8.640um-1.2m-116keV-masked.zarr'
cache=next(p for p in (W/'renderer-cache').glob('202505*.zarr-*') if p.is_dir())
version=1+max([0]+[int(p.stem.rsplit('v',1)[1]) for p in W.glob('cached-transport-v*.jsonl')])
observed_absent=set()
for old in W.glob('cached-transport*.jsonl'):
    for line in old.read_text().splitlines():
        event=json.loads(line)
        if event.get('status')==404:observed_absent.add(event['key'])
ledger=(W/f'cached-transport-v{version}.jsonl').open('x');guard=threading.Lock();locks={};fatal=[]
def record(v):
    with guard:ledger.write(json.dumps(v)+'\n');ledger.flush()
class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def do_HEAD(self):self.do_GET()
    def do_GET(self):
        key=self.path.lstrip('/')
        optional=key in ['.zmetadata','zarr.json','meta.json','metadata.json'] or bool(re.fullmatch(r'[0-5]/(?:zarr.json|\.zattrs)',key))
        if not (optional or key in ['.zgroup','.zattrs'] or re.fullmatch(r'[0-5]/\.z(?:attrs|array)',key) or re.fullmatch(r'0/\d+/\d+/\d+',key)):
            self.send_error(400);fatal.append(key);return
        if optional and key in observed_absent:
            record(dict(key=key,status=404,source='previous observed optional-metadata404'))
            self.send_error(404);return
        with guard:lock=locks.setdefault(key,threading.Lock())
        with lock:
            dest=cache/key
            if dest.exists():body=dest.read_bytes();origin='retained cache'
            else:
                body=None;origin=BASE+'/'+key
                for attempt in range(1,7):
                    start=time.monotonic()
                    try:
                        r=requests.get(origin,timeout=(20,60));status=r.status_code
                        record(dict(key=key,attempt=attempt,status=status,bytes=len(r.content),seconds=time.monotonic()-start))
                        if status==200:
                            body=r.content
                            if re.fullmatch(r'0/\d+/\d+/\d+',key):assert len(body)==128**3
                            else:json.loads(body)
                            dest.parent.mkdir(parents=True,exist_ok=True);dest.with_suffix('.partial').write_bytes(body);dest.with_suffix('.partial').rename(dest);break
                        if status not in [500,502,503,504]:
                            self.send_error(status)
                            if not(optional and status==404):fatal.append(dict(key=key,status=status))
                            return
                    except requests.RequestException as e:record(dict(key=key,attempt=attempt,error=type(e).__name__))
                    if attempt<6:time.sleep(min(2**(attempt-1),10))
                if body is None:
                    record(dict(key=key,error='exhausted transient retries'))
                    self.send_error(503);return
            record(dict(key=key,source=origin,bytes=len(body),sha256=hashlib.sha256(body).hexdigest()))
        self.send_response(200);self.send_header('Content-Length',str(len(body)));self.end_headers()
        if self.command!='HEAD':self.wfile.write(body)
server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
url=f'http://127.0.0.1:{server.server_port}'
# Probe actual server/retained bytes before passing the URL to the renderer.
probe=requests.get(url+'/.zgroup',timeout=10);assert probe.status_code==200 and probe.content==(cache/'.zgroup').read_bytes()
env=dict(os.environ,QT_QPA_PLATFORM='offscreen',VC3D_CONFIG_DIR=str(W/'vc3d-config'),OMP_NUM_THREADS='4')
cmd=[str(W/'squashfs-root/AppRun'),'vc_render_tifxyz','--volume',str(W/'ct-cache-loopback'),'--remote-url',url,'--group-idx','0','--scale','1','--segmentation',str(OLD/'ph343-mesh-replay'/f'{NAME}.tifxyz'),'--num-slices','66','--slice-step','1','--cache-gb','2','--zarr-output',str(W/f'{NAME}_66.zarr'),'--resume','--timeout','15']
start=time.monotonic()
try:
    with (W/f'render-cached-v{version}.log').open('xb') as log:r=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
finally:server.shutdown();server.server_close();ledger.close()
result=dict(exit_code=r.returncode,command=cmd,seconds=time.monotonic()-start,source_url=BASE,transport='loopback cache; upstream requests through inherited HTTPS proxy with TLS verification',fatal_source_errors=fatal)
(P/f'CACHED_RENDER_EXECUTION_V{version}.json').write_text(json.dumps(result,indent=2)+'\n')
assert r.returncode==0 and not fatal,result
subprocess.run(['/workspace/.venvs/pherc343/bin/python','-B',str(P/'verify_render.py')],check=True)
