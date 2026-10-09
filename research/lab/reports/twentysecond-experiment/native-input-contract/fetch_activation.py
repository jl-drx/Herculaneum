"""One separately authorized historical activation source. No fetched-code import."""
import hashlib,http.client,json,pathlib,signal,ssl,time,urllib.request,urllib.error
ROOT=pathlib.Path(__file__).resolve().parent
class Stop(Exception):pass
class NoRedirect(urllib.request.HTTPRedirectHandler):
 def redirect_request(self,*a,**kw):return None
def need(v,m):
 if not v:raise Stop(m)
def pin(p):
 b=p.read_bytes();return {'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'size_bytes':len(b)}
def alarm(*a):raise Stop('Absolute request deadline exceeded')
def main():
 go=ROOT.parent/'HISTORICAL_ACTIVATIONS_GO.json';g=json.loads(go.read_text());need(g['status']=='GO' and g['maximum_attempts']==1 and g['maximum_response_body_bytes']==65536,'Root literal bound')
 out=ROOT/'activation-source';out.mkdir(exist_ok=False);p=out/'activations.py';lp=out/'receipt.json';start=time.monotonic();r={'status':'SCHEDULED','url':g['url'],'root_go':pin(go),'reader':pin(pathlib.Path(__file__)),'scheduled_unix':time.time(),'observed_body_bytes':0,'attempts':1}
 def save():lp.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
 save();signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,30)
 try:
  op=urllib.request.build_opener(NoRedirect(),urllib.request.HTTPSHandler(context=ssl.create_default_context()));req=urllib.request.Request(g['url'],headers={'Accept-Encoding':'identity','User-Agent':'Herculaneum-source-contract/1'})
  try:response=op.open(req,timeout=30)
  except urllib.error.HTTPError as e:response=e
  with response:
   r['http_status']=response.status;need(response.geturl()==g['url'],'Unexpected response URL')
   need(response.headers.get('Content-Encoding','identity').lower()=='identity','Encoded response');n=response.headers.get('Content-Length');need(n is None or (n.isdecimal() and int(n)<65536),'Advertised body cap')
   with p.open('xb') as f:
    while r['observed_body_bytes']<65536:
     try:b=response.read(min(16384,65536-r['observed_body_bytes']))
     except http.client.IncompleteRead as e:f.write(e.partial);r['observed_body_bytes']+=len(e.partial);raise Stop('Truncated body') from None
     if not b:break
     need(len(b)<=65536-r['observed_body_bytes'],'Overread');f.write(b);r['observed_body_bytes']+=len(b)
   need(r['observed_body_bytes']<65536,'Cap reached without verified EOF');need(n is None or int(n)==r['observed_body_bytes'],'Length mismatch');need(response.status==200,'Unexpected HTTP status')
  b=p.read_bytes();s=b.decode('utf-8');need('\x00' not in s,'NUL in text');need(not s.lstrip().lower().startswith(('<html','<!doctype html')),'HTML response');r['git_blob_sha1']=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest();need(r['git_blob_sha1']==g['expected_git_blob_sha1'],'Git blob mismatch');r['status']='PASS'
 except BaseException as e:r['status']='STOP';r['reason']=str(e) if isinstance(e,Stop) else type(e).__name__
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);r['elapsed_seconds']=time.monotonic()-start
  if p.exists():r['body']=pin(p)
  save()
 print(json.dumps(r));return 0 if r['status']=='PASS' else 2
if __name__=='__main__':raise SystemExit(main())
