"""Bounded phase22 public source reader. Never imports fetched source."""
import hashlib, http.client, json, os, pathlib, signal, ssl, sys, time, urllib.error, urllib.parse, urllib.request
ROOT=pathlib.Path(__file__).resolve().parent
COMMIT='3ea17f54a9b3d5fd1aaf73e1d2c8386dbaa9f30e'
CAP=1048576
class Stop(Exception): pass
class NoRedirect(urllib.request.HTTPRedirectHandler):
 def redirect_request(self,*a,**kw): return None
def need(v,m):
 if not v: raise Stop(m)
def save(p,v):
 q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n');os.replace(q,p)
def pin(p):
 b=p.read_bytes();return {'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'size_bytes':len(b)}
def alarm(*a):raise Stop('Absolute request deadline exceeded')
def run(label,url,why):
 out=ROOT/'source';out.mkdir(exist_ok=True);lp=out/'request-ledger.json'
 l=json.loads(lp.read_text()) if lp.exists() else {'started_unix':time.time(),'status':'ACTIVE','attempts':[],'limits':{'attempts':16,'total_body_bytes':4194304,'response_body_bytes':CAP,'absolute_request_seconds':30,'absolute_total_seconds':480},'reader':pin(pathlib.Path(__file__))}
 u=urllib.parse.urlsplit(url)
 need(u.scheme=='https' and u.hostname in ['raw.githubusercontent.com','api.github.com'] and not u.username and not u.password and u.port in (None,443) and not u.fragment,'URL policy')
 need(COMMIT in url and '/ScrollPrize/villa/' in url,'Exact repository/commit binding')
 need(l['status']=='ACTIVE' and len(l['attempts'])<16,'Closed stage or attempt cap')
 need(time.time()-l['started_unix']<480,'Overall deadline exceeded')
 need(not any(a['url']==url for a in l['attempts']),'Retries forbidden')
 used=sum(a.get('observed_body_bytes',0) for a in l['attempts']);need(used+CAP<=4194304,'Response reservation cannot fit remaining total budget')
 need(label.replace('-','').replace('_','').isalnum(),'Bad output label')
 dst=out/(label+'.body');need(not dst.exists(),'Output exists')
 row={'url':url,'why':why,'label':label,'status':'SCHEDULED','observed_body_bytes':0,'scheduled_unix':time.time(),'reserved_body_bytes':CAP}
 l['attempts'].append(row);save(lp,l)
 opener=urllib.request.build_opener(NoRedirect(),urllib.request.HTTPSHandler(context=ssl.create_default_context()));signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,min(30,480-(time.time()-l['started_unix'])))
 try:
  req=urllib.request.Request(url,headers={'Accept-Encoding':'identity','User-Agent':'Herculaneum-native-source-contract/1'})
  try:r=opener.open(req,timeout=30)
  except urllib.error.HTTPError as e:r=e
  with r:
   row['http_status']=r.status;need(r.geturl()==url,'Redirect blocked')
   for k in ['Content-Length','Content-Encoding','Content-Type','ETag']:need(len(r.headers.get_all(k,[]))<=1,'Duplicate response header')
   row['headers']={k:r.headers[k] for k in ['Content-Length','Content-Encoding','Content-Type','ETag'] if k in r.headers}
   need(r.headers.get('Content-Encoding','identity').lower()=='identity','Encoded response blocked')
   n=r.headers.get('Content-Length');need(n is None or n.isdecimal(),'Invalid content length');need(n is None or int(n)<CAP,'Advertised body cap reached')
   with dst.open('xb') as f:
    while row['observed_body_bytes']<CAP:
     try:b=r.read(min(65536,CAP-row['observed_body_bytes']))
     except http.client.IncompleteRead as e:
      f.write(e.partial);row['observed_body_bytes']+=len(e.partial);raise Stop('Truncated response') from None
     if not b:break
     need(len(b)<=CAP-row['observed_body_bytes'],'Overread');f.write(b);row['observed_body_bytes']+=len(b)
   need(row['observed_body_bytes']<CAP,'Body cap reached without verified EOF');need(n is None or row['observed_body_bytes']==int(n),'Length mismatch')
   body=dst.read_bytes();s=body.decode('utf-8');need('\x00' not in s,'NUL in text')
   need(r.status in (200,404),'Unexpected HTTP status');row['status']='PASS' if r.status==200 else 'PATH_GAP'
   if r.status==200:need(not s.lstrip().lower().startswith(('<html','<!doctype html')),'Unexpected HTML');row['line_count']=len(s.splitlines())
 except BaseException as e:
  row['status']='STOP';row['reason']=str(e) if isinstance(e,Stop) else type(e).__name__;l['status']='STOP'
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);row['elapsed_seconds']=time.time()-row['scheduled_unix']
  if dst.exists():row['body']=pin(dst);row['observed_body_bytes']=dst.stat().st_size
  l['observed_body_bytes']=sum(a.get('observed_body_bytes',0) for a in l['attempts']);l['elapsed_seconds']=time.time()-l['started_unix'];save(lp,l)
 print(json.dumps(row));return 0 if row['status'] in ('PASS','PATH_GAP') else 2
if __name__=='__main__':raise SystemExit(run(*sys.argv[1:]))
