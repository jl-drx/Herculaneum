import hashlib,json,signal,ssl,time,urllib.request,urllib.error
from pathlib import Path
from urllib.parse import urlparse
HERE=Path(__file__).resolve().parent
class NoRedirect(urllib.request.HTTPRedirectHandler):
 def redirect_request(self,*args,**kwargs):return None

def fetch(url,name,cap=524288):
 assert urlparse(url).scheme=='https' and urlparse(url).hostname in ['api.github.com','github.com','raw.githubusercontent.com','scrollprize.org','html.duckduckgo.com','www.google.com','www.bing.com']
 ledger=HERE/'ledger.json';d=json.loads(ledger.read_text()) if ledger.exists() else {'requests':[],'reserved_body_bytes':0,'returned_body_bytes':0}
 assert len(d['requests'])<20 and 0<cap<=2097152 and d['reserved_body_bytes']+cap+1<=12*1024**2
 assert name not in [r['name'] for r in d['requests']]
 item={'name':name,'url':url,'cap':cap,'success':False};d['requests'].append(item);d['reserved_body_bytes']+=cap+1
 ledger.write_text(json.dumps(d,indent=2)+'\n');body=b'';r=None;start=time.monotonic()
 def expired(*_):raise TimeoutError('20second deadline')
 old=signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,20)
 try:
  opener=urllib.request.build_opener(NoRedirect(),urllib.request.HTTPSHandler(context=ssl.create_default_context()))
  try:r=opener.open(urllib.request.Request(url,headers={'Accept-Encoding':'identity','User-Agent':'Herculaneum-public-documentation-audit','Accept':'application/json,text/html,text/plain'}),timeout=15)
  except urllib.error.HTTPError as error:r=error
  item.update(status=r.status,headers={k:r.headers.get(k) for k in ['Content-Type','Content-Length','Content-Encoding','ETag','Last-Modified']})
  mime=r.headers.get('Content-Type','').split(';')[0].lower()
  assert mime in ['application/json','text/html','text/plain','application/vnd.github+json'],'Non-text response refused'
  assert r.headers.get('Content-Encoding','identity') in ['identity',''],'Encoded body refused'
  body=r.read(cap+1);assert len(body)<=cap,'Response cap exceeded'
  body.decode('utf-8');item['success']=r.status==200
 except Exception as error:item['error']=type(error).__name__+': '+str(error)
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
  if r:r.close()
  path=HERE/(name+'.body');path.write_bytes(body)
  item.update(returned_bytes=len(body),sha256=hashlib.sha256(body).hexdigest(),body=str(path),elapsed_seconds=time.monotonic()-start)
  d['returned_body_bytes']+=len(body);ledger.write_text(json.dumps(d,indent=2)+'\n')
 print(json.dumps({k:v for k,v in item.items() if k not in ['body','sha256','headers']}))
 return body if item['success'] else None
