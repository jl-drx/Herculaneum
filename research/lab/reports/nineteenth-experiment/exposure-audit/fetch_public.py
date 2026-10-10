"""Bounded public text and metadata retrieval; no credentials, retries or auto redirects."""
import datetime,hashlib,json,pathlib,sys,urllib.request,urllib.error
BASE=pathlib.Path(__file__).resolve().parent
class NoRedirect(urllib.request.HTTPRedirectHandler):
 def redirect_request(self,*args,**kwargs): return None
opener=urllib.request.build_opener(NoRedirect)
def fetch(label,url,cap=2*1024*1024):
 ledger=BASE/'request-ledger.json'; data=json.loads(ledger.read_text()) if ledger.exists() else {'max_requests':20,'max_response_body_bytes':12*1024*1024,'requests':[]}
 assert len(data['requests'])<20
 remaining=12*1024*1024-sum(r['bytes'] for r in data['requests']); limit=min(cap,remaining);assert limit>0
 now=datetime.datetime.now(datetime.timezone.utc).isoformat(); rec={'label':label,'url':url,'started_utc':now,'max_body_bytes':limit}
 try:
  req=urllib.request.Request(url,headers={'User-Agent':'Herculaneum-provenance-audit/1.0','Accept':'application/json,text/plain','Accept-Encoding':'identity'})
  try: response=opener.open(req,timeout=25)
  except urllib.error.HTTPError as e: response=e
  with response:
   rec.update(status=response.status,final_url=response.geturl(),headers={k:v for k,v in response.headers.items() if k.lower() in ['content-type','content-length','location','etag','last-modified','x-repo-commit']})
   body=response.read(limit)
   rec['truncated_or_at_cap']=len(body)==limit
 except Exception as e: rec.update(status=None,error=type(e).__name__+': '+str(e));body=b''
 rec.update(bytes=len(body),sha256=hashlib.sha256(body).hexdigest(),completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
 path=BASE/(label+'.body');path.write_bytes(body);rec['body_path']=str(path);data['requests'].append(rec);ledger.write_text(json.dumps(data,indent=2)+'\n');print(json.dumps({k:v for k,v in rec.items() if k not in ['headers']}));return rec
if __name__=='__main__':fetch(sys.argv[1],sys.argv[2])
