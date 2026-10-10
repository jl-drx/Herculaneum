import datetime, hashlib, json, pathlib, sys, urllib.request, urllib.error
ROOT=pathlib.Path(__file__).parent
LEDGER=ROOT/'request-ledger.json'
LIMIT=16*1024*1024
class NoRedirect(urllib.request.HTTPRedirectHandler):
 def redirect_request(self,*args,**kwargs): return None
opener=urllib.request.build_opener(NoRedirect())
ledger=json.loads(LEDGER.read_text()) if LEDGER.exists() else {'max_attempts':24,'max_body_bytes':LIMIT,'automatic_retries':0,'automatic_redirects':False,'scope':'Public text metadata only; no images, labels, CT, meshes or model payloads','attempts':[]}
for name,url in zip(sys.argv[1::2],sys.argv[2::2]):
 if len(ledger['attempts'])>=24: raise RuntimeError('attempt cap')
 remaining=LIMIT-sum(x.get('body_bytes',0) for x in ledger['attempts'])
 if remaining<=0: raise RuntimeError('byte cap')
 if (ROOT/name).exists(): raise RuntimeError('no overwrite')
 rec={'name':name,'url':url,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 ledger['attempts'].append(rec);LEDGER.write_text(json.dumps(ledger,indent=2)+'\n')
 try:
  try: response=opener.open(urllib.request.Request(url,headers={'Accept-Encoding':'identity','User-Agent':'Herculaneum-metadata-audit/1.0'}),timeout=30)
  except urllib.error.HTTPError as e: response=e
  with response:
   rec.update(http_status=response.status,headers={k:response.headers.get(k) for k in ['Content-Type','Content-Length','ETag','Last-Modified','Location']})
   ctype=(response.headers.get('Content-Type') or '').lower()
   if not any(t in ctype for t in ['text/','json','xml']): raise RuntimeError('not text metadata content type: '+ctype)
   body=response.read(min(remaining,2*1024*1024))
   rec['body_bytes']=len(body);rec['sha256']=hashlib.sha256(body).hexdigest()
   (ROOT/name).write_bytes(body)
   rec['path']=str(ROOT/name)
   declared=response.headers.get('Content-Length')
   rec['complete']=len(body)<min(remaining,2*1024*1024) and (declared is None or int(declared)==len(body))
   rec['status']='retained' if rec['complete'] else 'incomplete-stop'
 except Exception as e: rec['status']='error';rec['error']=str(e)
 rec['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
 LEDGER.write_text(json.dumps(ledger,indent=2)+'\n')
 print(json.dumps({k:rec.get(k) for k in ['name','http_status','body_bytes','status','error']}),flush=True)
 if rec['status']!='retained': raise RuntimeError('stopping on incomplete/error')
