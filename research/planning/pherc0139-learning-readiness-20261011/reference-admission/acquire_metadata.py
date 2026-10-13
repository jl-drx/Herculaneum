import json,pathlib,hashlib,urllib.request,urllib.error,concurrent.futures,datetime
R=pathlib.Path(__file__).parent; (R/'metadata').mkdir(exist_ok=True)
p=pathlib.Path('/workspace/herculaneum-lab/reports/pherc0139-correspondence-20261011/provenance/PHerc0139-retained-catalog.json');d=json.loads(p.read_text())
base='https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/'
segids={'w035':'20260317000000','w039':'20260302000000','w044':'20260115000000','w024':'20260215000000'}
extract={'catalog_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'catalog_local_source':str(p),'segments':{}}
requests=[]
for w,sid in segids.items():
 s=d['segments'][sid];extract['segments'][w]=s
 native=[a for a in s['data'] if a['type']=='layers-zarr' and a.get('parameters',{}).get('target_volume')=='20250728140407'][0]['origins'][0]['path'].rstrip('/')
 requests += [(w+'-native-attrs',base+native+'/.zattrs','GET'),(w+'-native-array',base+native+'/0/.zarray','GET')]
 if w!='w024':
  label=[a for a in s['data'] if a['type']=='ink-labels' and a.get('parameters',{}).get('target_volume')=='20250728140407'][0]['origins'][0]['path'].rstrip('/')
  for a in ['inklabels','supervision']:
   requests += [(w+'-s3-'+a+'-root',base+label+'/'+a+'.zarr/zarr.json','GET'),(w+'-s3-'+a+'-array',base+label+'/'+a+'.zarr/0/zarr.json','GET')]
  hf='https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/native9-scrollprizeorg-21slices/'+w+'/'+w+'_inklabels.zarr'
  requests += [(w+'-hf-label-attrs',hf+'/.zattrs','GET'),(w+'-hf-label-array',hf+'/0/.zarray','GET')]
 else:
  fine=[a for a in s['data'] if a['type']=='layers-zarr' and a.get('parameters',{}).get('target_volume')=='20260413113053'][0]['origins'][0]['path'].rstrip('/')
  requests += [(w+'-fine59-attrs',base+fine+'/.zattrs','GET'),(w+'-fine59-array',base+fine+'/0/.zarray','GET')]
  teacher=[a for a in s['data'] if a['type']=='ink-detection' and a.get('parameters',{}).get('target_volume')=='20260413113053'][0]['origins'][0]['path']
  requests += [(w+'-fine59-teacher-head',base+teacher,'HEAD')]
json.dump(extract,open(R/'CATALOG_EXTRACT.json','w'),indent=2)
json.dump([{'id':i,'url':u,'method':m} for i,u,m in requests],open(R/'REQUEST_PLAN.json','w'),indent=2)
def fetch(r):
 i,u,m=r;rec={'id':i,'url':u,'method':m,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 try:
  with urllib.request.urlopen(urllib.request.Request(u,method=m),timeout=40) as resp:
   b=resp.read(1048577) if m=='GET' else b'';assert len(b)<=1048576
   rec.update(status=resp.status,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),etag=resp.headers.get('ETag'),last_modified=resp.headers.get('Last-Modified'),content_length=resp.headers.get('Content-Length'))
   if m=='GET':json.loads(b);dest=R/'metadata'/(i+'.json');dest.write_bytes(b);rec['path']=str(dest.relative_to(R))
 except urllib.error.HTTPError as e:rec.update(status=e.code,error='HTTP error')
 except Exception as e:rec.update(error=type(e).__name__+': '+str(e))
 return rec
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool: out=list(pool.map(fetch,requests))
json.dump(out,open(R/'REQUEST_LEDGER.json','w'),indent=2)
print('Requests',len(out),'metadata bytes',sum(x.get('bytes',0) for x in out));print([(x['id'],x.get('status'),x.get('error')) for x in out])
