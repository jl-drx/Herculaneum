import urllib.request,urllib.error,pathlib,json,hashlib,datetime,concurrent.futures
R=pathlib.Path(__file__).parent;S=R/'metadata';S.mkdir(exist_ok=True)
hf='https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/native9-scrollprizeorg-21slices/w040/'
s3='https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0139/segments/20250831000000-w040_2025083102/ink-labels/'
plan=[]
for name in ['inklabels','supervision_mask']:
 for leaf in ['.zattrs','0/.zarray']:
  plan.append((f'hf-{name}-{leaf.replace("/","-")}',hf+f'w040_{name}.zarr/'+leaf))
for target in ['2.399um-volume-20260102150214','9.362um-volume-20250728140407']:
 plan.append((f's3-{target}-group',s3+target+'/20260918/zarr.json'))
 for arr in ['inklabels','supervision','mapping_validity']:
  plan.append((f's3-{target}-{arr}',s3+target+f'/20260918/{arr}.zarr/zarr.json'))
def fetch(q):
 name,url=q;rec={'url':url,'retrieved_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 try:
  with urllib.request.urlopen(url,timeout=40) as r:
   b=r.read(1000001);assert len(b)<=1000000;d=json.loads(b);dest=S/(name+'.json');dest.write_bytes(b)
   rec.update({'status':r.status,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'path':str(dest),'headers':{k:v for k,v in r.headers.items() if k.lower() in ['content-type','content-length','etag','last-modified']}})
 except Exception as e:rec['error']=str(e)
 return rec
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:ledger=list(pool.map(fetch,plan))
json.dump(ledger,open(R/'LABEL_METADATA_REQUESTS.json','w'),indent=2)
print([(a.get('status',a.get('error')),a['url']) for a in ledger])
