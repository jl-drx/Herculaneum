import urllib.request,hashlib,json,pathlib,datetime
R=pathlib.Path(__file__).parent;S=R/'sources';S.mkdir(exist_ok=True)
base='https://raw.githubusercontent.com/Nieuwlaar/ink9um-dense-native/217e7521670db49cde97872ea6a8bdfb80c449f1/'
paths=['training/teacher_sources.tsv','training/scripts/fetch_aligned_labels.py','training/scripts/calibrate_theta.py','README.md']
ledger=[]
for p in paths:
 url=base+p
 try:
  with urllib.request.urlopen(url,timeout=40) as r:
   b=r.read(2000001);assert len(b)<=2000000; rec={'url':url,'status':r.status,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'headers':dict(r.headers)}
  dest=S/p.replace('/','__');dest.write_bytes(b);rec['path']=str(dest)
 except Exception as e:rec={'url':url,'error':str(e)}
 rec['retrieved_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();ledger.append(rec)
json.dump(ledger,open(R/'SOURCE_REQUESTS.json','w'),indent=2)
p=pathlib.Path('/workspace/herculaneum-lab/reports/fortyfifth-experiment/source-evidence/PHerc0139-retained-catalog.json')
d=json.load(open(p));s=d['segments']['20250831000000'];items=[a for a in s['data'] if a['type'] in ['ink-detection','ink-labels','layers-zarr'] and a.get('parameters',{}).get('target_volume') in ['20260102150214','20250728140407']]
json.dump({'catalog_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'segment_id':s['long_id'],'data':items},open(R/'CATALOG_TARGET_EXTRACT.json','w'),indent=2)
for a in items: print(a['type'],a['parameters'],a['origins'][0]['path'])
print([(r.get('status'),r.get('bytes'),r['url']) for r in ledger])
