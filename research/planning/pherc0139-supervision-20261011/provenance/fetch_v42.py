import pathlib,urllib.request,json,hashlib,datetime,concurrent.futures
R=pathlib.Path(__file__).parent;S=R/'sources';base='https://raw.githubusercontent.com/ScrollPrize/villa/d4615b134ce1b62368bed1029ed47e5569ca03fb/ink-detection/optimized_inference/'
def one(f):
 u=base+f
 with urllib.request.urlopen(u,timeout=40) as r:b=r.read(1000001);assert len(b)<=1000000
 p=S/('v42-'+f);p.write_bytes(b);return {'url':u,'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'retrieved_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:led=list(pool.map(one,['inference.py','processing.py','entrypoint.py','build.sh']))
json.dump(led,open(R/'V42_SOURCE_REQUESTS.json','w'),indent=2);print(led)
