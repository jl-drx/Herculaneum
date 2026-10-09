"""Reconstruct fixed phase44 CPU inputs in a NEW directory; never start training."""
import argparse,hashlib,io,json,re,urllib.parse
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion

def check(ok,msg):
 if not ok:raise ValueError(msg)
def digest(b):return hashlib.sha256(b).hexdigest()
def _fetch(source):
 import requests
 url=source['url'];start,end=source['byte_range'];size=source['range_bytes']
 for hop in range(6):
  parsed=urllib.parse.urlsplit(url);host=parsed.hostname or '';check(parsed.scheme=='https' and (host=='huggingface.co' or host.endswith('.huggingface.co') or host.endswith('.hf.co')),'Official HTTPS source only')
  with requests.get(url,headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'},stream=True,allow_redirects=False,timeout=(10,30)) as r:
   if r.status_code in [301,302,303,307,308]:url=urllib.parse.urljoin(url,r.headers['Location']);continue
   check(r.status_code==206 and r.headers.get('Content-Range')==f"bytes {start}-{end}/{source['object_bytes']}",'Exact source range required')
   check(r.headers.get('Content-Length') is None or int(r.headers['Content-Length'])==size,'Range length')
   body=r.raw.read(size+1,decode_content=False);check(len(body)==size,'Bounded exact range bytes');return body
 raise ValueError('Redirect cap')
def fetch(source):
 import requests
 try:return _fetch(source)
 except requests.RequestException as exc:
  raise RuntimeError('Public range request failed ('+type(exc).__name__+'); no retry or URL logging') from None
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--offline-cache',type=Path);a=p.parse_args();plan=json.loads(a.plan.read_text());check(plan['schema_version']==1,'Known schema');a.output.mkdir(parents=True,exist_ok=False);cropdir=a.output/'crops';cropdir.mkdir();records=[];requests_count=0
 for case in plan['cases']:
  s=case['selection'];cid=s['case'];x,y,n=s['x'],s['y'],s['size'];sources={}
  for source in case['sources']:
   check(0<source['range_bytes']<=96*1024**2,'Range size cap')
   body=(a.offline_cache/source['cache_file']).read_bytes() if a.offline_cache else fetch(source)
   check(len(body)==source['range_bytes'] and digest(body)==source['range_sha256'],'Pinned range checksum');sources[source['kind']]=body
   if not a.offline_cache:requests_count+=1
  z=np.loadtxt(io.BytesIO(sources['raw_height_txt']),delimiter=';',usecols=tuple(range(x+1,x+n+1)),dtype=np.float32);coords=np.loadtxt(io.BytesIO(sources['raw_height_txt']),delimiter=';',usecols=(0,));check(z.shape==(n,n) and np.allclose(coords,np.arange(y,y+n)*case['coordinate_step'],atol=.0002,rtol=0),'Exact measured coordinates')
  labels=np.array(Image.open(io.BytesIO(sources['label_png'])));check(list(labels.shape)==case['full_shape'],'Source label shape');label=labels[y:y+n,x:x+n];finite=np.isfinite(z);block=finite.reshape(n//4,4,n//4,4).all((1,3)).repeat(4,0).repeat(4,1);binary=label!=0;support=block&(binary_erosion(binary,iterations=8,border_value=0)|binary_erosion(~binary,iterations=8,border_value=0))
  for kind,arr in [('height',z),('label',label),('common-support',support)]:
   out=cropdir/f'{cid}-{kind}.npy';np.save(out,arr);body=out.read_bytes();expect=case['outputs'][kind];check(len(body)==expect['size_bytes'] and digest(body)==expect['sha256'],'Reconstructed array identity');records.append(dict(path=str(out.relative_to(a.output)),sha256=digest(body),size_bytes=len(body)))
 (a.output/'CROP_SELECTION.json').write_text(json.dumps([v['selection'] for v in plan['cases']],indent=2)+'\n');report=dict(status='PASS',plan_sha256=digest(a.plan.read_bytes()),source_mode='offline retained ranges' if a.offline_cache else 'verified public ranges',records=records,logical_public_range_requests=requests_count,network_replay_tested=not bool(a.offline_cache),training_started=False,license=plan['license'],attribution=plan['attribution']);(a.output/'MATERIALIZATION.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='records'}))
if __name__=='__main__':main()
