"""Bounded public-release requests; signed redirect queries stay in memory only."""
import argparse,hashlib,json,re,urllib.parse
from datetime import datetime,timezone
from pathlib import Path
import requests
P=Path(__file__).resolve().parent
REV='a806bead2f3b9100c20e19de37814fb285cfeefd'
def pin(p):
 p=Path(p)
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return dict(path=str(p),sha256=h,size_bytes=p.stat().st_size)
def save(p,v):
 with Path(p).open('x') as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def now():return datetime.now(timezone.utc).isoformat()
def need(v,msg):
 if not v:raise ValueError(msg)
def safe_url(url):
 s=urllib.parse.urlsplit(url)
 return urllib.parse.urlunsplit((s.scheme,s.netloc,s.path,'',''))
def fetch(tag,url,byte_range=None):
 go=json.loads((P/'PIXEL_GO.json').read_text())
 for item in go['pins']:need(pin(item['path'])==item,'Frozen source changed')
 out=P/'pixel-responses';out.mkdir(exist_ok=True);ledger=P/'PIXEL_REQUESTS.json';rows=json.loads(ledger.read_text()) if ledger.exists() else [];source=url
 need(not any(r['tag']==tag for r in rows),'No automatic repeat')
 for hop in range(6):
  need(len(rows)<128,'Pixel response count cap');remaining=384*1024**2-sum(r['bytes'] for r in rows);need(remaining>0,'Pixel byte cap')
  host=urllib.parse.urlsplit(url).hostname
  need(host=='huggingface.co' or host.endswith('.huggingface.co') or host.endswith('.hf.co'),'Official HF storage route')
  cap=min(remaining,96*1024**2 if byte_range else 2*1024**2);headers={'Accept-Encoding':'identity'}
  if byte_range:
   a,b=byte_range;need(0<=a<=b and b-a+1<=96*1024**2,'Pixel range bound');headers['Range']=f'bytes={a}-{b}'
  row=dict(tag=tag,hop=hop,source_url=safe_url(source),request_url=safe_url(url),request_url_sha256=hashlib.sha256(url.encode()).hexdigest(),requested_range=list(byte_range) if byte_range else None,started_at_utc=now(),bytes=0,status=None,complete=False);body=bytearray();redirect=None
  try:
   with requests.get(url,headers=headers,stream=True,allow_redirects=False,timeout=(10,25)) as response:
    row['status']=response.status_code;row['headers']={k:response.headers.get(k) for k in ['Content-Length','Content-Range','Content-Encoding','Content-Type','ETag','X-Repo-Commit','X-Linked-Size','X-Linked-Etag']}
    if response.status_code in [301,302,303,307,308]:
     redirect=urllib.parse.urljoin(url,response.headers['Location']);row['redirect_url']=safe_url(redirect)
    elif byte_range:
     need(response.status_code==206,'Range must return 206')
     value=response.headers.get('Content-Range','');m=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',value);need(m is not None,'Explicit total-length Content-Range')
     start,end,total=map(int,m.groups());need(start==a and end==min(b,total-1),'Exact requested range');row['object_bytes']=total
    else:need(response.status_code==200,'Non-ranged request must return 200')
    declared=response.headers.get('Content-Length');need(declared is None or int(declared)<=cap,'Response exceeds cap before body read')
    while len(body)<cap:
     chunk=response.raw.read(min(65536,cap-len(body)),decode_content=False)
     if not chunk:row['complete']=True;break
     body.extend(chunk)
    if declared is not None and len(body)==int(declared):row['complete']=True
    need(row['complete'],'Unproven response completion')
    if byte_range and redirect is None:need(len(body)==end-start+1,'Exact range body length')
  except Exception as e:row['error']=type(e).__name__+': '+str(e).split('?')[0][:250]
  finally:
   dest=out/f'{tag}-hop{hop}.bin';need(not dest.exists(),'Write-once response');dest.write_bytes(body);row.update(bytes=len(body),body=pin(dest));rows.append(row);ledger.write_text(json.dumps(rows,indent=2,sort_keys=True)+'\n')
  print(json.dumps({k:row[k] for k in ['tag','hop','status','bytes','complete']}|({'error':row['error']} if 'error' in row else {})),flush=True)
  if 'error' in row:raise RuntimeError('Request failed; retained sanitized receipt')
  if redirect is None:return dest,row
  url=redirect
 raise RuntimeError('Redirect depth exceeded')
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('tag');a.add_argument('url');a.add_argument('--range',nargs=2,type=int);args=a.parse_args();fetch(args.tag,args.url,args.range)
