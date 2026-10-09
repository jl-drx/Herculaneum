"""Bounded, single-request metadata fetch with retained failures and no retries."""
import argparse,gzip,hashlib,json,time,urllib.parse
from datetime import datetime,timezone
from pathlib import Path
import requests
P=Path(__file__).resolve().parent
def pin(p):
 p=Path(p)
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return dict(path=str(p),sha256=h,size_bytes=p.stat().st_size)
def now():return datetime.now(timezone.utc).isoformat()
def fetch(tag,url):
 go=json.loads((P/'GO.json').read_text())
 for item in go['pins']:
  if pin(item['path'])!=item:raise ValueError('Frozen source changed')
 out=P/'sources';out.mkdir(exist_ok=True);ledger=P/'REQUESTS.json';rows=json.loads(ledger.read_text()) if ledger.exists() else []
 if len(rows)>=24 or any(r['url']==url for r in rows):raise ValueError('Request count or duplicate request')
 if rows and (datetime.now(timezone.utc)-datetime.fromisoformat(rows[0]['started_at_utc'])).total_seconds()>1800:raise ValueError('Search time expired')
 uri=urllib.parse.urlsplit(url)
 if uri.scheme!='https' or uri.hostname not in ['vesuvius-challenge-open-data.s3.amazonaws.com','raw.githubusercontent.com','api.github.com','huggingface.co','scrollprize.org']:raise ValueError('Metadata destination scope')
 available=16*1024**2-sum(r['bytes'] for r in rows);cap=min(4*1024**2,available)
 if cap<=0:raise ValueError('Byte budget exhausted')
 target=out/(tag+'.txt')
 if target.exists():raise ValueError('Write-once source destination')
 row=dict(tag=tag,url=url,started_at_utc=now(),bytes=0,complete=False,status=None,headers={});body=bytearray();start=time.monotonic()
 try:
  with requests.get(url,stream=True,allow_redirects=False,timeout=(10,25),headers={'Accept-Encoding':'identity','User-Agent':'Herculaneum-research-metadata/1.0'}) as r:
   row['status']=r.status_code;row['headers']={k:r.headers.get(k) for k in ['Content-Length','Content-Encoding','Content-Type','ETag','Location']}
   declared=r.headers.get('Content-Length')
   if declared is not None and int(declared)>cap:raise ValueError('Declared response exceeds remaining cap; body not read')
   while len(body)<cap:
    chunk=r.raw.read(min(65536,cap-len(body)),decode_content=False)
    if not chunk:row['complete']=True;break
    body.extend(chunk)
   if len(body)==cap and declared is not None and int(declared)==cap:row['complete']=True
   if not row['complete']:raise ValueError('Response reached byte cap before proven completion')
   if r.status_code!=200:raise ValueError('Non-200 HTTP status retained')
 except Exception as e:row['error']=type(e).__name__+': '+str(e)
 finally:
  target.write_bytes(body);row.update(bytes=len(body),elapsed_seconds=time.monotonic()-start,raw=pin(target));rows.append(row);ledger.write_text(json.dumps(rows,indent=2,sort_keys=True)+'\n')
 print(json.dumps(row))
 if 'error' in row:raise SystemExit(2)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('tag');a.add_argument('url');args=a.parse_args();fetch(args.tag,args.url)
