"""Bounded public document retrieval; no scan chunks or authenticated requests."""
from pathlib import Path
from datetime import datetime,timezone
import concurrent.futures,hashlib,json,urllib.request
from html.parser import HTMLParser
ROOT=Path(__file__).resolve().parent
SOURCES={
 'topography':'https://www.nature.com/articles/s41598-026-58467-1',
 'distillation':'https://arxiv.org/abs/1511.03643',
 'data_catalog':'https://scrollprize.org/data',
 'ink_tutorial':'https://scrollprize.org/tutorial5',
 'canonical_model':'https://huggingface.co/scrollprize/ink_canonical_2um/raw/main/README.md',
}
class Text(HTMLParser):
 def __init__(self):super().__init__();self.text=[];self.ignore=0
 def handle_starttag(self,tag,attrs):
  if tag in ['script','style']:self.ignore+=1
  if tag in ['p','h1','h2','h3','tr','li','br']:self.text.append('\n')
 def handle_endtag(self,tag):
  if tag in ['script','style']:self.ignore=max(0,self.ignore-1)
 def handle_data(self,s):
  if not self.ignore:self.text.append(s)
def fetch(item):
 name,url=item
 try:
  req=urllib.request.Request(url,headers={'User-Agent':'Plinius-research-source-audit/1.0'})
  with urllib.request.urlopen(req,timeout=25) as response:
   body=response.read(3*1024*1024+1)
   if len(body)>3*1024*1024:raise ValueError('Document size cap')
   status=response.status;final=response.url;ctype=response.headers.get('Content-Type','')
  (ROOT/(name+'.raw')).write_bytes(body)
  raw=body.decode('utf-8','replace')
  if 'html' in ctype:
   parser=Text();parser.feed(raw);raw=''.join(parser.text)
  (ROOT/(name+'.txt')).write_text(raw)
  return dict(name=name,url=url,final_url=final,status=status,bytes=len(body),sha256=hashlib.sha256(body).hexdigest(),retrieved_at_utc=datetime.now(timezone.utc).isoformat())
 except Exception as e:return dict(name=name,url=url,error=type(e).__name__+': '+str(e))
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:receipts=list(pool.map(fetch,SOURCES.items()))
 (ROOT/'SOURCES.json').write_text(json.dumps(receipts,indent=2)+'\n');print(json.dumps(receipts,indent=2))
