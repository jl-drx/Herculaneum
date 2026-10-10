"""Exact public Git blobs only, with no execution of downloaded source."""
import base64,hashlib,json,signal,ssl,time,urllib.request,urllib.error
from pathlib import Path

HERE=Path(__file__).resolve().parent
TREE=HERE.parent/'frag6-public-reference-recheck-20261009/fragment-benchmark-tree.body'
NAMES=['common.py','score_3d.py','step4_train_compare.py','step5_lofo.py','export_predictions.py',
       'predict_fragment.py','step1_ink_depth_profile.py','step2_surface_aligned_profile.py','requirements.txt','LICENSE']

def pin(path):
 p=Path(path);return dict(path=str(p.resolve()),size_bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def save(path,value):Path(path).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
class NoRedirect(urllib.request.HTTPRedirectHandler):
 def redirect_request(self,*args,**kwargs):return None

def main():
 go=json.loads((HERE/'ACQUISITION_GO.json').read_text())
 assert go['status']=='GO_TEXT_BLOBS_ONLY'
 for item in go['files']:assert pin(item['path'])==item
 root=HERE/'acquisition-v1';root.mkdir(exist_ok=False)
 tree=json.loads(TREE.read_text());assert tree['sha']=='191126e0e677b4cf17131a1f7bbaefb1fa8f92e1' and not tree['truncated']
 ledger=dict(requests=[],reserved_body_bytes=0,returned_body_bytes=0,redirects=0,retries=0)
 opener=urllib.request.build_opener(NoRedirect(),urllib.request.HTTPSHandler(context=ssl.create_default_context()))
 for name in NAMES:
  item=next(t for t in tree['tree'] if t['path']==name and t['type']=='blob')
  assert item['url']=='https://api.github.com/repos/kartoun/vesuvius-fragment-ink-depth/git/blobs/'+item['sha']
  cap=131072;assert len(ledger['requests'])<12 and ledger['reserved_body_bytes']+cap+1<=2*1024**2
  entry=dict(name=name,url=item['url'],expected_git_blob_sha1=item['sha'],expected_decoded_bytes=item['size'],success=False)
  ledger['requests'].append(entry);ledger['reserved_body_bytes']+=cap+1;save(root/'ledger.json',ledger)
  body=b'';response=None;start=time.monotonic()
  def expired(*_):raise TimeoutError('20second absolute deadline')
  previous=signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,20)
  try:
   try:response=opener.open(urllib.request.Request(item['url'],headers={'User-Agent':'Herculaneum-source-audit','Accept-Encoding':'identity'}),timeout=15)
   except urllib.error.HTTPError as error:response=error
   entry['status']=response.status
   assert response.headers.get('Content-Type','').split(';')[0] in ['application/json','application/vnd.github+json']
   assert response.headers.get('Content-Encoding','identity')=='identity'
   body=response.read(cap+1);assert len(body)<=cap and response.status==200
   obj=json.loads(body);assert obj['encoding']=='base64' and obj['sha']==item['sha']
   source=base64.b64decode(obj['content']);source.decode('utf-8');assert len(source)==item['size']==obj['size']
   assert hashlib.sha1(b'blob '+str(len(source)).encode()+b'\0'+source).hexdigest()==item['sha']
   path=root/(name+'.source');path.write_bytes(source);entry['source']=pin(path);entry['success']=True
  except Exception as error:
   entry['error']=type(error).__name__+': '+str(error);raise
  finally:
   signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
   if response:response.close()
   path=root/(name+'.response.json');path.write_bytes(body);entry['response']=pin(path)
   entry['elapsed_seconds']=time.monotonic()-start;ledger['returned_body_bytes']+=len(body);save(root/'ledger.json',ledger)
  print(json.dumps(dict(name=name,source_bytes=len(source),response_bytes=len(body),status='VERIFIED')),flush=True)
 for item in go['files']:assert pin(item['path'])==item
 save(root/'SOURCE_RECEIPT.json',dict(status='PASS',tree=pin(TREE),requests=len(ledger['requests']),returned_body_bytes=ledger['returned_body_bytes'],reserved_body_bytes=ledger['reserved_body_bytes'],sources=[r['source'] for r in ledger['requests']],candidate_pixels=0,executed_source=False))

if __name__=='__main__':main()
