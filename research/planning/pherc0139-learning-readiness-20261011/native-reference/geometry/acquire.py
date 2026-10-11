import argparse,datetime,hashlib,json,urllib.request
from pathlib import Path
P=Path(__file__).resolve().parent
class NoRedirect(urllib.request.HTTPRedirectHandler):
 def redirect_request(self,*args,**kwargs):return None

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def main(stage):
 plan=json.loads((P/'SOURCE_PLAN.json').read_text());assert sha(P/'PROTOCOL.md')==plan['protocol_sha256'];opener=urllib.request.build_opener(NoRedirect());folder=P/'responses';folder.mkdir(exist_ok=True)
 if stage=='metadata':
  out=P/'METADATA_LEDGER.json';assert not out.exists();rows=[]
  for req in plan['requests']:
   method='HEAD' if req['kind']=='coordinate' else 'GET';row=dict(req,method=method,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());rows.append(row)
   try:
    with opener.open(urllib.request.Request(req['url'],method=method),timeout=30) as r:
     assert r.status==200;row.update(status=r.status,etag=r.headers.get('ETag'),length=int(r.headers['Content-Length']))
     body=b'' if method=='HEAD' else r.read(65537);row['body_bytes']=len(body);assert len(body)<=65536
     if method=='GET':assert len(body)==row['length'];(folder/req['name']).write_bytes(body);row['sha256']=hashlib.sha256(body).hexdigest()
   except Exception as e:row['error']=repr(e);save(out,rows);raise
   save(out,rows)
  assert sum(r['length'] for r in rows)<plan['new_body_cap_bytes'];print('metadata_complete',sum(r['body_bytes'] for r in rows),'coordinate_bytes',sum(r['length'] for r in rows if r['method']=='HEAD'))
 elif stage=='coordinates':
  metadata=json.loads((P/'METADATA_LEDGER.json').read_text());byname={r['name']:r for r in metadata};rows=[];out=P/'COORDINATE_LEDGER.json';assert not out.exists();total=sum(r['body_bytes'] for r in metadata)
  save(P/'PAYLOAD_FREEZE.json',dict(protocol_sha256=sha(P/'PROTOCOL.md'),source_plan_sha256=sha(P/'SOURCE_PLAN.json'),metadata_ledger_sha256=sha(P/'METADATA_LEDGER.json'),acquire_code_sha256=sha(Path(__file__)),requests=[dict(r,etag=byname[r['name']]['etag'],exact_bytes=byname[r['name']]['length']) for r in plan['requests'] if r['kind']=='coordinate']))
  for req in plan['requests']:
   if req['kind']!='coordinate':continue
   head=byname[req['name']];row=dict(req,method='GET',started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());rows.append(row);path=folder/req['name'];assert not path.exists()
   try:
    assert total+head['length']<=plan['new_body_cap_bytes']
    with opener.open(urllib.request.Request(req['url'],headers={'If-Match':head['etag']}),timeout=30) as r:
     assert r.status==200 and r.headers['ETag']==head['etag'] and int(r.headers['Content-Length'])==head['length']
     body=r.read(head['length']+1);total+=len(body);row.update(status=r.status,etag=r.headers['ETag'],bytes=len(body));assert len(body)==head['length'];path.write_bytes(body);row['sha256']=sha(path)
   except Exception as e:row['error']=repr(e);save(out,rows);raise
   save(out,rows);print(req['name'],len(body),flush=True)
  save(P/'ACQUISITION_COMPLETE.json',dict(status='PASS',all_response_body_bytes=total,cap=plan['new_body_cap_bytes'],coordinate_files=len(rows),metadata_files=3,coordinate_ledger_sha256=sha(out)))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['metadata','coordinates']);main(p.parse_args().stage)
