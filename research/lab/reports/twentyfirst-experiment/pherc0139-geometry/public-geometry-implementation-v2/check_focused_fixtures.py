"""Only additive durability/framing correction checks; no real network/coordinates."""
import email.message,json,pathlib,socket,stat
import geometry_kernels as K
import bounded_transport as B
ROOT=pathlib.Path(__file__).resolve().parent;OUT=ROOT/'focused-fixtures';OUT.mkdir(exist_ok=False)
def forbidden(*a,**kw):raise AssertionError('network forbidden')
socket.socket.connect=forbidden;socket.socket.connect_ex=forbidden
plan=json.loads((ROOT.parent/'public-geometry-plan/COORDINATE_QUERY_PLAN.json').read_text());q=plan['requests'][0];cases=[]
def check(name,fn,bad=False):
 try:fn()
 except (K.ContractError,OSError):
  if not bad:raise
 else:
  if bad:raise AssertionError(name+' accepted')
 cases.append({'name':name,'status':'PASS'})
class Response:
 def __init__(self,change=None,remove=None):
  self.status=206;self.headers=email.message.Message();self.left=q['expected_body_bytes'];lo,hi=q['range_inclusive'];headers={'Content-Length':str(self.left),'Content-Range':f'bytes {lo}-{hi}/{q["source"]["object_size_bytes"]}','ETag':q['source']['strong_etag']};headers.update(change or {})
  for k,v in headers.items():
   if k!=remove:self.headers[k]=v
 def geturl(self):return q['source']['url']
 def read(self,n):size=min(n,self.left);self.left-=size;return b'\0'*size
 def __enter__(self):return self
 def __exit__(self,*a):return False
check('exact Content-Length206 accepted',lambda:B.validate_response(Response(),q))
check('missing Content-Length rejected',lambda:B.validate_response(Response(remove='Content-Length'),q),True)
check('wrong Content-Length rejected',lambda:B.validate_response(Response({'Content-Length':'0'}),q),True)
check('nondecimal Content-Length rejected',lambda:B.validate_response(Response({'Content-Length':'456624x'}),q),True)
r=Response();r.headers['Content-Length']=str(q['expected_body_bytes']);check('duplicate identical Content-Length rejected',lambda:B.validate_response(r,q),True)
for encoding in ['chunked','identity','']:
 check('Transfer-Encoding rejected '+repr(encoding),lambda e=encoding:B.validate_response(Response({'Transfer-Encoding':e}),q),True)
# Record real fsync ordering and ensure opener cannot run before file/replace/directory.
events=[];original_sync=B.os.fsync;original_replace=B.os.replace
class Opener:
 def __init__(self,folder):self.folder=folder;self.called=0
 def open(self,req,timeout):
  self.called+=1;assert events[:3]==['file_fsync','replace','directory_fsync'];assert events[-1]=='directory_fsync';ledger=json.loads((self.folder/'transport-ledger.json').read_text());assert ledger['requests'][0]['status']=='RESERVED' and ledger['reserved_bytes_all_attempts']==456625;events.append('open');return Response()
def spy_sync(fd):events.append('directory_fsync' if stat.S_ISDIR(B.os.fstat(fd).st_mode) else 'file_fsync');return original_sync(fd)
def spy_replace(a,b):events.append('replace');return original_replace(a,b)
out=OUT/'ordered';out.mkdir();opener=Opener(out)
B.os.fsync=spy_sync;B.os.replace=spy_replace
try:
 client=B.Client(plan,out,lambda:None,opener);client.fetch(q);assert opener.called==1 and events[:4]==['file_fsync','replace','directory_fsync','open'];assert client.ledger['charged_bytes']==456624
finally:B.os.fsync=original_sync;B.os.replace=original_replace
cases.append({'name':'durable file-replace-directory reservation precedes opener','status':'PASS'})
# Any file or directory durability failure must abort before opener.
for kind in ['file','directory']:
 folder=OUT/('failure-'+kind);folder.mkdir();opener=Opener(folder);client=B.Client(plan,folder,lambda:None,opener)
 def failure(fd,kind=kind):
  is_dir=stat.S_ISDIR(B.os.fstat(fd).st_mode)
  if is_dir==(kind=='directory'):raise OSError('synthetic durability failure')
  return original_sync(fd)
 B.os.fsync=failure
 try:
  check(kind+' fsync failure stops before opener',lambda:client.fetch(q),True);assert opener.called==0
 finally:B.os.fsync=original_sync
# The correction must leave geometry, driver and the broad original fixture code unchanged.
v1=ROOT.parent/'public-geometry-implementation'
for name in ['geometry_kernels.py','run_public_geometry.py','check_fixtures.py']:
 assert (ROOT/name).read_bytes()==(v1/name).read_bytes()
cases.append({'name':'geometry driver broad-fixture source byte-identical to v1','status':'PASS'})
receipt={'status':'PASS','case_count':len(cases),'cases':cases,'scope':'Focused fake transport/durable writes only; sockets forbidden; no real coordinates/network','source':B.record(ROOT/'bounded_transport.py'),'fixture':B.record(pathlib.Path(__file__)),'previous_code_freeze':B.record(v1/'CODE_FREEZE.json'),'previous_37_case_receipt':B.record(v1/'fixtures-v1/receipt.json'),'events':events}
B.write_json(OUT/'receipt.json',receipt);print(json.dumps({'cases':len(cases),'receipt':B.record(OUT/'receipt.json')}))
