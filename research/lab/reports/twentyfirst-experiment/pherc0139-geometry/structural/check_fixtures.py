import email.message,json,pathlib,socket,struct
import tiff_metadata as T
import run_structural as R
ROOT=pathlib.Path(__file__).resolve().parent;out=ROOT/'fixtures';out.mkdir(exist_ok=False)
def forbidden(*a,**k):raise AssertionError('network forbidden')
socket.socket.connect=forbidden;socket.socket.connect_ex=forbidden
cases=[]
def check(name,fn,bad=False):
 try:fn()
 except (T.ContractError,ValueError):
  if not bad:raise
 else:
  if bad:raise AssertionError(name+' accepted')
 cases.append({'name':name,'status':'PASS'})
def fixture(order='<',change=None,external=False):
 tags={256:(4,1,10),257:(4,1,8),258:(3,1,32),259:(3,1,1),262:(3,1,1),273:(4,2 if external else 1,300 if external else 512),277:(3,1,1),278:(4,1,4 if external else 8),279:(4,2 if external else 1,308 if external else 320),284:(3,1,1),339:(3,1,3)}
 if change:tags.update(change)
 rows=[]
 for tag,(typ,count,value) in tags.items():
  # Deliberately nonzero unused padding discriminates big-endian SHORT decode.
  field=struct.pack(order+'H',value)+b'\xa5\x5a' if typ==3 and count==1 else struct.pack(order+'I',value)
  rows.append(struct.pack(order+'HHI',tag,typ,count)+field)
 body=b''.join(rows)+struct.pack(order+'I',0);head=T.header((b'II' if order=='<' else b'MM')+struct.pack(order+'HI',42,8),1024);n=T.count(struct.pack(order+'H',len(rows)),head);doc=T.parse_ifd(body,head,n);lay=T.layout(doc)
 extra={273:struct.pack(order+'II',512,672),279:struct.pack(order+'II',160,160)} if external else {}
 return T.finish(doc,lay,extra)
for order in ('<','>'):
 check(order+' inline SHORT with dirty padding',lambda o=order:fixture(o))
 check(order+' external strip metadata',lambda o=order:fixture(o,external=True))
 assert fixture(order)['width']==10 and fixture(order)['bits_per_sample']==32
check('wrong magic BigTIFF',lambda:T.header(b'II'+struct.pack('<HI',43,8),1024),True)
check('invalid endian',lambda:T.header(b'XX'+struct.pack('<HI',42,8),1024),True)
check('IFD header overlap',lambda:T.header(b'II'+struct.pack('<HI',42,4),1024),True)
check('IFD beyond object',lambda:T.header(b'II'+struct.pack('<HI',42,1024),1024),True)
check('too many IFD entries',lambda:T.count(struct.pack('<H',257),{'byte_order':'<','first_ifd_offset':8,'total_bytes':99999}),True)
check('truncated IFD count',lambda:T.count(b'\x01',{'byte_order':'<'}),True)
check('IFD extent outside file',lambda:T.count(struct.pack('<H',256),{'byte_order':'<','first_ifd_offset':8,'total_bytes':100}),True)
for name,change in [('zero width',{256:(4,1,0)}),('nondefault orientation',{274:(3,1,2)}),('multichannel',{277:(3,1,3)}),('planar',{284:(3,1,2)}),('unsupported sample format',{339:(3,1,4)}),('bad bits',{258:(3,1,12)}),('zero rows per strip',{278:(4,1,0)}),('wrong bytecount',{279:(4,1,319)}),('pixel overlaps IFD',{273:(4,1,10)}),('pixel extent outside object',{273:(4,1,1000)}),('wrong cardinality',{273:(4,2,300)}),('unknown TIFF type',{262:(99,1,1)}),('ambiguous tile strip',{324:(4,1,512),325:(4,1,320)})]:check(name,lambda c=change:fixture(change=c),True)
def parse_duplicate():
 raw=struct.pack('<HHII',256,4,1,10);T.parse_ifd(raw+raw+b'\0'*4,{'byte_order':'<','first_ifd_offset':8,'total_bytes':1024},2)
check('duplicate tags',parse_duplicate,True)
def parse_multipage():
 T.parse_ifd(struct.pack('<HHII',256,4,1,10)+struct.pack('<I',512),{'byte_order':'<','first_ifd_offset':8,'total_bytes':1024},1)
check('multipage',parse_multipage,True)
def parse_overlap():
 T.parse_ifd(struct.pack('<HHII',273,4,2,8)+b'\0'*4,{'byte_order':'<','first_ifd_offset':8,'total_bytes':1024},1)
check('external metadata overlap',parse_overlap,True)
class Response:
 def __init__(self,status=206,**change):
  self.status=status;self.headers=email.message.Message();d={'Content-Range':'bytes 0-7/1024','Content-Length':'8','ETag':'"same"'};d.update(change)
  for k,v in d.items():self.headers[k]=v
obj={'object_size_bytes':1024,'strong_etag':'"same"'}
check('exact206',lambda:T.validate_http(Response(),obj,0,7))
for name,resp in [('ignored Range200',Response(200)),('wrong range',Response(**{'Content-Range':'bytes 1-8/1024'})),('wrong total',Response(**{'Content-Range':'bytes 0-7/1025'})),('wrong length',Response(**{'Content-Length':'9'})),('changed ETag',Response(ETag='"changed"')),('weak ETag',Response(ETag='W/"same"')),('encoded range',Response(**{'Content-Encoding':'gzip'}))]:check(name,lambda r=resp:T.validate_http(r,obj,0,7),True)
r=Response();r.headers['ETag']='"same"';check('duplicate ETag',lambda:T.validate_http(r,obj,0,7),True)
# Exercise parent reservation and STOP paths before any opener can be called.
m=R.load_helper(R.rec(ROOT.parent/'fetch_text.py'))
a=R.Acquisition({},out,m);a.ledger['reserved_bytes_all_attempts']=262144
check('reserve before opener global cap',lambda:a.get({'object_size_bytes':1024},'guard',0,8,9),True)
a.ledger['reserved_bytes_all_attempts']=0;a.ledger['requests']=[{}]*16
check('attempt cap before opener',lambda:a.get({'object_size_bytes':1024},'guard',0,8,9),True)
# Full fake one-request transport ensures truncation retains partial and reservation.
import http.client
class FakeFull(Response):
 def __init__(self,partial=False):super().__init__();self.data=b'II'+struct.pack('<HI',42,8);self.partial=partial
 def geturl(self):return 'https://example.invalid/x.tif'
 def read(self,n):
  if self.partial:self.partial=False;raise http.client.IncompleteRead(self.data[:3],8)
  b,self.data=self.data[:n],self.data[n:];return b
 def __enter__(self):return self
 def __exit__(self,*args):return False
class Opener:
 def __init__(self,partial=False):self.partial=partial
 def open(self,request,timeout):
  assert request.headers['If-match']=='"same"' and request.headers['Range']=='bytes=0-7';return FakeFull(self.partial)
fakeobj={'name':'x.tif','url':'https://example.invalid/x.tif',**obj}
for partial in (False,True):
 d=out/('partial' if partial else 'success');d.mkdir();a=R.Acquisition({},d,m);a.opener=Opener(partial)
 if partial:
  try:a.get(fakeobj,'header',0,8,9)
  except m.Stop:pass
  else:raise AssertionError('partial accepted')
  assert a.ledger['charged_bytes']==9 and a.ledger['requests'][0]['retained_body_bytes']==3 and a.ledger['requests'][0]['status']=='STOP'
 else:assert len(a.get(fakeobj,'header',0,8,9))==8 and a.ledger['charged_bytes']==8
 cases.append({'name':'fake HTTP partial retained/full charged' if partial else 'fake HTTP exact206/reservation release','status':'PASS'})
receipt={'status':'PASS','scope':'Pure synthetic TIFF metadata and fake HTTP, sockets forbidden; no coordinate/CT/model/scientific array decode','cases':cases,'count':len(cases),'sources':[R.rec(ROOT/n) for n in ('tiff_metadata.py','run_structural.py','check_fixtures.py')],'sealed_transport_helper':R.rec(ROOT.parent/'fetch_text.py')};m.write(out/'receipt.json',receipt);print(json.dumps({'count':len(cases),'receipt':R.rec(out/'receipt.json')}))
