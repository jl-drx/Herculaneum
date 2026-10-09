import email.message,hashlib,http.client,importlib.util,json,pathlib,socket,tempfile
ROOT=pathlib.Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('bounded_fetch',ROOT/'fetch_text.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def network_forbidden(*a,**k):raise AssertionError('network forbidden')
socket.socket.connect=network_forbidden;socket.socket.connect_ex=network_forbidden
class Fake:
 def __init__(self,b,status=200,headers=None,partial=False):
  self.b=b;self.status=status;self.headers=email.message.Message();self.partial=partial
  for k,v in (headers or {}).items():self.headers[k]=v
 def read(self,n):
  if self.partial:self.partial=False;raise http.client.IncompleteRead(self.b[:min(3,n)],99)
  a,self.b=self.b[:n],self.b[n:];return a
out=ROOT/'fetch-fixtures';out.mkdir(exist_ok=False);cases=[]
def check(name,fn,reject=False):
 try:fn()
 except (m.Stop,UnicodeDecodeError,json.JSONDecodeError):
  if not reject:raise
 else:
  if reject:raise AssertionError(name+' unexpectedly accepted')
 cases.append({'name':name,'status':'PASS'})
def read(name,b,**kw):
 return m.body_read(Fake(b,**kw),{'maximum_reserved_body_bytes':16,'expected_status':200},out/(name+'.body'),{})
check('exact content length',lambda:read('ok',b'abc',headers={'Content-Length':'3'}))
check('unknown length EOF',lambda:read('unknown',b'abc'))
check('oversized declared length',lambda:read('large',b'',headers={'Content-Length':'16'}),True)
check('unknown length reaches cap',lambda:read('cap',b'x'*16),True)
check('truncated declared',lambda:read('truncated',b'abc',headers={'Content-Length':'4'}),True)
check('incomplete partial retained',lambda:read('partial',b'abcdef',partial=True),True)
assert (out/'partial.body').read_bytes()==b'abc'
check('nonidentity encoding',lambda:read('encoding',b'a',headers={'Content-Encoding':'gzip'}),True)
check('failed HTTP status body retained',lambda:read('status',b'not found',status=404),True)
assert (out/'status.body').read_bytes()==b'not found'
check('invalid content length',lambda:read('badlength',b'a',headers={'Content-Length':'abc'}),True)
check('HTML rejected',lambda:m.parse_text(b'<html>bad</html>',{'kind':'UTF-8 source text'}),True)
check('UTF8 rejected',lambda:m.parse_text(b'\xff',{'kind':'UTF-8 source text'}),True)
check('JSON object',lambda:m.parse_text(b'{"scale":[1,1]}',{'kind':'JSON metadata'}))
check('JSON nonobject rejected',lambda:m.parse_text(b'[]',{'kind':'JSON metadata'}),True)
prefix='PHerc0139/segments/20250108000004-w029_2025010827/mesh/20250108000004-on-20260102150214-2.399um.tifxyz/'
xml=f'<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/"><Prefix>{prefix}</Prefix><Delimiter>/</Delimiter><IsTruncated>false</IsTruncated><Contents><Key>{prefix}meta.json</Key></Contents></ListBucketResult>'.encode()
check('complete exact inventory',lambda:m.parse_text(xml,{'kind':'XML object-name inventory'}))
check('truncated inventory',lambda:m.parse_text(xml.replace(b'false',b'true'),{'kind':'XML object-name inventory'}),True)
check('XML entity rejected',lambda:m.parse_text(b'<!DOCTYPE x>'+xml,{'kind':'XML object-name inventory'}),True)
assert m.NoRedirect().redirect_request(None,None,None,None,None,None) is None
cases.append({'name':'redirect refusal','status':'PASS'})
receipt={'status':'PASS','scope':'Synthetic text/transport only; socket connections forbidden; no model/scientific data','source':m.record(ROOT/'fetch_text.py'),'fixture':m.record(pathlib.Path(__file__)),'cases':cases,'case_count':len(cases)}
m.write(out/'receipt.json',receipt);print(json.dumps({'receipt':m.record(out/'receipt.json'),'case_count':len(cases)}))
