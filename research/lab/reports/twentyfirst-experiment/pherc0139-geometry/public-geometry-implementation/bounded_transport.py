"""Fixed public single-range transport with retained conservative reservations."""
import hashlib,http.client,json,os,pathlib,signal,ssl,time,urllib.error,urllib.request
from geometry_kernels import ContractError,require
class NoRedirect(urllib.request.HTTPRedirectHandler):
 def redirect_request(self,*args,**kwargs):return None
def record(p):
 b=p.read_bytes();return {'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'size_bytes':len(b)}
def write_json(path,value):
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');os.replace(tmp,path)
def expired(*args):raise ContractError('absolute request deadline')
def validate_response(response,request):
 source=request['source'];lo,hi=request['range_inclusive'];require(response.status==206,'expected206; no fullbodyfallback')
 for key in ['Content-Range','Content-Length','Content-Encoding','ETag']:
  require(len(response.headers.get_all(key,[]))<=1,'duplicate '+key)
 require(response.headers.get('Content-Range')==f'bytes {lo}-{hi}/{source["object_size_bytes"]}','wrong Content-Range')
 require(response.headers.get('ETag')==source['strong_etag'] and not source['strong_etag'].startswith('W/'),'changed/weak ETag')
 require(response.headers.get('Content-Encoding','identity').lower()=='identity','nonidentity encoding')
 if response.headers.get('Content-Length') is not None:require(response.headers['Content-Length']==str(request['expected_body_bytes']),'wrong Content-Length')
def read_bounded(response,request,path,row):
 expected=request['expected_body_bytes'];cap=request['reserved_body_bytes'];require(cap==expected+1,'exact range reservation');count=0
 with path.open('xb') as out:
  while count<cap:
   maximum=min(65536,cap-count)
   try:data=response.read(maximum)
   except http.client.IncompleteRead as e:
    require(len(e.partial)<=maximum,'partial exceeds bounded read');out.write(e.partial);count+=len(e.partial);row['retained_body_bytes']=count;raise ContractError('truncated response with partial bytes') from None
   if not data:break
   require(len(data)<=maximum,'response exceeded requested read');out.write(data);count+=len(data);row['retained_body_bytes']=count
 require(count==expected,'truncated/oversized range body');row['body']=record(path);return path
class Client:
 def __init__(self,plan,out,guard,opener=None):
  self.plan=plan;self.out=out;self.guard=guard;self.ledger={'status':'RUNNING','requests':[],'reserved_bytes_all_attempts':0,'charged_bytes':0};self.opener=opener or urllib.request.build_opener(NoRedirect(),urllib.request.HTTPSHandler(context=ssl.create_default_context()))
 def save(self):write_json(self.out/'transport-ledger.json',self.ledger)
 def fetch(self,request):
  self.guard();index=len(self.ledger['requests']);require(index<6 and request==self.plan['requests'][index],'request outside frozen six or wrong order');cap=request['reserved_body_bytes'];require(self.ledger['reserved_bytes_all_attempts']+cap<=4194304,'total reservation cap')
  source=request['source'];lo,hi=request['range_inclusive'];require(8<=lo<=hi<8109512 and hi-lo+1==request['expected_body_bytes'],'range not fixed coordinate strip');row={'id':request['id'],'url':source['url'],'range_inclusive':[lo,hi],'if_match':source['strong_etag'],'reserved_body_bytes':cap,'status':'RESERVED','retained_body_bytes':0};self.ledger['requests'].append(row);self.ledger['reserved_bytes_all_attempts']+=cap;self.ledger['charged_bytes']+=cap;self.save();self.guard()
  path=self.out/(request['id']+'.body');start=time.monotonic();old=signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,30)
  try:
   req=urllib.request.Request(source['url'],headers={'Range':f'bytes={lo}-{hi}','If-Match':source['strong_etag'],'Accept-Encoding':'identity','User-Agent':'PHerc0139-public-geometry/1'},method='GET')
   try:r=self.opener.open(req,timeout=30)
   except urllib.error.HTTPError as e:r=e
   with r:
    row['http_status']=r.status;row['response_headers']={k:r.headers.get(k) for k in ['Content-Range','Content-Length','Content-Encoding','ETag'] if r.headers.get(k) is not None};require(r.geturl()==source['url'],'URL changed');validate_response(r,request);read_bounded(r,request,path,row)
   row['status']='PASS';self.ledger['charged_bytes']-=cap-row['retained_body_bytes'];return path
  except BaseException as e:
   row['status']='STOP';row['error_type']=type(e).__name__;row['stop_reason']=str(e) if isinstance(e,ContractError) else type(e).__name__
   if path.exists():row['partial_body']=record(path);row['retained_body_bytes']=path.stat().st_size
   raise
  finally:
   signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old);row['elapsed_seconds']=time.monotonic()-start;self.save()
