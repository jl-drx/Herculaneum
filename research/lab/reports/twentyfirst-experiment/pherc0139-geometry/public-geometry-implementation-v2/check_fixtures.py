"""Only synthetic coordinates and fake-response transport; sockets forbidden."""
import copy,email.message,http.client,json,math,pathlib,signal,socket,struct,time,types
import numpy as np
import geometry_kernels as K
import bounded_transport as B
import run_public_geometry as R
ROOT=pathlib.Path(__file__).resolve().parent;OUT=ROOT/'fixtures-v1';OUT.mkdir(exist_ok=False)
def forbidden(*a,**k):raise AssertionError('network forbidden')
socket.socket.connect=forbidden;socket.socket.connect_ex=forbidden
plan=json.loads((ROOT.parent/'public-geometry-plan/COORDINATE_QUERY_PLAN.json').read_text());cases=[]
def test(name,fn,fail=False):
 try:fn()
 except K.ContractError:
  if not fail:raise
 else:
  if fail:raise AssertionError(name+' accepted')
 cases.append({'name':name,'status':'PASS'})
def yes(x):
 if not x:raise AssertionError('fixture comparison')
def xyz_window(context,tilt=False):
 x,y,w,h=context['mesh_window_xywh'];Y,X=np.mgrid[y:y+h,x:x+w];return np.stack([X+1000,Y+1000,500+3*X+4*Y if tilt else np.full_like(X,500)],axis=2).astype(np.float32)
scale=[.05000000074505806]*2;dims=[26511,26511,76953]
for c in plan['contexts']:
 xyz=xyz_window(c);mesh,bounds,count=K.reconstruct(xyz,c,scale,dims)
 test(c['id']+' full144tiles planar/allfine/all68',lambda m=mesh,b=bounds,n=count:yes(m.shape==(384,384) and m.dtype==bool and m.all() and b.all() and n['mesh_and_bounds_valid']==2359296 and n['level2_total']==147456))
# Independent scalar staged-F32 parity on a tilted synthetic surface.
def f(x):return struct.unpack('<f',struct.pack('<f',float(x)))[0]
def scalar(P,px,py,wx,wy,nx,ny):
 p00,p01,p10,p11=[P[y,x].tolist() for x,y in [(px,py),(px+1,py),(px,py+1),(px+1,py+1)]];ix=f(1-wx);iy=f(1-wy);base=[]
 for j in range(3):top=f(f(p00[j]*ix)+f(p01[j]*wx));bot=f(f(p10[j]*ix)+f(p11[j]*wx));base.append(f(f(top*iy)+f(bot*wy)))
 u=[f(float(P[ny,nx+1,j])-float(P[ny,nx-1,j])) for j in range(3)];v=[f(float(P[ny+1,nx,j])-float(P[ny-1,nx,j])) for j in range(3)];n=[f(f(u[1]*v[2])-f(u[2]*v[1])),f(f(u[2]*v[0])-f(u[0]*v[2])),f(f(u[0]*v[1])-f(u[1]*v[0]))];l=f(f(f(n[0]*n[0])+f(n[1]*n[1]))+f(n[2]*n[2]));inv=f(1/f(math.sqrt(l)));d=[f(-f(k*inv)) for k in n];l=f(f(f(d[0]*d[0])+f(d[1]*d[1]))+f(d[2]*d[2]));root=f(math.sqrt(l));return np.array(base,dtype=np.float32),np.array([f(k/root) for k in d],dtype=np.float32)
c=plan['contexts'][0];P=xyz_window(c,True);valid=K.valid_vertices(P);norm,nok=K.normal_cache(P,valid);a=K.axis_map(1444,28880,scale[0],7168,1536);b=K.axis_map(1404,28080,scale[1],19456,1536);base,direction,pok,geom=K.fine_tile(P,valid,norm,nok,a,b,c['mesh_window_xywh'],0,0)
for yy,xx in [(0,0),(1,3),(37,99),(127,127)]:
 px=int(a['floor'][xx])-357;py=int(b['floor'][yy])-972;nx=int(a['nearest'][xx])-357;ny=int(b['nearest'][yy])-972;sb,sd=scalar(P,px,py,float(a['weight'][xx]),float(b['weight'][yy]),nx,ny)
 test('scalar bitwise parity '+str((yy,xx)),lambda sb=sb,sd=sd,y=yy,x=xx:yes(np.array_equal(sb.view(np.uint32),base[y,x].view(np.uint32)) and np.array_equal(sd.view(np.uint32),direction[y,x].view(np.uint32))))
for name,index,value in [('x sentinel',(10,10,0),-1),('zeroz',(10,10,2),0),('NaN',(10,10,1),np.nan),('Inf',(10,10,1),np.inf)]:
 Q=P.copy();Q[index]=value;v=K.valid_vertices(Q);nn,ok=K.normal_cache(Q,v);test(name+' invalid vertex and neighbor normal',lambda v=v,o=ok:yes(not v[10,10] and not o[10,11]))
Q=P.copy();Q[:,:,0]=1000;Q[:,:,2]=500;v=K.valid_vertices(Q);_,ok=K.normal_cache(Q,v);test('degenerate normals',lambda:yes(not ok.any()))
rng=np.random.default_rng(210139);bb=rng.uniform(-50,150,(64,32,3)).astype(np.float32);dd=rng.uniform(-1,1,bb.shape).astype(np.float32);all68=np.stack([K.F(bb+K.F(dd*K.F(o))) for o in range(-33,35)]);full=(np.isfinite(all68).all(axis=(0,3))&((all68>=0)&(all68<99)).all(axis=(0,3)))
test('2048 all68 versus endpoints',lambda:yes(np.array_equal(K.endpoint_support(bb,dd,[100]*3),full)))
for name,value,expected in [('zero',0,True),('lastcorner',99,False),('belowlast',np.nextafter(np.float32(99),np.float32(0)),True),('negativeulp',np.nextafter(np.float32(0),np.float32(-1)),False)]:
 q=np.array([[[value,50,50]]],dtype=np.float32);test('endpoint '+name,lambda q=q,e=expected:yes(bool(K.endpoint_support(q,np.zeros_like(q),[100]*3)[0,0])==e))
# Gap bytes are synthetic NaN-like 0xff, and remain outside every decoded slice.
req=plan['requests'][0];raw=bytearray([255]*req['expected_body_bytes'])
for j in range(80):raw[j*5776:j*5776+320]=np.full(80,j+.25,dtype='<f4').tobytes()
window=K.decode_window(raw,req);test('opaque gaps excluded',lambda:yes(np.isfinite(window).all() and window[79,79]==79.25))
test('wrong range length rejected',lambda:K.decode_window(raw[:-1],req),True)
# Production plan cannot expand roles, windows or quota.
for name,key,value in [('quota','maximum_attempts',7),('bodycap','maximum_total_reserved_response_body_bytes',8388608)]:
 q=copy.deepcopy(plan);q[key]=value;test('tampered '+name,lambda q=q:R.verify_plan(q),True)
q=copy.deepcopy(plan);q['contexts'][0]['mesh_window_xywh'][0]-=1;test('support expansion rejected',lambda:R.verify_plan(q),True)
class Response:
 def __init__(self,req,status=206,change=None,partial=False,oversize=False,short=False,opaque_error=False):
  self.req=req;self.status=status;self.headers=email.message.Message();lo,hi=req['range_inclusive'];self.remaining=req['expected_body_bytes']+(1 if oversize else -1 if short else 0);self.partial=partial;self.opaque_error=opaque_error
  d={'Content-Range':f'bytes {lo}-{hi}/{req["source"]["object_size_bytes"]}','Content-Length':str(req['expected_body_bytes']),'ETag':req['source']['strong_etag']};d.update(change or {})
  for k,v in d.items():self.headers[k]=v
 def geturl(self):return self.req['source']['url']
 def read(self,n):
  if self.partial:self.partial=False;raise http.client.IncompleteRead(b'abc',99)
  if self.opaque_error:raise OSError('DO_NOT_RETAIN_ARBITRARY_EXCEPTION')
  amount=min(n,self.remaining);self.remaining-=amount;return b'\0'*amount
 def __enter__(self):return self
 def __exit__(self,*a):return False
class FakeOpener:
 def __init__(self,reqs,out,opts=None):self.reqs=reqs;self.out=out;self.opts=opts or {};self.i=0
 def open(self,request,timeout):
  row=json.loads((self.out/'transport-ledger.json').read_text())['requests'][-1];assert row['status']=='RESERVED' and row['reserved_body_bytes']==456625;assert 0<signal.getitimer(signal.ITIMER_REAL)[0]<=30;req=self.reqs[self.i];self.i+=1;assert request.headers['If-match']==req['source']['strong_etag'];return Response(req,**self.opts)
def fake_case(name,options=None,success=False):
 out=OUT/name;out.mkdir();client=B.Client(plan,out,lambda:None,FakeOpener(plan['requests'],out,options))
 try:client.fetch(plan['requests'][0])
 except (K.ContractError,OSError):
  if success:raise
 else:
  if not success:raise AssertionError('failed response accepted')
 row=client.ledger['requests'][0]
 if success:assert row['status']=='PASS' and client.ledger['charged_bytes']==456624
 else:assert row['status']=='STOP' and client.ledger['charged_bytes']==456625
 if options and options.get('partial'):assert row['retained_body_bytes']==3 and pathlib.Path(row['partial_body']['path']).read_bytes()==b'abc'
 assert 'DO_NOT_RETAIN_ARBITRARY_EXCEPTION' not in (out/'transport-ledger.json').read_text()
for name,options in [('ignoredRange',{'status':200}),('wrongrange',{'change':{'Content-Range':'bytes 0-7/8'}}),('wronglength',{'change':{'Content-Length':'1'}}),('changedETag',{'change':{'ETag':'"changed"'}}),('encoded',{'change':{'Content-Encoding':'gzip'}}),('partial',{'partial':True}),('extraByte',{'oversize':True}),('truncated',{'short':True}),('opaqueError',{'opaque_error':True})]:test('transport '+name,lambda n=name,o=options:fake_case(n,o))
test('transport exact206',lambda:fake_case('exact206',success=True))
out=OUT/'six';out.mkdir();opener=FakeOpener(plan['requests'],out);client=B.Client(plan,out,lambda:None,opener)
for q in plan['requests']:client.fetch(q)
test('exact six reservation totals',lambda:yes(len(client.ledger['requests'])==6 and client.ledger['reserved_bytes_all_attempts']==2739750 and client.ledger['charged_bytes']==2739744))
test('seventh request rejected before open',lambda:client.fetch(plan['requests'][0]),True)
test('redirect disabled',lambda:yes(B.NoRedirect().redirect_request(None,None,None,None,None,None) is None))
class FakeOut:
 def __init__(self,size):self.size=size
 def rglob(self,*a):return [self]
 def is_file(self):return True
 def stat(self):return types.SimpleNamespace(st_size=self.size)
test('output cap',lambda:R.budget(FakeOut(33554433),time.monotonic()),True)
test('wall cap',lambda:R.budget(FakeOut(0),time.monotonic()-231),True)
old=R.resource.getrusage;R.resource.getrusage=lambda *_:types.SimpleNamespace(ru_maxrss=524289)
try:test('RSS cap',lambda:R.budget(FakeOut(0),time.monotonic()),True)
finally:R.resource.getrusage=old
receipt={'status':'PASS','case_count':len(cases),'cases':cases,'scope':'Only synthetic XYZ/index arrays and fake HTTP; real socket connections forbidden; no retained coordinate/nativeCT/model/label/SV arrays read','sources':[B.record(ROOT/n) for n in ['geometry_kernels.py','bounded_transport.py','run_public_geometry.py','check_fixtures.py']],'runtime':R.runtime()};B.write_json(OUT/'receipt.json',receipt);print(json.dumps({'receipt':B.record(OUT/'receipt.json'),'cases':len(cases)}))
